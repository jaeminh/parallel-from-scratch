import argparse
import torch
import torch.nn as nn
import os
import torch.distributed as dist

os.environ["NCCL_DEBUG"] = "ERROR"


# ============================================================
# Router
# ============================================================
class UniformRouter(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, x):
        num_tokens = x.shape[0]
        expert_weights = torch.full((num_tokens, self.num_experts), 1.0 / self.num_experts, device=x.device)
        expert_indices = torch.randint(0, self.num_experts, (num_tokens,), device=x.device)
        return expert_indices, expert_weights


class Router(nn.Module):
    """Learned Router: Linear layer의 가중치로 Expert를 선택한다."""

    def __init__(self, embed_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(embed_dim, num_experts, bias=False)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        logits = self.gate(x)  # [num_tokens, num_experts]
        expert_weights = torch.softmax(logits, dim=-1)
        expert_indices = torch.argmax(expert_weights, dim=-1)  # Top-1 선택
        return expert_indices, expert_weights


# ============================================================
# Expert: FFN (Feed-Forward Network)
# ============================================================
class Expert(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.net(x)


# ============================================================
def run(step, router_type="uniform"):
    rank = dist.get_rank()  # current process
    world_size = dist.get_world_size()  # total number of processes (GPUs)

    experts_per_gpu = NUM_EXPERTS // world_size

    # ===========================================================
    # Step 1: Generate same data on all GPUs & Local flattening
    # ===========================================================
    torch.manual_seed(42)
    data = torch.randint(0, 10, (BATCH_SIZE, SEQ_LEN, EMBED_DIM)).float().cuda()
    local_tokens = data[rank]  # [SEQ_LEN, EMBED_DIM]
    num_local_tokens = local_tokens.shape[0]

    print(f"[Rank {rank}] {local_tokens.tolist()}")

    if step <= 1:
        dist.destroy_process_group()
        return

    # ===========================================================
    # Step 2: Define Router & Expert, Assign Experts
    # ===========================================================
    torch.manual_seed(123)
    if router_type == "uniform":
        router = UniformRouter(NUM_EXPERTS).cuda()
    else:
        router = Router(EMBED_DIM, NUM_EXPERTS).cuda()

    # Assign Experts to GPUs
    local_expert_ids = list(range(rank * experts_per_gpu, (rank + 1) * experts_per_gpu))
    local_expert_modules = nn.ModuleList()
    for expert_id in local_expert_ids:
        torch.manual_seed(456 + expert_id)
        local_expert_modules.append(Expert(EMBED_DIM, HIDDEN_DIM).cuda())
    # print(f"[Rank {rank}] {local_expert_ids=}")

    # ===========================================================
    # 2-1: Forward pass through Router to get expert assignment
    # ===========================================================
    expert_indices, router_probs = router(local_tokens)
    gpu_indices = expert_indices // experts_per_gpu

    print(f"[Rank {rank}] Expert ID that Router assigned: {expert_indices.tolist()}")
    print(f"[Rank {rank}] GPU indices for each token: {gpu_indices.tolist()}")

    if step <= 2:
        dist.destroy_process_group()
        return

    # ===========================================================
    # Step 3: Prepare All-to-All (send/recv counts exchange)
    # ===========================================================
    # TODO
    # 3-1) send_counts 계산: gpu_indices를 이용해 각 GPU로 보낼 토큰 수를 세기
    #      send_counts[i] = 이 Rank에서 GPU i로 보낼 토큰 수
    #
    # 3-2) recv_counts 교환: all_to_all_single으로 send_counts를 교환하여 recv_counts 얻기
    #      recv_counts[i] = GPU i로부터 이 Rank가 받을 토큰 수
    # ===========================================================
    send_counts = torch.zeros(1)
    recv_counts = torch.zeros(1)

    print(f"[Rank {rank}] Tokens to send: {send_counts.tolist()}")
    print(f"[Rank {rank}] Tokens to receive: {recv_counts.tolist()}")

    dist.barrier()
    if step <= 3:
        dist.destroy_process_group()
        return

    # ===========================================================
    # Step 4: Sort tokens by GPU index (for efficient All-to-All)
    # ===========================================================
    # NOTE: all_to_all_single은 send_splits 순서대로 텐서를 연속(contiguous) 슬라이싱하여 전송한다.
    #   예) send_splits=[3,2] → 앞 3개는 GPU0으로, 뒤 2개는 GPU1로
    #   따라서 토큰이 목적지 GPU 순서대로 정렬되어 있어야 올바르게 전송된다.
    sorted_indices = torch.argsort(expert_indices)
    sorted_tokens = local_tokens[sorted_indices]
    sorted_expert_ids = expert_indices[sorted_indices]

    if rank == 0:
        print(f"[Rnk {rank}]")
        print(f"expert_indices:    {expert_indices.tolist()}")
        print(f"gpu_indices:       {gpu_indices.tolist()}")
        print(f"sorted_indices:    {sorted_indices.tolist()}")
        print()
        print(f"local_tokens:      {local_tokens.tolist()}")
        print(f"sorted_tokens:     {sorted_tokens.tolist()}")
        print(f"sorted_expert_ids: {sorted_expert_ids.tolist()}")
        print()
        print(f"send_counts:       {send_counts.tolist()}")
        print(f"recv_counts:       {recv_counts.tolist()}")

    if step <= 4:
        dist.destroy_process_group()
        return

    # ===========================================================
    # Step 5: All-to-All Dispatch
    # ===========================================================
    # TODO
    # 5-1) split 크기 준비: send_counts, recv_counts를 list[int]로 변환
    #      all_to_all_single의 split 인자는 list[int] 형태여야 합니다.
    #
    # 5-2) 토큰 데이터 전송: sorted_tokens → recv_tensor
    #      recv_tensor 크기 = (sum(recv_splits), EMBED_DIM)
    #      힌트: dist.all_to_all_single(recv_tensor, sorted_tokens, recv_splits, send_splits)
    #
    # 5-3) Expert ID 전송: sorted_expert_ids → recv_expert_ids
    #      받는 GPU에서 어떤 Expert로 처리할지 알아야 하므로 Expert ID도 함께 전송
    #      힌트: sorted_expert_ids는 long이지만 all_to_all_single은 float 텐서만 지원
    # ===========================================================
    send_splits = send_counts.tolist()
    recv_splits = recv_counts.tolist()

    dist.barrier()
    print()
    if step <= 5:
        dist.destroy_process_group()
        return

    # ===========================================================
    # Step 6: Forward through Experts
    # NOTE: Remove lines below after implementing Step 5
    # ===========================================================
    # expert_output = torch.zeros_like(recv_tensor)
    # for local_i, expert_id in enumerate(local_expert_ids):
    #     mask = recv_expert_ids == expert_id
    #     if mask.any():
    #         expert_input = recv_tensor[mask]
    #         expert_out = local_expert_modules[local_i](expert_input)
    #         expert_output[mask] = expert_out

    # print(f"[Rank {rank}] Expert output: {expert_output.tolist()}")

    if step <= 6:
        dist.destroy_process_group()
        return

    # ===========================================================
    # Step 7: All-to-All Combine + Unsort + Gating weight
    # ===========================================================
    # TODO
    # 7-1) All-to-All로 Expert 출력을 원래 GPU로 반환: expert_output → result_recv
    #      result_recv 크기 = (num_local_tokens, EMBED_DIM)
    #      힌트: Step 5의 Dispatch와 반대 방향. send_splits/recv_splits의 역할이 뒤바뀝니다.
    #
    # 7-2) 원래 토큰 순서로 복원: result_recv → final_output
    #      Step 4에서 argsort로 정렬했으므로, sorted_indices를 역으로 적용하면 복원 가능
    #      힌트: final_output[sorted_indices] = result_recv
    # ===========================================================
    result_recv = torch.zeros(1)
    final_output = torch.zeros(1)

    if rank == 0:
        print(f"[Rank {rank}]")
        print(f"result_recv:       {result_recv.tolist()}")
        print(f"sorted_indices:    {sorted_indices.tolist()}")
        print(f"final_output:      {final_output.tolist()}")

    # ===========================================================
    # 7-3: Gating weight
    # NOTE: Remove lines below after implementing Step 7
    # ===========================================================
    # selected_weights = router_probs[torch.arange(num_local_tokens, device="cuda"), expert_indices]
    # final_output = final_output * selected_weights.unsqueeze(-1)
    # final_output = final_output.view(1, SEQ_LEN, EMBED_DIM)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoE Expert Parallel Study (Step by Step)")
    parser.add_argument("--step", type=int, default=7, help="Step to run (1-7)")
    parser.add_argument("--router", type=str, default="uniform", choices=["uniform", "gate"])
    args = parser.parse_args()

    # initialize process group & set device
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

    # Configuration (global constants)
    BATCH_SIZE = dist.get_world_size()  # mbs = 1
    SEQ_LEN = 10  # sequence length
    EMBED_DIM = 2  # embedding dimension

    NUM_EXPERTS = 4  # total number of experts (should be divisible by world_size)
    HIDDEN_DIM = 8  # hidden dimension inside each expert

    if args.step < 1 or args.step > 7:
        print("Invalid step. Please choose a step between 1 and 7.")
    else:
        run(args.step, args.router)

    # destroy process group
    if dist.is_initialized():
        dist.destroy_process_group()
