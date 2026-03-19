import torch
import torch.distributed as dist
import torch.nn as nn
from utils import Simple8LayerModel, create_local_data


# =============================================================================
# ZeRO-1: Optimizer State Partitioning (비교용)
# =============================================================================
class ZeRO1Optimizer:
    def __init__(self, model: nn.Module, lr: float, world_size: int, rank: int):
        self.model = model
        self.world_size = world_size
        self.rank = rank
        self.all_params = list(model.parameters())

        self.total_numel = sum(p.numel() for p in self.all_params)
        self.chunk_size = (self.total_numel + world_size - 1) // world_size
        self.padded_numel = self.chunk_size * world_size

        device = self.all_params[0].device
        dtype = self.all_params[0].dtype

        self.flat_param = torch.zeros(self.padded_numel, device=device, dtype=dtype)
        self._offsets = []
        offset = 0
        for p in self.all_params:
            numel = p.numel()
            self._offsets.append(offset)
            self.flat_param[offset : offset + numel].copy_(p.data.view(-1))
            p.data = self.flat_param[offset : offset + numel].view(p.shape)
            offset += numel

        self.flat_grad = torch.zeros(self.padded_numel, device=device, dtype=dtype)
        for idx, p in enumerate(self.all_params):
            p.grad = self.flat_grad[self._offsets[idx] : self._offsets[idx] + p.numel()].view(p.shape)

        self.grad_chunk = torch.zeros(self.chunk_size, device=device, dtype=dtype)

        start = rank * self.chunk_size
        self.param_chunk = nn.Parameter(self.flat_param[start : start + self.chunk_size].clone())
        self.optimizer = torch.optim.Adam([self.param_chunk], lr=lr)

    def step(self):
        dist.reduce_scatter_tensor(self.grad_chunk, self.flat_grad, op=dist.ReduceOp.SUM)
        self.grad_chunk /= self.world_size

        self.param_chunk.grad = self.grad_chunk
        self.optimizer.step()

        dist.all_gather_into_tensor(self.flat_param, self.param_chunk.data)

    def zero_grad(self):
        self.flat_grad.zero_()


# =============================================================================
# ZeRO-2: Optimizer State + Gradient Partitioning
#
# ZeRO-1과의 차이:
#   ZeRO-1: step()에서 reduce-scatter (backward 끝난 후)
#   ZeRO-2: backward 중 hook으로 async reduce-scatter (통신-연산 overlap)
#           → reduce-scatter 후 flat_grad 해제 가능 (gradient 메모리 1/N)
#
# 메모리 비교:
#   DP:     params N + grads N + optimizer 2N     = 4N
#   ZeRO-1: params N + grads N + optimizer 2N/W   = 2N + 2N/W
#   ZeRO-2: params N + grads N/W + optimizer 2N/W = N + 3N/W
# =============================================================================
class ZeRO2Optimizer:
    def __init__(self, model: nn.Module, lr: float, world_size: int, rank: int):
        self.model = model
        self.world_size = world_size
        self.rank = rank
        self.all_params = list(model.parameters())

        # === flat buffer 인프라 (ZeRO-1과 동일) ===
        self.total_numel = sum(p.numel() for p in self.all_params)
        self.chunk_size = (self.total_numel + world_size - 1) // world_size
        self.padded_numel = self.chunk_size * world_size

        device = self.all_params[0].device
        dtype = self.all_params[0].dtype

        # flat_param + view (ZeRO-1과 동일)
        self.flat_param = torch.zeros(self.padded_numel, device=device, dtype=dtype)
        self._offsets = []
        offset = 0
        for p in self.all_params:
            numel = p.numel()
            self._offsets.append(offset)
            self.flat_param[offset : offset + numel].copy_(p.data.view(-1))
            p.data = self.flat_param[offset : offset + numel].view(p.shape)
            offset += numel

        # flat_grad + view (ZeRO-1과 동일)
        self.flat_grad = torch.zeros(self.padded_numel, device=device, dtype=dtype)
        for idx, p in enumerate(self.all_params):
            p.grad = self.flat_grad[self._offsets[idx] : self._offsets[idx] + p.numel()].view(p.shape)

        # grad_chunk, param_chunk, optimizer (ZeRO-1과 동일)
        self.grad_chunk = torch.zeros(self.chunk_size, device=device, dtype=dtype)
        start = rank * self.chunk_size
        self.param_chunk = nn.Parameter(self.flat_param[start : start + self.chunk_size].clone())
        self.optimizer = torch.optim.Adam([self.param_chunk], lr=lr)

        # === ZeRO-2 추가: backward 중 async reduce-scatter ===
        self._grad_count = 0
        self._rs_handle: dist.Work | None = None
        for p in self.all_params:
            p.register_post_accumulate_grad_hook(self._grad_hook)

    def _grad_hook(self, p: nn.Parameter):
        """param의 grad가 준비될 때마다 호출. 모든 grad가 모이면 async reduce-scatter 시작."""
        self._grad_count += 1
        if self._grad_count == len(self.all_params):
            # 모든 grad가 flat_grad에 쌓임 → 바로 reduce-scatter 시작 (async)
            self._rs_handle = dist.reduce_scatter_tensor(
                self.grad_chunk, self.flat_grad, op=dist.ReduceOp.SUM, async_op=True
            )

    def step(self):
        # (1) reduce-scatter 완료 대기 (hook에서 시작됨)
        self._rs_handle.wait()
        self._rs_handle = None
        self._grad_count = 0
        self.grad_chunk /= self.world_size

        # ★ ZeRO-2 핵심: reduce-scatter 후 flat_grad 해제 가능 ★
        # 각 rank는 grad_chunk (N/W)만 있으면 됨
        # (실제 구현에서는 여기서 flat_grad를 free하고 다음 backward 때 재할당)
        # self.flat_grad = None  # 개념적으로 해제

        # (2) 자기 chunk만 optimizer로 업데이트 (ZeRO-1과 동일)
        self.param_chunk.grad = self.grad_chunk
        self.optimizer.step()

        # (3) all-gather → flat_param 갱신 (ZeRO-1과 동일)
        dist.all_gather_into_tensor(self.flat_param, self.param_chunk.data)

    def zero_grad(self):
        self.flat_grad.zero_()


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    loss_fn = nn.CrossEntropyLoss()
    num_steps = 5

    # =================================================================
    # (1) ZeRO-1로 학습 (비교 기준)
    # =================================================================
    with torch.device("meta"):
        model1 = Simple8LayerModel()
    model1.to_empty(device=device)
    for m in model1.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    for p in model1.parameters():
        dist.broadcast(p.data, src=0)
    init_state = {k: v.clone() for k, v in model1.state_dict().items()}

    zero1_opt = ZeRO1Optimizer(model1, lr=0.001, world_size=world_size, rank=rank)

    for step in range(num_steps):
        x, y = create_local_data(rank + step * world_size)
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model1(x), y)
        zero1_opt.zero_grad()
        loss.backward()
        zero1_opt.step()

    zero1_params = {name: p.data.clone() for name, p in model1.named_parameters()}
    del model1, zero1_opt

    # =================================================================
    # (2) ZeRO-2로 학습
    # =================================================================
    with torch.device("meta"):
        model2 = Simple8LayerModel()
    model2.to_empty(device=device)
    model2.load_state_dict(init_state)

    zero2_opt = ZeRO2Optimizer(model2, lr=0.001, world_size=world_size, rank=rank)

    for step in range(num_steps):
        x, y = create_local_data(rank + step * world_size)
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model2(x), y)
        zero2_opt.zero_grad()
        loss.backward()  # ← hook에서 async reduce-scatter 자동 시작
        zero2_opt.step()

        if rank == 0:
            print(f"  Step {step}: loss = {loss.item():.4f}")

    # =================================================================
    # 검증: ZeRO-1 vs ZeRO-2 결과 비교
    # =================================================================
    if rank == 0:
        print()
    all_match = True
    for name, p2 in model2.named_parameters():
        p1 = zero1_params[name]
        if torch.allclose(p1, p2.data):
            if rank == 0:
                print(f"  ✓ {name}: ZeRO-1 == ZeRO-2")
        else:
            all_match = False
            diff = (p1 - p2.data).abs().max().item()
            if rank == 0:
                print(f"  ✗ {name}: 불일치 (max diff = {diff:.6e})")

    if rank == 0 and all_match:
        print("  → ZeRO-1과 ZeRO-2의 결과가 완전히 동일합니다")

    dist.destroy_process_group()
