import argparse
import torch
import os
import torch.distributed as dist

os.environ["NCCL_DEBUG"] = "ERROR"


def init_process():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())


def destroy_process():
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================
def example_all_to_all_single():
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Rank0: [0, 1, 2, 3]
    # Rank1: [4, 5, 6, 7]
    # Rank2: [8, 9, 10, 11]
    # ...
    local_tensor = torch.arange(rank * 4, (rank + 1) * 4, device="cuda")
    print(f"[Rank {rank}] Local tensor: {local_tensor.tolist()}")

    recv_tensor = torch.zeros_like(local_tensor)
    print(f"[Rank {rank}] Before All-to-All, recv_tensor: {recv_tensor.tolist()}")
    dist.all_to_all_single(recv_tensor, local_tensor)
    # print(f"[Rank {rank}] After All-to-All, local_tensor: {local_tensor.tolist()}")
    print(f"[Rank {rank}] Received tensor after All-to-All: {recv_tensor.tolist()}")


def example_all_to_all_single_failed():
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Rank0: [0, 1, 2, 3, 4]
    # Rank1: [5, 6, 7, 8, 9]
    # Rank2: [10, 11, 12, 13, 14
    # ...
    local_tensor = torch.arange(rank * 4, (rank + 1) * 4 + 1, device="cuda")
    print(f"[Rank {rank}] Local tensor: {local_tensor.tolist()}")

    recv_tensor = torch.zeros_like(local_tensor)
    dist.all_to_all_single(recv_tensor, local_tensor)
    print(f"[Rank {rank}] Received tensor after All-to-All: {recv_tensor.tolist()}")


def example_all_to_all_single_with_split_sizes():
    """Example when world_size=2.

    [Rank 0] Local tensor: [0, 1, 2, 3, 4]
    [Rank 1] Local tensor: [10, 11, 12, 13, 14, 15, 16]

    [Rank 0] Input splits (tokens to send to each rank): [3, 2]
    [Rank 1] Input splits (tokens to send to each rank): [4, 3]

    [Rank 0] Output splits (tokens to receive from each rank): [3, 4]
    [Rank 1] Output splits (tokens to receive from each rank): [2, 3]

    [Rank 0] Received tensor after All-to-All: [0, 1, 2, 10, 11, 12, 13]
    [Rank 1] Received tensor after All-to-All: [3, 4, 14, 15, 16]

    Rank 0 send [3, 4] to Rank 1, while Rank 1 send [10, 11, 12, 13] to Rank 0.
    Because the second element of output splits is 4 for Rank 0, it means Rank 0 will receive 4 tokens from Rank 1, which are [10, 11, 12, 13].
    Then Rank 0 will concatenate the 3 tokens it keeps ([0, 1, 2]) with the 4 tokens it receives from Rank 1 to form the final received tensor [0, 1, 2, 10, 11, 12, 13].
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Rank0: [0, 1, 2, 3, 4]
    # Rank1: [10, 11, 12, 13, 14, 15, 16]
    # Rank2: [20, 21, 22, 23, 24, 25, 26, 27, 28]
    # ...
    local_size = (rank * 7) % 5 + 5
    local_tensor = torch.arange(rank * 10, rank * 10 + local_size, device="cuda")
    print(f"[Rank {rank}] Local tensor: {local_tensor.tolist()}")

    base = local_size // world_size
    remainder = local_size % world_size
    input_splits = torch.tensor(
        [base + (1 if i < remainder else 0) for i in range(world_size)], dtype=torch.long, device="cuda"
    )
    print(f"[Rank {rank}] Input splits (tokens to send to each rank): {input_splits.tolist()}")

    # exchange split sizes
    output_splits = torch.zeros(world_size, dtype=torch.long, device="cuda")
    dist.all_to_all_single(output_splits, input_splits)
    print(f"[Rank {rank}] Output splits (tokens to receive from each rank): {output_splits.tolist()}")

    # Because the number of tokens to receive from each rank can be different, we need to allocate a tensor that can hold all incoming tokens.
    recv_tensor = torch.zeros(output_splits.sum().item(), dtype=local_tensor.dtype, device="cuda")
    dist.all_to_all_single(recv_tensor, local_tensor, output_splits.tolist(), input_splits.tolist())
    print(f"[Rank {rank}] Received tensor after All-to-All: {recv_tensor.tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Distributed Example")
    parser.add_argument("--example", type=int, default=1, help="Example to run")
    args = parser.parse_args()

    init_process()

    match args.example:
        case 1:
            example_all_to_all_single()
        case 2:
            example_all_to_all_single_failed()
        case 3:
            example_all_to_all_single_with_split_sizes()
        case _:
            print(f"Invalid example number: {args.example}")

    destroy_process()
