import argparse
import torch
import os
import torch.distributed as dist

os.environ["NCCL_DEBUG"] = "ERROR"  # NCCL debug messages: ERROR, WARN, INFO, DEBUG


def init_process():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())


def destroy_process():
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================


def example_broadcast():
    if dist.get_rank() == 0:
        tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32).cuda()
    else:
        # NOTE: 3 -> 5 -> 7
        # tensor = torch.zeros(3, dtype=torch.float32).cuda()
        tensor = torch.zeros(5, dtype=torch.float32).cuda()
        # tensor = torch.zeros(7, dtype=torch.float32).cuda()
    print(f"Before broadcast on rank {dist.get_rank()}: {tensor}")

    dist.broadcast(tensor, src=0)
    # dist.broadcast(tensor, src=1) # if src=1, rank 0 will receive zeros
    print(f"After broadcast on rank {dist.get_rank()}: {tensor}")


def example_reduce():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()
    print(f"Before reduce on rank {dist.get_rank()}: {tensor}")

    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    # dist.reduce(tensor, dst=1, op=dist.ReduceOp.AVG)
    print(f"After reduce on rank {dist.get_rank()}: {tensor}")


def example_all_reduce():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()
    print(f"Before all_reduce on rank {dist.get_rank()}: {tensor}")

    # NOTE: In All-Reduce, all ranks will have the same result after the operation.
    # So, we don't need to specify a destination rank.
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"After all_reduce on rank {dist.get_rank()}: {tensor}")


def example_gather():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()
    if dist.get_rank() == 0:
        gather_list = [torch.zeros(5, dtype=torch.float32).cuda() for _ in range(dist.get_world_size())]
    else:
        gather_list = None
    print(f"Before gather on rank {dist.get_rank()}: {tensor}")

    # NOTE: tensor: torch.Tensor,
    # NOTE: gather_list: Optional[list[torch.Tensor]] = None,
    dist.gather(tensor, gather_list, dst=0)
    print(f"After gather on rank {dist.get_rank()}: {gather_list}")


def example_all_gather():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()
    gather_list = [torch.zeros(5, dtype=torch.float32).cuda() for _ in range(dist.get_world_size())]

    print(f"Before all_gather on rank {dist.get_rank()}: {tensor}")
    dist.all_gather(gather_list, tensor)
    print(f"After all_gather on rank {dist.get_rank()}: {gather_list}")


# def example_all_gather_variable():
#     tensor = torch.tensor([dist.get_rank() + 1] * (dist.get_rank() + 1), dtype=torch.float32).cuda()
#     gather_list = [torch.zeros(i + 1, dtype=torch.float32).cuda() for i in range(dist.get_world_size())]

#     print(f"Before all_gather_variable on rank {dist.get_rank()}: {tensor}")
#     dist.all_gather(gather_list, tensor)
#     print(f"After all_gather_variable on rank {dist.get_rank()}: {gather_list}")


def example_scatter():
    if dist.get_rank() == 0:
        scatter_list = [torch.tensor([i + 1] * 5, dtype=torch.float32).cuda() for i in range(dist.get_world_size())]
        print(f"Rank 0: Tensor to scatter: {scatter_list}")
    else:
        scatter_list = None
    tensor = torch.zeros(5, dtype=torch.float32).cuda()
    print(f"Before scatter on rank {dist.get_rank()}: {tensor}")

    # NOTE: tensor: torch.Tensor,
    # NOTE: scatter_list: Optional[list[torch.Tensor]] = None,
    dist.scatter(tensor, scatter_list, src=0)
    print(f"After scatter on rank {dist.get_rank()}: {tensor}")


def example_reduce_scatter():
    """This example demonstrates the Reduce-Scatter operation, which combines the reduction and scattering steps.

    input_tensor: list[torch.Tensor]
        Rank 0: [tensor([1., 2.]), tensor([1., 4.])]
        Rank 1: [tensor([2., 4.]), tensor([4., 16.])]

    output_tensor: torch.Tensor
        Rank 0: tensor([3., 6.])
        Rank 1: tensor([5., 20.])
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    input_tensor = [
        torch.tensor([(rank + 1) * i for i in range(1, 3)], dtype=torch.float32).cuda() ** (j + 1)
        for j in range(world_size)
    ]
    # input_tensor = [
    #     torch.tensor([(rank + 1) * i for i in range(1, 3)], dtype=torch.float32).cuda() ** (j + 1)
    #     for j in range(world_size + 1)
    # ]
    output_tensor = torch.zeros(2, dtype=torch.float32).cuda()
    print(f"Before ReduceScatter on rank {rank}: {input_tensor}")

    dist.reduce_scatter(output_tensor, input_tensor, op=dist.ReduceOp.SUM)
    print(f"After ReduceScatter on rank {rank}: {output_tensor}")


def example_all_to_all():
    """This example demonstrates the All-to-All operation, where each rank sends a unique tensor to every other rank and receives a unique tensor from every other rank.

    input_list (2 GPUs):
        Rank 0: [tensor([1.]), tensor([2.])]   → chunk[0]은 rank0에게, chunk[1]은 rank1에게
        Rank 1: [tensor([2.]), tensor([4.])]   → chunk[0]은 rank0에게, chunk[1]은 rank1에게

    output_list:
        Rank 0: [tensor([1.]), tensor([2.])]   ← rank0에서 온 것, rank1에서 온 것
        Rank 1: [tensor([2.]), tensor([4.])]   ← rank0에서 온 것, rank1에서 온 것
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    input_list = [torch.tensor([(rank + 1) * (i + 1)], dtype=torch.float32).cuda() for i in range(world_size)]
    output_list = [torch.zeros(1, dtype=torch.float32).cuda() for _ in range(world_size)]
    print(f"Before All-to-All on rank {rank}: {input_list}")

    dist.all_to_all(output_list, input_list)
    print(f"After All-to-All on rank {rank}: {output_list}")


def ring_all_reduce_from_scratch():
    """The Ring All-Reduce algorithm consists of two main phases: the Reduce-Scatter phase and the All-Gather phase.

    In the Reduce-Scatter phase, each rank sends a portion of its data to the next rank in a ring topology while receiving a portion of data from the previous rank. This process is repeated until all ranks have contributed to the reduction.

    In the All-Gather phase, each rank sends its reduced portion of data to all other ranks, allowing each rank to gather the complete reduced result.

    Note: This implementation assumes that the input tensor is divisible by the number of ranks for simplicity.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # e.g., rank 0: [1, 1, 1, 1], rank 1: [2, 2, 2, 2], rank 2: [3, 3, 3, 3], rank 3: [4, 4, 4, 4]
    tensor = torch.tensor([rank + 1] * (world_size * 2), dtype=torch.float32).cuda()
    print(f"Before Ring All-Reduce on rank {rank}: {tensor}")

    # Chunk the tensor into equal parts for each rank (assuming tensor size is divisible by world_size)
    chunks = list(tensor.chunk(world_size))

    left = (rank - 1) % world_size  # previous rank (receive from)
    right = (rank + 1) % world_size  # next rank (send to)

    # ================================================================
    # Phase 1: Reduce-Scatter (N-1 라운드)
    #   각 라운드에서: 자기 chunk를 right로 보내고, left에서 받아서 누적 합산
    #   결과: 각 rank가 자기 담당 chunk의 전체 합(SUM)을 가짐
    # ================================================================

    print(f"After Reduce-Scatter on rank {rank}: {chunks}")

    # ================================================================
    # Phase 2: All-Gather (N-1 라운드)
    #   각 라운드에서: 완성된 chunk를 right로 보내고, left에서 받아서 저장
    #   결과: 모든 rank가 전체 합산 결과를 가짐
    # ================================================================

    print(f"After All-Gather on rank {rank}: {chunks}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Distributed Example")
    parser.add_argument("--example", type=int, default=1, help="Example to run")
    args = parser.parse_args()

    init_process()

    match args.example:
        case 1:
            print("A0.1 Broadcast")
            example_broadcast()
        case 2:
            print("A0.2 Reduce")
            example_reduce()
        case 3:
            print("A0.3 All-Reduce")
            example_all_reduce()
        case 4:
            print("A0.4 Gather")
            example_gather()
        case 5:
            print("A0.5 All-Gather")
            example_all_gather()
        case 6:
            print("A0.6 Scatter")
            example_scatter()
        case 7:
            print("A0.7 Reduce-Scatter")
            example_reduce_scatter()
    destroy_process()
