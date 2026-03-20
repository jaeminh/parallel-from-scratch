import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


def example_mesh_1d():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    mesh_1d = init_device_mesh("cuda", mesh_shape=(world_size,))

    if rank == 0:
        print("=== 1D DeviceMesh ===")
        print(f"mesh.mesh = {mesh_1d.mesh}")
        print(f"mesh.mesh.shape = {mesh_1d.mesh.shape}")
        print(f"mesh_dim_names = {mesh_1d.mesh_dim_names}")
    dist.barrier(device_ids=[rank])

    # Each rank's coordinate in the mesh
    coord = mesh_1d.get_coordinate()
    local_rank = mesh_1d.get_local_rank()
    print(f"  [Rank {rank}] coordinate={coord}, local_rank={local_rank}")

    # Get the ProcessGroup for the single dimension
    pg = mesh_1d.get_group()
    print(f"  [Rank {rank}] ProcessGroup ranks={dist.get_process_group_ranks(pg)}")

    # ── Use the mesh's ProcessGroup for a collective ──
    tensor = torch.tensor([rank], dtype=torch.float32, device="cuda")
    dist.all_reduce(tensor, group=pg)
    # Expected: 0+1+2+3+4+5+6+7 = 28
    print(f"  [Rank {rank}] all_reduce result = {tensor.item():.0f} (expected 28)")

    dist.barrier(device_ids=[rank])
    dist.destroy_process_group()


def example_mesh_2d():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    mesh_2d = init_device_mesh("cuda", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    if rank == 0:
        print("\n=== 2D DeviceMesh ===")
        print(f"mesh.mesh =\n{mesh_2d.mesh}")
        print(f"mesh.mesh.shape = {mesh_2d.mesh.shape}")
        print(f"mesh_dim_names = {mesh_2d.mesh_dim_names}")
    dist.barrier(device_ids=[rank])

    # Coordinate and local rank per dimension
    coord = mesh_2d.get_coordinate()
    dp_local = mesh_2d.get_local_rank("dp")
    tp_local = mesh_2d.get_local_rank("tp")
    print(f"  [Rank {rank}] coord={coord}, dp_local={dp_local}, tp_local={tp_local}")

    # ProcessGroup per dimension
    dp_pg = mesh_2d.get_group("dp")
    tp_pg = mesh_2d.get_group("tp")
    print(
        f"  [Rank {rank}] dp_group={dist.get_process_group_ranks(dp_pg)}, "
        f"tp_group={dist.get_process_group_ranks(tp_pg)}"
    )
    dist.barrier(device_ids=[rank])

    # ── dim 0 (dp) collective: column-wise all-reduce ──
    # dp groups: {0,4}, {1,5}, {2,6}, {3,7}
    tensor_dp = torch.tensor([rank], dtype=torch.float32, device="cuda")
    dist.all_reduce(tensor_dp, group=dp_pg)
    if rank == 0:
        print("\n--- dim 0 (dp) all_reduce: column-wise ---")
    dist.barrier(device_ids=[rank])
    # Rank 0+4=4, Rank 1+5=6, Rank 2+6=8, Rank 3+7=10
    print(f"  [Rank {rank}] dp all_reduce = {tensor_dp.item():.0f}")
    dist.barrier(device_ids=[rank])

    # ── dim 1 (tp) collective: row-wise all-reduce ──
    # tp groups: {0,1,2,3}, {4,5,6,7}
    tensor_tp = torch.tensor([rank], dtype=torch.float32, device="cuda")
    dist.all_reduce(tensor_tp, group=tp_pg)
    if rank == 0:
        print("\n--- dim 1 (tp) all_reduce: row-wise ---")
    dist.barrier(device_ids=[rank])
    # Row 0: 0+1+2+3=6, Row 1: 4+5+6+7=22
    print(f"  [Rank {rank}] tp all_reduce = {tensor_tp.item():.0f}")

    dist.barrier(device_ids=[rank])
    dist.destroy_process_group()


if __name__ == "__main__":
    example_mesh_1d()
    # example_mesh_2d()
