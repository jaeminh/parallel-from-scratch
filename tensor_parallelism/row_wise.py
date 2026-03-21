import torch
import torch.distributed as dist

if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # =============================================
    bsz = 4  # batch size
    in_features = 16
    out_features = 64  # out_features should be divisible by world_size for row-wise parallelism

    torch.manual_seed(42)
    X = torch.randn(bsz, in_features, device=device)  # [bsz, in_features]
    W = torch.randn(in_features, out_features, device=device)  # [in_features, out_features]
    # Y will be [bsz, out_features]

    Y_normal = X @ W

    # =============================================
    # Row-wise Parallel MatMul
    # =============================================
    shard_size = in_features // world_size

    # scatter X to all processes
    X_row = X.clone()
    X_shard = torch.empty(bsz, shard_size, device=device)

    if rank == 0:
        scatter_list = [chunk.contiguous() for chunk in X_row.chunk(world_size, dim=1)]
    else:
        scatter_list = None
    dist.scatter(X_shard, scatter_list, src=0)  # [bsz, shard_size (in_features/world_size)]

    # matmul local shard
    W_shard = W[rank * shard_size : (rank + 1) * shard_size, :].contiguous()
    Y_local = X_shard @ W_shard  # [bsz, out_features]

    # all_reduce Y_local from all processes
    dist.all_reduce(Y_local, op=dist.ReduceOp.SUM)
    Y_row = Y_local

    dist.barrier()

    # =============================================
    is_close = torch.allclose(Y_normal, Y_row, atol=1e-5)

    if rank == 0:
        print(f"Row-wise Parallel MatMul matches Normal MatMul: {is_close}")

    dist.destroy_process_group()
