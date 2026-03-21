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
    out_features = 64  # out_features should be divisible by world_size for column-wise parallelism

    torch.manual_seed(42)
    X = torch.randn(bsz, in_features, device=device)  # [bsz, in_features]
    W = torch.randn(in_features, out_features, device=device)  # [in_features, out_features]
    # Y will be [bsz, out_features]

    Y_normal = X @ W

    # =============================================
    # Column-wise Parallel MatMul
    # =============================================
    shard_size = out_features // world_size

    # broadcast X to all processes
    X_col = X.clone()
    dist.broadcast(X_col, src=0)

    # matmul local shard
    W_shard = W[:, rank * shard_size : (rank + 1) * shard_size].contiguous()
    Y_local = X_col @ W_shard  # [bsz, shard_size (out_features/world_size)]

    # all_gather Y_local from all processes
    Y_ag = [torch.empty_like(Y_local) for _ in range(world_size)]
    dist.all_gather(Y_ag, Y_local)

    # concat Y_ag to get the final Y
    Y_column = torch.cat(Y_ag, dim=-1)

    dist.barrier()

    # =============================================
    is_close = torch.allclose(Y_normal, Y_column, atol=1e-5)

    if rank == 0:
        print(f"Column-wise Parallel MatMul matches Normal MatMul: {is_close}")

    dist.destroy_process_group()
