# https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from utils import Simple8LayerModel, create_local_data


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)  # set device for this process

    with torch.device("meta"):
        model = Simple8LayerModel()
    model.to_empty(device=device)
    for m in model.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    total_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"\n Total Params: {total_params:,} ({total_params * 4 / 1024**2:.2f} MB, fp32)", rank)

    # =============================================
    # (1) apply fully_shard to each layer (optional, for finer granularity)
    for layer_module in model.layers:
        if isinstance(layer_module, nn.Linear):
            fully_shard(layer_module)

    # (b) root module sharding
    fully_shard(model)

    sharded_numel = 0
    for name, p in model.named_parameters():
        local_numel = p._local_tensor.numel()
        sharded_numel += local_numel

    if rank == 0:
        print("\nSharded Parameters:", rank)
        print(f"  {name}: logical={list(p.shape)}, local={list(p._local_tensor.shape)} ({local_numel:,})")
        print(f"  → Total Params in this rank: {sharded_numel:,} ({total_params:,} / {world_size})", rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # =============================================
    num_steps = 5
    for step in range(num_steps):
        x, y = create_local_data(rank + step * world_size)
        x, y = x.to(device), y.to(device)

        # forward: per-layer all-gather → compute → reshard
        logits = model(x)
        loss = loss_fn(logits, y)

        # backward: per-layer all-gather → grad → reduce-scatter → reshard
        optimizer.zero_grad()
        loss.backward()

        # step: update sharded params with sharded grads
        optimizer.step()

    alloc_mb = torch.cuda.memory_allocated() / 1024**2
    if rank == 0:
        print(f"\n[Memory] GPU 할당: {alloc_mb:.1f} MB (rank 0)", rank)

    dist.destroy_process_group()
