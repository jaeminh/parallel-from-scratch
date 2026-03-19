import torch
import torch.distributed as dist
import torch.nn as nn
from data_parallelism.dp1 import allreduce_gradients
from data_parallelism.dp2 import GradientOverlapDP
from data_parallelism.dp3 import FlatBucketDP
from utils import create_local_data


class LargeModel(nn.Module):
    """Model for benchmarking, with ~25M parameters."""

    def __init__(self, input_dim=512, hidden_dim=2048, num_layers=6, output_dim=128):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def create_model(device: torch.device) -> nn.Module:
    with torch.device("meta"):
        model = LargeModel()
    model.to_empty(device=device)
    for m in model.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    return model


def benchmark_study1(model, optimizer, loss_fn, device, rank, world_size, num_steps):
    """study1: sync all-reduce"""
    for step in range(num_steps):
        x, y = create_local_data(rank + step * world_size, batch_size=64, input_dim=512)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        allreduce_gradients(model, world_size)
        optimizer.step()


def benchmark_study2(model, optimizer, loss_fn, overlap_dp, device, rank, world_size, num_steps):
    """study2: async overlap per-param"""
    for step in range(num_steps):
        x, y = create_local_data(rank + step * world_size, batch_size=64, input_dim=512)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        overlap_dp.finish_gradient_sync()
        optimizer.step()


def benchmark_study3(model, optimizer, loss_fn, flat_bucket, device, rank, world_size, num_steps):
    """study3: flat bucket"""
    for step in range(num_steps):
        x, y = create_local_data(rank + step * world_size, batch_size=64, input_dim=512)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        flat_bucket.zero_grad()
        loss.backward()
        flat_bucket.finish_gradient_sync()
        optimizer.step()


def time_fn(fn, warmup: int, repeat: int, device: torch.device) -> float:
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record(stream=torch.cuda.current_stream(device))
    fn()
    end.record(stream=torch.cuda.current_stream(device))
    torch.cuda.synchronize(device)

    total_ms = start.elapsed_time(end)
    return total_ms / repeat


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    warmup_steps = 5
    measure_steps = 20

    if rank == 0:
        print("=" * 60)
        print("  DP Benchmark: study1 vs study2 vs study3")
        print(f"  GPUs: {world_size}, warmup: {warmup_steps}, measure: {measure_steps}")
        print("=" * 60)

    loss_fn = nn.CrossEntropyLoss()

    # =============================================
    # Study 1: sync all-reduce
    # =============================================
    model1 = create_model(device)
    opt1 = torch.optim.SGD(model1.parameters(), lr=0.01)
    total_params = sum(p.numel() for p in model1.parameters())
    if rank == 0:
        print(f"\n  Model params: {total_params:,}")

    t1 = time_fn(
        lambda: benchmark_study1(model1, opt1, loss_fn, device, rank, world_size, measure_steps),
        warmup=warmup_steps,
        repeat=measure_steps,
        device=device,
    )
    if rank == 0:
        print(f"\n  [study1] sync all-reduce        : {t1:.3f} ms/step")

    # =============================================
    # Study 2: async overlap per-param
    # =============================================
    model2 = create_model(device)
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    overlap_dp = GradientOverlapDP(model2, world_size)

    t2 = time_fn(
        lambda: benchmark_study2(model2, opt2, loss_fn, overlap_dp, device, rank, world_size, measure_steps),
        warmup=warmup_steps,
        repeat=measure_steps,
        device=device,
    )
    if rank == 0:
        print(f"  [study2] async overlap per-param : {t2:.3f} ms/step")

    # =============================================
    # Study 3: flat bucket
    # =============================================
    model3 = create_model(device)
    opt3 = torch.optim.SGD(model3.parameters(), lr=0.01)
    flat_bucket = FlatBucketDP(model3, world_size, bucket_size=25_000_000, verbose=False)

    t3 = time_fn(
        lambda: benchmark_study3(model3, opt3, loss_fn, flat_bucket, device, rank, world_size, measure_steps),
        warmup=warmup_steps,
        repeat=measure_steps,
        device=device,
    )
    if rank == 0:
        print(f"  [study3] flat bucket             : {t3:.3f} ms/step")

    # =============================================
    # 결과 요약
    # =============================================
    if rank == 0:
        baseline = t1
        print(f"\n{'─' * 60}")
        print(f"  {'Method':<30} {'ms/step':>10} {'vs study1':>10}")
        print(f"{'─' * 60}")
        for name, t in [("study1: sync all-reduce", t1), ("study2: async overlap", t2), ("study3: flat bucket", t3)]:
            speedup = baseline / t
            print(f"  {name:<30} {t:>9.3f}  {speedup:>9.2f}x")
        print(f"{'─' * 60}")

    dist.destroy_process_group()
