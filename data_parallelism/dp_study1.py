import torch
import torch.distributed as dist
import torch.nn as nn
from utils import Simple3LayerModel, create_local_data, verify_params_sync


# NOTE: all-reduce를 이용해서 모든 GPU의 gradient를 평균
def allreduce_gradients(model: nn.Module, world_size: int):
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size
            # dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    # =============================================
    # initialize model
    # =============================================
    with torch.device("meta"):
        model = Simple3LayerModel()
    model.to_empty(device=device)

    for m in model.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # =============================================
    # training loop
    # =============================================
    num_steps = 5
    for step in range(num_steps):
        # (1) Data: 각 GPU가 다른 mini-batch 처리
        x, y = create_local_data(rank + step * world_size)
        x, y = x.to(device), y.to(device)

        # (2) Forward: 각 GPU에서 독립적으로 수행
        logits = model(x)
        loss = loss_fn(logits, y)

        # (3) Backward: 각 GPU에서 독립적으로 gradient 계산
        optimizer.zero_grad()
        loss.backward()

        # (4) All-Reduce: 모든 GPU의 gradient를 평균
        allreduce_gradients(model, world_size)

        # (5) Update: 모든 GPU가 동일한 gradient로 업데이트
        optimizer.step()

    verify_params_sync(model, rank)

    dist.destroy_process_group()
