import torch
import torch.distributed as dist
import torch.nn as nn
from utils import Simple3LayerModel, create_local_data, verify_params_sync


class GradientOverlapDP:
    def __init__(self, model: nn.Module, world_size: int):
        self.model = model
        self.world_size = world_size
        self._handles: list[dist.Work] = []
        self._hooks = []

        # register hooks for all parameters to start async all-reduce when grad is updated
        for param in model.parameters():
            hook = param.register_post_accumulate_grad_hook(self._make_hook())
            self._hooks.append(hook)

    def _make_hook(self):
        def hook(p: torch.nn.Parameter):
            handle = dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM, async_op=True)
            self._handles.append(handle)

        return hook

    def finish_gradient_sync(self):
        for handle in self._handles:
            handle.wait()
        self._handles.clear()

        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data /= self.world_size


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

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
    overlap_dp = GradientOverlapDP(model, world_size)

    # =============================================
    num_steps = 5
    for step in range(num_steps):
        x, y = create_local_data(rank + step * world_size)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()  # hook에서 async all-reduce 자동 시작
        overlap_dp.finish_gradient_sync()  # 통신 완료 대기 + 평균

        optimizer.step()

    # =============================================
    verify_params_sync(model, rank)

    dist.destroy_process_group()
