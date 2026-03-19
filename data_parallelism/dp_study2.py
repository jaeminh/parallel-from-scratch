import torch
import torch.distributed as dist
import torch.nn as nn
from utils import Simple3LayerModel, create_local_data, verify_models_match


# =============================================
def allreduce_gradients_sync(model: nn.Module, world_size: int):
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size


# =============================================
class GradientOverlapDP:
    def __init__(self, model: nn.Module, world_size: int):
        self.model = model
        self.world_size = world_size
        self._handles: list[dist.Work] = []
        self._hooks = []

        # register hooks for all parameters to start async all-reduce when grad is ready
        for param in model.parameters():
            hook = param.register_post_accumulate_grad_hook(self._make_hook())
            self._hooks.append(hook)

    def _make_hook(self):
        def hook(p: torch.nn.Parameter):
            handle = dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM, async_op=True)
            self._handles.append(handle)

        return hook

    def finish_gradient_sync(self):
        """모든 비동기 all-reduce가 끝날 때까지 대기 + 평균 계산."""
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

    # To verify that both Study 1 and Study 2 approaches yield the same final model parameters,
    # we save the initial state_dict before training.
    init_state = {k: v.clone() for k, v in model.state_dict().items()}

    # =============================================
    # 1) Sync All-Reduce (Study 1)
    # =============================================
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    num_steps = 5
    for step in range(num_steps):
        x, y = create_local_data(rank + step * world_size)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        allreduce_gradients_sync(model, world_size)
        optimizer.step()

    # =============================================
    # 2) Async All-Reduce (Study 2)
    # =============================================
    with torch.device("meta"):
        model2 = Simple3LayerModel()
    model2.to_empty(device=device)
    model2.load_state_dict(init_state)

    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    overlap_dp = GradientOverlapDP(model2, world_size)

    for step in range(num_steps):
        x, y = create_local_data(rank + step * world_size)
        x, y = x.to(device), y.to(device)

        logits = model2(x)
        loss = loss_fn(logits, y)

        optimizer2.zero_grad()
        loss.backward()  # hook에서 async all-reduce 자동 시작
        overlap_dp.finish_gradient_sync()  # 통신 완료 대기 + 평균

        optimizer2.step()

    # =============================================
    verify_models_match(model, model2, rank)

    dist.destroy_process_group()
