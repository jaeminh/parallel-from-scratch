import torch
import torch.distributed as dist
import torch.nn as nn
from utils import Simple3LayerModel, create_local_data, verify_models_match


class BucketDP:
    """
    param hook을 사용하되, module 단위로 묶어서 통신.
    - 각 param의 grad가 준비되면 해당 module의 pending counter 감소
    - module의 모든 param grad가 준비되면 → flat buffer로 복사 → async all-reduce
    - 통신 횟수: param 6개 → module 3개 (fc1, fc2, fc3)
    """

    def __init__(self, model: nn.Module, world_size: int):
        self.model = model
        self.world_size = world_size
        self._handles: list[tuple[dist.Work, nn.Module]] = []

        # 파라미터가 있는 leaf module 수집 + bucket buffer 할당
        self._bucket_params: dict[nn.Module, list[nn.Parameter]] = {}
        self._bucket_buffers: dict[nn.Module, torch.Tensor] = {}
        self._bucket_pending: dict[nn.Module, int] = {}
        self._param_to_bucket: dict[int, nn.Module] = {}

        for module in model.modules():
            params = list(module.parameters(recurse=False))
            if not params:
                continue
            self._bucket_params[module] = params
            self._bucket_buffers[module] = torch.zeros(
                sum(p.numel() for p in params), device=params[0].device, dtype=params[0].dtype
            )
            self._bucket_pending[module] = len(params)
            for p in params:
                self._param_to_bucket[id(p)] = module

        self._pending_reset = dict(self._bucket_pending)

        # param hook 등록
        for module, params in self._bucket_params.items():
            for p in params:
                p.register_post_accumulate_grad_hook(self._make_hook(p))

    def _make_hook(self, param: nn.Parameter):
        module = self._param_to_bucket[id(param)]

        def hook(p: nn.Parameter):
            self._bucket_pending[module] -= 1
            if self._bucket_pending[module] == 0:
                # bucket의 모든 grad 준비됨 → flat buffer에 복사 → async all-reduce
                buf = self._bucket_buffers[module]
                offset = 0
                for mp in self._bucket_params[module]:
                    numel = mp.numel()
                    buf[offset : offset + numel].copy_(mp.grad.data.view(-1))
                    offset += numel
                handle = dist.all_reduce(buf, op=dist.ReduceOp.SUM, async_op=True)
                self._handles.append((handle, module))

        return hook

    def finish_gradient_sync(self):
        for handle, module in self._handles:
            handle.wait()
            buf = self._bucket_buffers[module]
            buf /= self.world_size
            offset = 0
            for p in self._bucket_params[module]:
                numel = p.numel()
                p.grad.data.copy_(buf[offset : offset + numel].view_as(p.grad.data))
                offset += numel
        self._handles.clear()
        for m in self._bucket_pending:
            self._bucket_pending[m] = self._pending_reset[m]


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
        # allreduce_gradients_sync(model, world_size)
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size
        optimizer.step()

    # =============================================
    # 2) Module Bucketing (Study 3)
    # =============================================
    with torch.device("meta"):
        model2 = Simple3LayerModel()
    model2.to_empty(device=device)
    model2.load_state_dict(init_state)

    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    bucket_dp = BucketDP(model2, world_size)

    for step in range(num_steps):
        x, y = create_local_data(rank + step * world_size)
        x, y = x.to(device), y.to(device)

        logits = model2(x)
        loss = loss_fn(logits, y)

        optimizer2.zero_grad()
        loss.backward()
        bucket_dp.finish_gradient_sync()

        optimizer2.step()

    # =============================================
    verify_models_match(model, model2, rank)

    dist.destroy_process_group()
