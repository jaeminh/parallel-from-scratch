import torch
import torch.distributed as dist
import torch.nn as nn
from utils import Simple8LayerModel, create_local_data, verify_params_sync


class ZeRO2Optimizer:
    def __init__(self, model: nn.Module, lr: float, world_size: int, rank: int):
        self.model = model
        self.world_size = world_size
        self.rank = rank
        self.all_params = list(model.parameters())

        self.total_numel = sum(p.numel() for p in self.all_params)
        self.chunk_size = (self.total_numel + world_size - 1) // world_size
        self.padded_numel = self.chunk_size * world_size

        device = self.all_params[0].device
        dtype = self.all_params[0].dtype

        # flat_param + view (same as ZeRO-1)
        self.flat_param = torch.zeros(self.padded_numel, device=device, dtype=dtype)
        self._offsets = []
        offset = 0
        for p in self.all_params:
            numel = p.numel()
            self._offsets.append(offset)
            self.flat_param[offset : offset + numel].copy_(p.data.view(-1))
            p.data = self.flat_param[offset : offset + numel].view(p.shape)
            offset += numel

        self.device = device
        self.dtype = dtype

        # flat_grad: assigned before backward, released after reduce-scatter
        self.flat_grad: torch.Tensor | None = None

        # grad_chunk: reduce-scatter output (1/N gradients for this rank)
        self.grad_chunk = torch.zeros(self.chunk_size, device=device, dtype=dtype)

        # param_chunk, optimizer (same as ZeRO-1)
        start = rank * self.chunk_size
        self.param_chunk = nn.Parameter(self.flat_param[start : start + self.chunk_size].clone())
        self.optimizer = torch.optim.Adam([self.param_chunk], lr=lr)

        # async reduce-scatter during backward
        self._grad_count = 0
        self._rs_handle: dist.Work | None = None
        for i, p in enumerate(self.all_params):
            p.register_post_accumulate_grad_hook(self._make_hook(i))

    def _make_hook(self, param_idx: int):
        def hook(p: nn.Parameter):
            offset = self._offsets[param_idx]
            self.flat_grad[offset : offset + p.numel()].copy_(p.grad.data.view(-1))

            # release gradient
            p.grad = None

            self._grad_count += 1
            if self._grad_count == len(self.all_params):
                self._rs_handle = dist.reduce_scatter_tensor(
                    self.grad_chunk, self.flat_grad, op=dist.ReduceOp.SUM, async_op=True
                )

        return hook

    def step(self):
        # (1) wait for reduce-scatter to finish
        self._rs_handle.wait()
        self._rs_handle = None
        self._grad_count = 0

        # release flat_grad to save memory
        self.flat_grad = None

        self.grad_chunk /= self.world_size

        self.optimizer.step()

        # (3) all-gather updated parameters back to flat_param
        dist.all_gather_into_tensor(self.flat_param, self.param_chunk.data)

    def zero_grad(self):
        # assign flat_grad before backward
        self.flat_grad = torch.zeros(self.padded_numel, device=self.device, dtype=self.dtype)


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    # =============================================
    with torch.device("meta"):
        model = Simple8LayerModel()
    model.to_empty(device=device)
    for m in model.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    loss_fn = nn.CrossEntropyLoss()
    zero2_opt = ZeRO2Optimizer(model, lr=0.001, world_size=world_size, rank=rank)

    # =============================================
    num_steps = 5
    for step in range(num_steps):
        x, y = create_local_data(rank + step * world_size)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        zero2_opt.zero_grad()
        loss.backward()

        zero2_opt.step()

    # =============================================
    verify_params_sync(model, rank)

    dist.destroy_process_group()
