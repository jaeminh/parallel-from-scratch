import torch
import torch.distributed as dist
import torch.nn as nn
from utils import Simple8LayerModel, create_local_data, verify_params_sync


class ZeRO1Optimizer:
    def __init__(self, model: nn.Module, lr: float, world_size: int, rank: int):
        self.model = model
        self.world_size = world_size
        self.rank = rank
        self.all_params = list(model.parameters())

        # Compute size of flat buffer (with padding to be divisible by world_size)
        self.total_numel = sum(p.numel() for p in self.all_params)
        self.chunk_size = (self.total_numel + world_size - 1) // world_size
        self.padded_numel = self.chunk_size * world_size

        device = self.all_params[0].device
        dtype = self.all_params[0].dtype

        # flat_param + view
        self.flat_param = torch.zeros(self.padded_numel, device=device, dtype=dtype)
        self._offsets = []
        offset = 0
        for p in self.all_params:
            numel = p.numel()
            self._offsets.append(offset)
            self.flat_param[offset : offset + numel].copy_(p.data.view(-1))
            p.data = self.flat_param[offset : offset + numel].view(p.shape)
            offset += numel

        # flat_grad + view
        self.flat_grad = torch.zeros(self.padded_numel, device=device, dtype=dtype)
        for idx, p in enumerate(self.all_params):
            p.grad = self.flat_grad[self._offsets[idx] : self._offsets[idx] + p.numel()].view(p.shape)

        # grad_chunk: gradient chunk for reduce-scatter (optimizer input)
        self.grad_chunk = torch.zeros(self.chunk_size, device=device, dtype=dtype)

        # param_chunk: shard of flat_param for this rank (all-gather input)
        start = rank * self.chunk_size
        self.param_chunk = nn.Parameter(self.flat_param[start : start + self.chunk_size].clone())
        self.optimizer = torch.optim.Adam([self.param_chunk], lr=lr)

    def step(self):
        # (1) all gradients are already in flat_grad (as views) --- no copy needed!

        # (2) reduce-scatter: Each rank gets a chunk of the summed gradients
        dist.reduce_scatter_tensor(self.grad_chunk, self.flat_grad, op=dist.ReduceOp.SUM)
        self.grad_chunk /= self.world_size

        # (3) optimizer step on the local chunk
        self.param_chunk.grad = self.grad_chunk
        self.optimizer.step()

        # (4) all-gather updated parameters back to flat_param
        dist.all_gather_into_tensor(self.flat_param, self.param_chunk.data)

    def zero_grad(self):
        self.flat_grad.zero_()


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
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    loss_fn = nn.CrossEntropyLoss()
    zero1_opt = ZeRO1Optimizer(model, lr=0.001, world_size=world_size, rank=rank)

    # =============================================
    num_steps = 5
    for step in range(num_steps):
        x, y = create_local_data(rank + step * world_size)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        zero1_opt.zero_grad()
        loss.backward()

        zero1_opt.step()

    # =============================================
    verify_params_sync(model, rank)

    dist.destroy_process_group()
