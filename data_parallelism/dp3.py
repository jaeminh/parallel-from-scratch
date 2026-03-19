import torch
import torch.distributed as dist
import torch.nn as nn
from utils import Simple3LayerModel, create_local_data, verify_params_sync


class FlatBucketDP:
    def __init__(self, model: nn.Module, world_size: int, bucket_size: int = 8192, verbose: bool = True):
        self.model = model
        self.world_size = world_size
        self.all_params = list(model.parameters())

        total_numel = sum(p.numel() for p in self.all_params)
        device = self.all_params[0].device
        dtype = self.all_params[0].dtype

        # flat_grad buffer
        self.flat_grad = torch.zeros(total_numel, device=device, dtype=dtype)
        self._offsets = []
        offset = 0
        for p in self.all_params:
            self._offsets.append(offset)
            offset += p.numel()

        # make buckets
        self._buckets: list[tuple[int, int]] = []  # (start, end) in flat_grad
        self._param_to_bucket: dict[int, int] = {}  # param_index → bucket_index

        current_params: list[int] = []
        current_size = 0
        for i in reversed(range(len(self.all_params))):
            current_params.append(i)
            current_size += self.all_params[i].numel()
            if current_size >= bucket_size:
                self._create_bucket(current_params)
                current_params = []
                current_size = 0
        if current_params:
            self._create_bucket(current_params)

        # pending counter per bucket
        self._bucket_pending = [0] * len(self._buckets)
        for bi in self._param_to_bucket.values():
            self._bucket_pending[bi] += 1
        self._pending_reset = list(self._bucket_pending)
        self._handles: list[dist.Work] = []

        # hook 등록
        for i, p in enumerate(self.all_params):
            p.register_post_accumulate_grad_hook(self._make_hook(i))

        if dist.get_rank() == 0 and verbose:
            print(f"  FlatBucketDP: {len(self._buckets)} buckets (bucket_size={bucket_size})")
            for bi, (s, e) in enumerate(self._buckets):
                n_params = self._pending_reset[bi]
                print(f"    bucket {bi}: [{s}:{e}] ({e - s} elements, {n_params} params)")
                print(
                    f"    bucket {bi}: [{s}:{e}] memory={self.flat_grad[s:e].numel() * self.flat_grad.element_size() / 1024**2:.1f} MB"
                )

    def _create_bucket(self, param_indices: list[int]):
        min_idx = min(param_indices)
        max_idx = max(param_indices)
        start = self._offsets[min_idx]
        end = self._offsets[max_idx] + self.all_params[max_idx].numel()
        bucket_id = len(self._buckets)
        self._buckets.append((start, end))
        for pi in param_indices:
            self._param_to_bucket[pi] = bucket_id

    def _make_hook(self, param_idx: int):
        bucket_id = self._param_to_bucket[param_idx]

        def hook(p: nn.Parameter):
            # param.grad → flat_grad에 복사
            offset = self._offsets[param_idx]
            self.flat_grad[offset : offset + p.numel()].copy_(p.grad.data.view(-1))

            self._bucket_pending[bucket_id] -= 1
            if self._bucket_pending[bucket_id] == 0:
                start, end = self._buckets[bucket_id]
                handle = dist.all_reduce(self.flat_grad[start:end], op=dist.ReduceOp.SUM, async_op=True)
                self._handles.append(handle)

        return hook

    def finish_gradient_sync(self):
        for handle in self._handles:
            handle.wait()
        self._handles.clear()

        self.flat_grad /= self.world_size
        for idx, p in enumerate(self.all_params):
            offset = self._offsets[idx]
            p.grad.data.copy_(self.flat_grad[offset : offset + p.numel()].view(p.shape))
        self._bucket_pending = list(self._pending_reset)

    def zero_grad(self):
        self.flat_grad.zero_()
        for p in self.all_params:
            if p.grad is not None:
                p.grad.zero_()


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

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    flat_bucket = FlatBucketDP(model, world_size, bucket_size=8192)

    # =============================================
    num_steps = 5
    for step in range(num_steps):
        x, y = create_local_data(rank + step * world_size)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        flat_bucket.zero_grad()
        loss.backward()  # hook에서 bucket별 async all-reduce 시작
        flat_bucket.finish_gradient_sync()

        optimizer.step()

    # =============================================
    verify_params_sync(model, rank)

    dist.destroy_process_group()
