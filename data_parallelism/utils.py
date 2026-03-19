import torch
import torch.nn as nn
import torch.distributed as dist


class Simple3LayerModel(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class Simple8LayerModel(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


def create_local_data(rank: int, batch_size: int = 16, input_dim: int = 64):
    torch.manual_seed(42 + rank)
    x = torch.randn(batch_size, input_dim)
    y = torch.randint(0, 10, (batch_size,))
    return x, y


def verify_params_sync(model: nn.Module, rank: int):
    """모든 rank의 param이 rank 0과 동일한지 검증."""
    for name, param in model.named_parameters():
        ref = param.data.clone()
        dist.broadcast(ref, src=0)
        if torch.allclose(param.data, ref):
            if rank == 0:
                print(f"  ✓ {name}: 모든 GPU 동기화 확인")
        else:
            print(f"  ✗ {name}: Rank {rank}에서 불일치 발견!")


def verify_models_match(model1: nn.Module, model2: nn.Module, rank: int):
    """두 모델의 param이 동일한지 검증."""
    all_match = True
    for (name, p1), (_, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        if torch.allclose(p1.data, p2.data):
            if rank == 0:
                print(f"  ✓ {name}: 동일")
        else:
            all_match = False
            if rank == 0:
                diff = (p1.data - p2.data).abs().max().item()
                print(f"  ✗ {name}: 불일치 (max diff = {diff:.6e})")
    if rank == 0 and all_match:
        print("  → 두 방식의 결과가 완전히 동일합니다")
