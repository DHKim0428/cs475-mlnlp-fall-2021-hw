import torch

A = torch.rand((1, 768, 1))
after_tile = torch.tile(A, (3, 1, 1))
print(after_tile.shape)
B = torch.rand((3, 512, 768))

print(torch.matmul(B, after_tile).shape)