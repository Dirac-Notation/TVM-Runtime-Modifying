import torch

from tqdm import tqdm

def NM_pruning(tensor: torch.tensor, restore=False):
    NM = tensor.clone()

    NM = NM.view((int(tensor.shape[1]/4),tensor.shape[0],4))

    _, idx = torch.topk(torch.abs(NM), k=2, dim=-1)

    idx, _ = torch.sort(idx, dim=-1)

    if restore:
        mask = torch.zeros_like(NM)

        mask.scatter_(2, idx, 1)

        return (NM * mask).view(tensor.shape[0], tensor.shape[1])
    else:
        return torch.gather(NM, 2, idx).view(tensor.shape[0], int(tensor.shape[1]/2))

def vector_pruning(tensor: torch.tensor, vector_size, vector_window, remain_vector):
    if vector_window < remain_vector:
        raise ValueError("vector_window should be bigger than remain_vector")

    sparsity = remain_vector/vector_window

    new_shape = (tensor.shape[0], int(tensor.shape[1]*sparsity))

    vec = tensor.view(-1, vector_size, vector_window)

    score = torch.sum(torch.abs(vec), dim=1)

    _, idx = torch.topk(score, k=remain_vector, dim=1)

    idx, _ = torch.sort(idx, dim=-1)

    idx = idx.unsqueeze(1). expand(-1, vector_size, -1)

    result = torch.gather(vec, -1, idx)
    result = result.view(*new_shape)

    return result

# result = []

# for i in tqdm(range(100)):
#     dim = 1024
#     A = torch.randn((dim, dim))
#     B = vector_pruning(NM_pruning(A, True), 16, 8, 4)
#     C = NM_pruning(vector_pruning(A, 16, 8, 4))

#     result.append(B.norm() > C.norm())

# result = torch.tensor(result)

# print(torch.sum(result))

dim = 1024
A = torch.randn((dim, dim))
vector_size = 8
B = NM_pruning(vector_pruning(NM_pruning(A, True), vector_size, 8, 4))
C = NM_pruning(vector_pruning(A, vector_size, 8, 4))

print(B.numel())
print(B.norm())
print((abs(B)<0.3).sum())
print(B[:8,:8])

print(C.numel())
print(C.norm())
print((abs(C)<0.3).sum())
print(C[:8,:8])

# A = torch.randn((8, 16))
# B = vector_pruning(A, 4, 8, 2)

# print(A[0:4])
# print(torch.abs(A[0:4]).sum(dim=0))
# print(B[0:4])
# print()
# print(A[4:8])
# print(torch.abs(A[4:8]).sum(dim=0))
# print(B[4:8])