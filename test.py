import torch

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
    sparsity = remain_vector/vector_window

    result = torch.zeros((tensor.shape[0], int(tensor.shape[1]*sparsity)))

    for i in range(int(tensor.shape[0]/vector_size)):
        vec = tensor[i*vector_size:(i+1)*vector_size].clone()
        for j in range(int(tensor.shape[1]/vector_window)):
            vec_tmp = vec[:,j*vector_window:(j+1)*vector_window]

            score = torch.sum(torch.abs(vec_tmp), dim=0)

            if vector_window >= remain_vector:
                _, idx = torch.topk(score, k=remain_vector, dim=0)
            else:
                print("vector_window should be bigger than remain_vector")
                exit()

            # mask = torch.ones_like(score)

            # mask.scatter_(0, idx, 0)

            # vec = vec * mask

            idx, _ = torch.sort(idx, dim=-1)

            result[i*vector_size:(i+1)*vector_size, int(j*vector_window*sparsity):int((j+1)*vector_window*sparsity)] = torch.gather(vec_tmp, 1, idx.repeat(vec_tmp.shape[0],1))
    
    return result

# result = []

# for i in range(100):
#     A = torch.randn((1024,1024))
#     B = vector_pruning(NM_pruning(A, True), 2)
#     C = NM_pruning(vector_pruning(A, 4))

#     result.append(B.norm() > C.norm())

# result = torch.tensor(result)

# print(torch.sum(result))

dim = 1024
A = torch.randn((dim, dim))
B = vector_pruning(NM_pruning(A, True), 16, 8, 4)
C = NM_pruning(vector_pruning(A, 16, 8, 4))

print(B.norm())
print(B[:8,:8])

print(C.norm())
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