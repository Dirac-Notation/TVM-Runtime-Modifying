import torch
import matplotlib.pyplot as plt

from transformers import BertLMHeadModel, BertTokenizer
from tqdm import tqdm
from einops import rearrange

def unstructured(tensor: torch.tensor):
    us = tensor.flatten()
    us_abs = torch.abs(us)

    sorted_values, _ = torch.sort(us_abs, descending=True)
    threadhold = sorted_values[int(us.numel()*0.25)]

    mask = us_abs >= threadhold

    result = us * mask
    result = result.view(tensor.shape[0], tensor.shape[1])

    return result

def NM_pruning(tensor: torch.tensor, restore=False):
    NM = rearrange(tensor, "h (a m) -> a h m", m=4)

    _, idx = torch.topk(torch.abs(NM), k=2, dim=-1)

    idx, _ = torch.sort(idx, dim=-1)

    if restore:
        mask = torch.zeros_like(NM)

        mask.scatter_(2, idx, 1)

        return rearrange((NM * mask), "a h m -> h (a m)")
    else:
        return rearrange(torch.gather(NM, 2, idx), "a h n -> h (a n)")

def vector_pruning(tensor: torch.tensor, vector_size, vector_window, remain_vector):
    if vector_window < remain_vector:
        raise ValueError("vector_window should be bigger than remain_vector")

    b = int(tensor.shape[1]/vector_window)

    vec = rearrange(tensor, "(a h) (b w) -> (a b) h w", h=vector_size, w=vector_window)

    score = torch.sum(torch.abs(vec), dim=1)

    _, idx = torch.topk(score, k=remain_vector, dim=1)
    idx, _ = torch.sort(idx, dim=-1)

    idx = idx.unsqueeze(1).expand(-1, vector_size, -1)

    result = torch.gather(vec, -1, idx)
    result = rearrange(result, "(a b) h w -> (a h) (b w)", b=b)

    return result

def mod_vector_pruning(tensor: torch.tensor, vector_size, vector_window, remain_vector):
    if vector_window < remain_vector:
        raise ValueError("vector_window should be bigger than remain_vector")

    b = int(tensor.shape[1]/vector_window)

    vec = rearrange(tensor, "(a h) (b w) -> (a b) h w", h=vector_size, w=vector_window)

    tmp_abs = torch.abs(vec)
    tmp_sum = torch.sum(tmp_abs, dim=-1).unsqueeze(-1)
    score = torch.sum(tmp_abs/tmp_sum, dim=1)

    _, idx = torch.topk(score, k=remain_vector, dim=1)

    idx, _ = torch.sort(idx, dim=-1)

    idx = idx.unsqueeze(1).expand(-1, vector_size, -1)

    result = torch.gather(vec, -1, idx)
    result = rearrange(result, "(a b) h w -> (a h) (b w)", b=b)

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

sentence = "I love you."

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertLMHeadModel.from_pretrained("bert-base-uncased", is_decoder=True)
model.eval()

inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
print(tokenizer.decode(input_ids[0]))
print()

# for i in range(12):
#     model.bert.encoder.layer[i].attention.self.query.weight = torch.nn.Parameter(NM_pruning(mod_vector_pruning(model.bert.encoder.layer[i].attention.self.query.weight, 8, 8, 4)))
#     model.bert.encoder.layer[i].attention.self.key.weight = torch.nn.Parameter(NM_pruning(mod_vector_pruning(model.bert.encoder.layer[i].attention.self.key.weight, 8, 8, 4)))
#     model.bert.encoder.layer[i].attention.self.value.weight = torch.nn.Parameter(NM_pruning(mod_vector_pruning(model.bert.encoder.layer[i].attention.self.value.weight, 8, 8, 4)))
#     model.bert.encoder.layer[i].attention.output.dense.weight = torch.nn.Parameter(NM_pruning(mod_vector_pruning(model.bert.encoder.layer[i].attention.self.query.weight, 8, 8, 4)))
#     model.bert.encoder.layer[i].intermediate.dense.weight = torch.nn.Parameter(NM_pruning(mod_vector_pruning(model.bert.encoder.layer[i].intermediate.dense.weight, 8, 8, 4)))
#     model.bert.encoder.layer[i].output.dense.weight = torch.nn.Parameter(NM_pruning(mod_vector_pruning(model.bert.encoder.layer[i].output.dense.weight, 8, 8, 4)))

with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)

print(tokenizer.decode(torch.argmax(outputs["logits"], dim=-1).squeeze()))
print()

A = model.bert.encoder.layer[0].attention.self.query.weight
# print(A[:8,:8])
# print()

plt.imshow(A.detach().numpy(), cmap='gray_r')
plt.savefig("result.png")
exit()

# vector_size = 8
# B = NM_pruning(mod_vector_pruning(A, vector_size, 8, 4))
# C = NM_pruning(vector_pruning(A, vector_size, 8, 4))
# D = unstructured(A)

# print(torch.sum(torch.abs(B)))
# print(B[:8,:4])

# print(torch.sum(torch.abs(C)))
# print(C[:8,:4])

# print(torch.sum(torch.abs(D)))
# print(D[:8,:4])

# A = torch.randn((8, 16))
# B = vector_pruning(A, 4, 8, 2)

# print(A[0:4])
# print(torch.abs(A[0:4]).sum(dim=0))
# print(B[0:4])
# print()
# print(A[4:8])
# print(torch.abs(A[4:8]).sum(dim=0))
# print(B[4:8])