import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from collections import defaultdict
import numpy as np


def load_neighbor(args, memory, loader1, loader2, durations):
    memory.dict_mean()
    k = 1 # 1 + int(durations * 5)

    distance, neighbor_index = memory.search(k)
    # print('using block=1')

    loader1.dataset.update_neighbor(distance, neighbor_index)
    loader2.dataset.update_neighbor(distance, neighbor_index)


class MemoryBank(nn.Module):
    def __init__(self, N, c, cls_num, size=3):
        super(MemoryBank, self).__init__()
        self.memory = torch.FloatTensor(N, c)
        self.preds_mem = torch.LongTensor(N, )

        self.num_classes = cls_num
        self.size = size
        self.memory_mean = torch.FloatTensor(N, c)

    def update(self, index, features, pred):
        self.memory[index] = features
        self.preds_mem[index] = pred.long()

    def dict_mean(self):
        # TODO: normalization
        self.memory_mean = self.memory
        self.memory_mean = F.normalize(self.memory_mean, dim=1)

    def search(self, topk=3):
        memory = self.memory_mean.cpu().numpy()
        n, dim = memory.shape[0], memory.shape[1]
        index = faiss.IndexFlatIP(dim)
        # index = faiss.index_cpu_to_gpu(index)
        gpu_faiss = True
        if gpu_faiss:
            print('use gpu faiss')
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, device=0, index=index)
        else:
            print('use cpu faiss')
        index.add(memory)
        distances, nei_indices = index.search(memory, topk+1) # Sample itself is included
        randk = np.random.randint(1, topk+1, size=(n,))

        nei_indices = nei_indices[np.arange(n), randk]

        return  distances[np.arange(n), randk],\
                nei_indices
        # return nei_indices[:, 1]


if __name__ == '__main__':
    m = MemoryBank(N=100, c=512)
    idx, feats = torch.arange(100), torch.ones(100, 512)
    m.update(idx, feats)
    m.update(idx, feats)
    m.dict_mean()
    m.search(2)
