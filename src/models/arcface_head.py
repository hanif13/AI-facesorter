import math
import torch
from torch import nn
import torch.nn.functional as F

class ArcFaceHead(nn.Module):
    def __init__(self,num_classes,embedding_dim=512,s=64.,m=.5):
        super().__init__(); self.s=s; self.m=m
        self.weight=nn.Parameter(torch.empty(num_classes,embedding_dim)); nn.init.xavier_uniform_(self.weight)
    def cosine(self,embedding):
        return F.linear(F.normalize(embedding,dim=1),F.normalize(self.weight,dim=1)).clamp(-1+1e-7,1-1e-7)
    def forward(self,embedding,labels,margin=None):
        m=self.m if margin is None else margin
        cosine=self.cosine(embedding)
        if m==0: return self.s*cosine,cosine
        sine=torch.sqrt((1-cosine.square()).clamp_min(1e-7))
        phi=cosine*math.cos(m)-sine*math.sin(m)
        threshold=math.cos(math.pi-m); correction=math.sin(math.pi-m)*m
        phi=torch.where(cosine>threshold,phi,cosine-correction)
        mask=F.one_hot(labels,num_classes=self.weight.shape[0]).to(cosine.dtype)
        return self.s*(mask*phi+(1-mask)*cosine),cosine
