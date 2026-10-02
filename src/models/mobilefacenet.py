import hashlib
import logging
import torch
from torch import nn

class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout, kernel=1, stride=1, padding=0, groups=1, activation=True):
        layers=[nn.Conv2d(cin,cout,kernel,stride,padding,groups=groups,bias=False),nn.BatchNorm2d(cout)]
        if activation: layers.append(nn.PReLU(cout))
        super().__init__(*layers)

class Bottleneck(nn.Module):
    def __init__(self, cin, cout, expansion, stride):
        super().__init__(); hidden=cin*expansion
        self.body=nn.Sequential(ConvBlock(cin,hidden),ConvBlock(hidden,hidden,3,stride,1,hidden),
                                ConvBlock(hidden,cout,activation=False))
        self.residual=stride==1 and cin==cout
    def forward(self,x):
        y=self.body(x); return x+y if self.residual else y

class MobileFaceNet(nn.Module):
    def __init__(self, image_size=112, embedding_dim=512):
        super().__init__()
        if image_size not in {96,112}: raise ValueError("MobileFaceNet supports 96 or 112")
        self.image_size=image_size; self.embedding_dim=embedding_dim
        layers=[ConvBlock(3,64,3,2,1),ConvBlock(64,64,3,1,1,64)]
        cin=64
        for expansion,cout,repeats,stride in [(2,64,5,2),(4,128,1,2),(2,128,6,1),(4,128,1,2),(2,128,2,1)]:
            for i in range(repeats):
                layers.append(Bottleneck(cin,cout,expansion,stride if i==0 else 1)); cin=cout
        layers.extend([ConvBlock(128,512),ConvBlock(512,512,image_size//16,groups=512,activation=False),
                       nn.Conv2d(512,embedding_dim,1,bias=False),nn.Flatten(),nn.BatchNorm1d(embedding_dim)])
        self.body=nn.Sequential(*layers)
        for module in self.modules():
            if isinstance(module,nn.Conv2d): nn.init.kaiming_normal_(module.weight,mode="fan_out",nonlinearity="relu")
            elif isinstance(module,(nn.BatchNorm1d,nn.BatchNorm2d)):
                nn.init.ones_(module.weight); nn.init.zeros_(module.bias)
        self.initial_weight_hash=self.weight_hash()

    def weight_hash(self):
        digest=hashlib.sha256()
        for name,tensor in self.state_dict().items():
            digest.update(name.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def assert_random_init(self):
        assert self.weight_hash()==self.initial_weight_hash, "Weights changed since random initialization"
        provenance={"origin":"random_init","pretrained_weights_loaded":False,
                    "initial_sha256":self.initial_weight_hash,
                    "parameters":sum(p.numel() for p in self.parameters())}
        logging.info("No pretrained weights loaded: %s",provenance)
        return provenance

    def forward(self,x):
        if x.shape[-2:]!=(self.image_size,self.image_size): raise ValueError("Unexpected image dimensions")
        return self.body(x)
