from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class MemmapFaceDataset(Dataset):
    """Open memmaps lazily in each worker; tensor-only augmentation."""
    def __init__(self, root: Path, image_size=112, split="train", augment=True, gaussian_blur=False):
        self.root=Path(root); self.image_size=image_size; self.augment=augment; self.gaussian_blur=gaussian_blur
        if split not in {"train","val","all"}: raise ValueError(split)
        self.indices=np.load(self.root/f"{split}_indices.npy",allow_pickle=False) if split!="all" else None
        self.length=len(self.indices) if self.indices is not None else len(np.load(self.root/"labels.npy",mmap_mode="r"))
        self._images=None; self._labels=None
        import json
        self.stored_size=json.loads((self.root/"prep_config.json").read_text())["image_size"]

    def __len__(self): return self.length

    def __getstate__(self):
        state=self.__dict__.copy(); state["_images"]=state["_labels"]=None; return state

    def __getitem__(self,index):
        if self._images is None:
            self._images=np.load(self.root/f"images_{self.stored_size}.npy",mmap_mode="r",allow_pickle=False)
            self._labels=np.load(self.root/"labels.npy",mmap_mode="r",allow_pickle=False)
        i=int(self.indices[index]) if self.indices is not None else index
        x=torch.from_numpy(self._images[i].copy()).permute(2,0,1).float()
        if self.image_size!=self.stored_size:
            x=F.interpolate(x.unsqueeze(0),size=(self.image_size,self.image_size),mode="bilinear",align_corners=False).squeeze(0)
        if self.augment:
            if random.random()<0.5: x=x.flip(2)
            brightness=random.uniform(0.85,1.15); contrast=random.uniform(0.85,1.15)
            mean=x.mean(dim=(1,2),keepdim=True)
            x.sub_(mean).mul_(contrast).add_(mean).mul_(brightness).clamp_(0,255)
            side=random.randint(round(self.image_size*0.9**0.5),self.image_size)
            if side<self.image_size:
                top=random.randint(0,self.image_size-side); left=random.randint(0,self.image_size-side)
                x=F.interpolate(x[:,top:top+side,left:left+side].unsqueeze(0),size=(self.image_size,self.image_size),
                                mode="bilinear",align_corners=False).squeeze(0)
            if random.random()<0.25:
                h=random.randint(8,max(8,self.image_size//3)); w=random.randint(8,max(8,self.image_size//3))
                top=random.randint(0,self.image_size-h); left=random.randint(0,self.image_size-w)
                x[:,top:top+h,left:left+w]=127.5
            if self.gaussian_blur and random.random()<0.1:
                from torchvision.transforms.functional import gaussian_blur
                x=gaussian_blur(x,[3,3])
        label=int(self._labels[i])
        if hasattr(self,"label_map"): label=self.label_map[label]
        return x.sub_(127.5).div_(128), label

def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32
    np.random.seed(seed); random.seed(seed)
    torch.set_num_threads(1)
