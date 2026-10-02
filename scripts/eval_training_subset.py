import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.common.device import get_device,seed_everything
from src.models.mobilefacenet import MobileFaceNet
from src.models.arcface_head import ArcFaceHead
from src.train_embedding import make_dataset

def main():
    p=argparse.ArgumentParser(description="Measure memorization on the exact diagnostic training subset; not LFW")
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); ckpt=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    cfg=ckpt["config"].copy(); cfg["augment"]=False; seed_everything(cfg["seed"]); torch.set_num_threads(1)
    ds,classes=make_dataset(cfg); device=get_device()
    if device.type=="mps": torch.mps.set_per_process_memory_fraction(cfg.get("mps_memory_fraction",.7))
    model=MobileFaceNet(cfg["image_size"],cfg["embedding_dim"]).to(device)
    head=ArcFaceHead(classes,cfg["embedding_dim"],s=cfg["arcface"]["s"],m=cfg["arcface"]["m"]).to(device)
    model.load_state_dict(ckpt["model"]); head.load_state_dict(ckpt["head"]); model.eval(); head.eval()
    correct=0; total=0; loss=0.
    with torch.no_grad():
        for x,y in DataLoader(ds,batch_size=cfg["batch_size"],num_workers=0):
            y=y.to(device); logits=head.s*head.cosine(model(x.to(device)))
            correct+=int((logits.argmax(1)==y).sum()); total+=len(y)
            loss+=float(torch.nn.functional.cross_entropy(logits,y))*len(y)
    result={"training_subset_images":total,"training_subset_classes":classes,"top1":correct/total,
            "margin_free_loss":loss/total,"threshold":.9,"passed":correct/total>=.9,
            "checkpoint":str(a.checkpoint),"purpose":"memorization sanity test, not test-set accuracy"}
    a.output.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
