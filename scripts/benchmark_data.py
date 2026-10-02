import argparse
import json
import logging
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from src.common.device import seed_everything
from src.common.logging_utils import setup_logging
from src.data.dataset import MemmapFaceDataset,seed_worker

def main():
    p=argparse.ArgumentParser(description="DataLoader-only full epoch benchmark and augmented sample grid")
    p.add_argument("--data",type=Path,default=Path("data/casia_npy"))
    p.add_argument("--image-size",type=int,default=112)
    p.add_argument("--workers",type=int,default=6)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--max-batches",type=int,help="Omit to iterate a full epoch")
    p.add_argument("--no-augment",action="store_true")
    p.add_argument("--output",type=Path,default=Path("outputs/logs/data_benchmark.json"))
    a=p.parse_args(); setup_logging("benchmark_data"); seed_everything(42); torch.set_num_threads(1)
    dataset=MemmapFaceDataset(a.data,a.image_size,augment=not a.no_augment)
    assets=Path("outputs/report_assets"); assets.mkdir(parents=True,exist_ok=True)
    sheet=Image.new("RGB",(8*a.image_size,8*a.image_size))
    for n,index in enumerate(np.random.default_rng(42).choice(len(dataset),size=min(64,len(dataset)),replace=False)):
        x,_=dataset[int(index)]
        rgb=(x*128+127.5).clamp(0,255).byte().permute(1,2,0).numpy()
        sheet.paste(Image.fromarray(rgb),((n%8)*a.image_size,(n//8)*a.image_size))
    if not a.no_augment: sheet.save(assets/"aug_samples.png")
    loader=DataLoader(dataset,batch_size=a.batch_size,num_workers=a.workers,persistent_workers=a.workers>0,
                      shuffle=True,drop_last=True,pin_memory=False,worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(42))
    started=time.monotonic(); count=0; batches=0; steady_started=None; warm_count=0
    for x,y in loader:
        if not torch.isfinite(x).all(): raise ValueError("Nonfinite tensor")
        count+=len(x); batches+=1
        if batches==10: steady_started=time.monotonic(); warm_count=count
        if batches%100==0: logging.info("batches=%d images=%d elapsed=%.1fs",batches,count,time.monotonic()-started)
        if a.max_batches and batches>=a.max_batches: break
    elapsed=time.monotonic()-started
    steady=(count-warm_count)/(time.monotonic()-steady_started) if steady_started and count>warm_count else count/elapsed
    result={"dataset_images":len(dataset),"processed_images":count,"batches":batches,"seconds":elapsed,
            "images_per_second_including_startup":count/elapsed,"steady_images_per_second":steady,
            "full_epoch":batches==len(loader),"drop_last_images":len(dataset)%a.batch_size,
            "augment":not a.no_augment,"workers":a.workers,"batch_size":a.batch_size,
            "target_images_per_second":3000,"target_met_steady":steady>=3000}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2))
    logging.info("Result: %s",json.dumps(result)); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
