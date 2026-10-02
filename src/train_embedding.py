import argparse
import csv
from datetime import datetime
import hashlib
import json
import logging
import math
import random
import signal
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml
from src.common.device import get_device,seed_everything
from src.common.logging_utils import setup_logging
from src.data.dataset import MemmapFaceDataset,seed_worker
from src.models.mobilefacenet import MobileFaceNet
from src.models.arcface_head import ArcFaceHead

def sync(device):
    if device.type=="mps": torch.mps.synchronize()

def memory(device):
    return int(torch.mps.current_allocated_memory()) if device.type=="mps" else 0

def build_components(cfg,num_classes,device):
    model=MobileFaceNet(cfg["image_size"],cfg["embedding_dim"])
    provenance=model.assert_random_init()
    head=ArcFaceHead(num_classes,cfg["embedding_dim"],**{k:cfg["arcface"][k] for k in ["s","m"]})
    model.to(device); head.to(device)
    decay=[]; no_decay=[]
    for module in [model,head]:
        for name,param in module.named_parameters():
            (no_decay if param.ndim==1 or name.endswith("bias") else decay).append(param)
    optimizer=torch.optim.SGD([{"params":decay,"weight_decay":cfg["weight_decay"]},
                               {"params":no_decay,"weight_decay":0.}],lr=cfg["lr"],momentum=cfg["momentum"])
    return model,head,optimizer,provenance

def make_dataset(cfg):
    ds=MemmapFaceDataset(Path(cfg["data"]),cfg["image_size"],augment=cfg["augment"])
    labels=np.load(Path(cfg["data"])/"labels.npy",mmap_mode="r")
    classes=int(labels.max())+1
    rng=np.random.default_rng(cfg["seed"])
    if cfg.get("subset_identities"):
        available=ds.indices
        counts=np.bincount(labels[available],minlength=classes)
        ids=np.argsort(-counts,kind="stable")[:cfg["subset_identities"]]
        picked=[]
        for identity in ids:
            candidates=available[labels[available]==identity]
            picked.extend(rng.choice(candidates,size=min(len(candidates),cfg["images_per_identity"]),replace=False))
        ds.indices=np.asarray(picked,dtype=np.int64)
        ds.label_map={int(identity):i for i,identity in enumerate(ids)}
        classes=len(ids)
    elif cfg.get("max_images"):
        ds.indices=np.sort(rng.choice(ds.indices,size=min(len(ds.indices),cfg["max_images"]),replace=False))
    ds.length=len(ds.indices)
    cfg["dataset_indices_sha256"]=hashlib.sha256(ds.indices.tobytes()).hexdigest()
    return ds,classes

def make_loader(ds,cfg):
    return DataLoader(ds,batch_size=cfg["batch_size"],shuffle=True,num_workers=cfg["num_workers"],
                      persistent_workers=cfg["num_workers"]>0,drop_last=True,pin_memory=False,
                      worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(cfg["seed"]))

def epoch_batches(loader,steps):
    iterator=iter(loader)
    for batch in range(steps):
        try: x,y=next(iterator)
        except StopIteration: iterator=iter(loader); x,y=next(iterator)
        yield batch,(x,y)

def update(model,head,optimizer,x,y,device,margin,grad_clip):
    x=x.to(device); y=y.to(device)
    optimizer.zero_grad(set_to_none=True)
    embedding=model(x); logits,cosine=head(embedding,y,margin=margin)
    loss=nn.functional.cross_entropy(logits,y)
    if not torch.isfinite(loss): raise FloatingPointError("NaN/Inf loss; no optimizer update applied")
    loss.backward()
    norm=nn.utils.clip_grad_norm_([*model.parameters(),*head.parameters()],grad_clip)
    if not torch.isfinite(norm): raise FloatingPointError("NaN/Inf gradients; no optimizer update applied")
    optimizer.step()
    with torch.no_grad():
        free_loss=nn.functional.cross_entropy(head.s*cosine,y)
        accuracy=(cosine.argmax(1)==y).float().mean()
    return float(loss.detach()),float(free_loss),float(accuracy)

def benchmark(model,head,optimizer,loader,cfg,device,iterations):
    model.train(); head.train(); iterator=iter(loader); warm=min(20,iterations//4)
    peak=0; driver_peak=0; start=None; measured=0
    for step in range(iterations):
        try: x,y=next(iterator)
        except StopIteration: iterator=iter(loader); x,y=next(iterator)
        if step==warm: sync(device); start=time.monotonic()
        update(model,head,optimizer,x,y,device,0.,cfg["grad_clip"])
        peak=max(peak,memory(device))
        if device.type=="mps": driver_peak=max(driver_peak,int(torch.mps.driver_allocated_memory()))
        if step>=warm: measured+=len(x)
        if (step+1)%20==0: logging.info("Benchmark %d/%d allocated_memory=%d",step+1,iterations,peak)
    sync(device); elapsed=time.monotonic()-start
    result={"image_size":cfg["image_size"],"batch_size":cfg["batch_size"],"device":str(device),
            "iterations":iterations,"warmup_iterations":warm,"measured_images":measured,
            "measured_seconds":elapsed,"images_per_second":measured/elapsed,
            "peak_allocated_bytes":peak,"allocation_sampling":"at end of each update; not an activation peak measurement",
            "peak_driver_allocated_bytes":driver_peak,"epoch_steps":len(loader),
            "estimated_epoch_seconds":len(loader)*cfg["batch_size"]/(measured/elapsed)}
    logging.info("Benchmark result: %s",json.dumps(result)); return result

def atomic_checkpoint(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".tmp"); torch.save(payload,temp); temp.replace(path)

def rng_state(device):
    state={"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state()}
    if device.type=="mps": state["mps"]=torch.mps.get_rng_state()
    return state

def restore_rng(state,device):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if device.type=="mps" and "mps" in state: torch.mps.set_rng_state(state["mps"])

def main():
    p=argparse.ArgumentParser(description="Train MobileFaceNet from random init; benchmark/sanity before Gate 2")
    p.add_argument("--config",type=Path,default=Path("configs/train.yaml"))
    p.add_argument("--resume",type=Path)
    p.add_argument("--run-name")
    p.add_argument("--time_budget_hours",type=float)
    p.add_argument("--epochs",type=int)
    p.add_argument("--lr",type=float,help="Explicitly override learning rate, including on resume")
    p.add_argument("--steps-per-epoch",type=int)
    p.add_argument("--subset-identities",type=int)
    p.add_argument("--images-per-identity",type=int)
    p.add_argument("--max-images",type=int)
    p.add_argument("--no-augment",action="store_true")
    p.add_argument("--benchmark-only",action="store_true")
    p.add_argument("--benchmark-iterations",type=int,default=200)
    p.add_argument("--sanity",action="store_true",help="Short diagnostic run; permits missing LFW")
    a=p.parse_args(); setup_logging("train_embedding")
    if a.time_budget_hours is not None and a.time_budget_hours<=0: p.error("Budget must be positive")
    if a.benchmark_iterations<4: p.error("Benchmark needs at least 4 iterations")
    if a.lr is not None and a.lr<=0: p.error("Learning rate must be positive")
    if any(value is not None and value<1 for value in [a.epochs,a.steps_per_epoch,a.subset_identities,a.images_per_identity,a.max_images]):
        p.error("Epoch/step/subset limits must be positive")
    checkpoint=None
    if a.resume:
        checkpoint=torch.load(a.resume,map_location="cpu",weights_only=False)
        if checkpoint.get("provenance",{}).get("origin")!="random_init": p.error("Resume accepts only this project's random-init checkpoint")
        cfg=checkpoint["config"].copy()
    else:
        cfg=yaml.safe_load(a.config.read_text())
        for key,default in [("subset_identities",None),("images_per_identity",20),("max_images",None),("steps_per_epoch",None)]:
            cfg.setdefault(key,default)
            value=getattr(a,key)
            if value is not None: cfg[key]=value
        cfg["sanity"]=a.sanity or cfg.get("sanity",False)
        if a.no_augment: cfg["augment"]=False
        if a.epochs: cfg["epochs"]=a.epochs
    if a.lr is not None: cfg["lr"]=a.lr
    if cfg["amp"]: p.error("Use fp32 on MPS for this project")
    if cfg["optimizer"]!="sgd" or cfg["scheduler"]!="cosine": p.error("This trainer implements SGD and cosine scheduling")
    if cfg["eval_every_epochs"]<1 or cfg["log_every"]<1: p.error("Logging/evaluation intervals must be positive")
    seed_everything(cfg["seed"]); torch.set_num_threads(1); device=get_device()
    if device.type=="mps":
        torch.mps.set_per_process_memory_fraction(cfg.get("mps_memory_fraction",0.7))
        logging.info("MPS allocator fraction=%s",cfg.get("mps_memory_fraction",0.7))
    ds,classes=make_dataset(cfg); loader=make_loader(ds,cfg)
    if not len(loader): p.error("Dataset smaller than a batch")
    if checkpoint and cfg["dataset_indices_sha256"]!=checkpoint["config"]["dataset_indices_sha256"]:
        p.error("Dataset changed since checkpoint")
    stats=json.loads((Path(cfg["data"])/"prep_config.json").read_text())
    if checkpoint and checkpoint["config"]["data_fingerprint"]!=stats["fingerprint"]: p.error("Prepared data changed")
    cfg["data_fingerprint"]=stats["fingerprint"]; cfg["num_classes"]=classes
    run_name=a.run_name or datetime.now(ZoneInfo("Asia/Bangkok")).strftime("train_%Y%m%d_%H%M%S")
    if Path(run_name).name!=run_name: p.error("Run name must be a single folder name")
    run_dir=a.resume.parent if a.resume else Path("outputs/checkpoints")/run_name
    if not a.resume and run_dir.exists(): p.error("Run directory already exists")
    run_dir.mkdir(parents=True,exist_ok=True)
    model,head,optimizer,provenance=build_components(cfg,classes,device)
    if a.benchmark_only:
        (run_dir/"config.json").write_text(json.dumps(cfg,indent=2))
        result=benchmark(model,head,optimizer,loader,cfg,device,a.benchmark_iterations)
        result["provenance"]=provenance
        (run_dir/"benchmark.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); return
    if not cfg.get("sanity") and not Path(cfg["lfw"]).is_file(): p.error("Missing LFW evaluation file; provide it before full training")
    started=time.time()
    accumulated_seconds=checkpoint.get("elapsed_training_seconds",0.) if checkpoint else 0.
    budget_hours=a.time_budget_hours if a.time_budget_hours is not None else cfg.get("time_budget_hours")
    cfg["time_budget_hours"]=budget_hours
    deadline=started+budget_hours*3600-accumulated_seconds if budget_hours else None
    if deadline and deadline<=started:
        logging.warning("Original training time budget already exhausted; no updates applied"); return
    steps=cfg.get("steps_per_epoch") or len(loader)
    if steps<1: p.error("steps-per-epoch must be positive")
    if cfg["epochs"]=="auto":
        if checkpoint: p.error("Resumed checkpoint must have a fixed schedule horizon")
        if not a.time_budget_hours: p.error("epochs=auto requires --time_budget_hours")
        speed=benchmark(model,head,optimizer,loader,cfg,device,200)
        (run_dir/"budget_benchmark.json").write_text(json.dumps(speed,indent=2))
        cfg["epochs"]=math.floor(max(0,deadline-time.time())*.9/speed["estimated_epoch_seconds"])
        if cfg["epochs"]<1: p.error("Time budget cannot fit one epoch")
        del model,head,optimizer
        if device.type=="mps": torch.mps.empty_cache()
        seed_everything(cfg["seed"]); model,head,optimizer,provenance=build_components(cfg,classes,device)
    horizon=int(cfg["epochs"])*steps; warmup=min(cfg["warmup_epochs"]*steps,max(1,horizon//2))
    def multiplier(step):
        if step<warmup: return max(1e-3,(step+1)/max(1,warmup))
        fraction=min(1.,(step-warmup)/max(1,horizon-warmup))
        return .5*(1+math.cos(math.pi*fraction))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier)
    epoch=0; next_batch=0; global_step=0; best_lfw=-1.; history=[]
    if checkpoint:
        model.load_state_dict(checkpoint["model"]); head.load_state_dict(checkpoint["head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key,value in state.items():
                if torch.is_tensor(value): state[key]=value.to(device)
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch=checkpoint["epoch"]; next_batch=checkpoint["next_batch"]; global_step=checkpoint["global_step"]
        provenance=checkpoint["provenance"]; best_lfw=checkpoint["best_lfw"]; history=checkpoint["history"]
        if a.lr is not None:
            scheduler.base_lrs=[a.lr]*len(optimizer.param_groups)
            for group in optimizer.param_groups:
                group["initial_lr"]=a.lr; group["lr"]=a.lr*multiplier(global_step)
            scheduler._last_lr=[group["lr"] for group in optimizer.param_groups]
            logging.info("Explicit resume learning rate override=%s",a.lr)
        restore_rng(checkpoint["rng"],device)
        logging.info("Resumed epoch=%d next_batch=%d global_step=%d",epoch,next_batch,global_step)
    (run_dir/"config.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False))
    def save(name,status):
        sync(device)
        payload={"model":model.state_dict(),"head":head.state_dict(),"optimizer":optimizer.state_dict(),
                 "scheduler":scheduler.state_dict(),"epoch":epoch,"next_batch":next_batch,
                 "global_step":global_step,"config":cfg,"provenance":provenance,"rng":rng_state(device),
                 "best_lfw":best_lfw,"history":history,"status":status,
                 "elapsed_training_seconds":accumulated_seconds+time.time()-started}
        atomic_checkpoint(run_dir/name,payload)
        if name!="last.pt": shutil.copyfile(run_dir/name,run_dir/"last.pt")
    csv_path=run_dir/"iterations.csv"
    if checkpoint is None: save("initial.pt","random_init_before_training")
    fields=["epoch","step","loss","margin_free_loss","top1","lr","images_per_second","mps_memory_bytes"]
    first_csv=not csv_path.exists()
    stop_requested=False
    def request_stop(signum,frame):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGINT,request_stop)
    with csv_path.open("a",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        if first_csv: writer.writeheader()
        try:
            while epoch<int(cfg["epochs"]):
                if stop_requested:
                    save("crash.pt","interrupted"); logging.info("Stopped at a completed update boundary"); return
                model.train(); head.train(); loader.generator.manual_seed(cfg["seed"]+epoch)
                epoch_started=time.monotonic(); losses=[]; free_losses=[]; accuracies=[]
                margin=cfg["arcface"]["m"]*min(1.,epoch/max(1,cfg["arcface"]["margin_warmup_epochs"]))
                for batch,(x,y) in epoch_batches(loader,steps):
                    if batch<next_batch: continue
                    step_started=time.monotonic(); used_lr=optimizer.param_groups[0]["lr"]
                    loss,free,accuracy=update(model,head,optimizer,x,y,device,margin,cfg["grad_clip"])
                    scheduler.step(); global_step+=1; next_batch=batch+1
                    losses.append(loss); free_losses.append(free); accuracies.append(accuracy)
                    if stop_requested:
                        save("crash.pt","interrupted"); logging.info("Stopped at a completed update boundary"); return
                    if global_step%cfg["log_every"]==0 or batch==0:
                        row=dict(zip(fields,[epoch+1,global_step,loss,free,accuracy,used_lr,
                                len(x)/(time.monotonic()-step_started),memory(device)]))
                        writer.writerow(row); stream.flush(); logging.info("Train: %s",json.dumps(row))
                        if shutil.disk_usage(run_dir).free<2*1024**3:
                            raise OSError("Free disk below 2 GiB; stopped before filling the disk")
                    if deadline and time.time()>=deadline:
                        save("budget_stop.pt","budget_exhausted")
                        logging.warning("Stopped at time budget; schedule not necessarily completed"); return
                completed=epoch+1
                entry={"epoch":completed,"mean_loss":float(np.mean(losses)) if losses else None,
                       "mean_margin_free_loss":float(np.mean(free_losses)) if free_losses else None,
                       "mean_top1":float(np.mean(accuracies)) if accuracies else None,
                       "epoch_seconds":time.monotonic()-epoch_started,"margin":margin,"global_step":global_step}
                validation=MemmapFaceDataset(Path(cfg["data"]),cfg["image_size"],split="val",augment=False)
                if hasattr(ds,"label_map"):
                    original_labels=np.load(Path(cfg["data"])/"labels.npy",mmap_mode="r")
                    validation.indices=np.asarray([i for i in validation.indices if int(original_labels[i]) in ds.label_map])
                    validation.length=len(validation.indices); validation.label_map=ds.label_map
                if len(validation):
                    model.eval(); val_losses=[]; correct=0; total=0
                    with torch.no_grad():
                        for vx,vy in DataLoader(validation,batch_size=cfg.get("eval_batch_size",64),num_workers=0):
                            vy=vy.to(device); vcos=head.cosine(model(vx.to(device)))
                            val_losses.append(float(nn.functional.cross_entropy(head.s*vcos,vy))*len(vy))
                            correct+=int((vcos.argmax(1)==vy).sum()); total+=len(vy)
                    entry["sanity_validation"]={"images":total,"margin_free_loss":sum(val_losses)/total,"top1":correct/total,
                                                "purpose":"same-identity loss monitoring; not a generalization metric"}
                if completed%cfg["eval_every_epochs"]==0 and Path(cfg["lfw"]).is_file():
                    from src.eval_lfw import evaluate_own
                    if device.type=="mps": sync(device); torch.mps.empty_cache()
                    metrics=evaluate_own(model,cfg["lfw"],device,cfg["image_size"],run_dir/f"lfw_epoch{completed}",
                                         batch_size=cfg.get("eval_batch_size",64))
                    metrics["evaluation_scope"]="diagnostic_checkpoint" if cfg.get("sanity") else "trained_model"
                    (run_dir/f"lfw_epoch{completed}"/"metrics.json").write_text(json.dumps(metrics,indent=2))
                    entry["lfw"]=metrics
                history.append(entry); epoch=completed; next_batch=0
                save(f"ckpt_epoch{completed}.pt","epoch_complete")
                if "lfw" in entry and entry["lfw"]["mean_accuracy"]>best_lfw:
                    best_lfw=entry["lfw"]["mean_accuracy"]; save("best.pt","best_lfw")
                (run_dir/"history.json").write_text(json.dumps(history,indent=2)); logging.info("Epoch: %s",json.dumps(entry))
            save("final.pt","schedule_complete")
        except (Exception,KeyboardInterrupt) as error:
            save("crash.pt","interrupted" if isinstance(error,KeyboardInterrupt) else "failed")
            logging.exception("Training stopped; crash checkpoint saved")
            raise

if __name__=="__main__": main()
