import argparse
from io import BytesIO
import json
from pathlib import Path
import pickle
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import KFold
from src.common.logging_utils import setup_logging

def read_bin(path):
    # The benchmark's official .bin format is pickle; accept only trusted evaluation files.
    with Path(path).open("rb") as file: encoded,same=pickle.load(file,encoding="bytes")
    same=np.asarray(same,dtype=bool)
    if len(encoded)!=2*len(same): raise ValueError("Invalid verification bin")
    return encoded,same

def own_embeddings(model,encoded,device,image_size=112,batch_size=128,flip=True):
    import torch
    import torch.nn.functional as F
    previous=model.training; model.eval(); vectors=[]
    try:
        with torch.no_grad():
            for start in range(0,len(encoded),batch_size):
                images=[]
                for raw in encoded[start:start+batch_size]:
                    with Image.open(BytesIO(raw)) as image:
                        image=image.convert("RGB").resize((image_size,image_size),Image.Resampling.BILINEAR)
                        images.append(np.asarray(image).copy())
                x=torch.from_numpy(np.stack(images)).permute(0,3,1,2).float().to(device)
                x=(x-127.5)/128
                embedding=F.normalize(model(x),dim=1)
                if flip: embedding=F.normalize(embedding+F.normalize(model(x.flip(3)),dim=1),dim=1)
                vectors.append(embedding.cpu().numpy())
    finally: model.train(previous)
    return np.concatenate(vectors)

def verification(embeddings,same,output: Path):
    embeddings=embeddings/np.maximum(np.linalg.norm(embeddings,axis=1,keepdims=True),1e-12)
    scores=(embeddings[0::2]*embeddings[1::2]).sum(axis=1)
    if len(same)<10 or len(np.unique(same))<2: raise ValueError("Need at least 10 labeled pairs with both classes")
    thresholds=np.linspace(-1,1,4001); accuracies=[]; chosen=[]
    for train,test in KFold(n_splits=10,shuffle=False).split(same):
        accuracy=np.mean((scores[train,None]>=thresholds[None,:])==same[train,None],axis=0)
        threshold=thresholds[accuracy.argmax()]
        accuracies.append(float(np.mean((scores[test]>=threshold)==same[test]))); chosen.append(float(threshold))
    result={"pairs":len(same),"fold_accuracies":accuracies,"mean_accuracy":float(np.mean(accuracies)),
            "std_accuracy":float(np.std(accuracies)),"roc_auc":float(roc_auc_score(same,scores)),
            "fold_thresholds":chosen,"standard_lfw_6000_pairs":len(same)==6000,
            "protocol":"10 contiguous folds; threshold fitted only on the other 9 folds"}
    output.mkdir(parents=True,exist_ok=True)
    (output/"metrics.json").write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fpr,tpr,_=roc_curve(same,scores)
    fig,ax=plt.subplots(); ax.plot(fpr,tpr); ax.set(xlabel="False positive rate",ylabel="True positive rate",title="Verification ROC")
    fig.tight_layout(); fig.savefig(output/"roc.png",dpi=160); plt.close(fig)
    return result

def evaluate_own(model,path,device,image_size,output,flip=True,batch_size=64):
    encoded,same=read_bin(path)
    result=verification(own_embeddings(model,encoded,device,image_size,batch_size=batch_size,flip=flip),same,Path(output))
    result.update(embedder="own_random_init",flip_tta=flip,image_size=image_size)
    (Path(output)/"metrics.json").write_text(json.dumps(result,indent=2))
    return result

def main():
    p=argparse.ArgumentParser(description="10-fold LFW verification; trained own model or pretrained comparison baseline")
    p.add_argument("--bin",type=Path,required=True)
    p.add_argument("--embedder",choices=["own","baseline"],required=True)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--baseline-model",type=Path,default=Path("outputs/models/models/buffalo_l/w600k_r50.onnx"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--no-flip",action="store_true")
    p.add_argument("--batch-size",type=int,default=64)
    a=p.parse_args(); setup_logging("eval_lfw"); encoded,same=read_bin(a.bin)
    if a.embedder=="own":
        if not a.checkpoint: p.error("--checkpoint required for own model")
        import torch
        from src.common.device import get_device
        from src.models.mobilefacenet import MobileFaceNet
        ckpt=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
        if ckpt["provenance"]["origin"]!="random_init": raise ValueError("Checkpoint does not declare random init")
        cfg=ckpt["config"]; device=get_device()
        if device.type=="mps": torch.mps.set_per_process_memory_fraction(cfg.get("mps_memory_fraction",.7))
        model=MobileFaceNet(cfg["image_size"],cfg["embedding_dim"]); model.load_state_dict(ckpt["model"]); model.to(device)
        embeddings=own_embeddings(model,encoded,device,cfg["image_size"],batch_size=a.batch_size,flip=not a.no_flip)
    else:
        import onnxruntime as ort
        ort.disable_telemetry_events()
        from insightface.model_zoo import get_model
        model=get_model(str(a.baseline_model),providers=["CPUExecutionProvider"]); model.prepare(ctx_id=-1)
        vectors=[]
        for raw in encoded:
            with Image.open(BytesIO(raw)) as image:
                x=np.asarray(image.convert("RGB").resize((112,112)))[:,:,::-1].copy()
            v=model.get_feat(x).reshape(-1); v=v/max(np.linalg.norm(v),1e-12)
            if not a.no_flip:
                f=model.get_feat(x[:,::-1].copy()).reshape(-1); v=v+f/max(np.linalg.norm(f),1e-12)
            vectors.append(v/max(np.linalg.norm(v),1e-12))
        embeddings=np.stack(vectors)
    result=verification(embeddings,same,a.output)
    result.update(embedder="own_random_init" if a.embedder=="own" else "pretrained_baseline",flip_tta=not a.no_flip)
    if a.embedder=="own": result["evaluation_scope"]="diagnostic_checkpoint" if cfg.get("sanity") else "trained_model"
    (a.output/"metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
