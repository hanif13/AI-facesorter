import argparse
import json
from pathlib import Path
import subprocess
import sys
import yaml

def main():
    p=argparse.ArgumentParser(description="Compare three MPS training configurations before long training")
    p.add_argument("--iterations",type=int,default=300)
    p.add_argument("--prefix",default="gate2")
    a=p.parse_args()
    root=Path("outputs/logs"); root.mkdir(parents=True,exist_ok=True)
    base=yaml.safe_load(Path("configs/train.yaml").read_text()); results=[]
    for size,batch in [(112,128),(112,256),(96,256)]:
        name=f"{a.prefix}_{size}_b{batch}"
        cfg=base.copy(); cfg.update(image_size=size,batch_size=batch,lr=.1*batch/256)
        config=root/f"{name}.yaml"; config.write_text(yaml.safe_dump(cfg))
        with (root/f"{name}.log").open("w") as log:
            status=subprocess.call([sys.executable,"-m","src.train_embedding","--config",str(config),
                                    "--benchmark-only","--benchmark-iterations",str(a.iterations),"--run-name",name],stdout=log,stderr=log)
        if status==0:
            result=json.loads((Path("outputs/checkpoints")/name/"benchmark.json").read_text())
            result["status"]="ok"
        else: result={"image_size":size,"batch_size":batch,"status":"failed","exit_code":status}
        results.append(result)
        (root/"training_benchmarks.json").write_text(json.dumps(results,indent=2))
        print(json.dumps(result),flush=True)

if __name__=="__main__": main()
