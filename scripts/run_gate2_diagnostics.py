import argparse
import json
from pathlib import Path
import subprocess
import sys
import yaml

def main():
    p=argparse.ArgumentParser(description="Run overfit, mini-epoch and actual interrupt/resume diagnostics")
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--prefix",default="gate2")
    a=p.parse_args(); root=Path("outputs/logs")
    base=yaml.safe_load(a.config.read_text()); base["eval_every_epochs"]=1000
    base["log_every"]=10
    cfg=root/f"{a.prefix}_diagnostics.yaml"; cfg.write_text(yaml.safe_dump(base))
    def run(name,args):
        with (root/f"{name}.log").open("w") as log:
            subprocess.run([sys.executable,"-m",*args],stdout=log,stderr=log,check=True)
    overfit=f"{a.prefix}_overfit"
    run(overfit,["src.train_embedding","--config",str(cfg),"--run-name",overfit,"--sanity","--epochs","2",
                 "--steps-per-epoch","100","--subset-identities","50","--images-per-identity","20","--no-augment"])
    run("overfit_evaluation",["scripts.eval_training_subset","--checkpoint",f"outputs/checkpoints/{overfit}/final.pt",
                             "--output",str(root/"overfit_result.json")])
    result=json.loads((root/"overfit_result.json").read_text())
    if not result["passed"]: raise RuntimeError(f"Overfit test failed: {result}; investigate before long training")
    mini=f"{a.prefix}_mini"
    run(mini,["src.train_embedding","--config",str(cfg),"--run-name",mini,"--sanity","--epochs","1","--max-images","24521"])
    run("verify_resume",["scripts.verify_resume","--config",str(cfg),"--run-name",f"{a.prefix}_resume"])
    print("All diagnostics completed",flush=True)

if __name__=="__main__": main()
