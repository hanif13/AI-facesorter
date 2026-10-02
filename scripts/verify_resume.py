import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import torch
import yaml

def main():
    p=argparse.ArgumentParser(description="Interrupt a diagnostic run then resume its optimizer and schedule")
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--run-name",default="gate2_resume")
    a=p.parse_args()
    cfg=yaml.safe_load(a.config.read_text()); cfg.update(log_every=1,eval_every_epochs=1000)
    config=Path("outputs/logs/resume_config.yaml"); config.write_text(yaml.safe_dump(cfg))
    root=Path("outputs/checkpoints")/a.run_name
    log_path=Path("outputs/logs/resume_interrupt.log")
    with log_path.open("w") as log:
        process=subprocess.Popen([sys.executable,"-m","src.train_embedding","--config",str(config),
                "--run-name",a.run_name,"--sanity","--epochs","2","--steps-per-epoch","5",
                "--subset-identities","50","--images-per-identity","20","--no-augment"],stdout=log,stderr=log)
        deadline=time.monotonic()+180
        interrupted=False
        while process.poll() is None and time.monotonic()<deadline:
            iterations=root/"iterations.csv"
            if iterations.exists() and len(iterations.read_text().splitlines())>=3:
                process.send_signal(signal.SIGINT); interrupted=True; break
            time.sleep(.1)
        if not interrupted:
            if process.poll() is None: process.send_signal(signal.SIGINT)
            process.wait(timeout=60); raise RuntimeError("Did not interrupt within the diagnostic run")
        process.wait(timeout=60)
    crash=torch.load(root/"crash.pt",map_location="cpu",weights_only=False)
    assert crash["status"]=="interrupted" and 0<crash["global_step"]<10
    assert crash["optimizer"]["state"] and crash["scheduler"]["last_epoch"]==crash["global_step"]
    with Path("outputs/logs/resume_continue.log").open("w") as log:
        subprocess.run([sys.executable,"-m","src.train_embedding","--resume",str(root/"crash.pt")],stdout=log,stderr=log,check=True)
    final=torch.load(root/"final.pt",map_location="cpu",weights_only=False)
    assert final["global_step"]==10 and final["epoch"]==2 and final["status"]=="schedule_complete"
    assert final["provenance"]==crash["provenance"]
    result={"interrupted_step":crash["global_step"],"resumed_final_step":final["global_step"],
            "optimizer_and_scheduler_state_present":True,"schedule_completed":True,
            "random_init_provenance_preserved":True,"verified":True,
            "limitation":"worker augmentation RNG is not restored bit-for-bit; diagnostic uses no augmentation"}
    Path("outputs/logs/resume_verified.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
