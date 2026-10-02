import argparse
from io import BytesIO
import json
from pathlib import Path,PurePosixPath
import pickle
from zipfile import ZipFile
from PIL import Image
from src.common.logging_utils import setup_logging

def main():
    p=argparse.ArgumentParser(description="Aligned verification ZIP + annotation pairs to trusted local LFW bin")
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("data/eval/lfw.bin"))
    a=p.parse_args(); setup_logging("prepare_lfw")
    if a.output.exists(): p.error("Output exists; will not overwrite")
    encoded=[]; same=[]
    with ZipFile(a.source) as archive:
        annotation=[i.filename for i in archive.infolist() if PurePosixPath(i.filename).name=="lfw_ann.txt"]
        if len(annotation)!=1: p.error("Need exactly one lfw_ann.txt")
        base=PurePosixPath(annotation[0]).parent
        for line in archive.read(annotation[0]).decode().splitlines():
            label,left,right=line.split()
            if label not in {"0","1"}: raise ValueError("Unexpected pair label")
            same.append(label=="1")
            for relative in [left,right]:
                raw=archive.read(str(base/relative))
                with Image.open(BytesIO(raw)) as image:
                    if image.size!=(112,112): raise ValueError("Expected aligned 112x112 images")
                    # Lossless PNG preserves the aligned BMP pixels exactly.
                    buffer=BytesIO(); image.convert("RGB").save(buffer,format="PNG")
                    encoded.append(buffer.getvalue())
    if len(same)!=6000 or sum(same)!=3000: raise ValueError("Expected 6000 pairs, 3000 same + 3000 different")
    fold_counts=[sum(same[i:i+600]) for i in range(0,6000,600)]
    if fold_counts!=[300]*10: raise ValueError("Pair ordering does not match contiguous 10-fold protocol")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temp=a.output.with_suffix(".tmp")
    with temp.open("wb") as output: pickle.dump((encoded,same),output,protocol=4)
    temp.replace(a.output)
    info={"source":str(a.source),"output":str(a.output),"pairs":len(same),"images":len(encoded),
          "same_pairs":sum(same),"fold_same_counts":fold_counts,"lossless_png":True,
          "pair_order":"preserved exactly from lfw_ann.txt; 600 pairs per contiguous fold",
          "limitation":"aligned Kaggle mirror; no independent canonical-image checksum comparison"}
    Path("outputs/logs/lfw_preparation.json").write_text(json.dumps(info,indent=2)); print(json.dumps(info,indent=2))

if __name__=="__main__": main()
