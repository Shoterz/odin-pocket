"""Local long run. Safe to rerun: resumes an existing matching training recipe."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from odin.model import ModelConfig
from odin.release import verify_prepared,export,package
from odin.train import train


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/prepared')
    p.add_argument('--run',default='runs/pocket')
    p.add_argument('--steps',type=int,default=60000)
    p.add_argument('--wiki-test',required=True)
    p.add_argument('--dataset-cache',help='Use the same datasets cache supplied to preparation')
    p.add_argument('--wait-for-data',action='store_true')
    a=p.parse_args()
    os.chdir(ROOT)
    data,run=Path(a.data),Path(a.run)
    if a.wait_for_data:
        begin=time.monotonic()
        print('Waiting for atomic completion of corpus preparation.',flush=True)
        while not (data/'manifest.json').exists():
            if time.monotonic()-begin>3600:
                raise TimeoutError('Corpus did not finish within one hour')
            time.sleep(10)
    verify_prepared(data)
    config=ModelConfig(**json.loads((ROOT/'configs/pocket.json').read_text()))
    checkpoint=train(data=data,config=config,output=run,steps=a.steps,batch_size=16,accumulation=2,lr=0.0006,device='cuda',eval_every=250,resume=run/'latest.pt' if (run/'latest.pt').exists() else None)
    final=export(checkpoint,run/'submission.pt')
    env={**os.environ,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'}
    command=[sys.executable,'-m','odin.evaluate','--checkpoint',str(final),'--output','results/official.json','--device','cuda','--batch-size','8','--wiki-test',a.wiki_test]
    if a.dataset_cache:
        command.extend(['--dataset-cache',a.dataset_cache])
    subprocess.run(command,env=env,check=True)
    package(final,'results/official.json',run,data,'submission/evidence')
    print('Training, full evaluation and local evidence package completed. Public submission still requires verification.',flush=True)
