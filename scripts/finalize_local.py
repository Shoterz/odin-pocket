"""Verify and capture the final local product after training; no publication."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from odin.data import sha256
from odin.release import package,validate_report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--wait',action='store_true')
    p.add_argument('--browser-python',default=sys.executable)
    p.add_argument('--chromium')
    p.add_argument('--port',type=int,default=8780)
    args=p.parse_args()
    checkpoint=ROOT/'runs/pocket/submission.pt'
    report_path=ROOT/'results/official.json'
    begin=time.monotonic()
    # The training pipeline writes its first package after evaluation completes.
    while not (ROOT/'submission/evidence/model-card.md').is_file():
        if not args.wait:
            raise RuntimeError('Training and full evaluation must finish first')
        if time.monotonic()-begin>8*3600:
            raise TimeoutError('Final evidence did not arrive within eight hours')
        print('Waiting for completed training and full evaluation.',flush=True)
        time.sleep(60)
    report=json.loads(report_path.read_text())
    digest=sha256(checkpoint)
    validate_report(report,digest)
    package(checkpoint,report_path,ROOT/'runs/pocket',ROOT/'data/prepared',ROOT/'submission/evidence')
    commands=[
        [sys.executable,'-m','pytest','-q'],
        [sys.executable,'scripts/measure_product.py','--checkpoint',str(checkpoint)],
        [sys.executable,'scripts/benchmark_inference.py','--checkpoint',str(checkpoint)],
    ]
    for command in commands:
        print('Running: '+str(command),flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
    browser_options=['--chromium',args.chromium] if args.chromium else []
    url=f'http://127.0.0.1:{args.port}'
    with (ROOT/'runs/final-capture-server.log').open('w') as log:
        server=subprocess.Popen([sys.executable,'-m','odin.server','--checkpoint',str(checkpoint),
            '--evidence','submission/evidence','--port',str(args.port)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        try:
            start=time.monotonic()
            while True:
                if server.poll() is not None:
                    raise RuntimeError('Capture server failed; see runs/final-capture-server.log')
                try:
                    with urlopen(url+'/api/status',timeout=2) as response:
                        status=json.load(response)
                    if status.get('checkpoint_sha256')!=digest:
                        raise ValueError('Capture server has the wrong checkpoint')
                    break
                except OSError:
                    if time.monotonic()-start>60:
                        raise TimeoutError('Capture server did not start')
                    time.sleep(1)
            subprocess.run([args.browser_python,'scripts/browser_check.py','--url',url,'--require-model',*browser_options],cwd=ROOT,check=True)
            subprocess.run([args.browser_python,'scripts/record_demo.py','--url',url,*browser_options],cwd=ROOT,check=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    evidence={'checkpoint_sha256':digest,'verified_at':datetime.now(timezone.utc).isoformat(),
              'tests_passed':True,'browser_checks_passed':True,'video_recorded':True,
              'remaining':'Human inspection of final text, screenshots and video; final documentation; publication and Devpost submission.'}
    (ROOT/'submission/evidence/local-verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence),flush=True)


if __name__=='__main__':
    main()
