"""Package reviewable source and evidence without caches, optimizer state or credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from odin.reporting import validate_report

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='submission/odin-pocket-source.zip')
    parser.add_argument('--require-final',action='store_true')
    args=parser.parse_args()
    if args.require_final:
        required=['submission/evidence/model-card.md','submission/evidence/local-verification.json',
                  'results/official.json','results/product.json','results/efficiency.json',
                  'runs/pocket/submission.pt','submission/video/odin-pocket-demo.mp4']
        if any(not (ROOT/name).is_file() for name in required):
            raise SystemExit('Final evidence or captures are missing; refusing a final package')
        digest=hashlib.sha256()
        with (ROOT/'runs/pocket/submission.pt').open('rb') as checkpoint:
            for chunk in iter(lambda:checkpoint.read(1024*1024),b''):
                digest.update(chunk)
        checkpoint_hash=digest.hexdigest()
        validate_report(json.loads((ROOT/'results/official.json').read_text()),checkpoint_hash)
        for name in ('submission/evidence/local-verification.json','results/product.json','results/efficiency.json'):
            if json.loads((ROOT/name).read_text()).get('checkpoint_sha256')!=checkpoint_hash:
                raise ValueError('Final artifact checkpoint mismatch: '+name)
        if len(list((ROOT/'submission/screenshots').glob('*.png')))<3:
            raise ValueError('At least three screenshots are required')
    paths=[]
    for name in ('README.md','LICENSE','PRODUCT.md','DESIGN.md','requirements.txt','pyproject.toml','run.sh','.gitignore'):
        paths.append(ROOT/name)
    for folder in ('odin','tests','scripts','configs','web','docs'):
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in {'.py','.json','.md','.js','.css','.html'} and '__pycache__' not in p.parts and p.name!='progress.md')
    paths.extend(p for p in (ROOT/'submission').glob('*') if p.is_file() and p.suffix in {'.json','.md','.txt','.srt'})
    paths.extend(p for p in (ROOT/'submission/video').glob('*') if p.is_file() and p.suffix in {'.srt','.ass','.mp4'})
    for folder in ('submission/evidence','submission/screenshots'):
        if (ROOT/folder).exists():
            paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in {'.json','.jsonl','.md','.png'})
    paths.extend(p for p in (ROOT/'results').glob('*.json') if p.name in {'official.json','product.json','efficiency.json','pilot-smoke.json','midtraining-product.json','preview-product-cached.json'})
    files=sorted(set(paths))
    out=ROOT/args.output
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in files:
            if path.is_symlink():
                raise ValueError('Refusing symbolic link in source package: '+str(path))
            archive.write(path,'odin-pocket/'+path.relative_to(ROOT).as_posix())
    digest=hashlib.sha256(out.read_bytes()).hexdigest()
    (out.with_suffix('.zip.sha256')).write_text(digest+'  '+out.name+'\n')
    print(json.dumps({'archive':str(out),'files':len(files),'bytes':out.stat().st_size,'sha256':digest}),flush=True)
