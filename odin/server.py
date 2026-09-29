"""Loopback workbench. No external inference or prompt transmission."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
from urllib.parse import urlsplit
import torch
from odin.data import sha256
from odin.evaluate import Scorer
from odin.train import load_checkpoint

ROOT=Path(__file__).resolve().parent.parent


class Workbench:
    def __init__(self, checkpoint, evidence, device='cpu'):
        self.checkpoint,self.evidence,self.device=Path(checkpoint),Path(evidence),device
        self.lock=threading.Lock()
        self.scorer,self.metadata,self.checkpoint_hash=None,{},None
        self.reload()

    def reload(self):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('Model is busy; try again after the current request')
        try:
            if not self.checkpoint.exists():
                return self.status()
            before=sha256(self.checkpoint)
            model,tok,state=load_checkpoint(self.checkpoint,self.device)
            if sha256(self.checkpoint)!=before:
                raise RuntimeError('Checkpoint changed while loading; retry reload')
            self.scorer=Scorer(model,tok,batch_size=4)
            self.checkpoint_hash=before
            self.metadata={k:state.get(k) for k in ('run_id','parameter_count','tokens','step','training_seconds','provenance','config')}
            self.metadata['parameter_count']=model.parameter_count
            return self.status()
        finally:
            self.lock.release()

    def status(self):
        return {'name':'ODIN Pocket','ready':self.scorer is not None,'checkpoint_sha256':self.checkpoint_hash,'inference_device':self.device,'model':self.metadata,'message':'Local model ready' if self.scorer else 'No trained checkpoint found. Train a model, then choose Reload checkpoint.'}

    @staticmethod
    def text(payload,name,max_length=16000):
        value=payload.get(name)
        if not isinstance(value,str) or not value.strip() or len(value)>max_length:
            raise ValueError(f'{name} must contain 1–{max_length:,} characters')
        return value

    def run(self,fn):
        if self.scorer is None:
            raise RuntimeError('A trained checkpoint is required; train and reload first')
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('Model is busy; wait for the current request to finish')
        try:
            return fn(self.scorer)
        finally:
            self.lock.release()

    def generate(self,payload):
        prompt=self.text(payload,'prompt')
        count=payload.get('max_new_tokens',96)
        temperature=payload.get('temperature',0.7)
        if type(count) is not int or not 1<=count<=256:
            raise ValueError('Choose 1–256 new tokens')
        if type(temperature) not in (int,float) or not math.isfinite(temperature) or not 0<=temperature<=2:
            raise ValueError('Temperature must be between 0 and 2')
        return self.run(lambda s:{**s.generate(prompt,count,temperature),'checkpoint_sha256':self.checkpoint_hash})

    def compare(self,payload):
        prompt=self.text(payload,'prompt')
        candidates=payload.get('candidates')
        if not isinstance(candidates,list) or not 2<=len(candidates)<=4:
            raise ValueError('Provide between 2 and 4 candidate endings')
        for candidate in candidates:
            self.text({'candidate':candidate},'candidate',2000)
        def score(s):
            separator='' if prompt[-1].isspace() else ' '
            pairs=[s.encode_pair(prompt,separator+c.strip()) for c in candidates]
            values=s.score_tokens(pairs)
            result=[{'text':c,'log_likelihood':ll,'tokens':len(pair[1]),'mean_log_likelihood':ll/max(1,len(pair[1]))} for c,(ll,_),pair in zip(candidates,values,pairs)]
            winner=max(range(len(result)),key=lambda i:result[i]['mean_log_likelihood'])
            return {'candidates':result,'preferred':winner,'method':'mean log likelihood per candidate token; preference is not factual verification','checkpoint_sha256':self.checkpoint_hash}
        return self.run(score)

    def tokenize(self,payload):
        text=self.text(payload,'prompt')
        return self.run(lambda s:{'count':len(s.tokenizer.encode(text)),'tokens':[{'id':i,'text':s.tokenizer.decode([i])} for i in s.tokenizer.encode(text)[:256]],'display_limit':256})

    def results(self):
        metrics=[]
        path=self.evidence/'metrics.jsonl'
        if not path.exists():
            path=self.evidence/'training-metrics.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    row=json.loads(line)
                    if row.get('run_id') and row['run_id']==self.metadata.get('run_id') and row['step']<=self.metadata.get('step',-1):
                        metrics.append(row)
                except (ValueError,KeyError):
                    continue
        reports=[]
        for path in sorted((ROOT/'results').glob('*.json')):
            try:
                report=json.loads(path.read_text())
            except (OSError,ValueError):
                continue
            if report.get('checkpoint_sha256')==self.checkpoint_hash:
                reports.append({'file':path.name,**report})
        return {'training':metrics,'reports':reports,'checkpoint_sha256':self.checkpoint_hash,'benchmark_note':'Only reports matching the loaded checkpoint appear here.'}


def make_server(app,host='127.0.0.1',port=8766):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)
        def send(self,status,value,content_type='application/json'):
            body=json.dumps(value,allow_nan=False).encode() if content_type=='application/json' else value
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)
        def allowed(self):
            host_header=self.headers.get('Host','')
            allowed={f'127.0.0.1:{self.server.server_address[1]}',f'localhost:{self.server.server_address[1]}'}
            if host_header not in allowed:
                self.send(403,{'error':'Use this workbench through its localhost address'})
                return False
            origin=self.headers.get('Origin')
            if origin and origin not in {f'http://{h}' for h in allowed}:
                self.send(403,{'error':'Cross-origin requests are disabled'})
                return False
            return True
        def do_GET(self):
            if not self.allowed():
                return
            path=urlsplit(self.path).path
            if path=='/api/status':
                return self.send(200,app.status())
            if path=='/api/evidence':
                return self.send(200,app.results())
            static={'/':('index.html','text/html; charset=utf-8'),'/app.js':('app.js','text/javascript; charset=utf-8'),'/style.css':('style.css','text/css; charset=utf-8')}
            if path in static:
                file,kind=static[path]
                return self.send(200,(ROOT/'web'/file).read_bytes(),kind)
            self.send(404,{'error':'Not found'})
        def do_POST(self):
            if not self.allowed():
                return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=65536:
                    return self.send(413,{'error':'Request must be between 1 and 65,536 bytes'})
                payload=json.loads(self.rfile.read(length))
                if not isinstance(payload,dict):
                    raise ValueError('Request must be a JSON object')
                route={'/api/generate':app.generate,'/api/compare':app.compare,'/api/tokenize':app.tokenize,'/api/reload':lambda _:app.reload()}.get(urlsplit(self.path).path)
                if route is None:
                    return self.send(404,{'error':'Not found'})
                self.send(200,route(payload))
            except (ValueError,TypeError) as e:
                self.send(400,{'error':str(e)})
            except RuntimeError as e:
                self.send(503,{'error':str(e)})
            except Exception:
                import traceback
                traceback.print_exc()
                self.send(500,{'error':'Local inference failed. See the server log for details.'})
        def log_message(self,format,*args):
            # Do not put user passages in logs.
            pass
    return ThreadingHTTPServer((host,port),Handler)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='runs/pocket/submission.pt' if (ROOT/'runs/pocket/submission.pt').exists() else 'runs/pocket/latest.pt')
    p.add_argument('--evidence',default='submission/evidence' if (ROOT/'submission/evidence/training-metrics.jsonl').exists() else 'runs/pocket')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--port',type=int,default=8766)
    args=p.parse_args()
    torch.set_num_threads(4)
    app=Workbench(args.checkpoint,args.evidence,args.device)
    server=make_server(app,port=args.port)
    print(f'ODIN Pocket: http://127.0.0.1:{args.port}',flush=True)
    server.serve_forever()
