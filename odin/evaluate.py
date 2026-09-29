"""Causal likelihood, official harness adapter, and explicit WikiText-103 report."""
import argparse
from contextlib import nullcontext
import importlib.metadata
import hashlib
import json
import math
import os
from pathlib import Path
import time
from datetime import datetime,timezone
import torch
from odin.data import sha256, wiki_documents
from odin.train import load_checkpoint
from odin.inference import CachedDecoder


def scoring_windows(context, continuation, max_length):
    if not context:
        raise ValueError('Provide at least one context/BOS token')
    if max_length < 1:
        raise ValueError('Context length must be positive')
    sequence = list(context) + list(continuation)
    position = len(context)
    # Half-context stride retains context while scoring every target exactly once.
    stride = max(1, max_length//2)
    while position < len(sequence):
        end = min(len(sequence), position+stride)
        start = max(0, end-1-max_length)
        yield sequence[start:end-1], sequence[position:end], position-1-start
        position = end


class Scorer:
    def __init__(self, model, tokenizer, batch_size=8):
        if batch_size < 1:
            raise ValueError('Batch size must be positive')
        self.model, self.tokenizer, self.batch_size = model.eval(), tokenizer, batch_size
        self.device = next(model.parameters()).device

    def amp(self):
        return torch.autocast('cuda',dtype=torch.bfloat16) if self.device.type=='cuda' else nullcontext()

    @torch.inference_mode()
    def score_tokens(self, pairs):
        scores = [[0.0,True] for _ in pairs]
        jobs = [(i,x,y,start) for i,(ctx,cont) in enumerate(pairs) for x,y,start in scoring_windows(ctx,cont,self.model.config.context)]
        jobs.sort(key=lambda x:len(x[1]))
        for offset in range(0,len(jobs),self.batch_size):
            group = jobs[offset:offset+self.batch_size]
            inputs = torch.full((len(group),max(len(j[1]) for j in group)), self.tokenizer.eos_id, dtype=torch.long, device=self.device)
            for row,(_,ids,_,_) in enumerate(group):
                inputs[row,:len(ids)] = torch.tensor(ids,device=self.device)
            with self.amp():
                logits,_ = self.model(inputs)
            for row,(i,_,targets,start) in enumerate(group):
                selected = logits[row,start:start+len(targets)].float()
                target = torch.tensor(targets,device=self.device)
                ll = selected.log_softmax(-1).gather(-1,target[:,None]).sum().item()
                scores[i][0] += ll
                scores[i][1] = scores[i][1] and bool((selected.argmax(-1)==target).all())
        return [tuple(x) for x in scores]

    def encode_pair(self, context, continuation):
        if continuation=='':
            return self.tokenizer.encode(context) or [self.tokenizer.eos_id],[]
        if not context:
            return [self.tokenizer.eos_id],self.tokenizer.encode(continuation)
        spaces = len(context)-len(context.rstrip())
        if spaces:
            continuation=context[-spaces:]+continuation
            context=context[:-spaces]
        whole = self.tokenizer.encode(context+continuation)
        ctx = self.tokenizer.encode(context)
        # BPE can merge across a mid-word boundary: score that complete token too.
        boundary = min(len(ctx),len(whole))
        while boundary and ctx[:boundary] != whole[:boundary]:
            boundary -= 1
        return whole[:boundary] or [self.tokenizer.eos_id],whole[boundary:]

    def score(self, pairs):
        return self.score_tokens([self.encode_pair(a,b) for a,b in pairs])

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens=96, temperature=0.7, top_k=40, seed=42, use_cache=True):
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 256:
            raise ValueError('Generation length must be between 1 and 256 tokens')
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError('Temperature must be between 0 and 2')
        if type(top_k) is not int or top_k < 1:
            raise ValueError('top_k must be positive')
        ids = self.tokenizer.encode(prompt) or [self.tokenizer.eos_id]
        rng = torch.Generator(device=self.device).manual_seed(seed)
        generated, trace = [], []
        decoder=CachedDecoder(self.model) if use_cache else None
        begin=time.perf_counter()
        for _ in range(max_new_tokens):
            with self.amp():
                if decoder is None:
                    x=torch.tensor([ids[-self.model.config.context:]],device=self.device)
                    logits=self.model(x)[0][0,-1].float()
                else:
                    logits=decoder.next_logits(ids).float()
            raw_probs=logits.softmax(-1)
            if temperature==0:
                token=int(logits.argmax())
            else:
                values,indices=torch.topk(logits,min(top_k,len(logits)))
                pick=torch.multinomial((values/temperature).softmax(-1),1,generator=rng)
                token=int(indices[pick].item())
            trace.append({'id':token,'text':self.tokenizer.decode([token]),'probability':float(raw_probs[token])})
            if token==self.tokenizer.eos_id:
                break
            generated.append(token)
            ids.append(token)
        elapsed=time.perf_counter()-begin
        return {'text':self.tokenizer.decode(generated),'tokens':len(generated),'seconds':elapsed,'tokens_per_second':len(generated)/max(elapsed,1e-9),'trace':trace,'context_truncated':len(self.tokenizer.encode(prompt))>self.model.config.context}


def harness_adapter(scorer):
    from lm_eval.api.model import LM
    class PocketLM(LM):
        def __init__(self):
            super().__init__()
            self._device=scorer.device
        @property
        def tokenizer_name(self):
            return 'odin-pocket-byte-bpe'
        def loglikelihood(self, requests, **kwargs):
            result=[]
            for i in range(0,len(requests),128):
                result.extend(scorer.score([r.args for r in requests[i:i+128]]))
                if i%2048==0:
                    print(f'Likelihood requests {i}/{len(requests)}',flush=True)
            return result
        def loglikelihood_rolling(self, requests, **kwargs):
            return [scorer.score_tokens([([scorer.tokenizer.eos_id],scorer.tokenizer.encode(r.args[0]))])[0][0] for r in requests]
        def generate_until(self, requests, **kwargs):
            outputs=[]
            for r in requests:
                context,opts=r.args
                text=scorer.generate(context,max_new_tokens=min(256,opts.get('max_gen_toks',128)),temperature=0)['text']
                until=opts.get('until',[])
                for stop in ([until] if isinstance(until,str) else until):
                    text=text.split(stop)[0]
                outputs.append(text)
            return outputs
    return PocketLM()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--limit',type=int)
    p.add_argument('--tasks',default='hellaswag,arc_easy,piqa,winogrande')
    p.add_argument('--dataset-cache',help='Datasets cache prepared by scripts/cache_datasets.py')
    p.add_argument('--wiki-test',help='Cached WikiText-103 test arrow file, never training data')
    args=p.parse_args()
    if args.dataset_cache:
        os.environ['HF_DATASETS_CACHE']=str(Path(args.dataset_cache).resolve())
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    torch.set_num_threads(4)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    before=sha256(args.checkpoint)
    model,tok,state=load_checkpoint(args.checkpoint,args.device)
    if sha256(args.checkpoint)!=before:
        raise RuntimeError('Checkpoint changed while loading; evaluate an immutable snapshot')
    scorer=Scorer(model,tok,args.batch_size)
    report={'schema':1,'created_at':datetime.now(timezone.utc).isoformat(),'checkpoint_sha256':before,'parameter_count':model.parameter_count,'trained_tokens':state['tokens'],'device':args.device,'limit':args.limit,'status':'partial' if args.limit else 'full','harness_version':importlib.metadata.version('lm_eval'),'results':{}}
    start=time.perf_counter()
    if args.tasks:
        import lm_eval
        result=lm_eval.simple_evaluate(model=harness_adapter(scorer),tasks=args.tasks.split(','),num_fewshot=0,limit=args.limit,log_samples=False,bootstrap_iters=1000,random_seed=0,numpy_random_seed=1234,torch_random_seed=1234,fewshot_random_seed=1234)
        report['results']=result['results']
        for key in ('versions','n-samples','configs','config'):
            report[key]=result.get(key)
    if args.wiki_test:
        if 'test' not in Path(args.wiki_test).name:
            raise ValueError('WikiText report requires test split')
        nll,tokens,words,byte_count,documents=0.0,0,0,0,0
        text_digest=hashlib.sha256()
        for text in wiki_documents([args.wiki_test]):
            if args.limit and documents>=args.limit:
                break
            ids=tok.encode(text)
            if not ids:
                continue
            value=scorer.score_tokens([([tok.eos_id],ids)])[0][0]
            nll-=value
            tokens+=len(ids)
            words+=len(text.split())
            byte_count+=len(text.encode())
            text_digest.update(text.encode())
            documents+=1
        if not tokens:
            raise ValueError('Empty WikiText evaluation')
        report['wikitext_103']={'token_perplexity':math.exp(nll/tokens),'word_perplexity':math.exp(nll/words),'bits_per_byte':nll/(byte_count*math.log(2)),'tokens':tokens,'words':words,'bytes':byte_count,'text_sha256':text_digest.hexdigest(),'documents':documents,'nll':nll,'source_sha256':sha256(args.wiki_test),'subset':'wikitext-103-raw-v1','split':'test','context':model.config.context,'stride':max(1,model.config.context//2),'protocol':'raw article text, EOS prefix per article; every token scored once; token perplexity tokenizer-dependent'}
    report['elapsed_seconds']=time.perf_counter()-start
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,indent=2,default=str,allow_nan=False)+'\n')
    print(json.dumps({'output':args.output,'results':report['results'],'wikitext_103':report.get('wikitext_103')},indent=2),flush=True)


if __name__=='__main__':
    main()
