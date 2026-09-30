"""Verified source separation and frozen validation inputs; never touches old data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from odin.data import sha256, encode_documents, wiki_documents
from odin.experiment import atomic_json
from odin.tokenizer import Tokenizer


def ensure_link(path,target):
    path,target=Path(path),Path(target).resolve()
    if path.is_symlink() or path.exists():
        if not path.is_symlink() or path.resolve()!=target:
            raise ValueError('Existing link differs from prepared source: '+str(path))
    else:
        path.symlink_to(target)


def split_tokens(path, output, document_counts, eos):
    a=np.memmap(path,dtype=np.uint16,mode='r')
    ends=np.flatnonzero(a==eos)+1
    if len(ends)!=sum(document_counts.values()) or not len(ends) or ends[-1]!=len(a):
        raise ValueError('Packed document count or final EOS mismatch')
    start,doc=0,0
    for name,count in document_counts.items():
        if type(count) is not int or count<1:
            raise ValueError('Invalid document count')
        doc+=count
        end=int(ends[doc-1])
        target=Path(output)/f'{name}.bin'
        tmp=target.with_suffix('.bin.tmp')
        a[start:end].tofile(tmp)
        tmp.replace(target)
        start=end


def make_mixture(common, output, weights):
    from odin.sampling import _weights
    if set(weights)!={'fineweb','wiki','stories'} or any(w<0 for w in weights.values()):
        raise ValueError('Specify nonnegative weights for all three sources')
    entries=[{'file':k+'.bin','weight':v} for k,v in weights.items() if v>0]
    _weights([r['weight'] for r in entries])
    common,output=Path(common).resolve(),Path(output)
    manifest={'schema':3,'train_sampling':entries,'expected_source_weights':weights,
              'parent_manifest_sha256':sha256(common/'manifest.json')}
    if (output/'manifest.json').exists():
        if json.loads((output/'manifest.json').read_text())!=manifest:
            raise ValueError('Mixture already exists with different identity')
    output.mkdir(parents=True,exist_ok=True)
    for name in ('fineweb.bin','wiki.bin','stories.bin','dev.bin','tokenizer.json'):
        ensure_link(output/name,common/name)
    ensure_link(output/'train.bin',common/'fineweb.bin')
    atomic_json(output/'manifest.json',manifest)


def sample_audit(path, source_counts, per_source=100):
    # Uniform reservoir within each source, independent of model outcomes.
    import random
    rng=random.Random(20260930)
    seen=Counter()
    samples={name:[] for name in source_counts}
    for line in Path(path).open():
        row=json.loads(line)
        name=row['source']
        if name not in samples:
            raise ValueError('Unexpected source in prepared JSONL')
        seen[name]+=1
        bucket=samples[name]
        if len(bucket)<per_source: bucket.append(row['text'])
        else:
            i=rng.randrange(seen[name])
            if i<per_source: bucket[i]=row['text']
    if dict(seen)!=source_counts:
        raise ValueError('JSONL source document counts differ from manifest')
    result={}
    for name,texts in samples.items():
        shingles=[set(zip(*[t.lower().split()[i:] for i in range(5)])) for t in texts]
        near=[]
        for i in range(len(shingles)):
            for j in range(i):
                union=shingles[i]|shingles[j]
                similarity=len(shingles[i]&shingles[j])/len(union) if union else 0.
                if similarity>=.8: near.append([j,i,similarity])
        result[name]={'sample_size':len(texts),'near_duplicate_pairs_jaccard_0.8':near,
                      'median_characters':float(np.median([len(t) for t in texts])),
                      'samples':texts}
    return result


def prepare(common, output, cache, prompts):
    from datasets import Dataset
    common,output,cache=Path(common),Path(output),Path(cache)
    if (output/'manifest.json').exists():
        manifest=json.loads((output/'manifest.json').read_text())
        if manifest['parent_manifest_sha256']!=sha256(common/'manifest.json') or manifest['prompts_sha256']!=sha256(prompts):
            raise ValueError('Prepared input identity changed')
        for name,digest in manifest['files'].items():
            if sha256(output/name)!=digest: raise ValueError('Prepared file changed: '+name)
        return manifest
    output.mkdir(parents=True,exist_ok=True)
    parent=json.loads((common/'manifest.json').read_text())
    # Verify full parent inventory before reconstructing source boundaries.
    for name,digest in parent['files'].items():
        if sha256(common/name)!=digest: raise ValueError('Parent file changed: '+name)
    if sha256(prompts)!=parent['decontamination']['additional_protected']['prompts_sha256']:
        raise ValueError('Generation prompts were not excluded from training')
    tok=Tokenizer.load(common/'tokenizer.json')
    stats=parent['statistics']
    split_tokens(common/'train.bin',output,{'fineweb':stats['fineweb_edu_train_documents'],
                                         'wiki':stats['wikitext_103_train_train_documents']},tok.eos_id)
    for name,source in [('stories.bin','stories-train.bin'),('dev.bin','dev.bin'),
                        ('fineweb-dev.bin','extra-dev.bin'),('stories-dev.bin','stories-dev.bin'),('tokenizer.json','tokenizer.json')]:
        ensure_link(output/name,common/source)
    # No WikiText extra-dev documents survived preparation; check this assumption.
    if stats.get('wikitext_103_train_dev_documents',0)!=0:
        raise ValueError('Extra development stream contains WikiText; separate it first')
    protected={r['file']:r['sha256'] for r in parent['decontamination']['protected_files']}
    def protected_file(pattern):
        paths=sorted(cache.glob(pattern))
        if len(paths)!=1 or sha256(paths[0])!=protected.get(paths[0].name):
            raise ValueError('Validation source differs from excluded source: '+pattern)
        return paths[0]
    wiki=protected_file('Salesforce___wikitext/wikitext-103-raw-v1/**/wikitext-validation.arrow')
    arc=protected_file('allenai___ai2_arc/ARC-Easy/**/ai2_arc-validation.arrow')
    encode_documents(wiki_documents([wiki]),tok,output/'wiki-dev.bin')
    rows=list(Dataset.from_file(str(arc)))
    if len(rows)!=570 or len({r['id'] for r in rows})!=570:
        raise ValueError('Expected 570 distinct ARC-Easy validation questions')
    rows.sort(key=lambda r:hashlib.sha256(('odin-controlled-v1:'+r['id']).encode()).hexdigest())
    questions=[]
    for i,r in enumerate(rows):
        labels=r['choices']['label']
        questions.append({'id':r['id'],'partition':'screen' if i<285 else 'confirmation',
                          'prompt':'Question: '+r['question']+'\nAnswer:',
                          'choices':[' '+s for s in r['choices']['text']],
                          'expected':labels.index(r['answerKey'])})
    atomic_json(output/'questions.json',questions)
    # Audit samples, not an assertion of corpus-wide near-duplicate removal.
    audit=sample_audit(common/'train.jsonl',{'fineweb_edu':stats['fineweb_edu_train_documents'],
                                          'wikitext_103_train':stats['wikitext_103_train_train_documents']})
    audit.update(sample_audit(common/'stories-train.jsonl',{'tinystories_gpt4':stats['tinystories_gpt4_train_documents']}))
    atomic_json(output/'source-audit.json',{'method':'100 deterministic reservoir samples/source; within-source exact 5-word-shingle Jaccard >=0.8; no corpus-wide near-duplicate guarantee','sources':audit})
    manifest={'schema':3,'parent_manifest_sha256':sha256(common/'manifest.json'),
              'prompts_sha256':sha256(prompts),'validation_sources':{p.name:sha256(p) for p in (wiki,arc)},
              'source_tokens':{n:(output/(n+'.bin')).stat().st_size//2 for n in ('fineweb','wiki','stories')},
              'files':{p.name:sha256(p) for p in sorted(output.iterdir()) if p.suffix in ('.json','.bin')}}
    atomic_json(output/'manifest.json',manifest)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--common',default='data/common')
    p.add_argument('--output',default='data/controlled/common')
    p.add_argument('--cache',default=str(Path.home()/'.cache/huggingface/datasets'))
    p.add_argument('--prompts',default='experiments/prompts.json')
    args=p.parse_args()
    print(json.dumps(prepare(**vars(args)),indent=2),flush=True)
