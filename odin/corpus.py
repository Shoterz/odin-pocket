"""Prepare pinned public corpora for the ODIN refinement experiment."""
import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import numpy as np
from odin.data import normalized, grams, build_exclusions, require_benchmark_coverage, sha256, wiki_documents, strings
from odin.tokenizer import Tokenizer

STORY_REVISION = 'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64'
STORY_SHA256 = '6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443'
_blocked = None
_tokenizer = None


def verify_refinement_inputs(baseline, exclusions, wiki_paths, fineweb, stories, story_expected_sha=STORY_SHA256):
    baseline=Path(baseline)
    original=json.loads((baseline/'manifest.json').read_text())
    inventory=lambda rows:sorted((row['file'],row['sha256']) for row in rows)
    if inventory(exclusions)!=inventory(original['decontamination']['protected_files']):
        raise ValueError('Protected benchmark inventory differs from baseline')
    for name in ('dev.bin','dev.jsonl','tokenizer.json'):
        if sha256(baseline/name)!=original['files'][name]:
            raise ValueError('Baseline file changed: '+name)
    wiki_actual=[{'file':Path(p).name,'sha256':sha256(p)} for p in wiki_paths]
    if inventory(wiki_actual)!=inventory(original['sources'][1]['files']):
        raise ValueError('WikiText training files differ from pinned baseline')
    if sha256(fineweb)!=original['sources'][0]['sha256']:
        raise ValueError('FineWeb shard differs from pinned baseline')
    if Path(stories).name!='TinyStoriesV2-GPT4-train.txt' or sha256(stories)!=story_expected_sha:
        raise ValueError('TinyStories file differs from pinned revision')
    return {'verified':True,'baseline_manifest_sha256':sha256(baseline/'manifest.json'),
            'protected_files':exclusions,'wiki_train_files':wiki_actual,
            'fineweb_sha256':original['sources'][0]['sha256'],
            'stories_revision':STORY_REVISION,'stories_sha256':story_expected_sha,
            'story_hash_authority':'Pinned Hugging Face download metadata / LFS SHA256',
            'baseline_files':{name:original['files'][name] for name in ('dev.bin','dev.jsonl','tokenizer.json')}}


def iter_stories(path):
    story = []
    with open(path) as f:
        for line in f:
            # The public text format separates complete stories with this token.
            parts = line.split('<|endoftext|>')
            for i, part in enumerate(parts):
                story.append(part)
                if i < len(parts)-1:
                    text = normalized(''.join(story))
                    if text:
                        yield text
                    story = []
    text = normalized(''.join(story))
    if text:
        yield text


def filter_document(text, blocked):
    text = normalized(text)
    if not 200 <= len(text) <= 200000:
        return {'reason':'length'}
    key = hashlib.sha256(text.encode()).hexdigest()
    if any(g in blocked for g in grams(text)):
        return {'reason':'overlap'}
    return {'text':text, 'sha256':key, 'split':'dev' if int(key[:8],16)%100==0 else 'train'}


def _process(text):
    row = filter_document(text, _blocked)
    if 'reason' not in row:
        row['ids'] = np.asarray(_tokenizer.encode(row['text'])+[_tokenizer.eos_id], dtype=np.uint16)
    return row


def prepare(args):
    import pyarrow.parquet as pq
    global _blocked, _tokenizer
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    out, baseline = Path(args.output), Path(args.baseline)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        raise ValueError('Corpus already prepared; choose a fresh directory')
    _blocked, exclusions = build_exclusions(args.benchmark_cache)
    require_benchmark_coverage(exclusions)
    for line in (baseline/'dev.jsonl').open():
        _blocked.update(grams(json.loads(line)['text']))
    prompt_file = Path(args.prompts)
    for text in strings(json.loads(prompt_file.read_text())):
        _blocked.update(grams(text))
    _tokenizer = Tokenizer.load(baseline/'tokenizer.json')
    if _tokenizer.vocab_size > 65536:
        raise ValueError('uint16 vocabulary overflow')
    print(json.dumps({'protected_ngrams':len(_blocked), 'workers':args.workers}), flush=True)
    stats, counts, seen = Counter(), Counter(), set()
    def fineweb():
        for batch in pq.ParquetFile(args.fineweb).iter_batches(batch_size=512, columns=['text']):
            yield from batch.column(0).to_pylist()
    wiki_paths = sorted(Path(args.benchmark_cache).glob('Salesforce___wikitext/wikitext-103-raw-v1/**/wikitext-train*.arrow'))
    if not wiki_paths:
        raise ValueError('WikiText training files missing')
    audit=verify_refinement_inputs(baseline,exclusions,wiki_paths,args.fineweb,args.stories)
    # Fork shares the read-only exclusion index; each worker encodes complete documents.
    handles = {name:(open(out/f'{name}.bin','wb'),open(out/f'{name}.jsonl','w'))
               for name in ('train','extra-dev','stories-train','stories-dev')}
    try:
        with mp.get_context('fork').Pool(args.workers) as pool:
            for source, texts in [('fineweb_edu',fineweb()),('wikitext_103_train',wiki_documents(wiki_paths)),('tinystories_gpt4',iter_stories(args.stories))]:
                for i,row in enumerate(pool.imap(_process,texts,chunksize=16),1):
                    stats[source+'_scanned'] += 1
                    if 'reason' in row:
                        stats[source+'_'+row['reason']+'_rejected'] += 1
                        continue
                    if row['sha256'] in seen:
                        stats[source+'_duplicate_rejected'] += 1
                        continue
                    seen.add(row['sha256'])
                    split = row.pop('split')
                    name = ('stories-'+split) if source=='tinystories_gpt4' else ('train' if split=='train' else 'extra-dev')
                    ids = row.pop('ids')
                    ids.tofile(handles[name][0])
                    handles[name][1].write(json.dumps({**row,'source':source},ensure_ascii=False)+'\n')
                    counts[name] += len(ids)
                    stats[source+'_'+split+'_documents'] += 1
                    if i%10000==0:
                        print(json.dumps({'source':source,'scanned':i,'stored_tokens':dict(counts)}),flush=True)
    finally:
        for pair in handles.values():
            for handle in pair:
                handle.close()
    for name in ('dev.bin','dev.jsonl','tokenizer.json'):
        shutil.copyfile(baseline/name,out/name)
    counts['dev'] = (out/'dev.bin').stat().st_size//2
    original = json.loads((baseline/'manifest.json').read_text())
    sources = original['sources']
    sources[0]['scanned_documents'] = stats['fineweb_edu_scanned']
    if sha256(args.fineweb) != sources[0]['sha256']:
        raise ValueError('FineWeb shard differs from pinned baseline')
    sources.append({'dataset':'roneneldan/TinyStories','revision':STORY_REVISION,'file':Path(args.stories).name,'split':'train','license':'CDLA-Sharing-1.0','sha256':sha256(args.stories),'synthetic':True,'origin':'Public GPT-4 generated dataset; no private teacher calls or weight distillation'})
    manifest = {'schema':2,'seed':20260929,'statistics':dict(stats),'tokens':dict(counts),'tokenizer_vocab_size':_tokenizer.vocab_size,'sources':sources,
        'input_verification':audit,
        'tokenizer_policy':'Reuse v0.1.0 training-only BPE unchanged',
        'split_policy':'SHA256(normalized complete document) modulo100; bucket0 held out. Original development set reused, its 13-grams excluded from all training sources.',
        'decontamination':{'method':'Exact whole-document dedup across sources and reject any normalized13-word overlap with protected benchmark, original development or frozen prompt text','protected_files':exclusions,'unique_ngrams':len(_blocked),'additional_protected':{'original_dev_jsonl_sha256':sha256(baseline/'dev.jsonl'),'prompts_sha256':sha256(prompt_file)},'limitations':'No near-duplicate/paraphrase or short-overlap guarantee; packed EOS does not isolate document attention.'},
        'files':{p.name:sha256(p) for p in sorted(out.iterdir()) if p.suffix in ('.bin','.jsonl') or p.name=='tokenizer.json'}}
    temporary = out/'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest,indent=2)+'\n')
    temporary.replace(out/'manifest.json')
    print(json.dumps({'complete':True,'tokens':dict(counts),'statistics':dict(stats)}),flush=True)


def make_view(common, output, story_weight):
    common, output = Path(common).resolve(), Path(output)
    if not 0 <= story_weight < 1:
        raise ValueError('Invalid story weight')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists():
        raise ValueError('Mixture already exists')
    manifest = json.loads((common/'manifest.json').read_text())
    for name in manifest['files']:
        (output/name).symlink_to(common/name)
    manifest['train_sampling'] = [{'file':'train.bin','weight':1-story_weight}]
    if story_weight:
        manifest['train_sampling'].append({'file':'stories-train.bin','weight':story_weight})
    manifest['mixture_policy'] = 'Independent source choice per training window; weights are expected token proportions because all windows have equal length.'
    manifest['common_manifest_sha256'] = sha256(common/'manifest.json')
    temporary=output/'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest,indent=2)+'\n')
    temporary.replace(output/'manifest.json')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--fineweb',required=True)
    parser.add_argument('--stories',required=True)
    parser.add_argument('--benchmark-cache',required=True)
    parser.add_argument('--prompts',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=6)
    prepare(parser.parse_args())
