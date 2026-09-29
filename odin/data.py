"""Public corpus preparation; benchmark rows are exclusion data only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import numpy as np
from odin.tokenizer import Tokenizer


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def normalized(text):
    return unicodedata.normalize('NFC', text).replace('\r\n', '\n').strip()


def partition_documents(texts):
    train, dev, seen = [], [], set()
    stats = Counter()
    for text in texts:
        text = normalized(text)
        if not text:
            stats['empty'] += 1
            continue
        key = hashlib.sha256(text.encode()).hexdigest()
        if key in seen:
            stats['duplicates'] += 1
            continue
        seen.add(key)
        (dev if int(key[:8], 16) % 100 == 0 else train).append(text)
    return train, dev, dict(stats)


def encode_documents(texts, tokenizer, path):
    if tokenizer.vocab_size > 65536:
        raise ValueError('uint16 token store supports at most 65536 entries')
    total = 0
    with open(path, 'wb') as f:
        for text in texts:
            ids = tokenizer.encode(text) + [tokenizer.eos_id]
            np.asarray(ids, dtype=np.uint16).tofile(f)
            total += len(ids)
    return total


def arrow_rows(path):
    import pyarrow as pa
    with pa.memory_map(str(path), 'r') as source:
        for batch in pa.ipc.open_stream(source):
            yield from batch.to_pylist()


def wiki_documents(paths):
    # Split on top-level article titles, never split paragraphs across partitions.
    article = []
    for path in sorted(paths):
        for row in arrow_rows(path):
            text = row['text']
            if re.match(r'^\s*= [^=].*[^=] =\s*$', text) and article:
                yield ''.join(article)
                article = []
            article.append(text)
    if article:
        yield ''.join(article)


def grams(text, n=13):
    words = re.findall(r'\w+', text.casefold())
    for i in range(len(words)-n+1):
        yield hashlib.blake2b(' '.join(words[i:i+n]).encode(), digest_size=8).digest()


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from strings(v)


def build_exclusions(cache):
    blocked, sources = set(), []
    for path in sorted(Path(cache).rglob('*.arrow')):
        # Protect evaluation splits for the five specified benchmarks, not training rows.
        if not any(n in str(path) for n in ('hellaswag', 'ai2_arc', 'winogrande', 'piqa', 'wikitext')):
            continue
        if not any(n in path.name for n in ('validation', 'test')):
            continue
        for row in arrow_rows(path):
            for s in strings(row):
                blocked.update(grams(s))
        sources.append({'file': path.name, 'sha256': sha256(path)})
    return blocked, sources


def require_benchmark_coverage(sources):
    required={'hellaswag-validation.arrow','ai2_arc-test.arrow','piqa-validation.arrow','winogrande-validation.arrow','wikitext-test.arrow'}
    missing=required-{source['file'] for source in sources}
    if missing:
        raise ValueError('Missing protected benchmark splits: '+', '.join(sorted(missing)))


def prepare(args):
    import pyarrow.parquet as pq
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        raise ValueError('Output already prepared; choose a fresh directory')
    blocked, exclusions = build_exclusions(args.benchmark_cache)
    require_benchmark_coverage(exclusions)
    print(f'Protected {len(blocked):,} benchmark 13-grams', flush=True)
    stats, seen = Counter(), set()
    handles = {name: open(out/f'{name}.jsonl', 'w') for name in ('train', 'dev')}
    def accept(text, source):
        text = normalized(text)
        if len(text) < 200 or len(text) > 200_000:
            stats['length_rejected'] += 1
            return
        key = hashlib.sha256(text.encode()).hexdigest()
        if key in seen:
            stats['duplicate_rejected'] += 1
            return
        seen.add(key)
        if any(g in blocked for g in grams(text)):
            stats['benchmark_overlap_rejected'] += 1
            return
        split = 'dev' if int(key[:8], 16) % 100 == 0 else 'train'
        handles[split].write(json.dumps({'text': text, 'source': source, 'sha256': key}, ensure_ascii=False)+'\n')
        stats[f'{source}_{split}_documents'] += 1
    try:
        rows = 0
        for batch in pq.ParquetFile(args.fineweb).iter_batches(batch_size=512, columns=['text']):
            for text in batch.column(0).to_pylist():
                if rows >= args.fineweb_documents:
                    break
                accept(text, 'fineweb_edu')
                rows += 1
            if rows % 10240 == 0:
                print(f'Filtered {rows:,} FineWeb documents', flush=True)
            if rows >= args.fineweb_documents:
                break
        wiki_paths = sorted(Path(args.benchmark_cache).glob('Salesforce___wikitext/wikitext-103-raw-v1/**/wikitext-train*.arrow'))
        if not wiki_paths:
            raise ValueError('WikiText-103 training files missing')
        for text in wiki_documents(wiki_paths):
            accept(text, 'wikitext_103_train')
    finally:
        for f in handles.values():
            f.close()
    def docs(split):
        with open(out/f'{split}.jsonl') as f:
            for line in f:
                yield json.loads(line)['text']
    # A deterministic uniform sample across both sources, training split only.
    sample = []
    rng = np.random.default_rng(20260929)
    for i, text in enumerate(docs('train')):
        if i < 20000:
            sample.append(text)
        else:
            j = int(rng.integers(i+1))
            if j < len(sample):
                sample[j] = text
    print('Training byte-level BPE on training-only sample', flush=True)
    tokenizer = Tokenizer.train(sample, args.vocab_size)
    tokenizer.save(out/'tokenizer.json')
    counts = {}
    for split in ('train', 'dev'):
        print(f'Encoding {split}', flush=True)
        counts[split] = encode_documents(docs(split), tokenizer, out/f'{split}.bin')
    manifest = {
        'schema': 1, 'seed': 20260929, 'statistics': dict(stats), 'tokens': counts,
        'tokenizer_vocab_size': tokenizer.vocab_size,
        'sources': [
            {'dataset':'HuggingFaceFW/fineweb-edu', 'revision':'87f09149ef4734204d70ed1d046ddc9ca3f2b8f9', 'subset':'sample-10BT', 'shard':'sample/10BT/000_00000.parquet', 'split':'train', 'license':'ODC-By-1.0', 'sha256':sha256(args.fineweb), 'scanned_documents':rows},
            {'dataset':'Salesforce/wikitext', 'revision':'b08601e04326c79dfdd32d625aee71d232d685c3', 'subset':'wikitext-103-raw-v1', 'split':'train', 'license':'CC-BY-SA-3.0/GFDL', 'files':[{'file':p.name, 'sha256':sha256(p)} for p in wiki_paths]}
        ],
        'split_policy':'SHA256(normalized complete document) modulo 100; bucket 0 development',
        'decontamination': {'method':'reject any normalized 13-word overlap with protected benchmark text', 'protected_files':exclusions, 'unique_ngrams':len(blocked), 'limitations':'Exact 13-gram filter does not establish absence of paraphrased or short overlaps.'},
        'files': {p.name:sha256(p) for p in (out/'train.bin', out/'dev.bin', out/'tokenizer.json', out/'train.jsonl', out/'dev.jsonl')}
    }
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'tokens':counts, 'statistics':dict(stats)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fineweb', required=True)
    p.add_argument('--fineweb-documents', type=int, default=100000)
    p.add_argument('--benchmark-cache', required=True)
    p.add_argument('--output', default='data/prepared')
    p.add_argument('--vocab-size', type=int, default=16384)
    prepare(p.parse_args())
