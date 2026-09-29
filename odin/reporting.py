"""Dependency-free validation of published evaluation evidence."""
import math


def validate_report(report,checkpoint_hash):
    if report.get('status')!='full' or report.get('limit') is not None:
        raise ValueError('A full evaluation is required')
    if report.get('checkpoint_sha256')!=checkpoint_hash:
        raise ValueError('Report checkpoint does not match submitted weights')
    expected={'hellaswag':10042,'arc_easy':2376,'piqa':1838,'winogrande':1267}
    for task,count in expected.items():
        result=report.get('results',{}).get(task)
        if not result:
            raise ValueError('Missing official task: '+task)
        if result.get('sample_len')!=count:
            raise ValueError(f'Incomplete {task} split: expected {count} samples')
        if not 0 <= result.get('acc,none',float('nan')) <= 1:
            raise ValueError('Invalid accuracy for '+task)
    wiki=report.get('wikitext_103')
    # The pinned raw stream produces 62 nonempty scored documents plus an
    # empty prefix. The empty prefix has no targets and is not an article.
    identity={'subset':'wikitext-103-raw-v1','split':'test','documents':62,
              'words':241211,'bytes':1287656,
              'text_sha256':'bbf94c53a05abe9ee670d3b6343608095822c85e26de37c70b24fc571964574a'}
    if not wiki or any(wiki.get(key)!=value for key,value in identity.items()):
        raise ValueError('Missing full canonical WikiText-103 test evidence')
    if wiki.get('tokens',0)<1 or not wiki.get('source_sha256'):
        raise ValueError('Invalid WikiText token count or source identity')
    nll=wiki.get('nll',float('nan'))
    if not math.isfinite(nll) or nll<0:
        raise ValueError('Invalid WikiText likelihood')
    expected={'token_perplexity':math.exp(nll/wiki['tokens']),
              'word_perplexity':math.exp(nll/wiki['words']),
              'bits_per_byte':nll/(wiki['bytes']*math.log(2))}
    for key,value in expected.items():
        if not math.isclose(wiki.get(key,float('nan')),value,rel_tol=1e-8):
            raise ValueError('Inconsistent WikiText metric: '+key)
