"""Standard harness metrics over observed development data; historical code stays frozen."""
import argparse
import json
import math
from pathlib import Path
from odin import controlled_eval
from odin.data import sha256
from odin.experiment import atomic_json


def standard_question(row, question):
    raw = row['raw_scores']
    choices = question['choices']
    if (row['id'] != question['id'] or row['expected'] != question['expected'] or
            len(raw) != len(choices) or not raw or
            any(not math.isfinite(s) for s in raw) or
            any(not c.startswith(' ') or len(c) < 2 for c in choices)):
        raise ValueError('Invalid question identity, scores, or answer separator')
    normalized = [score / len(choice[1:]) for score, choice in zip(raw, choices)]
    pred = max(range(len(raw)), key=raw.__getitem__)
    norm = max(range(len(raw)), key=normalized.__getitem__)
    return {**row, 'token_mean_correct':row['correct'], 'token_mean_predicted':row['predicted'],
            'character_scores':normalized, 'predicted':pred, 'correct':pred == row['expected'],
            'predicted_norm':norm, 'correct_norm':norm == row['expected']}


def protocol(data, prompts, device, windows=256):
    return {'schema':1, 'partition':'observed-development', 'choice_score':'raw summed log likelihood',
            'secondary_score':'likelihood / answer character count excluding inserted separator',
            'wrapper_sha256':sha256(__file__),
            'parts':[controlled_eval.protocol(data, prompts, part, device, windows, part == 'screen')
                     for part in ('screen', 'confirmation')]}


def validate_report(report, checkpoint, data, prompts, device, windows=256):
    p = protocol(data, prompts, device, windows)
    if (report['checkpoint_sha256'] != sha256(checkpoint) or report['protocol'] != p or
            report['protocol_sha256'] != controlled_eval.digest(p)):
        raise ValueError('Stale checkpoint or evaluation protocol')
    source = json.loads((Path(data)/'questions.json').read_text())
    questions = [q for part in ('screen', 'confirmation') for q in source if q['partition'] == part]
    if [r['id'] for r in report['questions']] != [q['id'] for q in questions]:
        raise ValueError('Incomplete question coverage')
    for row, q in zip(report['questions'], questions):
        original = {**row, 'correct':row['token_mean_correct'], 'predicted':row['token_mean_predicted']}
        if standard_question(original, q) != row:
            raise ValueError('Invalid derived question score')
    for name, field in (('accuracy','correct'), ('accuracy_norm','correct_norm')):
        if not questions or report[name] != sum(r[field] for r in report['questions'])/len(questions):
            raise ValueError('Invalid accuracy aggregate')
    if set(report['domains']) != {'fineweb','wiki','stories'}:
        raise ValueError('Incomplete domains')
    if any(not math.isfinite(v['nll']) or v['nll'] < 0 or v['tokens'] != windows * 512 for v in report['domains'].values()):
        raise ValueError('Invalid domain loss or coverage')
    suite = json.loads(Path(prompts).read_text())
    if [(r['prompt'],r['seed']) for r in report['generation']] != [(p,s) for p in suite['generation'] for s in suite['seeds']]:
        raise ValueError('Incomplete generation suite')
    return True


def evaluate(checkpoint, data, prompts, device='cpu', windows=256):
    p = protocol(data, prompts, device, windows)
    parts = [controlled_eval.evaluate(checkpoint, data, prompts, partition=part, device=device,
                                     windows=windows, generation=part == 'screen')
             for part in ('screen','confirmation')]
    if parts[0]['checkpoint_sha256'] != parts[1]['checkpoint_sha256'] or parts[0]['domains'] != parts[1]['domains']:
        raise ValueError('Checkpoint or deterministic domains changed between partitions')
    source = {q['id']:q for q in json.loads((Path(data)/'questions.json').read_text())}
    questions = [standard_question(r, source[r['id']]) for part in parts for r in part['questions']]
    result = {k:parts[0][k] for k in ('checkpoint_sha256','trained_tokens','parameter_count','domains','generation')}
    result.update(protocol=p, protocol_sha256=controlled_eval.digest(p), questions=questions,
                  accuracy=sum(r['correct'] for r in questions)/len(questions),
                  accuracy_norm=sum(r['correct_norm'] for r in questions)/len(questions),
                  status='Observed ARC validation development set; not official test or untouched confirmation')
    validate_report(result, checkpoint, data, prompts, device, windows)
    return result


def compare(control, candidate, expected_questions=570):
    if control['protocol_sha256'] != candidate['protocol_sha256']:
        raise ValueError('Mismatched evaluation protocol')
    for report in (control, candidate):
        if report['trained_tokens'] != 1000013824 or report['parameter_count'] != 49295872 or len(report['questions']) != expected_questions:
            raise ValueError('Mismatched model budget or question coverage')
        if set(report['domains']) != {'fineweb','wiki','stories'} or any(not math.isfinite(v['nll']) or v['nll'] < 0 for v in report['domains'].values()):
            raise ValueError('Invalid evaluation domains')
    intervals = {}
    for metric in ('correct','correct_norm'):
        rows = [[{**r, 'correct':r[metric]} for r in report['questions']] for report in (control,candidate)]
        intervals[metric] = controlled_eval.paired_interval(*rows)
    ratios = {d:math.exp(candidate['domains'][d]['nll']-control['domains'][d]['nll']) for d in ('fineweb','wiki','stories')}
    gates = {'fineweb_ppl_reduction_at_least_3pct':ratios['fineweb'] <= .97,
             'wiki_ppl_regression_at_most_2pct':ratios['wiki'] <= 1.02,
             'raw_accuracy_improves':intervals['correct']['difference'] > 0,
             'normalized_regression_at_most_1pp':intervals['correct_norm']['difference'] >= -.01}
    return {'paired_intervals':intervals, 'perplexity_ratios_candidate_over_control':ratios,
            'gates':gates, 'numeric_gate_passed':all(gates.values()),
            'decision':'Pending complete generation review; no automatic continuation or promotion',
            'limitations':'One training seed; observed development questions. Intervals cover question sampling only.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default='data/controlled/common')
    parser.add_argument('--prompts', default='experiments/beta-prompts.json')
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--output', required=True)
    args = vars(parser.parse_args()); output = args.pop('output')
    report = evaluate(**args)
    atomic_json(output, report)
    print(json.dumps({k:report[k] for k in ('checkpoint_sha256','accuracy','accuracy_norm','domains')}), flush=True)
