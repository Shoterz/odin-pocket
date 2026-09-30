"""Reconstruct deadline-research measurements from saved artifacts; no inference."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {}


def read(relative):
    path = ROOT / relative
    content = path.read_bytes()
    INPUTS[relative] = hashlib.sha256(content).hexdigest()
    return json.loads(content)


def main():
    questions = {q['id']: q for q in read('data/controlled/common/questions.json')}
    scores = {}
    for name in ('published', 'previous-B', 'short', 'long'):
        report = read(f'results/warmup-confirmation/confirmation-{name}.json')
        counts = dict(token_mean=0, raw_sum=0, character_normalized=0, raw_token_disagreements=0)
        for row in report['questions']:
            question = questions[row['id']]
            assert question['partition'] == 'confirmation'
            assert question['expected'] == row['expected']
            assert all(c.startswith(' ') and len(c) > 1 for c in question['choices'])
            raw = row['raw_scores']
            assert all(math.isfinite(x) for x in raw)
            # controlled_data adds exactly one separator space; harness doc_to_choice excludes it.
            char_scores = [s / len(c[1:]) for s, c in zip(raw, question['choices'])]
            choose = lambda values: max(range(len(values)), key=values.__getitem__)
            preds = {'token_mean': choose(row['scores']), 'raw_sum': choose(raw),
                     'character_normalized': choose(char_scores)}
            assert preds['token_mean'] == row['predicted']
            for metric, pred in preds.items():
                counts[metric] += int(pred == row['expected'])
            counts['raw_token_disagreements'] += int(preds['raw_sum'] != preds['token_mean'])
        scores[name] = dict(n=len(report['questions']), **counts)
    expected = {'published': (77, 100, 79), 'previous-B': (89, 120, 103),
                'short': (78, 100, 84), 'long': (81, 117, 87)}
    for name, values in expected.items():
        assert tuple(scores[name][k] for k in ('token_mean', 'raw_sum', 'character_normalized')) == values
        assert scores[name]['n'] == 285
    runs = {}
    for warmup in (200, 1000):
        stem = f'runs/controlled/warmup-confirm-{warmup}'
        summary = read(stem + '/summary.json')
        path = ROOT / stem / 'metrics.jsonl'
        INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        last = json.loads(path.read_text().splitlines()[-1])
        assert last['tokens'] == summary['tokens'] == 1000013824
        runs[str(warmup)] = {'tokens': summary['tokens'], 'seconds': summary['training_seconds'],
                            'peak_allocated_gib': last['peak_vram_bytes'] / 2**30,
                            'hours_per_billion_tokens': summary['training_seconds'] / 3600 * 1e9 / summary['tokens']}
    manifest = read('data/controlled/common/manifest.json')
    long = read('runs/controlled/warmup-confirm-1000/summary.json')
    recipe, config = long['recipe'], long['config']
    batch_tokens = recipe['batch_size'] * recipe['accumulation'] * config['context']
    hours = sum(r['hours_per_billion_tokens'] for r in runs.values()) / len(runs)
    result = {
        'schema': 1, 'audit_date': '2026-09-30',
        'deadline': '2026-10-01T23:45:00+08:00',
        'deadline_source': 'https://gibc-v2.devpost.com/rules',
        'status': 'Read-only reconstruction; scoring is posthoc on observed validation questions, not an official harness run or independent confirmation.',
        'scoring': scores, 'runs': runs, 'config': config, 'recipe': recipe,
        'parameter_count': long['parameter_count'], 'batch_tokens': batch_tokens,
        'adam_second_moment_half_life_tokens': {str(b): batch_tokens * math.log(.5) / math.log(b) for b in (.95, .99, .999)},
        'beta2_for_10m_token_half_life': 2 ** (-batch_tokens / 10_000_000),
        'source_token_inventory': manifest['source_tokens'],
        'source_exposure_equivalents_1b_run': {k: v / manifest['source_tokens'][k.removesuffix('.bin')] for k, v in long['source_tokens'].items()},
        'estimated_hours_same_recipe': {str(b) + 'B': hours * b for b in (1, 1.5, 2, 3, 4)},
        'hypothetical_22_layer_384_width_ff1152_params': 16384 * 384 + 22 * (4 * 384**2 + 3 * 384 * 1152 + 2 * 384) + 384,
        'input_sha256': INPUTS,
    }
    output = ROOT / 'results/research/deadline-audit-2026-09-30.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('scoring', 'estimated_hours_same_recipe', 'adam_second_moment_half_life_tokens', 'source_exposure_equivalents_1b_run')}, indent=2))


if __name__ == '__main__':
    main()
