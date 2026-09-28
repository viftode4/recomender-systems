"""Export aggregate-only evidence from a completed negative-information audit."""
import argparse
import json
from pathlib import Path

import numpy as np

from study import digest, write_json


GROUPS = ('positive', 'plain', 'surprise', 'weight_permutation', 'placement_null')
LABELS = {'positive': 'Positive only', 'plain': 'Plain dislike channels',
          'surprise': 'Surprise-weighted dislikes', 'weight_permutation': 'Permuted weights (mean of 5)',
          'placement_null': 'Placement null (mean of 5)'}


def names(group):
    return [f'{group}_{r}' for r in range(5)] if group in ('weight_permutation', 'placement_null') else [group]


def export(source, destination):
    plan = json.loads((source/'plan.json').read_text())
    seeds = plan['seeds']
    summary = json.loads((source/'summary.json').read_text())
    if set(summary) != set(map(str, seeds)):
        raise ValueError('Missing seed summaries')
    evidence = {'plan': plan, 'summary': summary, 'seeds': {}}
    code_signature = None
    for seed in seeds:
        folder = source/str(seed)
        manifest = json.loads((folder/'manifest.json').read_text())
        diagnostics = json.loads((folder/'diagnostics.json').read_text())
        if (manifest['status'] != 'complete' or manifest['test_read'] or manifest['test_evaluated']
                or manifest['validation_used_for_decoder_or_teacher_fit']):
            raise ValueError('Invalid completed audit manifest')
        if code_signature is not None and code_signature != manifest['source_sha256']:
            raise ValueError('Seeds used different source code')
        code_signature = manifest['source_sha256']
        for filename, expected in code_signature.items():
            if digest(Path(__file__).parent/filename) != expected:
                raise ValueError(f'Current source differs from recorded audit: {filename}')
        for control in diagnostics['placement_controls'].values():
            checks = [control['combined_invariants'], control['context']['invariants'], control['probe']['invariants']]
            if not all(all(part.values()) for part in checks):
                raise ValueError('Control invariant failure')
        # These source objects contain aggregate counts, hashes, settings and
        # intervals only. Per-user metrics, IDs and rating records stay in runs/.
        evidence['seeds'][str(seed)] = {
            'selection': json.loads((folder/'selection.json').read_text()),
            'contrasts': json.loads((folder/'contrasts.json').read_text()),
            'diagnostics': diagnostics,
            'elapsed_seconds': manifest['elapsed_seconds'],
            'split_sha256': manifest['split_sha256'],
            'data_sha256': manifest['data_sha256'],
            'artifacts_sha256': {name: digest(folder/name) for name in
                                 ('protocol.json', 'manifest.json', 'selection.json', 'selection-grid.json',
                                  'metrics.json', 'contrasts.json', 'diagnostics.json', 'timings.json')},
        }
    evidence['source_sha256'] = code_signature
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination/'aggregate.json', evidence)
    primary = {group: float(np.mean([summary[str(s)]['fixed']['context'][n]['liked_ndcg10']
                                   for s in seeds for n in names(group)])) for group in GROUPS}
    findings = (f"The primary comparison gives mean liked nDCG@10 {primary['positive']:.4f} for positive-only, "
                f"{primary['plain']:.4f} for plain dislike channels, and {primary['surprise']:.4f} for surprise weights. "
                f"The matched placement controls score {primary['placement_null']:.4f}, and shuffled weights "
                f"score {primary['weight_permutation']:.4f}.")
    placement_intervals = [evidence['seeds'][str(s)]['contrasts'][budget][mode]['plain_minus_placement_null']
                           ['descriptive_user_bootstrap_interval_95']
                           for s in seeds for budget in ('fixed', 'selected') for mode in ('context', 'full_train')]
    if all(lo <= 0 <= hi for lo, hi in placement_intervals):
        findings += (' Every plain-minus-placement-control interval includes zero, across both penalty choices '
                     'and both inference modes. This audit does not demonstrate a stable benefit from the '
                     'personalized placement of rejection labels under this estimator; it does not establish '
                     'that dislikes lack useful information for every model.')
    lines = [
        '# Personalized rejection information audit', '',
        findings, '',
        'Validation-only exploratory evidence under a protocol fixed before decoder fitting. '
        'The strict context-only comparison is primary; full-history prediction is a predeclared secondary check. '
        'The same decoder is used in both modes. No test labels or test metrics were opened.', '',
        'The decoder predicts disjoint probe likes from context ratings. The positive teacher also fits context only. '
        'All original training items, including probe observations, are masked from validation recommendations. '
        'Likes mean actual ratings >=4; dislikes mean ratings <=2; rating 3 is neutral. Missing ratings are unknown.', '',
        'The common-penalty comparison uses ridge 250. Equal-grid selection uses [50,250,1000] and only '
        'strict-context meta-fit liked nDCG@10. Each control branch selects one common penalty from the average '
        'of all five control replicates; no replicate is selected. Both inference modes reuse that choice.', '',
    ]
    for budget in ('fixed', 'selected'):
        for mode in ('context', 'full_train'):
            lines += [f'## {mode}: {budget} penalty', '',
                      '| Branch | '+' | '.join(map(str, seeds))+' | Mean nDCG@10 | Known dislikes/slot |',
                      '|---|'+'---:|'*(len(seeds)+2)]
            for group in GROUPS:
                values = [np.mean([summary[str(s)][budget][mode][n]['liked_ndcg10'] for n in names(group)]) for s in seeds]
                dislike = np.mean([summary[str(s)][budget][mode][n]['known_dislike_rate_per_slot']
                                   for s in seeds for n in names(group)])
                lines.append('| '+LABELS[group]+' | '+' | '.join(f'{v:.4f}' for v in values)
                             +f' | {np.mean(values):.4f} | {dislike:.2%} |')
            lines += ['', 'Control replicate ranges (not confidence intervals):', '',
                      '| Control | '+' | '.join(map(str, seeds))+' |', '|---|'+'---:|'*len(seeds)]
            for group in ('weight_permutation', 'placement_null'):
                ranges = []
                for s in seeds:
                    values = [summary[str(s)][budget][mode][n]['liked_ndcg10'] for n in names(group)]
                    ranges.append(f'{min(values):.4f}–{max(values):.4f}')
                lines.append('| '+LABELS[group]+' | '+' | '.join(ranges)+' |')
            lines += ['', 'Paired liked nDCG@10 differences with descriptive user-bootstrap 95% intervals:', '',
                      '| Contrast | '+' | '.join(map(str, seeds))+' |', '|---|'+'---:|'*len(seeds)]
            for comparison in ('plain_minus_positive', 'surprise_minus_plain',
                               'surprise_minus_weight_permutation', 'plain_minus_placement_null'):
                values = []
                for s in seeds:
                    result = evidence['seeds'][str(s)]['contrasts'][budget][mode][comparison]
                    lo, hi = result['descriptive_user_bootstrap_interval_95']
                    values.append(f"{result['mean_delta']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
                lines.append('| '+comparison.replace('_', ' ')+' | '+' | '.join(values)+' |')
            lines.append('')
    lines += ['## Control and label-access diagnostics', '',
              '| Seed | Decoder fit / excluded users | Fitting users without probe likes | Context / probe observations | Context moved dislikes | Context acceptance |',
              '|---|---:|---:|---:|---:|---:|']
    for seed in seeds:
        d = evidence['seeds'][str(seed)]['diagnostics']
        movement = [v['context']['moved_negative_fraction'] for v in d['placement_controls'].values()]
        acceptance = [v['context']['acceptance_fraction'] for v in d['placement_controls'].values()]
        lines.append(f"| {seed} | {d['decoder_fit_users']} / {d['decoder_fit_excluded_users']} | "
                     f"{d['decoder_fit_users_without_probe_likes']} | {d['context_observations']} / {d['probe_observations']} | "
                     f'{min(movement):.1%}–{max(movement):.1%} | {min(acceptance):.1%}–{max(acceptance):.1%} |')
    lines += ['',
        'All exact support, positive-label, per-user and per-item dislike margins passed, including item margins '
        'within the context-defined decoder fitting stratum. Full-history null queries retain the exact switched '
        'context and add independently switched probe labels. Complete aggregate diagnostics and hashes are in `aggregate.json`.', '',
        'A first preparation was interrupted after independent review identified a fitted-subset margin issue, '
        'before any decoder fit or validation metric. The corrected, documented protocol bases fitting eligibility '
        'only on context and retains zero-probe-like target rows. The interrupted record is preserved in ignored runs/.', '',
        '## Limits', '',
        'The five switch chains are descriptive perturbations. Structural zeros can disconnect their state space; '
        'acceptance and moved-label fractions do not prove uniform sampling or convergence. User-bootstrap '
        'intervals are conditional on these draws, are not multiplicity-adjusted, and are not randomization '
        'p-values. Repeated splits share users, and these validation users were used in earlier project studies.', '',
        'Known-dislike rates measure only rated held-out dislikes and cannot certify satisfaction. The full-history '
        'mode adds known input history and shifts its length relative to fitting; it is never evaluated as held-out '
        'probe reconstruction. Surprise versus shuffled weights alone cannot separate personalized surprise from '
        'item-popularity-correlated teacher scores. Hard-negative weighting has prior art; no method-priority or '
        'state-of-the-art claim follows from this audit.', '',
        '## Reproduction', '',
        'Run `negative_information_experiment.py --source-root runs/research-v2 --ratings PATH/ml-100k.inter '
        '--item-metadata PATH/ml-100k.item --out runs/negative-information-v2 --seeds 2026 2027 2028`, '
        'then `negative_information_report.py --source runs/negative-information-v2 --out evidence/negative-information-v2`. '
        'Use the pinned project environment and set OPENBLAS_NUM_THREADS, OMP_NUM_THREADS, '
        'VECLIB_MAXIMUM_THREADS, MKL_NUM_THREADS and NUMEXPR_NUM_THREADS to 1. Output directories must be new. '
        'The default records score hashes without storing large raw score archives; `--save-scores` retains them.', '',
        'The pre-fit specification is [NEGATIVE_INFORMATION_PROTOCOL.md](../../NEGATIVE_INFORMATION_PROTOCOL.md). '
        'Primary references: [LAGCL4Rec](https://aclanthology.org/2025.findings-emnlp.61.pdf) and '
        '[Rapallo and Yoshida](https://arxiv.org/abs/0905.4841).', '',
    ]
    (destination/'RESULTS.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.out)


if __name__ == '__main__':
    main()
