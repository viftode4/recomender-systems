"""Create aggregate development evidence, with protocol facts read from artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


FAMILIES = ['expert', 'constrained', 'calibrated', 'static', 'user', 'item', 'disagreement',
            'context', 'static-pairwise', 'context-pairwise']
MODES = {'diversity': ('diversity', 'Genre diversity (higher is more diverse)'),
         'calibration': ('calibration_jsd', 'Genre JSD (lower is better calibrated)'),
         'exposure': ('head_exposure', 'Raw head-item exposure share'),
         'popularity_calibration': ('popularity_jsd', 'Popularity JSD / UPD (lower is better)')}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def aggregate_policy(policy):
    """Keep policy decisions/bounds/counts while excluding individual identifiers."""
    if isinstance(policy, list):
        return [aggregate_policy(value) for value in policy]
    if not isinstance(policy, dict):
        return policy
    result = {}
    for key, value in policy.items():
        if key == 'per_user':
            continue
        if key in ('calibration_users', 'fit_users', 'selection_users') and isinstance(value, list):
            result[key.removesuffix('_users') + '_user_count'] = len(value)
        else:
            result[key] = aggregate_policy(value)
    return result


def cohort_summary(manifest, cohorts):
    """Validate counts before describing a two- or three-cohort protocol."""
    labels = ('meta_fit', 'development', 'calibration')
    groups = {label: set(cohorts.get(label, [])) for label in labels}
    if any(len(groups[label]) != len(cohorts.get(label, [])) for label in labels):
        raise ValueError('Duplicate cohort users')
    if any(groups[a] & groups[b] for a, b in [('meta_fit', 'development'),
            ('meta_fit', 'calibration'), ('development', 'calibration')]):
        raise ValueError('Study cohorts overlap')
    counts = {label: len(groups[label]) for label in labels}
    if any(counts[label] != manifest.get(label + '_users', 0) for label in labels):
        raise ValueError('Cohort counts differ from study manifest')
    return counts


def load_evidence(runs):
    plan = read_json(runs / 'plan.json')
    if plan['status'] != 'complete' or not plan['seeds']:
        raise ValueError('Experiment suite is incomplete or has no seeds')
    data = {key: {} for key in ('selected', 'aggregates', 'candidate-feasibility',
            'candidate-budgets', 'group-policies', 'provenance', 'cohort-counts', 'individual-models')}
    studies = {}
    for seed in plan['seeds']:
        directory = runs / f'study-{seed}'
        manifest = read_json(directory / 'manifest.json')
        if manifest['status'] != 'complete' or manifest['test_read']:
            raise ValueError('Expected completed validation-only study')
        if any(v['manifest'].get('validation_used_for_training', True) for v in manifest['sources'].values()):
            raise ValueError('Primary evidence requires fixed-budget expert training')
        results = read_json(directory / 'results.json')
        selection = read_json(directory / 'selection.json')
        if selection['metric'] != f"ndcg@{manifest['k']}":
            raise ValueError('Study cutoff differs from selection metric')
        names = {'expert': selection['best_expert'], **selection['families']}
        data['selected'][seed] = {family: {'name': name, **results[name]['aggregate']}
                                  for family, name in names.items()}
        data['aggregates'][seed] = {name: {key: value[key] for key in
            ('aggregate', 'groups', 'item_groups', 'item_exposure') if key in value}
            for name, value in results.items()}
        data['candidate-feasibility'][seed] = read_json(directory / 'candidate-feasibility.json')
        data['candidate-budgets'][seed] = read_json(directory / (names['context'] + '-candidate-budget.json'))
        data['group-policies'][seed] = aggregate_policy(read_json(directory / 'group-policy.json'))
        data['cohort-counts'][seed] = cohort_summary(manifest, read_json(directory / 'cohorts.json'))
        data['provenance'][seed] = {'manifest': manifest,
            **{key + '_sha256': digest(directory / (key + '.json')) for key in ('results', 'coefficients', 'cohorts')}}
        individual = {}
        for name, source in manifest['sources'].items():
            description = source['manifest']
            label = description['model']
            objective = description.get('selection', {}).get('selection_objective')
            if objective:
                label += ':' + objective
            if label in individual:
                raise ValueError(f'Ambiguous individual-model label {label}; export distinct model identities')
            individual[label] = {'run': name, 'model': description['model'],
                'settings': description.get('settings', {}), 'selection': description.get('selection'),
                'protocol': description.get('protocol'), 'timing_seconds': description.get('timing_seconds'),
                **data['aggregates'][seed][name]}
        data['individual-models'][seed] = individual
        studies[seed] = directory
    if len({v['manifest']['k'] for v in data['provenance'].values()}) != 1:
        raise ValueError('Cannot aggregate different ranking cutoffs')
    families = set(next(iter(data['selected'].values())))
    if any(set(row) != families for row in data['selected'].values()):
        raise ValueError('Model families differ across seeds')
    return plan, data, studies


def mean_available(rows, metric):
    values = [row[metric] for row in rows if row.get(metric) is not None]
    return statistics.mean(values) if values else None


def formatted(value):
    return '-' if value is None else f'{value:.4f}'


def summary_text(plan, data):
    seeds, rows, full = plan['seeds'], data['selected'], data['aggregates']
    families = [f for f in FAMILIES if f in rows[seeds[0]]]
    families += sorted(set(rows[seeds[0]]) - set(families))
    k = data['provenance'][seeds[0]]['manifest']['k']
    metric = f'ndcg@{k}'
    means = {f: statistics.mean(rows[s][f][metric] for s in seeds) for f in families}
    text = ['# Research checkpoint: development evidence', '',
        f'{len(seeds)} MovieLens 100K split(s), with cohort sizes verified against each study manifest. '
        'Expert settings and regression coefficients use meta-fit labels. Hybrid settings use development '
        'labels, so reported development comparisons are selected and exploratory, not unbiased test estimates.', '',
        '| Seed | Meta-fit users | Development / model-selection users | Independent policy-calibration users |',
        '|---|---:|---:|---:|']
    for seed in seeds:
        c = data['cohort-counts'][seed]
        text.append(f'| {seed} | {c["meta_fit"]} | {c["development"]} | {c["calibration"]} |')
    text += ['', 'These cohorts share the train-fitted recommenders; this is not user cold-start. '
        'Repeated split seeds share users and interactions and are not independent populations.', '',
        '![Research figures](research-figures.png)', '', '## Individual models', '',
        'Each row is a source model passed into the hybrid study; model identities and settings come from '
        'source manifests. Values average available seeds, whose count is explicit. Rating-aware custom '
        'models may use richer signals than implicit baselines; their matched-information controls belong '
        'in the supplementary contrast study. No item-rating satisfaction claim follows from the main '
        'all-observed-interactions ranking protocol.', '',
        f'| Model | Seeds | nDCG@{k} | Recall@{k} | Diversity | Genre JSD | Popularity JSD | Exposure Gini |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    labels = sorted(set().union(*(set(v) for v in data['individual-models'].values())))
    for label in labels:
        values = [data['individual-models'][s][label]['aggregate'] for s in seeds if label in data['individual-models'][s]]
        metrics = [metric, f'recall@{k}', 'diversity', 'calibration_jsd', 'popularity_jsd', 'exposure_gini']
        text.append('| ' + label + f' | {len(values)} | ' + ' | '.join(formatted(mean_available(values, m)) for m in metrics) + ' |')
    text += ['', 'Exact selected settings, timings and group metrics are in `individual-models.json`. '
        'A dash means the historical run did not record that metric.', '', '## Architecture and ablations', '',
        'Static and contextual regression use the same expert-score inputs. Contextual weights depend on '
        'training history size, genre entropy and item popularity; the disagreement feature and squared '
        'pairwise margin loss have separate ablations. These techniques have prior art.', '']
    if 'constrained' in families:
        text += ['The lecture-aligned constrained ridge baseline enforces sum(weights)=1 with an unpenalized '
                 'intercept. Nonnegativity is not imposed; this is an affine combination, not a convex mixture.', '']
    if 'calibrated' in families:
        text += ['The calibrated constrained variant first fits a nonnegative-slope affine response transform '
                 'for each expert on the same meta-fit observations, then fits sum-to-one ridge weights. '
                 'This tests sensitivity to the scale imposed by z-scores and binary regression targets; '
                 'the original constrained result is retained. The transformed scores need not be probabilities. '
                 'The sum-to-one constraint applies before composing calibration with the final score; '
                 'both calibration and effective coefficients are saved. This correction followed development '
                 'inspection and remains exploratory.', '']
    text += ['| Family | ' + ' | '.join(map(str, seeds)) + f' | Mean nDCG@{k} |',
             '|---|' + '---:|' * (len(seeds) + 1)]
    for family in families:
        text.append('| ' + family + ' | ' + ' | '.join(formatted(rows[s][family][metric]) for s in seeds)
                    + f' | {means[family]:.4f} |')
    best = max(means, key=means.get)
    text += ['', f'The highest mean selected development score is **{best}** ({means[best]:.4f}). '
        f'Full contextual ridge differs from the selected individual expert by **{means["context"] - means["expert"]:+.4f} nDCG**. '
        'This is a description of development results, not a generalization or significance claim. '
        'Paired-comparison intervals remain descriptive because this cohort selected the models.', '',
        '## Candidate feasibility', '',
        'With H head and T tail candidates, feasible head counts in a length-k list lie between '
        'max(0,k-T) and min(k,H). A target q is feasible exactly when it lies in that interval. '
        'Reranking cannot recover an item absent from its candidate set.', '',
        '| Seed | Quota checks | Target head / tail | Mean pool | Users expanded | Same top-k as full pool |',
        '|---|---|---:|---:|---:|---:|']
    for seed in seeds:
        b = data['candidate-budgets'][seed]
        q = b.get('target_head_items', k - b['target_tail_items'])
        checks = b.get('quota_checks', 'tail_only (historical)')
        text.append(f'| {seed} | {checks} | {q} / {b["target_tail_items"]} | {b["mean_pool"]:.1f} '
                    f'| {b["fraction_expanded"]:.1%} | {b["same_topk_as_full"]:.1%} |')
    text += ['', 'New runs expand to the shortest prefix supporting both target counts, with minimum '
        'pool 100. Historical runs explicitly labelled tail-only did not check head support. The measured '
        'comparison uses exposure strength 1. Full-catalog expert scores are already computed, so pool size '
        'is reranking work, not a measured total latency speedup. Agreement is empirical.', '',
        '## Reranking objectives and order', '',
        'Genre calibration compares genre distributions. Popularity calibration is a user-side JSD/UPD '
        'objective matching each user\'s training head/tail distribution. Item exposure instead uses the '
        'catalog head/tail proportions as a chosen shared target. Exposure Gini and entropy include '
        'zero-exposure items and use logarithmic position discounts; raw head share does not.', '',
        f'| Contextual reranker (strength 0.5) | Seeds | nDCG@{k} | Diversity | Genre JSD | Popularity JSD | Raw head share | Exposure Gini |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for mode in ['base', *MODES]:
        values = []
        for seed in seeds:
            name = rows[seed]['context']['name'] + ('' if mode == 'base' else f'/{mode}-0.5')
            if name in full[seed]:
                values.append(full[seed][name]['aggregate'])
        if values:
            metrics = [metric, 'diversity', 'calibration_jsd', 'popularity_jsd', 'head_exposure', 'exposure_gini']
            text.append('| ' + mode + f' | {len(values)} | ' + ' | '.join(formatted(mean_available(values, m)) for m in metrics) + ' |')
    text += ['', '![Reranker trade-offs](reranker-tradeoffs.png)', '',
        '| Order comparison at strength 0.5 | Seeds | Before RRF nDCG | After RRF nDCG | Before objective | After objective |',
        '|---|---:|---:|---:|---:|---:|']
    for mode, (objective, _) in MODES.items():
        pairs = [(full[s][f'rerank-experts-then-rrf/{mode}']['aggregate'], full[s][f'rrf-then-rerank/{mode}']['aggregate'])
                 for s in seeds if f'rerank-experts-then-rrf/{mode}' in full[s] and f'rrf-then-rerank/{mode}' in full[s]]
        if pairs:
            before, after = zip(*pairs)
            text.append(f'| {mode} ({objective}) | {len(pairs)} | ' + ' | '.join(formatted(mean_available(v, m))
                        for v, m in [(before, metric), (after, metric), (before, objective), (after, objective)]) + ' |')
    text += ['', 'Both orderings retain full ranking support. Before-fusion reranking promotes each expert\'s '
        'reranked top-k and appends remaining candidates before RRF. Each seed\'s individual and contextual '
        'reranker metrics, discounted item-group exposure and activity-group diagnostics remain in '
        '`aggregates.json`, including settings that perform poorly.', '', '## Group policy audit', '',
        '| Seed | Policy fit | Target retention | Sparse development | Medium development | Dense development |',
        '|---|---|---:|---:|---:|---:|']
    for seed in seeds:
        policy = data['group-policies'][seed]
        independent = policy.get('independent_calibration')
        method = 'independent calibration / ' + independent.get('bound_method', 'unspecified bound') if independent else 'meta-fit reuse (historical)'
        retention = policy['development_retention']
        text.append(f'| {seed} | {method} | {policy["fit_retention_floor"]:.1%} | '
                    + ' | '.join('-' if str(g) not in retention else f'{retention[str(g)]:.1%}' for g in range(3)) + ' |')
    text += ['', 'Independent calibration users are excluded from fitting and model selection. The new policy '
        'uses paired, approximately Bonferroni-adjusted bootstrap lower bounds to screen retention loss '
        'before minimizing exposure gap; small groups fall back to the baseline. This is not a distribution-free '
        'guarantee. Historical meta-fit reuse is labelled separately. Development selected the base model, '
        'so development retention is still a descriptive audit; reserved test is the final check. Report '
        'absolute and worst-group quality, since a smaller group gap can accompany harm to every group.', '',
        '## Evidence and remaining work', '',
        'Source manifests, score/split hashes, coefficient artifacts, cohort counts and model identities '
        'accompany this summary. Raw user histories and recommendation lists stay in ignored run directories. '
        'Clean-environment reproduction, temporal sensitivity and frozen held-out evaluation require their '
        'own evidence artifacts; their completion is not inferred from these development results. '
        'This evidence summary is not the task-formatted '
        'final report and does not establish novelty, emotional understanding, or a grade.']
    return '\n'.join(text) + '\n'


def render_figures(plan, data, out):
    import os
    import tempfile
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'recsys-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    seeds, rows, full = plan['seeds'], data['selected'], data['aggregates']
    metric = f"ndcg@{data['provenance'][seeds[0]]['manifest']['k']}"
    families = [f for f in FAMILIES if f in rows[seeds[0]]]
    colors = plt.get_cmap('tab10')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout='constrained')
    for j, seed in enumerate(seeds):
        color = colors(j % 10)
        axes[0, 0].plot(range(len(families)), [rows[seed][f][metric] for f in families], 'o-', color=color, label=str(seed))
        context = rows[seed]['context']['name']
        values = [v['aggregate'] for n, v in full[seed].items() if n == context or n.startswith(context + '/exposure-') or n == context + '/adaptive-pool-exposure']
        values.sort(key=lambda v: v['head_exposure'])
        axes[0, 1].plot([v['head_exposure'] for v in values], [v[metric] for v in values], 'o-', color=color, label=str(seed))
        feasibility = [r for r in data['candidate-feasibility'][seed] if r['model'] == context]
        axes[1, 0].plot([r['pool'] for r in feasibility], [r['fraction_quota_feasible'] for r in feasibility], 'o-', color=color, label=str(seed))
        retention = data['group-policies'][seed]['development_retention']
        groups = sorted(map(int, retention))
        axes[1, 1].plot(groups, [retention[str(g)] for g in groups], 'o-', color=color, label=str(seed))
    axes[0, 0].set_xticks(range(len(families)), families, rotation=35, ha='right')
    axes[0, 0].set(title='Feature, constraint and loss comparisons', ylabel='Selected development ' + metric)
    axes[0, 1].axvline(.2, color='gray', ls='--', lw=1, label='Approximate catalog head share')
    axes[0, 1].set(title='Item-exposure trade-off', xlabel='Raw head-item exposure share', ylabel='Development ' + metric)
    axes[1, 0].set(xscale='log', ylim=(-.03, 1.03), title='Candidate support limits quota feasibility', xlabel='Candidate pool size', ylabel='Users with feasible quota')
    floors = sorted({p['fit_retention_floor'] for p in data['group-policies'].values()})
    for floor in floors:
        axes[1, 1].axhline(floor, color='gray', ls='--', lw=1, label=f'Policy target {floor:.0%}')
    axes[1, 1].set_xticks([0, 1, 2], ['Sparse', 'Medium', 'Dense'])
    axes[1, 1].set(title='Policy audit after calibration', ylabel='Development group nDCG / base nDCG')
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Hybrid recommendation: exploratory development evidence\nCohort sizes and policy fit scope are documented in RESULTS.md', fontsize=14)
    for suffix in ('png', 'pdf'):
        fig.savefig(out / ('research-figures.' + suffix), dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for ax, (mode, (objective, label)) in zip(axes.flat, MODES.items()):
        plotted = False
        for j, seed in enumerate(seeds):
            context = rows[seed]['context']['name']
            values = [v['aggregate'] for name, v in full[seed].items()
                      if (name == context or name.startswith(context + '/' + mode + '-')) and objective in v['aggregate']]
            if len(values) < 2:
                continue
            values.sort(key=lambda v: v[objective])
            ax.plot([v[objective] for v in values], [v[metric] for v in values], 'o-', color=colors(j % 10), label=str(seed))
            plotted = True
        ax.set(title=mode.replace('_', ' '), xlabel=label, ylabel='Development ' + metric)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.15)
        if plotted:
            ax.legend(frameon=False, fontsize=8)
        else:
            ax.text(.5, .5, 'Not recorded by this historical study', transform=ax.transAxes, ha='center')
    fig.suptitle('Distinct reranking objectives: fixed strengths 0, 0.2, 0.5, 0.8', fontsize=14)
    for suffix in ('png', 'pdf'):
        fig.savefig(out / ('reranker-tradeoffs.' + suffix), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan, data, studies = load_evidence(args.runs)
    report = summary_text(plan, data)
    args.out.mkdir(parents=True, exist_ok=False)
    for name, value in data.items():
        write_json(args.out / (name + '.json'), value)
    for seed, directory in studies.items():
        for filename in ('coefficients.json', 'selection.json', 'paired-comparisons.json', 'switching.json'):
            (args.out / f'{seed}-{filename}').write_text((directory / filename).read_text())
    render_figures(plan, data, args.out)
    (args.out / 'RESULTS.md').write_text(report)
    hashes = {p.name: digest(p) for p in args.out.iterdir() if p.is_file()}
    write_json(args.out / 'MANIFEST.json', {'artifact_sha256': hashes, 'plan': plan,
               'generator_sha256': digest(Path(__file__))})
    print(args.out / 'RESULTS.md')


if __name__ == '__main__':
    main()
