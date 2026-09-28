"""Create shareable aggregate evidence and publication-format research figures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'recsys-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


FAMILIES = ['expert','static','user','item','disagreement','context','static-pairwise','context-pairwise']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads((args.runs/'plan.json').read_text())
    if plan['status']!='complete':
        raise ValueError('Experiment suite is incomplete')
    args.out.mkdir(parents=True,exist_ok=False)
    rows,full,provenance={}, {}, {}
    feasibility,budgets,policies={}, {}, {}
    for seed in plan['seeds']:
        directory=args.runs/f'study-{seed}'
        manifest=json.loads((directory/'manifest.json').read_text())
        if manifest['status']!='complete' or manifest['test_read']:
            raise ValueError('Expected completed validation-only study')
        if any(v['manifest'].get('validation_used_for_training',True) for v in manifest['sources'].values()):
            raise ValueError('Primary evidence requires fixed-budget expert training')
        results=json.loads((directory/'results.json').read_text())
        selection=json.loads((directory/'selection.json').read_text())
        names={'expert':selection['best_expert'],**selection['families']}
        rows[seed]={family:{'name':name,**results[name]['aggregate']} for family,name in names.items()}
        full[seed]={n:{'aggregate':v['aggregate'],'groups':v['groups']} for n,v in results.items()}
        feasibility[seed]=json.loads((directory/'candidate-feasibility.json').read_text())
        budgets[seed]=json.loads((directory/(names['context']+'-candidate-budget.json')).read_text())
        policies[seed]=json.loads((directory/'group-policy.json').read_text())
        provenance[seed]={'manifest':manifest,'results_sha256':digest(directory/'results.json'),
                          'coefficients_sha256':digest(directory/'coefficients.json'),
                          'cohorts_sha256':digest(directory/'cohorts.json')}
        for filename in ['coefficients.json','selection.json','paired-comparisons.json','switching.json']:
            (args.out/f'{seed}-{filename}').write_text((directory/filename).read_text())
    for name,value in [('selected',rows),('aggregates',full),('candidate-feasibility',feasibility),
                       ('candidate-budgets',budgets),('group-policies',policies),('provenance',provenance)]:
        (args.out/f'{name}.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    seeds=plan['seeds']
    means={f:statistics.mean(rows[s][f]['ndcg@10'] for s in seeds) for f in FAMILIES}
    fig,axes=plt.subplots(2,2,figsize=(12,8.5),layout='constrained')
    palette=['#276779','#ba6131','#77893d']
    ax=axes[0,0]
    for j,seed in enumerate(seeds):
        ax.plot(range(len(FAMILIES)),[rows[seed][f]['ndcg@10'] for f in FAMILIES],
                'o-',color=palette[j%len(palette)],label=str(seed),alpha=.85)
    ax.set_xticks(range(len(FAMILIES)),['Expert','Static','User','Item','Disagr.','Context','Pair-static','Pair-context'],rotation=35,ha='right')
    ax.set_ylabel('Development nDCG@10')
    ax.set_title('Feature and loss ablations')
    ax.legend(title='Data seed',frameon=False,fontsize=8)
    ax=axes[0,1]
    for j,seed in enumerate(seeds):
        contextual=rows[seed]['context']['name']
        metrics=[v['aggregate'] for n,v in full[seed].items() if n==contextual or n.startswith(contextual+'/exposure-') or n==contextual+'/adaptive-pool-exposure']
        metrics.sort(key=lambda v:v['head_exposure'])
        ax.plot([v['head_exposure'] for v in metrics],[v['ndcg@10'] for v in metrics],
                'o-',color=palette[j%len(palette)],label=str(seed))
    ax.axvline(.2,color='gray',ls='--',lw=1,label='Catalog head share ≈20%')
    ax.set_xlabel('Head-item exposure fraction')
    ax.set_ylabel('Development nDCG@10')
    ax.set_title('Exposure correction has an accuracy cost')
    ax.legend(frameon=False,fontsize=8)
    ax=axes[1,0]
    for j,seed in enumerate(seeds):
        entries=[r for r in feasibility[seed] if r['model']==rows[seed]['context']['name']]
        ax.plot([r['pool'] for r in entries],[r['fraction_quota_feasible'] for r in entries],
                'o-',color=palette[j%len(palette)],label=str(seed))
    ax.set_xscale('log')
    ax.set_ylim(-.03,1.03)
    ax.set_xlabel('Candidate pool size (log scale)')
    ax.set_ylabel('Users with sufficient tail candidates')
    ax.set_title('Can the candidate pool meet the exposure quota?')
    ax=axes[1,1]
    for j,seed in enumerate(seeds):
        retention=policies[seed]['development_retention']
        ax.plot([0,1,2],[retention[str(g)] for g in range(3)],'o-',
                color=palette[j%len(palette)],label=str(seed))
    ax.axhline(.95,color='gray',ls='--',lw=1,label='Meta-fit retention constraint')
    ax.set_xticks([0,1,2],['Sparse histories','Medium histories','Dense histories'])
    ax.set_ylabel('Development group nDCG / base nDCG')
    ax.set_title('Training utility budgets need a held-out audit')
    ax.legend(frameon=False,fontsize=8)
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15)
    fig.suptitle('Hybrid recommendation: accuracy, context and exposure\nExploratory development results; test set reserved',fontsize=15)
    fig.savefig(args.out/'research-figures.png',dpi=180)
    fig.savefig(args.out/'research-figures.pdf')
    plt.close(fig)
    report=['# Research checkpoint: development evidence','',
        'Three overlapping MovieLens 100K data splits; 471 meta-fit users and 472 development users per split. '
        'Expert variants were selected on meta-fit users. Hybrid variants were selected on development users. '
        'The table therefore reports selected development performance, not unbiased test accuracy.','',
        '![Research figures](research-figures.png)','',
        '## Architecture and ablations','',
        'The contextual model learns expert score weights that depend on training-history size, genre entropy '
        'and item popularity, with an additional disagreement feature. Static weights, isolated feature groups '
        'and pairwise score-margin regression provide controls. These techniques have prior art; the contribution '
        'is the implementation, controlled comparison and analysis.','',
        '| Family | '+' | '.join(map(str,seeds))+' | Mean nDCG@10 |',
        '|---|'+'---:|'*(len(seeds)+1)]
    for family in FAMILIES:
        report.append('| '+family+' | '+' | '.join(f'{rows[s][family]["ndcg@10"]:.4f}' for s in seeds)+f' | {means[family]:.4f} |')
    best=max(means,key=means.get)
    delta=means['context']-means['expert']
    report += ['',f'The highest mean selected development score is **{best}** ({means[best]:.4f}). '
               f'The full contextual ridge differs from the selected expert by **{delta:+.4f} nDCG** on average. '
               'The additional architecture does not earn a performance claim merely by being more elaborate. '
               'See paired-comparisons files for descriptive intervals and fractions of users helped/harmed.', '',
               '## Candidate feasibility','',
               'For a length-k list with only T tail items in its candidate pool, the head fraction cannot be '
               'below max(0,k−T)/k. No reranking weight can overcome missing candidates. '
               'Selective expansion uses the shortest score prefix that contains the required tail quota, '
               'with minimum pool 100; full support remains available upstream.','',
               '| Seed | Mean expanded pool | Users expanded | Same top-k as full-pool exposure reranking |',
               '|---|---:|---:|---:|']
    for seed in seeds:
        b=budgets[seed]
        report.append(f'| {seed} | {b["mean_pool"]:.1f} | {b["fraction_expanded"]:.1%} | {b["same_topk_as_full"]:.1%} |')
    report += ['', 'This comparison uses exposure strength 1 and an explicit catalog-parity objective. '
               'It measures reranking candidate work, not total retrieval latency: expert full-catalog scores '
               'were already computed. Exact agreement here is not a claim about all rerankers or unseen datasets.', '',
               '## Fairness and accuracy','',
               'Head means the most popular 20% of items by training count. Catalog-proportional exposure is '
               'a diagnostic choice, not an assertion that equal item exposure is always fair. '
               'Strong correction can raise tail recall while lowering nDCG; all settings, including poor ones, '
               'are retained in aggregates.json. Genre calibration uses JSD and diversity uses Jaccard distance.','',
               'The group-aware policy chooses an exposure strength that retains at least 95% of each '
               'activity group’s base nDCG on meta-fit users. The lower-right plot audits this on development '
               'users. A training constraint is not a held-out guarantee, and smaller utility gaps alone can '
               'hide degradation of every group. Both worst-group utility and absolute accuracy are reported.', '',
               '## Evidence and remaining work','',
               'Source manifests, score/split hashes and coefficient/selection artifacts accompany this summary. '
               'Raw recommendations and user-level data remain in ignored local run directories. '
               'The three seeds overlap in users/data, and the bootstrap intervals do not correct for selection. '
               'These are development findings. Required next checks are an independent implementation review, '
               'clean environment reproduction, a chronological split sensitivity analysis, lecturer feedback '
               'on fairness definitions and scope, and a frozen final test evaluation. '
               'Do not submit this checkpoint as the final task-structured report.']
    (args.out/'RESULTS.md').write_text('\n'.join(report)+'\n')
    hashes={p.name:digest(p) for p in args.out.iterdir() if p.is_file()}
    (args.out/'MANIFEST.json').write_text(json.dumps({'artifact_sha256':hashes,'plan':plan,
        'generator_sha256':digest(Path(__file__))},indent=2,sort_keys=True)+'\n')
    print(args.out/'RESULTS.md')


if __name__=='__main__':
    main()
