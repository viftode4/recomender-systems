"""Standalone, synthetic-profile inspection of a selected categorical checkpoint.

No interaction file, user identifier, real user history or test file is read.
The catalog input may be a scores archive, but only its item-token array is
loaded. The profile and candidates below are fixed before checkpoint inspection.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


# Hand-authored fiction, not copied or selected from a real user's history.
SYNTHETIC_HISTORY = (('50', 5), ('1', 4), ('100', 2), ('56', 1), ('77', 3), ('237', 3))
VARIABLE_ITEM = '2'  # GoldenEye (1995)
CANDIDATES = ('172', '181', '183', '195', '82', '272')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_catalog(path):
    path = Path(path)
    if path.suffix == '.npz':
        # Lazy archive access: user and score arrays are never loaded.
        with np.load(path, allow_pickle=False) as archive:
            items = archive['items'].tolist()
    else:
        data = json.loads(path.read_text())
        items = data['items'] if isinstance(data, dict) else data
    if not items or items[0] != '[PAD]' or len(set(items)) != len(items):
        raise ValueError('Require unique catalog item tokens with padding at zero')
    return items


def load_metadata(path):
    with Path(path).open() as stream:
        result = {row['item_id:token']: {'title': row['movie_title:token_seq'],
                                       'year': row['release_year:token']}
                  for row in csv.DictReader(stream, delimiter='\t')}
    required = {item for item, _ in SYNTHETIC_HISTORY} | {VARIABLE_ITEM} | set(CANDIDATES)
    if not required.issubset(result):
        raise ValueError('Movie metadata is missing a predeclared demonstration item')
    return result


def infer_payload(model, items, metadata, provenance):
    import torch
    from categorical_field import summarize_diagnostics

    lookup = {item: j for j, item in enumerate(items)}
    required = {i for i, _ in SYNTHETIC_HISTORY} | {VARIABLE_ITEM} | set(CANDIDATES)
    if not required.issubset(lookup) or model.n_items != len(items):
        raise ValueError('Model catalog differs from the demonstration catalog')
    if set(CANDIDATES) & ({i for i, _ in SYNTHETIC_HISTORY} | {VARIABLE_ITEM}):
        raise AssertionError('Candidate appears in synthetic input history')
    context = torch.zeros((1, len(items)), dtype=torch.long)
    for item, rating in SYNTHETIC_HISTORY:
        context[0, lookup[item]] = rating
    model.eval()
    scenarios = {}
    with torch.no_grad():
        for rating in range(6):
            context[0, lookup[VARIABLE_ITEM]] = rating
            prediction = model(context, return_diagnostics=True)
            probabilities = prediction['rating_logits'].softmax(-1)[0].cpu().numpy()
            if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                    or not np.allclose(probabilities.sum(axis=-1), 1., atol=2e-6, rtol=0)):
                raise ValueError('Checkpoint returned invalid rating probabilities')
            diagnostics = prediction['diagnostics']
            scenario = {'rating': rating, 'candidates': [],
                        'diagnostics': summarize_diagnostics(diagnostics),
                        'variable_source_weights': (diagnostics['fidelity'][0, :, lookup[VARIABLE_ITEM]].cpu().tolist()
                                                    if rating else None)}
            for item in CANDIDATES:
                distribution = probabilities[lookup[item]].astype(float)
                trace = diagnostics['rating_probabilities'][0, :, lookup[item]].cpu().tolist()
                scenario['candidates'].append({
                    'item': item, **metadata[item], 'probabilities': distribution.tolist(),
                    'probability_rating_at_least_four': float(distribution[3:].sum()),
                    'expected_rating': float(distribution @ np.arange(1., 6.)),
                    'iteration_probabilities': trace,
                })
            scenarios[str(rating)] = scenario
    return {
        'schema_version': 1, 'profile_kind': 'hand_authored_synthetic',
        'history': [{'item': item, 'rating': rating, **metadata[item]} for item, rating in SYNTHETIC_HISTORY],
        'variable_item': {'item': VARIABLE_ITEM, **metadata[VARIABLE_ITEM]},
        'candidate_order': list(CANDIDATES), 'scenarios': scenarios,
        'model_config': model.config, 'provenance': provenance,
        'scope': 'Predictive sensitivities of one fictional profile. These are not causal effects, identified emotions, reliability estimates, or performance evidence.',
    }


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>One rating, six forecasts · Categorical field</title>
<style>
:root{color-scheme:light;--ink:#172d37;--muted:#52626b;--paper:#f5f3ed;--line:#d9dfda;--accent:#176859;--soft:#e6eee8}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.55}
main{max-width:1240px;margin:auto;padding:clamp(22px,4vw,60px)}h1,h2,h3,p{margin-top:0}h1{font-size:clamp(2.1rem,5vw,4rem);line-height:1.08;letter-spacing:-.04em;margin-bottom:20px;max-width:850px}h2{font-size:1.2rem;letter-spacing:-.01em}h3{font-size:1.05rem;line-height:1.3;margin-bottom:4px}
.eyebrow{font-size:.75rem;font-weight:750;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);margin-bottom:15px}.intro{max-width:760px;font-size:1.05rem;color:var(--muted);margin-bottom:32px}.layout{display:grid;grid-template-columns:290px 1fr;gap:30px;align-items:start}.panel,.card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:22px}.panel{position:sticky;top:20px}.history{list-style:none;padding:0;margin:0 0 24px}.history li{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid var(--line);padding:9px 0;font-size:.9rem}.score{font-variant-numeric:tabular-nums;white-space:nowrap;font-weight:650}.small{font-size:.82rem;color:var(--muted)}fieldset{margin:0;padding:0;border:0}legend{font-size:1rem;font-weight:700;margin-bottom:6px}.choices{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:14px}.choice{position:relative}.choice input{position:absolute;opacity:0;width:100%;height:100%;margin:0;cursor:pointer}.choice span{display:block;text-align:center;border:1px solid var(--line);padding:10px 6px;border-radius:6px;font-size:.9rem;font-weight:650}.choice input:checked+span{background:var(--accent);color:#fff;border-color:var(--accent)}.choice input:focus-visible+span{outline:3px solid #df8c24;outline-offset:3px}.status{min-height:3em;margin:16px 0 0;font-size:.83rem;color:var(--muted)}.candidate-head{display:flex;justify-content:space-between;gap:15px;align-items:baseline;margin-bottom:15px}.candidate-head h2{margin:0}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.year{color:var(--muted);font-size:.8rem}.probability{font-size:2.15rem;font-weight:720;letter-spacing:-.04em;margin-top:14px;font-variant-numeric:tabular-nums;line-height:1.1}.prob-label{font-size:.77rem;margin:4px 0 10px;color:var(--muted)}.delta{display:inline-block;background:var(--soft);color:#21544b;border-radius:4px;padding:4px 7px;font-size:.76rem;font-variant-numeric:tabular-nums}.bars{display:grid;grid-template-columns:repeat(5,1fr);gap:6px;align-items:end;height:78px;margin:19px 0 6px}.bar{height:100%;position:relative;background:#eef1ee;border-radius:3px 3px 0 0;overflow:hidden}.bar:after{content:"";position:absolute;left:0;right:0;bottom:0;height:var(--fill);background:#789c91}.bar:nth-child(4):after,.bar:nth-child(5):after{background:var(--accent)}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}.distribution th,.distribution td{text-align:center;font-size:.7rem;padding:2px 0}.distribution th{font-weight:500;color:var(--muted)}.details{margin-top:30px;border-top:1px solid var(--line);padding-top:20px}summary{cursor:pointer;font-weight:700;padding:5px 0}summary:focus-visible{outline:3px solid #df8c24}.table-wrap{overflow-x:auto}.diagnostics th,.diagnostics td{text-align:right;font-size:.81rem;padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap}.diagnostics th:first-child,.diagnostics td:first-child{text-align:left}.note{max-width:900px;margin:17px 0 10px;font-size:.87rem;color:var(--muted)}footer{margin-top:34px;border-top:1px solid var(--line);padding-top:22px;font-size:.8rem;color:var(--muted)}code{font-size:.78rem;overflow-wrap:anywhere}.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)}
@media(max-width:850px){.layout{grid-template-columns:1fr}.panel{position:static}.history{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 22px}.choices{grid-template-columns:repeat(6,1fr)}}@media(max-width:520px){.grid{grid-template-columns:1fr}.history{display:block}.choices{grid-template-columns:repeat(3,1fr)}.candidate-head{display:block}.card{padding:20px}.candidate-head .small{margin-top:5px}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
</style></head><body><main>
<header><p class="eyebrow">Synthetic profile · Categorical evidence field</p><h1>One rating.<br>Six forecasts.</h1><p class="intro">Change one fictional rating and inspect how the selected model redistributes its predictions. Each forecast retains all five rating categories.</p></header>
<div class="layout"><aside class="panel"><h2>A fictional movie history</h2><ul id="history" class="history"></ul><fieldset><legend id="variable-title"></legend><p class="small">Keep the six ratings above fixed. Choose how this profile rates one more film.</p><div class="choices" id="choices"></div></fieldset><p id="status" class="status" aria-live="polite"></p></aside>
<section aria-labelledby="candidate-heading"><div class="candidate-head"><h2 id="candidate-heading">Six fixed candidates</h2><span class="small">Same films and order in every scenario</span></div><div id="candidates" class="grid"></div></section></div>
<details class="details"><summary>Inspect the learned updates</summary><p class="note">Source weight is a learned coefficient that balances supplied evidence and a routed message. Routing change is the mean total-variation distance from the first update. These are internal computations; they do not measure trust, emotion or physical flow.</p><div class="table-wrap"><table class="diagnostics"><thead><tr><th scope="col">Update</th><th scope="col">Mean source weight</th><th scope="col">Changed film weight</th><th scope="col">Routing change</th><th scope="col">Distribution change</th></tr></thead><tbody id="diagnostics"></tbody></table></div><p class="note">Distribution change averages the change from the previous update across the catalog. Small values are shown faithfully; this view does not exaggerate the model's response.</p></details>
<footer><p><strong>Interpretation:</strong> this is a hand-authored profile with no real user's history. Differences are predictive sensitivities, not causal preference changes. The demonstration makes no claim about accuracy, calibration or novelty. Probabilities are model outputs conditional on observed-rating training data.</p><details><summary>Checkpoint and reproducibility</summary><p id="provenance"></p><p class="small">Six scenarios were computed locally from one selected checkpoint. Switching the control chooses among those exact outputs. This file works offline and sends no requests.</p></details></footer>
</main><script type="application/json" id="demo-data">__PAYLOAD__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('demo-data').textContent);
const node=(tag,text,cls)=>{const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(cls)el.className=cls;return el;};
const pct=x=>(x*100).toFixed(3)+'%';
for(const movie of data.history){const li=node('li');li.append(node('span',movie.title),node('span',movie.rating+' / 5','score'));document.getElementById('history').append(li);}
document.getElementById('variable-title').textContent=data.variable_item.title+' ('+data.variable_item.year+')';
for(let rating=0;rating<=5;rating++){const label=node('label',undefined,'choice');const input=node('input');input.type='radio';input.name='rating';input.value=String(rating);input.setAttribute('aria-label',rating===0?'No rating supplied':rating+' out of 5');label.append(input,node('span',rating===0?'Absent':String(rating)+' / 5'));input.addEventListener('change',()=>render(rating));document.getElementById('choices').append(label);}
function render(rating){const scenario=data.scenarios[String(rating)];const root=document.getElementById('candidates');root.replaceChildren();document.querySelector('input[value="'+rating+'"]').checked=true;
 for(let j=0;j<scenario.candidates.length;j++){const film=scenario.candidates[j],base=data.scenarios['0'].candidates[j];const article=node('article',undefined,'card');article.append(node('h3',film.title),node('div',film.year,'year'),node('div',pct(film.probability_rating_at_least_four),'probability'),node('p','Predicted probability of rating 4 or 5','prob-label'));const delta=100*(film.probability_rating_at_least_four-base.probability_rating_at_least_four);article.append(node('span',(delta>=0?'+':'')+delta.toFixed(4)+' percentage points vs absent','delta'));const bars=node('div',undefined,'bars');bars.setAttribute('aria-hidden','true');for(const p of film.probabilities){const bar=node('span',undefined,'bar');bar.style.setProperty('--fill',(p*100)+'%');bars.append(bar);}article.append(bars);const table=node('table',undefined,'distribution');const caption=node('caption','Full predicted rating distribution','sr-only');table.append(caption);const thead=node('thead'),head=node('tr'),tbody=node('tbody'),row=node('tr');film.probabilities.forEach((p,i)=>{const th=node('th',String(i+1)+' / 5');th.scope='col';head.append(th);row.append(node('td',pct(p)));});thead.append(head);tbody.append(row);table.append(thead,tbody);article.append(table);root.append(article);}
 const body=document.getElementById('diagnostics');body.replaceChildren();scenario.diagnostics.steps.forEach((step,i)=>{const row=node('tr');const values=[step.step,step.source_gate_mean.toFixed(6),scenario.variable_source_weights?scenario.variable_source_weights[i].toFixed(6):'Absent',step.up_routing_tv_from_initial.toFixed(6),step.rating_distribution_tv_from_previous.toFixed(6)];for(const value of values)row.append(node('td',String(value)));body.append(row);});document.getElementById('status').textContent=(rating===0?'No rating supplied for ':rating+' / 5 for ')+data.variable_item.title+'. Six forecasts updated.';}
const provenance=data.provenance;document.getElementById('provenance').append(node('span','Variant: '+data.model_config.variant+' · epoch '+provenance.epoch+' · training seed '+provenance.seed+'. '),node('br'),node('code','Checkpoint SHA-256: '+provenance.checkpoint_sha256));
const initial=Number(location.hash.slice(1));render(Number.isInteger(initial)&&initial>=0&&initial<=5?initial:0);
</script></body></html>'''


def write_demo(payload, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    embedded = encoded.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    (output/'predictions.json').write_text(encoded+'\n')
    (output/'index.html').write_text(HTML.replace('__PAYLOAD__', embedded))
    manifest = {'profile_kind': 'hand_authored_synthetic', 'test_read': False, 'real_user_history_read': False,
                'generator_sha256': sha256(__file__), 'provenance': payload['provenance'],
                'files_sha256': {name: sha256(output/name) for name in ('predictions.json', 'index.html')}}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True, help='Expected hash from the committed validation selection')
    parser.add_argument('--catalog', type=Path, required=True, help='Item-only JSON or source NPZ; only items are read')
    parser.add_argument('--item-metadata', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError('Selected checkpoint hash mismatch')
    import torch
    from categorical_field import CategoricalEvidenceField
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    model = CategoricalEvidenceField(**checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    provenance = {'checkpoint_sha256': args.checkpoint_sha256,
                  'model_source_sha256': sha256(Path(__file__).with_name('categorical_field.py')),
                  'catalog_sha256': sha256(args.catalog), 'metadata_sha256': sha256(args.item_metadata),
                  'epoch': checkpoint['epoch'], 'seed': checkpoint['seed']}
    payload = infer_payload(model, load_catalog(args.catalog), load_metadata(args.item_metadata), provenance)
    write_demo(payload, args.out)
    print(json.dumps({'status': 'complete', 'output': str(args.out), 'synthetic_profile': True}))


if __name__ == '__main__':
    main()
