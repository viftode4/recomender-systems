"""Independent verification of the sealed context-interaction experiment.

The measured runner and evaluation helpers are never imported. Selected models
are replayed only after the global selection barrier and completed manifests
pass. Additional direct kernel solves do not use the model's downdate formula.
Outputs contain aggregate checks only, with no user or item identifiers.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# These are independent audit utilities, not the measured model/evaluator.
# Their exact source hashes are recorded separately in the audit receipt.
from exploratory.categorical_reconstruction.verify_results import (
    array_digest, check_hashes, close, digest, json_read, json_write,
    load_inputs, rank, require, safe_path, selected_arrays)
from exploratory.categorical_reconstruction.verify_groups import (
    compare as compare_group_fields, recompute as independent_groups)

SEEDS = (2026, 2027, 2028)
LAMBDAS = (10., 30., 50., 100., 250., 300., 1000., 3000., 10000.)
PAIR_WEIGHTS = (0., .01, .1, 1., 10.)
ROLES = ("binary", "pair_only", "nested")
K = 10


def independent_columns(binary, scores, penalty, pair_weight):
    """Delete the candidate column, construct its kernel, and solve directly.

The target sample covers the TRAIN frequency distribution, without inspecting
development labels or selecting examples based on numerical disagreement.
"""
    x = np.asarray(binary, dtype=np.float64)
    counts = x.sum(axis=0)
    ordered = np.argsort(counts, kind="stable")
    indices = sorted(set(ordered[np.linspace(0, len(ordered)-1, 9, dtype=int)].tolist()))
    sizes = x.sum(axis=1)
    normalizer = float(np.sum(sizes*(sizes-1)/2)/sizes.sum()) if sizes.sum() else 1.
    if normalizer <= 0:
        normalizer = 1.
    maximum, residual = 0., 0.
    for item in indices:
        target = x[:, item]
        source = x.copy()
        source[:, item] = 0
        gram = source @ source.T
        kernel = (gram + pair_weight/normalizer*(gram*gram-gram)/2)/penalty
        system = kernel + np.eye(len(x))
        dual = cho_solve(cho_factor(system, lower=True, check_finite=True), target)
        predicted = kernel @ dual
        maximum = max(maximum, close(scores[:, item], predicted,
            "Independent complete-target-exclusion kernel", atol=2e-9, rtol=2e-9))
        residual = max(residual, float(np.max(np.abs(system @ dual-target))))
    return {"target_columns_checked": len(indices), "maximum_absolute_score_error": maximum,
            "maximum_direct_solve_residual": residual,
            "method": "Separate direct SPD solve after deleting each sampled target column",
            "column_rule": "Nine equally spaced order statistics of TRAIN item frequency; unique columns"}


def full_metrics(scores, inputs, cohort, truth):
    """Independent macro ranking metrics, including MRR, novelty and coverage."""
    recommendation = rank(scores, inputs["eligible"])
    counts = Counter(item for user, item in inputs["train"])
    total = sum(counts.values()) + len(inputs["items"])-1
    discount = 1/np.log2(np.arange(2, K+2))
    rows, exposed = {}, set()
    for user in sorted(cohort):
        positives = truth.get(user, set())
        if not positives:
            continue
        columns = recommendation[inputs["user_index"][user]]
        items = [inputs["items"][i] for i in columns]
        hits = np.array([item in positives for item in items], dtype=float)
        ranks = np.flatnonzero(hits)
        rows[user] = {"ndcg@10": float(hits@discount/discount[:min(K,len(positives))].sum()),
            "precision@10": float(hits.mean()), "recall@10": float(hits.sum()/len(positives)),
            "mrr@10": 1/float(ranks[0]+1) if len(ranks) else 0., "hit@10": float(bool(len(ranks))),
            "novelty@10": float(np.mean([math.log2(total/(counts[item]+1)) for item in items]))}
        exposed.update(items)
    require(rows, "Endpoint has no truth-bearing users")
    aggregate = {key: sum(row[key] for row in rows.values())/len(rows) for key in next(iter(rows.values()))}
    aggregate["coverage@10"] = len(exposed)/(len(inputs["items"])-1)
    return {"users": len(rows), **aggregate}, rows


def independent_diagnostics(scores, inputs, cohort, ratings_path):
    """Recompute rated endpoint denominators and TRAIN-frequency diagnostics."""
    allowed = set(inputs["valid"])
    labels = {}
    with Path(ratings_path).open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            pair = row["user_id:token"], row["item_id:token"]
            if pair in allowed:
                require(pair not in labels, "Duplicate validation rating")
                labels[pair] = float(row["rating:float"])
    require(set(labels) == allowed, "Validation ratings are incomplete")
    cohort = set(cohort)
    labels = {pair:value for pair,value in labels.items() if pair[0] in cohort}
    rankings = rank(scores, inputs["eligible"])
    recs = {user:[inputs["items"][i] for i in rankings[inputs["user_index"][user]]]
            for user in cohort}
    slots = len(cohort)*K
    rated_hits = sum((user,item) in labels for user,items in recs.items() for item in items)
    dislike_hits = sum(labels.get((user,item), 6) <= 2 for user,items in recs.items() for item in items)
    liked_users = {user for (user,item),value in labels.items() if value >= 4}
    denominators = {"all_observed_users":len(cohort), "liked_ratings_users":len(liked_users),
        "users_without_liked_validation_excluded_from_liked_metrics":len(cohort)-len(liked_users),
        "validation_observations":len(labels), "validation_likes":sum(value>=4 for value in labels.values()),
        "validation_dislikes":sum(value<=2 for value in labels.values()), "recommendation_slots":slots,
        "recommended_items_with_validation_rating":rated_hits,"recommended_known_dislikes":dislike_hits}
    counts = Counter(item for user,item in inputs["train"])
    catalog = inputs["items"][1:]
    head_count = math.ceil(len(catalog)*.2)
    head = set(sorted(catalog,key=lambda item:(-counts[item],item))[:head_count])
    recommended = [item for user in sorted(cohort) for item in recs[user]]
    frequency = np.array([counts[item] for item in recommended])
    training_frequency = {"users":len(cohort),"recommendation_slots":slots,
        "mean_training_count":float(frequency.mean()),"median_training_count":float(np.median(frequency)),
        "zero_training_count_slot_share":float(np.mean(frequency==0)),
        "head_slot_share":sum(item in head for item in recommended)/slots,
        "head_catalog_size":head_count,"catalog_items":len(catalog)}
    return {"denominators":denominators,"known_dislike_rate_per_slot":dislike_hits/slots,
        "dislike_rate_among_validation_rated_recommendations":dislike_hits/rated_hits if rated_hits else None,
        "training_frequency":training_frequency}


def check_barrier(research):
    """Validate all sealed selections before reading prediction or label arrays."""
    frozen = json_read(research/"SELECTIONS-FROZEN.json")
    require(frozen.get("status") == "all_selections_frozen"
        and frozen.get("test_read") is False and frozen.get("development_evaluated") is False
        and tuple(frozen.get("seeds",[])) == SEEDS,"Invalid global selection barrier")
    protocol = json_read(research/"protocol.json")
    require(digest(research/"protocol.json") == frozen["protocol_sha256"],"Protocol changed after freeze")
    require(tuple(protocol["seeds"]) == SEEDS and tuple(protocol["lambdas"]) == LAMBDAS
        and tuple(protocol["pair_weights"]) == PAIR_WEIGHTS[1:]
        and tuple(protocol["models"]) == ROLES
        and protocol["test_read"] is False,"Unexpected declared interaction grid")
    require(digest(research/"reference-input-signature.json")==frozen["reference_input_signature_sha256"],
        "Locked reference signature changed")
    require(protocol["source_sha256"] == frozen["source_sha256"],"Source seal differs")
    check_hashes(ROOT,frozen["source_sha256"])
    runtime = frozen["runtime"]
    expected = {"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
        "numerical_threads":1,"machine":platform.machine(),"system":platform.system()}
    require(all(runtime[key] == value for key,value in expected.items()),"Use the original numerical runtime")
    selections = {}
    for seed in SEEDS:
        directory = research/str(seed)
        require(digest(directory/"selection-manifest.json") == frozen["selection_manifest_sha256"][str(seed)],
            "Selection changed after global freeze")
        manifest = json_read(directory/"selection-manifest.json")
        require(manifest["status"] == "selected" and manifest["seed"] == seed
            and manifest["test_read"] is False and manifest["development_evaluated"] is False
            and manifest["source_sha256"] == frozen["source_sha256"]
            and manifest["runtime"] == runtime and manifest["unique_fits"] == 45,"Invalid selection manifest")
        check_hashes(directory,manifest["output_sha256"])
        selections[seed] = manifest
    complete = json_read(research/"manifest.json")
    require(complete["status"] == "complete" and complete["test_read"] is False
        and complete["test_evaluated"] is False and complete["fresh_confirmation"] is False
        and complete["source_sha256"] == frozen["source_sha256"] and complete["runtime"] == runtime
        and complete["selection_freeze_sha256"] == digest(research/"SELECTIONS-FROZEN.json"),
        "Invalid completed study manifest")
    check_hashes(research,complete["output_sha256"])
    opened = json_read(research/"DEVELOPMENT-OPENED.json")
    require(opened["selection_freeze_sha256"] == digest(research/"SELECTIONS-FROZEN.json")
        and opened["test_read"] is False,"Development marker does not bind the selection freeze")
    return frozen,protocol,selections,complete


def check_grid(directory):
    grid,chosen = json_read(directory/"candidate-grid.json"),json_read(directory/"selection.json")
    specs = [("binary",penalty,0.) for penalty in LAMBDAS]
    specs += [("pair",penalty,weight) for penalty in LAMBDAS for weight in PAIR_WEIGHTS[1:]]
    require(len(grid)==45 and set(chosen)==set(ROLES),"Unexpected grid or selected roles")
    require(len({row["candidate_id"] for row in grid})==45,"Duplicate candidate identifier")
    for row,spec in zip(grid,specs):
        require((row["feature_kind"],row["penalty"],row["pair_weight"])==spec,"Candidate order or settings differ")
        require(row["candidate_id"]==f"lambda-{row['penalty']:g}-beta-{row['pair_weight']:g}",
            "Candidate identifier differs")
        require(math.isfinite(row["meta_fit_all_observed_ndcg@10"])
            and 0<=row["meta_fit_all_observed_ndcg@10"]<=1,"Invalid candidate selection value")
    pools = {"binary":grid[:9],"pair_only":grid[9:],"nested":grid}
    for name,rows in pools.items():
        winner = max(rows,key=lambda row:row["meta_fit_all_observed_ndcg@10"])
        require(chosen[name]["candidate_count"]==len(rows),"Wrong model-selection budget")
        for field in ("candidate_id","feature_kind","penalty","pair_weight", "scores_array_sha256",
                      "meta_fit_all_observed_ndcg@10"):
            require(chosen[name][field]==winner[field],"Selected candidate is not the first exact maximum")
        require(chosen[name]["selection_cohort"]=="reused_meta_fit","Unexpected selection cohort")
    return grid,chosen


def explicit_feature_check():
    """Small synthetic primal check, with no dataset or experiment-seed access."""
    from exploratory.context_interactions.model import fit, prepare_features
    rng = np.random.default_rng(901)
    x = rng.integers(0, 2, size=(9, 7)).astype(float)
    x[:, 0] = 0
    query = rng.integers(0, 2, size=(5, 7)).astype(float)
    sizes = x.sum(axis=1)
    normalizer = np.sum(sizes*(sizes-1)/2)/sizes.sum()
    pairs = [(i,j) for i in range(x.shape[1]) for j in range(i+1,x.shape[1])]
    maximum = 0.
    for penalty, weight in ((.7,0.), (.7,.3), (4.,10.)):
        model = fit(prepare_features(x), lambda_binary=penalty, pair_weight=weight)
        actual = model.predict(query)
        gamma = weight/normalizer
        features = np.column_stack([x]+[np.sqrt(gamma)*x[:,i]*x[:,j] for i,j in pairs])
        query_features = np.column_stack([query]+[np.sqrt(gamma)*query[:,i]*query[:,j] for i,j in pairs])
        descriptors = [(i,) for i in range(x.shape[1])] + pairs
        expected = np.zeros_like(actual)
        for item in range(x.shape[1]):
            allowed = [index for index,descriptor in enumerate(descriptors) if item not in descriptor]
            f, q = features[:,allowed], query_features[:,allowed]
            coefficients = np.linalg.solve(f.T@f+penalty*np.eye(len(allowed)), f.T@x[:,item])
            close(f.T@model.dual_coefficients[:,item]/penalty,coefficients,
                  "Literal feature coefficients",atol=2e-11,rtol=2e-11)
            expected[:,item] = q@coefficients
            changed = query.copy(); changed[:,item] = 1-changed[:,item]
            close(actual[:,item],model.predict(changed)[:,item],"Own-target query invariance",atol=2e-11,rtol=2e-11)
        maximum = max(maximum,close(actual,expected,"Literal complete pair-feature primal",atol=2e-11,rtol=2e-11))
    return {"synthetic_configurations":3,"maximum_absolute_score_error":maximum,
            "own_target_query_invariance":True}


def selected_replay(inputs, grid, selections, arrays):
    from exploratory.context_interactions.model import fit, prepare_features
    x = (inputs["matrix"] > 0).astype(float)
    prepared = prepare_features(x)
    configurations, result = {}, {}
    for role in ROLES:
        row = selections[role]
        key = row["candidate_id"]
        if key not in configurations:
            replayed = fit(prepared,lambda_binary=row["penalty"],pair_weight=row["pair_weight"]).predict()
            require(array_digest(replayed)==row["scores_array_sha256"],"Selected model refit is not byte-exact")
            meta,_ = full_metrics(replayed,inputs,inputs["cohorts"]["meta_fit"],inputs["truth"])
            close(meta["ndcg@10"],row["meta_fit_all_observed_ndcg@10"],"Selected candidate meta nDCG")
            checks = independent_columns(x,replayed,row["penalty"],row["pair_weight"])
            configurations[key] = (replayed,checks)
        replayed,checks = configurations[key]
        require(np.array_equal(arrays[role],replayed),"Stored selected scores differ from exact refit")
        result[role] = {"candidate_id":key,"penalty":row["penalty"],"pair_weight":row["pair_weight"],
            "exact_selected_score_replay":True,"direct_kernel_check":checks}
    penalty = selections["binary"]["penalty"]
    factor = cho_factor(x.T@x+penalty*np.eye(x.shape[1]),lower=True)
    inverse = cho_solve(factor,np.eye(x.shape[1]))
    coefficients = -inverse/np.diag(inverse)[None,:]
    np.fill_diagonal(coefficients,0.)
    ease = x@coefficients
    error = close(arrays["binary"],ease,"Independent item-space EASE",atol=2e-9,rtol=2e-9)
    first,second = rank(arrays["binary"],inputs["eligible"]),rank(ease,inputs["eligible"])
    return {"selected":result,"unique_selected_configurations":len(configurations),
        "binary_primal_ease":{"maximum_absolute_score_error":error,
            "exact_top10_order_users":int(np.all(first==second,axis=1).sum()),"users":len(x)}}


def evaluate_scores(scores, inputs, published, detail, metadata, ratings):
    report, errors = {}, []
    for endpoint,truth in (("all_observed",inputs["truth"]),("liked_ratings",inputs["likes"])):
        aggregate,rows = full_metrics(scores,inputs,inputs["cohorts"]["development"],truth)
        original = detail[endpoint]
        require(original["users"]==aggregate["users"] and set(original["per_user"])==set(rows),
                "Detailed evaluation users differ")
        for key,value in aggregate.items():
            if key == "users":
                continue
            errors.append(close(published[endpoint][key],value,"Published "+endpoint+" "+key))
            errors.append(close(original["aggregate"][key],value,"Detailed "+endpoint+" "+key))
        for user,row in rows.items():
            for key,value in row.items():
                errors.append(close(original["per_user"][user][key],value,"Per-user "+endpoint+" "+key))
        report[endpoint] = aggregate
    diagnostics = independent_diagnostics(scores,inputs,inputs["cohorts"]["development"],ratings)
    error,fields = compare_group_fields(published,diagnostics)
    errors.append(error)
    with Path(metadata).open() as stream:
        genres = {row["item_id:token"]:set(row["class:token_seq"].split()) for row in csv.DictReader(stream,delimiter="\t")}
    report.update(diagnostics)
    for key,truth in (("groups",inputs["truth"]),("liked_groups",inputs["likes"])):
        chosen_users = {user for user in inputs["cohorts"]["development"] if truth.get(user)}
        recomputed = independent_groups(scores,inputs["users"],inputs["items"],inputs["history"],truth,
            chosen_users,genres)
        error,group_fields = compare_group_fields(published[key],recomputed)
        errors.append(error)
        report[key] = {"numeric_fields_verified":group_fields,"maximum_absolute_error":error,
            "user_group_counts":{name:row["users"] for name,row in recomputed["user_groups"].items()},
            "item_group_positive_users":{name:row["users_with_positives"] for name,row in recomputed["item_groups"].items()}}
    return report,max(errors,default=0.)


def reference_arrays(research, reference_root, seed, inputs, selected):
    """Bind copied references to their original selected files and input IDs."""
    expected = json_read(research/"reference-input-signature.json")
    require(expected=={"manifest_sha256":digest(reference_root/"manifest.json"),
        "selection_freeze_sha256":digest(reference_root/"SELECTIONS-FROZEN.json")},"Reference root differs")
    original_manifest = json_read(reference_root/"manifest.json")
    require(original_manifest["input_signatures"][str(seed)]==inputs["signature"],"Reference inputs differ")
    source = reference_root/str(seed)
    selections = json_read(source/"selection.json")
    directory = research/str(seed)
    references = json_read(directory/"locked-references.json")
    roles = {"locked_binary":"binary_expanded","locked_categorical":"categorical"}
    require(set(references)==set(roles),"Reference roles differ")
    arrays = selected_arrays(directory,references,inputs["matrix"].shape)
    for name,old_role in roles.items():
        row,original = references[name],selections[old_role]
        require(row["source_model"]==old_role
            and row["source_selection_manifest_sha256"]==digest(source/"selection-manifest.json"),
            "Reference source selection differs")
        for key in ("candidate_id","penalty","category_ratio","feature_kind","scores_array_sha256","scores_file_sha256"):
            require(row[key]==original[key],"Reference metadata differs")
        original_path = safe_path(source,original["scores_file"])
        require(digest(original_path)==row["scores_file_sha256"]
            and np.array_equal(np.load(original_path,allow_pickle=False),arrays[name]),"Reference predictions differ")
    old,new = arrays["locked_binary"],selected["binary"]
    error = close(old,new,"Locked binary reference",atol=2e-9,rtol=2e-9)
    exact = np.array_equal(rank(old,inputs["eligible"]),rank(new,inputs["eligible"]))
    require(exact,"Locked binary top10 ranking changed")
    return arrays,{"maximum_absolute_binary_score_error":error,"exact_binary_top10_order":True,
        "reference_manifest_sha256":expected["manifest_sha256"]}


def check_comparisons(comparisons, details):
    expected_names = {f"{first}_minus_{second}:{endpoint}"
        for first,second in (("pair_only","binary"),("nested","binary"),
                             ("pair_only","locked_categorical"),("nested","locked_categorical"),
                             ("binary","locked_binary"))
        for endpoint in ("all_observed","liked_ratings")}
    require(set(comparisons)==expected_names,"Comparison coverage differs")
    maximum = 0.
    for name,original in comparisons.items():
        contrast,endpoint = name.split(":")
        first,second = contrast.split("_minus_")
        a,b = details[first][endpoint]["per_user"],details[second][endpoint]["per_user"]
        require(set(a)==set(b),"Paired comparison users differ")
        delta = np.array([a[user]["ndcg@10"]-b[user]["ndcg@10"] for user in sorted(a)])
        expected = {"users":len(delta),"mean_ndcg_delta":float(delta.mean()),
                    "fraction_users_improved":float(np.mean(delta>0))}
        error,_ = compare_group_fields(original,expected)
        maximum = max(maximum,error)
    return {"comparisons_verified":len(expected_names),"maximum_absolute_error":maximum}


def audit(args):
    started = time.monotonic()
    research,evidence = args.research.resolve(),args.evidence.resolve()
    require(not args.out.exists(),"Refuse to overwrite audit evidence")
    frozen,protocol,manifests,complete = check_barrier(research)
    from exploratory.categorical_reconstruction.verify_results import check_barrier as check_reference_barrier
    check_reference_barrier(args.reference_run)
    check_hashes(evidence,json_read(evidence/"SHA256.json"))
    require(digest(evidence/"aggregates.json")==digest(research/"aggregates.json")
        and digest(evidence/"protocol.json")==digest(research/"protocol.json"),"Curated outputs differ")
    freeze_hash,manifest_hash = digest(research/"SELECTIONS-FROZEN.json"),digest(research/"manifest.json")
    provenance = json_read(evidence/"provenance.json")
    require(provenance["manifest_sha256"]==manifest_hash
        and provenance["selection_freeze_sha256"]==freeze_hash
        and provenance["source_sha256"]==frozen["source_sha256"],"Curated provenance differs")
    artifacts = {name:digest(evidence/name) for name in ("aggregates.json","protocol.json","provenance.json")}
    verifier_sources = {str(Path(__file__).relative_to(ROOT)):digest(__file__)}
    for name in ("verify_results.py","verify_groups.py"):
        path = ROOT/"exploratory/categorical_reconstruction"/name
        verifier_sources[str(path.relative_to(ROOT))] = digest(path)
    synthetic = explicit_feature_check()
    published = json_read(evidence/"aggregates.json")
    require(set(published["seeds"])==set(map(str,SEEDS)),"Missing published seeds")
    result = {"schema_version":1,"status":"complete","stage":"reused_development",
        "fresh_confirmation":False,"test_read":False,"test_evaluated":False,
        "source_sha256":frozen["source_sha256"],"verifier_source_sha256":verifier_sources,
        "selection_freeze_sha256":freeze_hash,"completed_manifest_sha256":manifest_hash,
        "artifact_sha256":artifacts,"synthetic_primal_check":synthetic,"seeds":{}}
    for seed in SEEDS:
        directory = research/str(seed)
        inputs = load_inputs(directory,manifests[seed],args.data_path,args.source_root,seed)
        grid,chosen = check_grid(directory)
        arrays = selected_arrays(directory,chosen,inputs["matrix"].shape)
        replay = selected_replay(inputs,grid,chosen,arrays)
        references,reference_checks = reference_arrays(research,args.reference_run,seed,inputs,arrays)
        details = json_read(directory/"development-metrics.json")
        records,error = {},0.
        for role,scores in {**arrays,**references}.items():
            section = "models" if role in ROLES else "references"
            records[role],difference = evaluate_scores(scores,inputs,
                published["seeds"][str(seed)][section][role],details[role],
                args.data_path/"ml-100k.item",args.data_path/"ml-100k.inter")
            error = max(error,difference)
        comparisons = check_comparisons(published["seeds"][str(seed)]["comparison"],details)
        result["seeds"][str(seed)] = {"candidate_rows":len(grid),"selection_pools":
            {role:chosen[role]["candidate_count"] for role in ROLES},"replay":replay,
            "references":reference_checks,"comparisons":comparisons,
            "verified_development_metrics":records,"maximum_metric_absolute_error":error}
        print(f"Verified seed {seed}: 45 grid rows, selected predictions and all development metrics",flush=True)
    check_barrier(research)
    check_reference_barrier(args.reference_run)
    require(digest(research/"SELECTIONS-FROZEN.json")==freeze_hash and digest(research/"manifest.json")==manifest_hash,
        "Sealed study changed during audit")
    check_hashes(evidence,artifacts)
    check_hashes(ROOT,verifier_sources)
    result["unique_candidate_rows_verified"] = 135
    result["elapsed_seconds"] = time.monotonic()-started
    result["limits"] = ["All135 candidate rows and exact selection maxima are checked; only selected configurations are refitted.",
        "Selected fits reuse the tested solver; literal feature primal and sampled direct target kernels are separate implementations.",
        "All per-user ranking metrics, aggregate coverage, known-dislike/frequency diagnostics and numeric group fields are independently recomputed.",
        "No TEST split or final-result artifact is opened. Reused development is not fresh confirmation."]
    args.out.mkdir(parents=True,exist_ok=False)
    json_write(args.out/"audit.json",result)
    (args.out/"RESULTS.md").write_text("# Independent interaction results audit\n\nAll selected-model replay, direct kernel, grid, barrier and independent metric checks passed.\n\n135 candidate rows; three splits; reused development only. No fresh confirmation or TEST access.\n")
    json_write(args.out/"SHA256.json",{path.name:digest(path) for path in sorted(args.out.iterdir())})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("research","evidence","data-path","out"):
        parser.add_argument("--"+flag,type=Path,required=True)
    parser.add_argument("--source-root",type=Path,default=ROOT/"runs/research-v2")
    parser.add_argument("--reference-run",type=Path,default=ROOT/"runs/categorical-reconstruction-v1")
    result = audit(parser.parse_args())
    print(json.dumps({"status":result["status"],"elapsed_seconds":result["elapsed_seconds"]},indent=2))


if __name__ == "__main__":
    main()
