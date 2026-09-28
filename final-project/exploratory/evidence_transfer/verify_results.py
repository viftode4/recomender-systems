"""Independent sealed-run audit for the shared evidence interpreter.

Rebuilds TRAIN episodes/features, replays every checkpoint with explicit network
operations, and recomputes ranking metrics without importing the measured
runner. No fitting, TEST access, new selection, or individual audit output.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import scipy
import torch
import torch.nn.functional as functional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from exploratory.categorical_reconstruction.verify_results import (
    array_digest,check_hashes,close,digest,json_read,json_write,load_inputs,
    require,safe_path,selected_arrays,check_barrier as reference_barrier)
from exploratory.context_interactions.verify_results import full_metrics,evaluate_scores
from exploratory.categorical_reconstruction.verify_groups import compare as compare_fields

SEEDS = (2026,2027,2028)
ARMS = ("full_pattern","no_pattern","marginal_only")
MODELS = (*ARMS,"analytic_donor","full_zero_pattern")
CHECKPOINTS = (0,10,30,60,100)
FEATURES = ("log_context_count","log_donor_support","donor_support_fraction",
    "log_weighted_support","weighted_support_fraction","mean_supporter_weight",
    "log_kish_count","log_pattern_count","pattern_coverage")
CONTRASTS = ("full_pattern_minus_no_pattern", "full_pattern_minus_marginal_only",
    "full_pattern_minus_analytic_donor", "full_pattern_minus_locked_binary",
    "no_pattern_minus_locked_binary", "full_zero_pattern_minus_full_pattern")


def check_barrier(research):
    frozen = json_read(research/"SELECTIONS-FROZEN.json")
    require(frozen.get("status")=="all_selections_frozen" and frozen.get("test_read") is False
        and frozen.get("development_evaluated") is False and tuple(frozen.get("seeds",[]))==SEEDS,
        "Invalid global all-seed selection barrier")
    protocol = json_read(research/"protocol.json")
    require(digest(research/"protocol.json")==frozen["protocol_sha256"],"Protocol changed after freeze")
    expected = {"seeds":list(SEEDS),"arms":list(ARMS),"models":list(MODELS),
        "checkpoints":list(CHECKPOINTS),"retentions":[.8,.9],"feature_names":list(FEATURES),
        "width":16,"batch_size":64,"learning_rate":.001,"weight_decay":.0001,"optimizer":"Adam",
        "mask_seed_offset":41001,"initialization_seed_offset":42001,"batch_seed_offset":43001,
        "test_read":False,"test_evaluated":False,"fresh_confirmation":False,
        "stage":"reused_development","dataset_test_previously_evaluated":True}
    require(all(protocol.get(key)==value for key,value in expected.items()),"Study differs from declared pilot")
    require(protocol["source_sha256"]==frozen["source_sha256"],"Scientific source seals differ")
    check_hashes(ROOT,frozen["source_sha256"])
    require(digest(research/"reference-input-signature.json")==frozen["reference_input_signature_sha256"],
        "Locked reference signature changed")
    runtime = {"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
        "machine":platform.machine(),"system":platform.system(),"numerical_threads":1,
        "torch":str(torch.__version__),"device":"cpu","model_dtype":"float32","deterministic_algorithms":True}
    require(all(frozen["runtime"][key]==value for key,value in runtime.items()),"Use the frozen numerical runtime")
    manifests = {}
    for seed in SEEDS:
        directory = research/str(seed)
        require(digest(directory/"selection-manifest.json")==frozen["selection_manifest_sha256"][str(seed)],
            "Selection manifest changed")
        row = json_read(directory/"selection-manifest.json")
        require(row["status"]=="selected" and row["seed"]==seed and row["test_read"] is False
            and row["development_evaluated"] is False and row["source_sha256"]==frozen["source_sha256"]
            and row["runtime"]==frozen["runtime"] and row["arms_trained"]==3
            and row["checkpoints_per_arm"]==5,"Invalid seed selection manifest")
        check_hashes(directory,row["output_sha256"])
        manifests[seed] = row
    complete = json_read(research/"manifest.json")
    require(complete["status"]=="complete" and complete["test_read"] is False
        and complete["test_evaluated"] is False and complete["fresh_confirmation"] is False
        and complete["source_sha256"]==frozen["source_sha256"] and complete["runtime"]==frozen["runtime"]
        and complete["selection_freeze_sha256"]==digest(research/"SELECTIONS-FROZEN.json"),"Invalid completion")
    check_hashes(research,complete["output_sha256"])
    opened = json_read(research/"DEVELOPMENT-OPENED.json")
    require(opened["status"]=="complete" and opened["test_read"] is False
        and opened["selection_freeze_sha256"]==digest(research/"SELECTIONS-FROZEN.json"),"Development marker differs")
    return frozen,protocol,manifests,complete


def rebuild_episodes(history,seed):
    rng = np.random.default_rng(seed+41001)
    contexts,targets,excluded = [],[],[]
    omitted = np.flatnonzero(history.sum(axis=1)<2).astype(np.int64)
    for keep_fraction in (.8,.9):
        for user in range(len(history)):
            present = np.flatnonzero(history[user])
            if len(present)<2:
                continue
            retained = min(len(present)-1,max(1,math.floor(keep_fraction*len(present))))
            query = np.zeros(history.shape[1],dtype=bool)
            query[rng.choice(present,retained,replace=False)] = True
            target = history[user]&~query
            require(not np.any(query&target) and target.any(),"Invalid hidden TRAIN target")
            contexts.append(query);targets.append(target);excluded.append(user)
    contexts,targets = np.array(contexts),np.array(targets)
    eligible = ~contexts;eligible[:,0] = False
    return {"contexts":contexts,"targets":targets,"eligible":eligible,
        "exclude_rows":np.array(excluded,dtype=np.int64),"excluded_training_rows":omitted}


def rebuild_features(directory,inputs,seed):
    # Feature algebra is tested against scalar donor loops separately. Here the
    # tested kernel is replayed; episodes and scaling are reconstructed directly.
    from exploratory.evidence_transfer.model import prepare_donors,extract_features
    history = inputs["matrix"]>0
    episodes = rebuild_episodes(history,seed)
    hashes = json_read(directory/"feature-hashes.json")
    require(tuple(hashes["feature_names"])==FEATURES,"Feature ordering differs")
    with np.load(directory/"episodes.npz",allow_pickle=False) as stored:
        require(set(stored.files)==set(episodes),"Episode array coverage differs")
        for key,value in episodes.items():
            require(np.array_equal(stored[key],value)
                and hashes["episode_arrays_sha256"][key]==array_digest(value),"TRAIN episode replay differs")
    bank = prepare_donors(history)
    train = extract_features(bank,episodes["contexts"],episodes["exclude_rows"])
    query = extract_features(bank,history,np.arange(len(history),dtype=np.int64))
    require(array_digest(train)==hashes["raw_train_features_sha256"]
        and array_digest(query)==hashes["raw_query_features_sha256"],"Raw feature replay differs")
    eligible = train[episodes["eligible"]]
    mean = eligible.mean(axis=0,dtype=np.float64)
    scale = eligible.std(axis=0,dtype=np.float64);scale[scale==0] = 1
    del eligible
    require(array_digest(mean)==hashes["normalizer_mean_sha256"]
        and array_digest(scale)==hashes["normalizer_scale_sha256"],"TRAIN-only scaling differs")
    with np.load(directory/"normalizer.npz",allow_pickle=False) as stored:
        require(np.array_equal(mean,stored["mean"]) and np.array_equal(scale,stored["scale"]),"Stored scaler differs")
    normalized_train = ((train-mean)/scale).astype(np.float32)
    normalized_query = ((query-mean)/scale).astype(np.float32)
    require(array_digest(normalized_train)==hashes["normalized_train_features_sha256"]
        and array_digest(normalized_query)==hashes["normalized_query_features_sha256"],"Standardized replay differs")
    audit = {"episodes":len(episodes["contexts"]),"excluded_training_users":len(episodes["excluded_training_rows"]),
        "episode_replay_exact":True,"raw_features_replay_exact":True,"normalization_replay_exact":True,
        "feature_hashes_sha256":digest(directory/"feature-hashes.json")}
    return normalized_query,query[:,:,4].copy(),audit


def checkpoint_scores(path,features,seed,source_arm,mask_arm=None):
    state = torch.load(path,map_location="cpu",weights_only=True)
    require(state["config"]=={"n_features":9,"width":16,"variant":source_arm,"seed":seed+42001},
        "Checkpoint scorer configuration differs")
    weights = state["state_dict"]
    shapes = {"network.0.weight":(16,9),"network.0.bias":(16,),
        "network.2.weight":(1,16),"network.2.bias":(1,)}
    require(set(weights)==set(shapes) and all(tuple(weights[key].shape)==shape for key,shape in shapes.items()),
        "Unexpected item-specific parameters or scorer shape")
    keep = torch.ones(9)
    variant = source_arm if mask_arm is None else mask_arm
    if variant=="no_pattern":keep[7:] = 0
    elif variant=="marginal_only":keep[3:] = 0
    rows = []
    with torch.no_grad():
        for start in range(0,len(features),64):
            batch = torch.from_numpy(np.ascontiguousarray(features[start:start+64]))*keep
            hidden = functional.silu(functional.linear(batch,weights["network.0.weight"],weights["network.0.bias"]))
            scores = functional.linear(hidden,weights["network.2.weight"],weights["network.2.bias"]).squeeze(-1)
            rows.append(scores.numpy())
    result = np.concatenate(rows).astype(np.float32,copy=False);result[:,0] = 0
    require(np.isfinite(result).all(),"Nonfinite checkpoint replay")
    return result


def replay_choices(directory,inputs,seed):
    chosen = json_read(directory/"selection.json")
    require(set(chosen)==set(MODELS),"Model roles differ")
    arrays = selected_arrays(directory,chosen,inputs["matrix"].shape)
    features,analytic,feature_checks = rebuild_features(directory,inputs,seed)
    controls = json_read(directory/"training-audit.json")
    require(set(controls)==set(ARMS) and all(controls[arm]==controls[ARMS[0]] for arm in ARMS),
        "Matched arm training budgets or initialization differ")
    expected_updates = 100*math.ceil(feature_checks["episodes"]/64)
    require(controls[ARMS[0]]["updates"]==expected_updates
        and controls[ARMS[0]]["epochs"]==100 and controls[ARMS[0]]["parameter_count"]==177
        and controls[ARMS[0]]["episodes"]==feature_checks["episodes"],
        "Declared allocated training budget differs")
    checks = {}
    for arm in ARMS:
        grid = json_read(directory/"checkpoints"/arm/"checkpoint-metrics.json")
        require(tuple(row["epoch"] for row in grid)==CHECKPOINTS,"Checkpoint grid incomplete")
        require(all(math.isfinite(row["meta_fit_all_observed_ndcg@10"])
            and 0<=row["meta_fit_all_observed_ndcg@10"]<=1 for row in grid),"Invalid meta metric")
        winner = max(grid,key=lambda row:row["meta_fit_all_observed_ndcg@10"])
        for key in ("epoch","meta_fit_all_observed_ndcg@10","scores_array_sha256","checkpoint_file_sha256"):
            require(chosen[arm][key]==winner[key],"Choice is not the earliest exact meta maximum")
        require(chosen[arm]["candidate_count"]==5 and chosen[arm]["selection_cohort"]=="reused_meta_fit",
            "Checkpoint selection boundary differs")
        require(chosen[arm]["checkpoint_file"]==f"checkpoints/{arm}/"+winner["checkpoint_file"],
            "Selected checkpoint path differs")
        max_error = 0.
        for row in grid:
            path = safe_path(directory,f"checkpoints/{arm}/"+row["checkpoint_file"])
            require(digest(path)==row["checkpoint_file_sha256"],"Checkpoint file changed")
            scores = checkpoint_scores(path,features,seed,arm)
            require(array_digest(scores)==row["scores_array_sha256"],"Checkpoint prediction replay differs")
            meta,_ = full_metrics(scores,inputs,inputs["cohorts"]["meta_fit"],inputs["truth"])
            max_error = max(max_error,close(meta["ndcg@10"],row["meta_fit_all_observed_ndcg@10"],"Meta nDCG"))
            if row["epoch"]==winner["epoch"]:
                require(np.array_equal(scores,arrays[arm]),"Selected checkpoint scores differ")
            if row["epoch"]==0:
                state = torch.load(path,map_location="cpu",weights_only=True)["state_dict"]
                require({key:array_digest(value.numpy()) for key,value in state.items()}==controls[arm]["initial_state_sha256"],
                    "Initial-state fingerprints differ")
        epochs = json_read(directory/"checkpoints"/arm/"epoch-metrics.json")
        require([row["epoch"] for row in epochs]==list(range(1,101))
            and all(math.isfinite(row["mean_online_episode_loss"]) for row in epochs),"Training trajectory incomplete")
        checks[arm] = {"checkpoint_predictions_replayed":5,"selected_epoch":winner["epoch"],
            "maximum_meta_metric_error":max_error,"exact_selected_score_replay":True}
    analytic[:,0] = 0
    require(chosen["analytic_donor"]["selection_cohort"]=="none_untrained"
        and np.array_equal(analytic,arrays["analytic_donor"]),"Unfitted donor control differs")
    zero,full = chosen["full_zero_pattern"],chosen["full_pattern"]
    require(zero["selection_cohort"]=="none_fixed_intervention" and zero["source_arm"]=="full_pattern"
        and zero["epoch"]==full["epoch"] and zero["checkpoint_file"]==full["checkpoint_file"]
        and zero["checkpoint_file_sha256"]==full["checkpoint_file_sha256"],"Intervention source differs")
    intervened = checkpoint_scores(safe_path(directory,full["checkpoint_file"]),features,seed,"full_pattern","no_pattern")
    require(np.array_equal(intervened,arrays["full_zero_pattern"]),"Frozen-weight intervention differs")
    return arrays,{"feature_checks":feature_checks,"trained_arms":checks,
        "analytic_replay_exact":True,"frozen_weight_intervention_replay_exact":True,
        "common_updates_per_arm":expected_updates}


def audit(args):
    started = time.monotonic()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    research,evidence = args.research.resolve(),args.evidence.resolve()
    require(not args.out.exists(),"Refuse to overwrite audit evidence")
    frozen,_,manifests,_ = check_barrier(research)
    reference_barrier(args.reference_run)
    check_hashes(evidence,json_read(evidence/"SHA256.json"))
    frozen_hash,completed_hash = digest(research/"SELECTIONS-FROZEN.json"),digest(research/"manifest.json")
    artifacts = {name:digest(evidence/name) for name in ("aggregates.json","protocol.json","provenance.json")}
    for name in ("aggregates.json","protocol.json"):
        require(artifacts[name]==digest(research/name),"Curated artifact differs")
    provenance = json_read(evidence/"provenance.json")
    require(provenance["manifest_sha256"]==completed_hash and provenance["selection_freeze_sha256"]==frozen_hash
        and provenance["source_sha256"]==frozen["source_sha256"],"Curated provenance differs")
    reference = json_read(research/"reference-input-signature.json")
    require(reference=={"manifest_sha256":digest(args.reference_run/"manifest.json"),
        "selection_freeze_sha256":digest(args.reference_run/"SELECTIONS-FROZEN.json")},"Reference root changed")
    verifier_paths = (Path(__file__),ROOT/"exploratory/categorical_reconstruction/verify_results.py",
        ROOT/"exploratory/categorical_reconstruction/verify_groups.py",ROOT/"exploratory/context_interactions/verify_results.py")
    verifier_hashes = {str(path.relative_to(ROOT)):digest(path) for path in verifier_paths}
    published = json_read(evidence/"aggregates.json")
    require(set(published["seeds"])=={str(seed) for seed in SEEDS}
        and published["stage"]=="reused_development" and published["test_read"] is False
        and published["fresh_confirmation"] is False,"Aggregate scope differs")
    result = {"schema_version":1,"status":"complete","stage":"reused_development", "test_read":False,
        "test_evaluated":False,"fresh_confirmation":False,"artifact_sha256":artifacts,
        "source_sha256":frozen["source_sha256"],"verifier_source_sha256":verifier_hashes,
        "selection_freeze_sha256":frozen_hash,"completed_manifest_sha256":completed_hash,"seeds":{}}
    for seed in SEEDS:
        directory = research/str(seed)
        inputs = load_inputs(directory,manifests[seed],args.data_path,args.source_root,seed)
        arrays,replay = replay_choices(directory,inputs,seed)
        ref = json_read(directory/"locked-reference.json")
        original = json_read(args.reference_run/str(seed)/"selection.json")["binary_expanded"]
        require(ref["source_model"]=="binary_expanded" and ref["penalty"]==original["penalty"]
            and ref["candidate_id"]==original["candidate_id"]
            and ref["scores_file_sha256"]==original["scores_file_sha256"]
            and ref["source_selection_manifest_sha256"]==digest(args.reference_run/str(seed)/"selection-manifest.json"),
            "Locked binary source selection differs")
        old_signature = json_read(args.reference_run/str(seed)/"input-signature.json")
        require(old_signature==inputs["signature"],"Locked binary inputs differ")
        arrays.update(selected_arrays(directory,{"locked_binary":ref},inputs["matrix"].shape))
        details = json_read(directory/"development-metrics.json")
        section = published["seeds"][str(seed)]
        require(set(details)==set(arrays) and set(section["models"])==set(MODELS)
            and set(section["references"])=={"locked_binary"},"Aggregate model coverage differs")
        require(section["cohorts"]=={name:len(inputs["cohorts"][name]) for name in ("meta_fit","development")},
            "Aggregate cohort sizes differ")
        records,error = {},0.
        for name,scores in arrays.items():
            section = "references" if name=="locked_binary" else "models"
            records[name],difference = evaluate_scores(scores,inputs,published["seeds"][str(seed)][section][name],details[name],
                args.data_path/"ml-100k.item",args.data_path/"ml-100k.inter")
            error = max(error,difference)
        contrasts = published["seeds"][str(seed)]["comparison"]
        require(set(contrasts)=={f"{pair}:{endpoint}" for pair in CONTRASTS
            for endpoint in ("all_observed","liked_ratings")},"Declared comparison coverage differs")
        for name,original in contrasts.items():
            pair,endpoint = name.split(":");first,second = pair.split("_minus_")
            a,b = details[first][endpoint]["per_user"],details[second][endpoint]["per_user"]
            require(set(a)==set(b),"Comparison user sets differ")
            delta = np.array([a[user]["ndcg@10"]-b[user]["ndcg@10"] for user in sorted(a)])
            compare_fields(original,{"users":len(delta),"mean_ndcg_delta":float(delta.mean()),
                "fraction_users_improved":float(np.mean(delta>0))})
        result["seeds"][str(seed)] = {"replay":replay,"verified_development_metrics":records,
            "maximum_metric_absolute_error":error,"comparisons_verified":len(contrasts)}
        print(f"Verified seed {seed}: TRAIN features,15 checkpoints,6 score rows and all development metrics",flush=True)
    check_barrier(research);reference_barrier(args.reference_run)
    require(digest(research/"SELECTIONS-FROZEN.json")==frozen_hash and digest(research/"manifest.json")==completed_hash,
        "Audit input changed")
    check_hashes(evidence,artifacts);check_hashes(ROOT,verifier_hashes)
    result["checkpoint_predictions_replayed"] = 45
    result["elapsed_seconds"] = time.monotonic()-started
    result["limits"] = ["Weights are replayed, not retrained; all15 checkpoint predictions per seed are checked.",
        "TRAIN features reuse the separately algebra-tested extractor; episode generation, scaling, network operations and metrics are separately implemented.",
        "Original TEST is not opened; results remain exploratory reused development."]
    args.out.mkdir(parents=True,exist_ok=False)
    json_write(args.out/"audit.json",result)
    (args.out/"RESULTS.md").write_text("# Independent shared-evidence audit\n\nAll sealed feature, checkpoint, ranking and metric checks passed.\n\n45 checkpoint predictions replayed; three overlapping splits. No TEST access or fresh confirmation.\n")
    json_write(args.out/"SHA256.json",{path.name:digest(path) for path in sorted(args.out.iterdir())})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("research","evidence","data-path","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--source-root",type=Path,default=ROOT/"runs/research-v2")
    parser.add_argument("--reference-run",type=Path,default=ROOT/"runs/categorical-reconstruction-v1")
    result = audit(parser.parse_args())
    print(json.dumps({"status":result["status"],"seconds":result["elapsed_seconds"]},indent=2))


if __name__=="__main__":
    main()
