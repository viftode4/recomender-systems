"""RecBole split/train adapter with independent evaluation and portable scores."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.utils import get_model, get_trainer, init_seed

from metrics import evaluate

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def split_pairs(loader, dataset, user_field, item_field):
    features = loader.dataset.inter_feat
    users = dataset.id2token(user_field, features[user_field].cpu().numpy())
    items = dataset.id2token(item_field, features[item_field].cpu().numpy())
    return sorted((str(u), str(i)) for u, i in zip(users, items))


def by_user(pairs):
    result = defaultdict(set)
    for user, item in pairs:
        result[user].add(item)
    return result


def top_k(scores, item_ids, seen, k):
    """Stable ties use internal item order; never emit padding/seen items."""
    eligible = [i for i, item in enumerate(item_ids) if i != 0 and item not in seen]
    if len(eligible) < k:
        raise ValueError("Fewer than k unseen candidates")
    values = np.asarray(scores)[eligible]
    if not np.isfinite(values).all():
        raise ValueError("Non-finite candidate score")
    order = np.argsort(-values, kind="stable")[:k]
    return [item_ids[eligible[i]] for i in order]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["Random", "ExactPop", "ItemKNN", "EASE", "BPR"], required=True)
    parser.add_argument("--data-path", type=Path, required=True, help="Parent containing ml-100k atomic files")
    parser.add_argument("--out", type=Path, required=True, help="New run directory; existing paths are refused")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--test", action="store_true", help="Explicitly evaluate held-out test after settings are frozen")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    data_path, output = args.data_path.resolve(), args.out.resolve()
    dataset_name = settings["dataset"]
    if not (data_path / dataset_name / f"{dataset_name}.inter").is_file():
        parser.error("Missing local atomic dataset; see README")
    output.mkdir(parents=True, exist_ok=False)
    # Keep RecBole's TensorBoard/checkpoint artifacts inside this run.
    os.chdir(output)
    baseline = args.model in {"Random", "ExactPop"}
    config = Config(model="Random" if baseline else args.model, dataset=dataset_name,
                    config_dict={**settings, "data_path": str(data_path),
                                 "checkpoint_dir": str(output / "checkpoints")})
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train, valid, test = data_preparation(config, dataset)
    uf, itf = config["USER_ID_FIELD"], config["ITEM_ID_FIELD"]
    splits = {name: split_pairs(loader, dataset, uf, itf)
              for name, loader in [("train", train), ("valid", valid), ("test", test)]}
    sets = {name: set(pairs) for name, pairs in splits.items()}
    if any(len(sets[name]) != len(pairs) for name, pairs in splits.items()):
        raise ValueError("Duplicate interactions: define an aggregation policy first")
    if any(sets[a] & sets[b] for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]):
        raise ValueError("Split overlap")
    fingerprints = {}
    for name, pairs in splits.items():
        body = "user_id\titem_id\n" + "".join(f"{u}\t{i}\n" for u, i in pairs)
        (output / f"{name}.tsv").write_text(body)
        fingerprints[name] = hashlib.sha256(body.encode()).hexdigest()
    counts = Counter(item for _, item in splits["train"])
    item_ids = [str(i) for i in dataset.id2token(itf, np.arange(dataset.item_num))]
    user_ids = [str(u) for u in dataset.id2token(uf, np.arange(dataset.user_num))]
    write_json(output / "ids.json", {"users": user_ids, "items": item_ids, "padding_index": 0})
    write_json(output / "training_counts.json", dict(counts))
    (output / "effective-config.txt").write_text(str(config))
    manifest = {
        "model": args.model, "settings": settings, "split_sha256": fingerprints,
        "split_sizes": {key: len(value) for key, value in splits.items()},
        "test_evaluated": args.test, "python": sys.version,
        "versions": {p: importlib.metadata.version(p) for p in ["recbole", "numpy", "torch", "scipy"]},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [ROOT / "run.py", ROOT / "metrics.py"]},
        "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted((data_path / dataset_name).glob(f"{dataset_name}.*")) if p.is_file()},
        "protocol": "All observed ratings are positives. Full catalog; padding/seen items excluded. Validation selects models.",
        "status": "started",
    }
    write_json(output / "manifest.json", manifest)
    model = None
    if not baseline:
        init_seed(config["seed"], config["reproducibility"])
        model = get_model(config["model"])(config, train.dataset).to(config["device"])
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        trainer.fit(train, valid, saved=True, show_progress=False)
        # Only deserialize the checkpoint produced by this process.
        checkpoint = torch.load(trainer.saved_model_file, map_location=config["device"], weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.load_other_parameter(checkpoint.get("other_parameter"))
        model.eval()
    k = config["topk"][0]
    for phase in (["valid", "test"] if args.test else ["valid"]):
        truth = by_user(splits[phase])
        history = by_user(splits["train"] + (splits["valid"] if phase == "test" else []))
        users = [i for i, user in enumerate(user_ids) if i and user in truth]
        recommendations, raw = {}, []
        with torch.no_grad():
            for uid in users:
                user = user_ids[uid]
                if args.model == "ExactPop":
                    scores = np.asarray([counts.get(item, 0) for item in item_ids], dtype=float)
                elif args.model == "Random":
                    # Independent per-user randomness, unaffected by iteration order or training draws.
                    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], uid, int(phase == "test")]))
                    scores = rng.random(len(item_ids))
                else:
                    interaction = Interaction({uf: torch.tensor([uid], device=config["device"])})
                    scores = model.full_sort_predict(interaction).detach().cpu().numpy().reshape(-1)
                recommendations[user] = top_k(scores, item_ids, history[user], k)
                raw.append(scores)
        # Unmasked scores preserve future hybrid features. Consumers must apply phase-specific masks.
        np.savez_compressed(output / f"{phase}-scores.npz", scores=np.stack(raw),
                            users=np.asarray([user_ids[i] for i in users]), items=np.asarray(item_ids))
        write_json(output / f"{phase}-recommendations.json", recommendations)
        result = evaluate(recommendations, truth, history, item_ids[1:], counts, k)
        write_json(output / f"{phase}-metrics.json", result)
        print(phase, json.dumps(result["aggregate"], sort_keys=True))
    manifest["status"] = "complete"
    write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
