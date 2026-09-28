"""RecBole split/train adapter with independent evaluation and portable scores."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.utils import get_model, get_trainer, init_seed
from recbole.model.general_recommender.fism import FISM
from recbole.model.general_recommender.ngcf import NGCF

from metrics import evaluate

ROOT = Path(__file__).resolve().parent
MODELS = ('Random', 'ExactPop', 'ItemKNN', 'UserKNN', 'EASE', 'BPR',
          'SLIMElastic', 'FISMCorrected', 'GenreContent', 'LightGCN', 'NeuMF', 'NGCF')
BASELINES = {'Random', 'ExactPop', 'GenreContent'}


class NGCFCompatible(NGCF):
    """Same normalized binary graph using public SciPy sparse operations."""

    def get_norm_adj_mat(self):
        matrix = self.interaction_matrix
        rows = np.concatenate([matrix.row, matrix.col + self.n_users])
        cols = np.concatenate([matrix.col + self.n_users, matrix.row])
        size = self.n_users + self.n_items
        adjacency = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(size, size)).tocsr()
        adjacency.data[:] = 1.0
        inverse_sqrt = (np.asarray(adjacency.sum(axis=1)).ravel() + 1e-7) ** -0.5
        normalized = (sp.diags(inverse_sqrt) @ adjacency @ sp.diags(inverse_sqrt)).tocoo()
        indices = torch.tensor(np.stack([normalized.row, normalized.col]), dtype=torch.long)
        return torch.sparse_coo_tensor(indices, torch.tensor(normalized.data, dtype=torch.float32), (size, size)).coalesce()


class FISMCorrected(FISM):
    """Local correction: raw-logit BCE and leave-target-out history in every path.

    The instructor's initialization and non-squared embedding regularizer remain
    unchanged. This is a documented correction, not a canonical-FISM claim.
    Full-sort exports logits; point prediction returns sigmoid(logit) once.
    """

    def inter_forward(self, user, item):
        history = self.history_item_matrix[user]
        mask = self.mask_mat[user] * (history != item.unsqueeze(1)).float()
        source = self.item_src_embedding(history)
        target = self.item_dst_embedding(item)
        similarities = (source * target.unsqueeze(1)).sum(dim=2)
        count = mask.sum(dim=1).clamp_min(1.0)
        return (count.pow(-self.alpha) * (mask * similarities).sum(dim=1)
                + self.user_bias[user] + self.item_bias[item])

    def predict(self, interaction):
        return torch.sigmoid(self.inter_forward(interaction[self.USER_ID], interaction[self.ITEM_ID]))

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID]
        history = self.history_item_matrix[users]
        mask = self.mask_mat[users]
        source = self.item_src_embedding(history)
        summed = (source * mask.unsqueeze(2)).sum(dim=1)
        membership = torch.zeros((len(users), self.n_items), device=source.device, dtype=source.dtype)
        membership.scatter_add_(1, history, mask)
        # Removing a scored training item also changes the history normalization.
        dots = summed @ self.item_dst_embedding.weight.T
        self_dots = (self.item_src_embedding.weight * self.item_dst_embedding.weight).sum(dim=1)
        counts = (mask.sum(dim=1, keepdim=True) - membership).clamp_min(1.0)
        logits = counts.pow(-self.alpha) * (dots - membership * self_dots.unsqueeze(0))
        return (logits + self.user_bias[users, None] + self.item_bias[None, :]).reshape(-1)


def recbole_model_name(name):
    return 'Random' if name in BASELINES else {'UserKNN': 'ItemKNN', 'FISMCorrected': 'FISM'}.get(name, name)


def construct_model(name, config, train_dataset):
    model_class = {'FISMCorrected': FISMCorrected, 'NGCF': NGCFCompatible}.get(name)
    if model_class is None:
        model_class = get_model(config['model'])
    return model_class(config, train_dataset).to(config['device'])


def full_catalog_scores(model, name, user, user_field, item_field, item_count, device):
    """NeuMF provides point prediction only; score the identical full catalog."""
    if name == 'NeuMF':
        chunks = []
        for start in range(0, item_count, 1024):
            items = torch.arange(start, min(start + 1024, item_count), device=device)
            interaction = Interaction({user_field: torch.full_like(items, user), item_field: items})
            chunks.append(model.predict(interaction))
        scores = torch.cat(chunks)
    else:
        interaction = Interaction({user_field: torch.tensor([user], device=device)})
        scores = model.full_sort_predict(interaction)
    return scores.detach().cpu().numpy().reshape(-1)


def genre_content_features(path, item_ids, user_ids, train_pairs, idf_power=1.0):
    """Cosine between genre TF-IDF items and profiles from training items only.

    IDF uses known catalog metadata (no interaction labels). Uniform weighting
    of all observed training items matches the project's implicit protocol.
    """
    if not np.isfinite(idf_power) or idf_power < 0:
        raise ValueError('genre_idf_power must be finite and nonnegative')
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    mapping = {r['item_id:token']: set(r['class:token_seq'].split()) for r in rows}
    if any(not mapping.get(i) for i in item_ids[1:]):
        raise ValueError('Missing catalog genres')
    genres = sorted(set().union(*(mapping[i] for i in item_ids[1:])))
    features = np.zeros((len(item_ids), len(genres)))
    for index, item in enumerate(item_ids[1:], 1):
        features[index] = [genre in mapping[item] for genre in genres]
    idf = (np.log((len(item_ids)) / (1.0 + features.sum(axis=0))) + 1.0) ** idf_power
    features *= idf
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    item_index = {item: index for index, item in enumerate(item_ids)}
    user_index = {user: index for index, user in enumerate(user_ids)}
    profiles = np.zeros((len(user_ids), len(genres)))
    for user, item in train_pairs:
        profiles[user_index[user]] += features[item_index[item]]
    profiles /= np.maximum(np.linalg.norm(profiles, axis=1, keepdims=True), 1e-12)
    return features, profiles, genres


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
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--data-path", type=Path, required=True, help="Parent containing ml-100k atomic files")
    parser.add_argument("--out", type=Path, required=True, help="New run directory; existing paths are refused")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--fixed-epochs", action="store_true", help="Train fixed budget without consulting validation labels")
    parser.add_argument("--test", action="store_true", help="Explicitly evaluate held-out test after settings are frozen")
    args = parser.parse_args()
    started = time.perf_counter()
    settings = json.loads(args.config.read_text())
    data_path, output = args.data_path.resolve(), args.out.resolve()
    dataset_name = settings["dataset"]
    if not (data_path / dataset_name / f"{dataset_name}.inter").is_file():
        parser.error("Missing local atomic dataset; see README")
    output.mkdir(parents=True, exist_ok=False)
    # Keep RecBole's TensorBoard/checkpoint artifacts inside this run.
    os.chdir(output)
    baseline = args.model in BASELINES
    adapter_settings = {'knn_method': 'user'} if args.model == 'UserKNN' else {}
    if args.model == 'ItemKNN':
        adapter_settings = {'knn_method': 'item'}
    config = Config(model=recbole_model_name(args.model), dataset=dataset_name,
                    config_dict={**settings, **adapter_settings, "data_path": str(data_path),
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
        "model": args.model, "settings": {**settings, **adapter_settings}, "split_sha256": fingerprints,
        "recbole_model": config['model'],
        "implementation": f'run.{args.model}' if args.model in {'Random', 'ExactPop'} else ('run.FISMCorrected' if args.model == 'FISMCorrected' else (
            'run.genre_content_features' if args.model == 'GenreContent' else (
                'run.NGCFCompatible' if args.model == 'NGCF' else f"recbole.{config['model']}"))),
        "split_sizes": {key: len(value) for key, value in splits.items()},
        "test_evaluated": args.test, "validation_used_for_training": not baseline and not args.fixed_epochs, "python": sys.version,
        "versions": {p: importlib.metadata.version(p) for p in ["recbole", "numpy", "torch", "scipy", "scikit-learn", "pandas"]},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [ROOT / "run.py", ROOT / "metrics.py"]},
        "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted((data_path / dataset_name).glob(f"{dataset_name}.*")) if p.is_file()},
        "protocol": "All observed ratings are positives. Full catalog; padding/seen items excluded. Validation selects models.",
        "status": "started",
    }
    if args.model == 'FISMCorrected':
        manifest['corrections'] = ['raw-logit BCEWithLogitsLoss', 'leave-target-out history and normalization',
                                   'one sigmoid in point prediction; full-sort emits logits']
    if args.model == 'SLIMElastic':
        manifest['limitations'] = ['Instructor ElasticNet uses max_iter=100 and suppresses convergence warnings.']
    if args.model == 'NGCF':
        manifest['compatibility'] = 'Public SciPy sparse adjacency construction replaces removed dok_matrix._update; same graph normalization.'
    if not baseline:
        upstream_source = Path(inspect.getfile(get_model(config['model'])))
        manifest['upstream_model_source'] = {'filename': upstream_source.name,
                                             'sha256': hashlib.sha256(upstream_source.read_bytes()).hexdigest()}
    manifest['score_space'] = {'FISMCorrected': 'raw logits', 'NeuMF': 'sigmoid probability',
                                'GenreContent': 'genre-profile cosine'}.get(args.model, 'model-native score')
    write_json(output / "manifest.json", manifest)
    model = None
    fit_started = time.perf_counter()
    if not baseline:
        init_seed(config["seed"], config["reproducibility"])
        model = construct_model(args.model, config, train.dataset)
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        trainer.fit(train, None if args.fixed_epochs else valid, saved=True, show_progress=False)
        # Only deserialize the checkpoint produced by this process.
        checkpoint = torch.load(trainer.saved_model_file, map_location=config["device"], weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.load_other_parameter(checkpoint.get("other_parameter"))
        model.eval()
        manifest['checkpoint'] = str(Path(trainer.saved_model_file).relative_to(output))
        manifest['checkpoint_sha256'] = hashlib.sha256(Path(trainer.saved_model_file).read_bytes()).hexdigest()
        trainer.tensorboard.close()
    elif args.model == 'GenreContent':
        genre_features, genre_profiles, genres = genre_content_features(
            data_path / dataset_name / f'{dataset_name}.item', item_ids, user_ids, splits['train'],
            idf_power=settings.get('genre_idf_power', 1.0))
        np.savez_compressed(output / 'content-model.npz', features=genre_features,
                            profiles=genre_profiles, genres=np.asarray(genres),
                            users=np.asarray(user_ids), items=np.asarray(item_ids))
        manifest['content_model_sha256'] = hashlib.sha256((output / 'content-model.npz').read_bytes()).hexdigest()
    manifest['timing_seconds'] = {'fit': time.perf_counter() - fit_started, 'inference': {}}
    k = config["topk"][0]
    for phase in (["valid", "test"] if args.test else ["valid"]):
        truth = by_user(splits[phase])
        history = by_user(splits["train"] + (splits["valid"] if phase == "test" else []))
        users = [i for i, user in enumerate(user_ids) if i and user in truth]
        recommendations, raw = {}, []
        inference_started = time.perf_counter()
        with torch.no_grad():
            for uid in users:
                user = user_ids[uid]
                if args.model == "ExactPop":
                    scores = np.asarray([counts.get(item, 0) for item in item_ids], dtype=float)
                elif args.model == "Random":
                    # Independent per-user randomness, unaffected by iteration order or training draws.
                    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], uid, int(phase == "test")]))
                    scores = rng.random(len(item_ids))
                elif args.model == 'GenreContent':
                    scores = genre_features @ genre_profiles[uid]
                else:
                    scores = full_catalog_scores(model, args.model, uid, uf, itf, len(item_ids), config['device'])
                recommendations[user] = top_k(scores, item_ids, history[user], k)
                raw.append(scores)
        manifest['timing_seconds']['inference'][phase] = time.perf_counter() - inference_started
        # Unmasked scores preserve future hybrid features. Consumers must apply phase-specific masks.
        np.savez_compressed(output / f"{phase}-scores.npz", scores=np.stack(raw),
                            users=np.asarray([user_ids[i] for i in users]), items=np.asarray(item_ids))
        write_json(output / f"{phase}-recommendations.json", recommendations)
        result = evaluate(recommendations, truth, history, item_ids[1:], counts, k)
        write_json(output / f"{phase}-metrics.json", result)
        print(phase, json.dumps(result["aggregate"], sort_keys=True))
    manifest["status"] = "complete"
    manifest['timing_seconds']['total'] = time.perf_counter() - started
    write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
