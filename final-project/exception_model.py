"""Train-only matched liked/disliked contrast representations.

These are operational preference contrasts, not validated psychological exceptions.
The candidate model transfers signed ratings through users with similar distributions
of oriented contrasts. All ablations share its neighborhood voting decoder.
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit


def unit_rows(values):
    values = np.asarray(values, dtype=float)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def rating_matrices(users, items, train_ratings):
    """Build matrices only from the explicitly supplied training triples."""
    ui, ii = {u: i for i, u in enumerate(users)}, {x: i for i, x in enumerate(items)}
    observed = np.zeros((len(users), len(items)), dtype=bool)
    signed = np.zeros(observed.shape, dtype=float)
    for user, item, rating in train_ratings:
        if user not in ui or item not in ii or ii[item] == 0:
            raise ValueError('Unknown user/item or padding in training ratings')
        row, col = ui[user], ii[item]
        if observed[row, col] or not np.isfinite(rating) or not 1 <= rating <= 5:
            raise ValueError('Duplicate training pair or invalid rating')
        observed[row, col] = True
        signed[row, col] = 1 if rating >= 4 else -1 if rating <= 2 else 0
    return signed, observed


def fit_geometry(signed, genres, dimensions=32, seed=2026):
    """Fixed-dimensional signed-SVD plus genre geometry; uses no held-out labels.

    Zero-filled missing entries are an explicit approximation, not dislikes.
    Genres and latent factors receive equal block norm before concatenation.
    """
    if dimensions < 1 or signed.shape[1] != len(genres):
        raise ValueError('Invalid geometry dimensions')
    k = min(dimensions, min(signed.shape) - 1)
    if k < 1 or not np.any(signed):
        raise ValueError('Need nonempty signed training evidence')
    _, singular, vt = svds(csr_matrix(signed), k=k, rng=np.random.default_rng(seed))
    order = np.argsort(-singular)
    latent = vt[order].T * np.sqrt(singular[order])
    # Remove SVD's arbitrary sign, for stable explanations and random features.
    for column in range(latent.shape[1]):
        pivot = np.argmax(np.abs(latent[:, column]))
        if latent[pivot, column] < 0:
            latent[:, column] *= -1
    geometry = unit_rows(np.concatenate([unit_rows(latent), unit_rows(genres)], axis=1))
    geometry[0] = 0
    return geometry


def matched_pairs(signed, geometry, genres):
    """For each like, choose nearest dislike with at least one shared genre.

    Positive cosine and a nonzero oriented difference are required. A dislike
    can support several likes; counts must therefore include distinct counterparts.
    Stable item order resolves ties without using held-out outcomes.
    """
    result = []
    for row in signed:
        likes, dislikes = np.flatnonzero(row > 0), np.flatnonzero(row < 0)
        pairs = []
        if len(likes) and len(dislikes):
            affinity = geometry[likes] @ geometry[dislikes].T
            overlap = genres[likes] @ genres[dislikes].T > 0
            for index, positive in enumerate(likes):
                eligible = np.flatnonzero(overlap[index] & (affinity[index] > 0))
                if not len(eligible):
                    continue
                negative = dislikes[eligible[np.argmax(affinity[index, eligible])]]
                if np.linalg.norm(geometry[positive] - geometry[negative]) > 1e-12:
                    pairs.append((int(positive), int(negative)))
        result.append(pairs)
    return result


def pair_features(pairs, geometry, projection, phase, mode):
    """Mean random Fourier features approximate an RBF relation-distribution kernel.

    The same frozen random map is used in all relation ablations. 'unordered'
    uses p+d so switching preference direction cannot change its representation.
    'anchor' uses the same liked anchors, without disliked counterparts.
    'first_moment' removes the distributional mechanism before the nonlinear map.
    """
    if mode not in ('oriented', 'unordered', 'anchor', 'first_moment'):
        raise ValueError('Unknown relation representation')
    if not pairs:
        return np.zeros(len(phase))
    positive, negative = np.asarray(pairs).T
    if mode == 'unordered':
        relations = geometry[positive] + geometry[negative]
    elif mode == 'anchor':
        relations = geometry[positive]
    else:
        relations = geometry[positive] - geometry[negative]
    relations = unit_rows(relations)
    if mode == 'first_moment':
        relations = unit_rows(relations.mean(axis=0, keepdims=True))
    return (np.sqrt(2 / len(phase)) * np.cos(relations @ projection + phase)).mean(axis=0)


def shuffled_partners(pairs, signed, seed):
    """Preserve each user's anchors/count and shuffle observed dislike assignments.

    Sampling is from that user's full explicit-dislike history, never other users
    or missing ratings. This ablates similarity matching, not rating access.
    """
    rng = np.random.default_rng(seed)
    result = []
    for row, user_pairs in zip(signed, pairs):
        negatives = np.flatnonzero(row < 0)
        result.append([(p, int(rng.choice(negatives))) for p, _ in user_pairs])
    return result


def permuted_partners(pairs, seed):
    """Exact pairing ablation: preserve each user's anchor and counterpart multisets."""
    rng = np.random.default_rng(seed)
    result = []
    for user_pairs in pairs:
        if not user_pairs:
            result.append([])
            continue
        anchors,counterparts = np.asarray(user_pairs).T
        result.append([(int(p),int(d)) for p,d in zip(anchors,rng.permutation(counterparts))])
    return result


def cosine_similarity(features):
    normed = unit_rows(features)
    result = np.maximum(normed @ normed.T, 0)
    np.fill_diagonal(result, 0)
    return result


@dataclass
class ContrastRepresentations:
    similarities: dict
    pairs: list
    geometry: np.ndarray
    fallback: np.ndarray
    shared_pair_mask: np.ndarray


def representations(signed, genres, dimensions=32, features=128, seed=2026):
    geometry = fit_geometry(signed, genres, dimensions, seed)
    pairs = matched_pairs(signed, geometry, genres)
    positive, negative = (signed > 0).astype(float), (signed < 0).astype(float)
    like_mean = positive @ geometry / np.maximum(positive.sum(axis=1, keepdims=True), 1)
    dislike_mean = negative @ geometry / np.maximum(negative.sum(axis=1, keepdims=True), 1)
    # Same signed ratings and genre/SVD geometry, without pairing information.
    generic = cosine_similarity(np.concatenate([like_mean, dislike_mean], axis=1))
    similarities = {
        'positive_knn': cosine_similarity(positive),
        'signed_knn': cosine_similarity(signed),
        'signed_geometry': generic,
    }
    rng = np.random.default_rng(seed)
    projection = rng.normal(0, np.sqrt(2), size=(geometry.shape[1], features))
    phase = rng.uniform(0, 2*np.pi, size=features)
    randomized = shuffled_partners(pairs, signed, seed + 991)
    fallback = np.asarray([not p for p in pairs])
    for name, mode, selected_pairs in [
        ('contrast_transfer', 'oriented', pairs),
        ('unordered_pairs', 'unordered', pairs),
        ('anchor_only', 'anchor', pairs),
        ('first_moment', 'first_moment', pairs),
        ('random_partners', 'oriented', randomized),
    ]:
        encoded = np.asarray([pair_features(p, geometry, projection, phase, mode) for p in selected_pairs])
        similarity = cosine_similarity(encoded)
        # One documented fallback for every relation variant: generic signed
        # geometry. Unsupported users are not dropped from evaluation.
        similarity[fallback] = generic[fallback]
        similarities[name] = similarity
    owner = {}
    shared = np.zeros((len(pairs), len(pairs)), dtype=bool)
    for user, user_pairs in enumerate(pairs):
        for pair in set(user_pairs):
            for other in owner.get(pair, []):
                shared[user, other] = shared[other, user] = True
            owner.setdefault(pair, []).append(user)
    return ContrastRepresentations(similarities, pairs, geometry, fallback, shared)


def neighborhood_scores(similarity, signed, neighbors=40, dislike_weight=1.0,
                        forbidden_neighbors=None):
    """Weighted signed evidence, normalized by total neighbor mass.

    All variants share this decoder. Missing votes are zero evidence. Users
    with no positive neighbor mass get training-only positive popularity.
    """
    n = signed.shape[0]
    if similarity.shape != (n, n) or not 1 <= neighbors < n or dislike_weight < 0:
        raise ValueError('Invalid neighborhood settings')
    if not np.isfinite(similarity).all() or (similarity < 0).any():
        raise ValueError('Similarity must be finite and nonnegative')
    weights = similarity.copy()
    np.fill_diagonal(weights, 0)
    if forbidden_neighbors is not None:
        if forbidden_neighbors.shape != weights.shape:
            raise ValueError('Invalid forbidden-neighbor mask')
        weights[forbidden_neighbors] = 0
    selected = np.argsort(-weights, axis=1, kind='stable')[:, :neighbors]
    limited = np.zeros_like(weights)
    rows = np.arange(n)[:, None]
    limited[rows, selected] = weights[rows, selected]
    vote = (signed > 0).astype(float) - dislike_weight*(signed < 0)
    mass = limited.sum(axis=1, keepdims=True)
    scores = limited @ vote / np.maximum(mass, 1e-12)
    unsupported = mass[:, 0] <= 1e-12
    scores[unsupported] = (signed > 0).mean(axis=0)
    scores[:, 0] = 0
    return scores


def constrained_linear_scores(features, target, excluded, penalty):
    """Exact ridge reconstruction with one/two self-item coefficients constrained zero.

    Solves in user space, then applies the constrained-ridge Schur correction.
    This is EASE for features==target and one excluded feature per item. With
    separate likes/dislikes channels, BOTH features for target item are excluded.
    Missing entries remain zero evidence rather than explicit dislike labels.
    """
    x, y = np.asarray(features,dtype=float), np.asarray(target,dtype=float)
    excluded = np.asarray(excluded,dtype=int)
    if penalty <= 0 or x.shape[0] != y.shape[0] or excluded.shape[0] != y.shape[1] or excluded.shape[1] not in (1,2):
        raise ValueError('Invalid constrained ridge settings')
    if np.any(excluded < 0) or np.any(excluded >= x.shape[1]):
        raise ValueError('Excluded feature outside feature matrix')
    factor = cho_factor(x @ x.T + penalty*np.eye(len(x)))
    inverse_x = cho_solve(factor,x)
    inverse_y = cho_solve(factor,y)
    scores = (x @ x.T) @ inverse_y
    nitems = y.shape[1]
    constraints = excluded.shape[1]
    block = np.empty((nitems,constraints,constraints))
    bias = np.empty((nitems,constraints))
    for a in range(constraints):
        xa = x[:,excluded[:,a]]
        bias[:,a] = np.sum(xa * inverse_y,axis=0)
        for b in range(constraints):
            block[:,a,b] = ((1 if a==b else 0)-np.sum(xa*inverse_x[:,excluded[:,b]],axis=0))/penalty
    correction = np.linalg.solve(block,bias[:,:,None])[:,:,0]
    for a in range(constraints):
        scores -= inverse_x[:,excluded[:,a]] * correction[None,:,a]
    scores[:,0] = 0
    return scores


def linear_rating_scores(signed, penalty, mode):
    positive, negative = (signed>0).astype(float),(signed<0).astype(float)
    m = signed.shape[1]
    if mode == 'positive_ease':
        return constrained_linear_scores(positive,positive,np.arange(m)[:,None],penalty)
    if mode == 'signed_ease':
        return constrained_linear_scores(signed,signed,np.arange(m)[:,None],penalty)
    if mode == 'signed_channels':
        return constrained_linear_scores(np.concatenate([positive,negative],axis=1),positive,
                                         np.stack([np.arange(m),m+np.arange(m)],axis=1),penalty)
    raise ValueError('Unknown explicit-rating linear model')


def pair_candidate_scores(geometry, pairs, signed, mode, temperature=.1):
    """Candidate-specific relation gate preserves location of the liked anchor.

    K is a fixed RBF on train-fitted signed-SVD+genre geometry. Proposed gate:
    mean_(p,d) K(candidate,p) * sigmoid((K(candidate,p)-K(candidate,d))/T).
    Anchor and signed-kernel controls use identical matched evidence and K.
    Empty-pair fallback is all-liked-anchor K, or positive popularity if no likes.
    """
    if mode not in ('gate','anchor','signed') or temperature <= 0:
        raise ValueError('Invalid pair candidate mode')
    distance = np.maximum(np.sum(geometry**2,axis=1)[:,None]
                          + np.sum(geometry**2,axis=1)[None,:]-2*geometry@geometry.T,0)
    kernel = np.exp(-distance)
    scores = np.zeros(signed.shape)
    for user,user_pairs in enumerate(pairs):
        if not user_pairs:
            positive = np.flatnonzero(signed[user]>0)
            scores[user] = kernel[:,positive].mean(axis=1) if len(positive) else (signed>0).mean(axis=0)
            continue
        positive,negative = np.asarray(user_pairs).T
        kp,kd = kernel[:,positive],kernel[:,negative]
        if mode=='gate':
            scores[user] = (kp * expit((kp-kd)/temperature)).mean(axis=1)
        elif mode=='anchor':
            scores[user] = kp.mean(axis=1)
        else:
            scores[user] = (kp-kd).mean(axis=1)
    scores[:,0] = 0
    return scores
