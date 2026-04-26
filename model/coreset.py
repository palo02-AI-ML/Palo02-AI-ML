import torch


def greedy_coreset_indices(
    features: torch.Tensor,
    ratio: float = 0.01,
    max_candidates: int = 50_000,
) -> torch.Tensor:
    """Greedy k-center coreset subsampling.

    Selects a diverse subset of patches that covers the feature space
    as uniformly as possible.

    Args:
        features: (N, D) feature matrix.
        ratio: fraction of N to keep.
        max_candidates: pre-sample to this size before greedy search for speed.

    Returns:
        Indices of selected points, shape (k,).
    """
    n = features.shape[0]
    k = max(1, int(n * ratio))

    if n > max_candidates:
        perm = torch.randperm(n, device=features.device)[:max_candidates]
        sub_indices = _k_center_greedy(features[perm], k)
        return perm[sub_indices]

    return _k_center_greedy(features, k)


def _k_center_greedy(features: torch.Tensor, k: int) -> torch.Tensor:
    n = features.shape[0]
    device = features.device

    if k >= n:
        return torch.arange(n, device=device)

    feat_f = features.float()
    first = int(torch.randint(0, n, (1,)).item())
    selected = [first]

    min_dists = torch.full((n,), float("inf"), device=device)
    _update_min_dists(feat_f, feat_f[first], min_dists)

    for _ in range(k - 1):
        next_idx = int(min_dists.argmax().item())
        selected.append(next_idx)
        _update_min_dists(feat_f, feat_f[next_idx], min_dists)

    return torch.tensor(selected, device=device, dtype=torch.long)


def _update_min_dists(
    features: torch.Tensor,
    center: torch.Tensor,
    min_dists: torch.Tensor,
) -> None:
    dists = torch.cdist(features, center.unsqueeze(0)).squeeze(1)
    torch.minimum(min_dists, dists, out=min_dists)
