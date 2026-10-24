"""Fixed image-moment tasks and affine readout diagnostics.

These routines never select tasks from a learned trajectory. Risks are finite-
sample affine OLS errors, not exact natural-image population risks. All fits,
scales and whitening transforms use training rows only.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def _matrix(value):
    value = value.detach().cpu().double()
    if value.ndim != 2 or not bool(value.isfinite().all()):
        raise ValueError("Expected a finite two-dimensional matrix")
    return value


def _indices(value):
    return torch.as_tensor(value, dtype=torch.long, device="cpu")


def fixed_moment_filters(dtype=torch.float64, device="cpu"):
    """11x11 filters: DC, then four orientations at 1 and 3 cycles/11 px.

    A direction names the wave vector, rather than the stripe orientation.
    Each non-DC pair is centered, orthogonalized, and normalized to unit norm.
    The DC filter averages each patch. Pair energies sum the two squared
    responses, making the task independent of the sine/cosine phase basis.
    """
    size = 11
    yy, xx = torch.meshgrid(torch.arange(size, dtype=dtype, device=device),
                            torch.arange(size, dtype=dtype, device=device), indexing="ij")
    filters = [torch.ones_like(xx) / size**2]
    names = []
    for direction, kx, ky in (("x", 1, 0), ("y", 0, 1),
                              ("diagonal", 1, 1), ("antidiagonal", 1, -1)):
        for band, radius in (("low", 1), ("high", 3)):
            phase = 2 * math.pi * radius * (kx * xx + ky * yy) / size
            cosine, sine = phase.cos(), phase.sin()
            cosine = cosine - cosine.mean()
            cosine = cosine / cosine.norm()
            sine = sine - sine.mean()
            sine = sine - (sine * cosine).sum() * cosine
            sine = sine / sine.norm()
            filters.extend((cosine, sine))
            names.append(f"{direction}_{band}_energy")
    return names, torch.stack(filters)[:, None]


@torch.no_grad()
def fixed_moment_dictionary(images, batch_size=128):
    """Return eleven fixed tasks as (names, N x 11 CPU float64 values).

    Every task is a linear functional of the same mean valid 11x11 patch
    second moment used by the square CNN. ``patch_dc_energy`` is the mean
    squared patch mean; contrast is patch mean-square minus this DC term.
    Frequencies are fixed at 1 and 3 cycles per 11 pixels for each wave-vector
    component. Images use the caller's globally standardized intensity units.
    """
    if images.ndim != 4 or images.shape[1] != 1 or min(images.shape[-2:]) < 11:
        raise ValueError("Expected N x 1 grayscale images with size at least 11")
    if len(images) < 1 or batch_size < 1 or not images.is_floating_point():
        raise ValueError("Expected nonempty floating-point images and positive batch size")
    band_names, filters = fixed_moment_filters(images.dtype, images.device)
    rows = []
    for first in range(0, len(images), batch_size):
        batch = images[first:first + batch_size]
        if not bool(batch.isfinite().all()):
            raise ValueError("Nonfinite image values")
        responses = F.conv2d(batch, filters).square().mean((-2, -1)).double()
        dc = responses[:, :1]
        energy = F.avg_pool2d(batch.square(), 11, stride=1).mean((-2, -1)).double()
        bands = responses[:, 1:].reshape(len(batch), 8, 2).sum(-1)
        rows.append(torch.cat((dc, energy, (energy - dc).clamp_min(0), bands), 1).cpu())
    return ["patch_dc_energy", "patch_mean_square", "patch_contrast", *band_names], torch.cat(rows)


def _standardized_targets(targets, train):
    mean = targets[train].mean(0)
    scale = (targets[train] - mean).square().mean(0).sqrt()
    # Exactly constant targets are kept constant, with zero normalized risk.
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return (targets - mean) / scale, mean, scale


def _feature_design(features, train):
    mean = features[train].mean(0)
    scale = (features[train] - mean).square().mean(0).sqrt()
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    standardized = (features - mean) / scale
    return torch.cat((torch.ones(len(features), 1, dtype=torch.float64), standardized), 1)


def _spectrum(features, return_leading=False):
    centered = features - features.mean(0)
    eigenvalues, vectors = torch.linalg.eigh(centered.T @ centered / len(centered))
    values = eigenvalues.clamp_min(0).flip(0)
    total = float(values.sum())
    shares = values / total if total > 0 else values
    result = dict(covariance_eigenvalues=values.tolist(), total_variance=total,
                  pc1_variance_share=float(shares[0]) if len(shares) else 0.,
                  participation_ratio=float(1 / shares.square().sum()) if total > 0 else 0.)
    leading = vectors[:, -1] if len(values) else features.new_zeros((0,))
    return (result, leading) if return_leading else result


@torch.no_grad()
def fit_readouts(features, projection, targets, labels, train, evaluation, classes):
    """Train-fitted unregularized affine OLS on both representations.

    Moment errors use per-task training-variance units. Class errors average
    across one-hot coordinates. The evaluation split never influences fitting.
    ``rank`` counts centered feature directions and excludes the intercept.
    """
    input_dtypes = dict(encoder=str(features.dtype), projection=str(projection.dtype))
    features, projection, targets = map(_matrix, (features, projection, targets))
    train, evaluation = _indices(train), _indices(evaluation)
    labels = _indices(labels)
    if not (len(features) == len(projection) == len(targets) == len(labels)):
        raise ValueError("Features, targets and labels must have the same rows")
    if classes < 2 or not len(train) or not len(evaluation) or int(labels.min()) < 0 or int(labels.max()) >= classes:
        raise ValueError("Invalid split or class labels")
    standardized, mean, scale = _standardized_targets(targets, train)
    onehot = F.one_hot(labels, classes).double()
    all_targets = torch.cat((standardized, onehot), 1)
    output = dict(target_mean=mean.tolist(), target_scale=scale.tolist())
    for name, matrix in (("encoder", features), ("projection", projection)):
        design = _feature_design(matrix, train)
        fit = torch.linalg.lstsq(design[train], all_targets[train], driver="gelsd")
        prediction = design @ fit.solution
        spectrum, leading = _spectrum(matrix[train], return_leading=True)
        pc1 = ((matrix - matrix[train].mean(0)) @ leading)[:, None]
        pc1_design = _feature_design(pc1, train)
        pc1_fit = torch.linalg.lstsq(pc1_design[train], standardized[train], driver="gelsd")
        pc1_prediction = pc1_design @ pc1_fit.solution
        pc1_train_mse = (pc1_prediction[train] - standardized[train]).square().mean(0)
        row = dict(rank=max(0, int(fit.rank) - 1), design_rank_with_intercept=int(fit.rank),
                   singular_values=fit.singular_values.tolist(),
                   input_dtype=input_dtypes[name], ols_dtype="torch.float64",
                   singular_value_ratio=float(fit.singular_values[-1] / fit.singular_values[0]),
                   smallest_retained_singular_value=float(fit.singular_values[int(fit.rank) - 1]),
                   training_feature_spectrum=spectrum,
                   leading_component_moments=dict(
                       source="PC1 of unstandardized centered clean training features",
                       train_r2=(1 - pc1_train_mse).tolist(),
                       evaluation_normalized_mse=(pc1_prediction[evaluation] - standardized[evaluation]).square().mean(0).tolist()))
        for split, index in (("train", train), ("evaluation", evaluation)):
            moment_prediction, class_prediction = prediction[index].split((targets.shape[1], classes), 1)
            row[f"moment_{split}_mse"] = (moment_prediction - standardized[index]).square().mean(0).tolist()
            row[f"class_{split}_onehot_mse"] = float((class_prediction - onehot[index]).square().mean())
            row[f"class_{split}_accuracy"] = float((class_prediction.argmax(1) == labels[index]).double().mean())
        # Alternative rank cutoffs expose sensitivity of small directions in the
        # supplied features; converting float32 features cannot restore lost bits.
        sensitivity = {}
        for cutoff in (1e-12, 1e-10, 1e-8):
            check = torch.linalg.lstsq(design[train], all_targets[train], driver="gelsd", rcond=cutoff)
            check_prediction = design @ check.solution
            item = dict(rank=max(0, int(check.rank) - 1), design_rank_with_intercept=int(check.rank))
            for split, index in (("train", train), ("evaluation", evaluation)):
                item[f"moment_{split}_mse"] = (check_prediction[index, :targets.shape[1]] - standardized[index]).square().mean(0).tolist()
                item[f"class_{split}_onehot_mse"] = float((check_prediction[index, targets.shape[1]:] - onehot[index]).square().mean())
            sensitivity[f"{cutoff:g}"] = item
        row['rank_threshold_checks'] = sensitivity
        output[name] = row
    return output


@torch.no_grad()
def directional_readout_slopes(features, projection, feature_direction,
                               projection_direction, targets, train):
    """Envelope-theorem derivative of per-task optimum training OLS risk.

    At a constant-rank point, dR=2 mean[(F decoder-Y)*(dF decoder)]. Feature
    centering/scaling is held fixed at the source, which gives the same affine
    OLS derivative because its fitted residual is orthogonal to that design.
    These are source-local task predictions, not finite-time rebound claims.
    """
    features, projection, feature_direction, projection_direction, targets = map(
        _matrix, (features, projection, feature_direction, projection_direction, targets))
    train = _indices(train)
    standardized, _, _ = _standardized_targets(targets, train)
    if features.shape != feature_direction.shape or projection.shape != projection_direction.shape:
        raise ValueError("A feature direction must have the source feature shape")
    if not (len(features) == len(projection) == len(targets)) or not len(train):
        raise ValueError("Invalid training rows or feature sizes")
    result = dict(rank={}, constant_rank_required=True)
    for name, matrix, direction in (("encoder", features, feature_direction),
                                     ("projection", projection, projection_direction)):
        source = matrix[train]
        scale = (source - source.mean(0)).square().mean(0).sqrt()
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        design = _feature_design(matrix, train)[train]
        fit = torch.linalg.lstsq(design, standardized[train], driver="gelsd")
        residual = design @ fit.solution - standardized[train]
        predicted_direction = (direction[train] / scale) @ fit.solution[1:]
        result[name] = (2 * (residual * predicted_direction).mean(0)).tolist()
        result['rank'][name] = dict(rank=max(0, int(fit.rank) - 1), columns=matrix.shape[1],
                                   full_column_rank=int(fit.rank) == design.shape[1])
    return result


def _sample_basis(features):
    centered = features - features.mean(0)
    scale = centered.square().mean(0).sqrt()
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    u, values, _ = torch.linalg.svd(centered / scale, full_matrices=False)
    threshold = max(centered.shape) * torch.finfo(torch.float64).eps * values.max() if len(values) else 0.
    keep = values > threshold
    return u[:, keep], values, float(threshold)


def signed_geometry(initial, current, targets, train):
    """Exact training-risk change: lost source component - gained residual + cross.

    Uses two thin sample-space bases and their small overlap matrix, never an
    N-by-N projector. This is a retrospective identity, not a source forecast.
    Test error of a train-fitted decoder has no such projector decomposition.
    """
    initial, current, targets = map(_matrix, (initial, current, targets))
    train = _indices(train)
    if not len(train) or not (len(initial) == len(current) == len(targets)):
        raise ValueError('Features and targets need matching rows and nonempty train')
    first, _, _ = _sample_basis(initial[train])
    later, _, _ = _sample_basis(current[train])
    standardized, _, scale = _standardized_targets(targets, train)
    y = standardized[train] / math.sqrt(len(train))
    overlap = later.T @ first
    a, b = first.T @ y, later.T @ y
    projected_source = overlap @ a
    projected_residual = b - projected_source
    source_power = a.square().sum(0)
    total = y.square().sum(0)
    initial_risk = total - source_power
    current_risk = total - b.square().sum(0)
    lost = source_power - projected_source.square().sum(0)
    gained = projected_residual.square().sum(0)
    cross = -2 * (projected_source * projected_residual).sum(0)
    delta = current_risk - initial_risk
    rank0, rank1 = first.shape[1], later.shape[1]
    d2 = max(0., float((rank0 + rank1) / 2 - overlap.square().sum()))
    rows = []
    for j in range(y.shape[1]):
        reconstructed = lost[j] - gained[j] + cross[j]
        rows.append(dict(initial_risk=float(initial_risk[j]), current_risk=float(current_risk[j]),
            delta_risk=float(delta[j]), lost_readable=float(lost[j]),
            gained_residual=float(gained[j]), signed_cross=float(cross[j]),
            reconstructed_delta=float(reconstructed), identity_error=float(reconstructed-delta[j]),
            a0=float((initial_risk[j].clamp_min(0)*source_power[j]).sqrt())))
    return dict(scope='exact empirical training-risk identity; retrospective, not a prediction or test bound',
                initial_rank=rank0, current_rank=rank1, equal_rank=rank0 == rank1,
                distance_frobenius_over_sqrt2=math.sqrt(d2), task_scale=scale.tolist(), tasks=rows)


def _aggregate_bound(y, first, current, d, equal_rank):
    """Y is centered and divided by sqrt(number of training observations)."""
    coefficient = first.T @ y
    residual = y - first @ coefficient
    current_residual = y - current @ (current.T @ y)
    # ||R C^T||_F^2 = tr((R^T R)(C^T C)), without an N-by-N matrix.
    a0 = float(((residual.T @ residual) * (coefficient.T @ coefficient)).sum().clamp_min(0).sqrt())
    variance = float(y.square().sum())
    initial_risk = float(residual.square().sum())
    current_risk = float(current_residual.square().sum())
    osc = float(torch.linalg.eigvalsh(y.T @ y).max().clamp_min(0)) if y.shape[1] else 0.
    delta = current_risk - initial_risk
    bound = 2 * a0 * d + osc * d**2 if equal_rank else None
    return dict(target_total_variance=variance, initial_risk=initial_risk,
                current_risk=current_risk, delta_risk=delta, a0=a0, osc=osc,
                bound=bound, bound_to_abs_change=bound / abs(delta) if bound is not None and abs(delta) > 1e-15 else None,
                bound_below_total_variance=bool(bound < variance) if bound is not None else None)


@torch.no_grad()
def geometry_bound(initial, current, targets, train):
    """Exact finite-training-sample function-space motion and risk bounds.

    Implements the paper's d=||P1-P0||_F/sqrt(2), a0=||Q0 T P0||_F.
    Operator displacement is included as a separate diagnostic. The theorem
    requires equal ranks; its bound is null if ranks differ. No N-by-N matrix
    or full patch moment dictionary is constructed. For class labels this
    sample-space extension can be more conservative than projecting labels
    into the complete moment dictionary first, and is not a population claim.
    """
    initial, current, targets = map(_matrix, (initial, current, targets))
    train = _indices(train)
    if len(initial) != len(current) or len(initial) != len(targets) or not len(train):
        raise ValueError("Invalid training rows or matrix sizes")
    first, first_values, first_threshold = _sample_basis(initial[train])
    later, later_values, later_threshold = _sample_basis(current[train])
    rank0, rank1 = first.shape[1], later.shape[1]
    overlap = first.T @ later
    # Residual bases avoid cancellation in rank - ||overlap||_F^2 for small motion.
    first_residual = first - later @ overlap.T
    later_residual = later - first @ overlap
    d = math.sqrt(float((first_residual.square().sum() + later_residual.square().sum()) / 2))
    equal_rank = rank0 == rank1
    if not equal_rank:
        distance_op = 1.
    elif not rank0:
        distance_op = 0.
    else:
        distance_op = math.sqrt(float(torch.linalg.eigvalsh(first_residual.T @ first_residual).max().clamp_min(0)))
    angle = math.degrees(math.asin(min(1., distance_op)))
    standardized, _, scale = _standardized_targets(targets, train)
    y = standardized[train] / math.sqrt(len(train))
    tasks = [_aggregate_bound(y[:, j:j + 1], first, later, d, equal_rank) for j in range(y.shape[1])]
    raw = targets[train] - targets[train].mean(0)
    return dict(scope="finite-training-sample function-space bound; not a population bound",
                initial_rank=rank0, current_rank=rank1, equal_rank=equal_rank,
                distance_frobenius_over_sqrt2=d, distance_op=distance_op,
                largest_principal_angle_degrees=angle,
                initial_singular_values=first_values.tolist(), current_singular_values=later_values.tolist(),
                initial_svd_threshold=first_threshold, current_svd_threshold=later_threshold,
                bound_status="equal-rank theorem applies" if equal_rank else "rank changed; equal-rank bound not applied",
                task_scale=scale.tolist(), tasks=tasks,
                raw_aggregate=_aggregate_bound(raw / math.sqrt(len(train)), first, later, d, equal_rank))


@torch.no_grad()
def knn_metrics(features, labels, train, evaluation, classes, device="cpu", k=20, query_batch=256):
    """Exact Euclidean majority-vote KNN with three training-only transforms.

    All evaluation rows are queried; train rows are all candidates. Ties in
    the class vote select the lowest class index. This is a scale diagnostic,
    not an exact replication of a particular paper's KNN implementation.
    Whitening retains the numerical training covariance support; it uses no
    ridge or labels. Distances use float64 and a bounded query block.
    """
    features = _matrix(features)
    labels, train, evaluation = map(_indices, (labels, train, evaluation))
    if not len(train) or not len(evaluation) or k < 1 or query_batch < 1:
        raise ValueError("KNN needs nonempty splits and positive k/batch")
    if len(features) != len(labels) or int(labels.min()) < 0 or int(labels.max()) >= classes:
        raise ValueError("Invalid class labels")
    mean = features[train].mean(0)
    centered = features - mean
    std = centered[train].square().mean(0).sqrt()
    std = torch.where(std > 0, std, torch.ones_like(std))
    _, singular, vh = torch.linalg.svd(centered[train], full_matrices=False)
    keep = singular > torch.finfo(torch.float64).eps * max(centered[train].shape) * singular.max()
    # Whiten the raw covariance rather than standardized correlation matrix.
    whitening = vh[keep].T / (singular[keep] / math.sqrt(len(train)))
    transformed = {"raw": centered, "standardized": centered / std, "whitened": centered @ whitening}
    labels_device = labels[train].to(device)
    effective_k = min(k, len(train))
    output = {}
    for name, matrix in transformed.items():
        candidates = matrix[train].to(device)
        candidate_norm = candidates.square().sum(1)
        correct = 0
        for start in range(0, len(evaluation), query_batch):
            index = evaluation[start:start + query_batch]
            queries = matrix[index].to(device)
            distances = (queries.square().sum(1, keepdim=True) + candidate_norm[None] - 2 * queries @ candidates.T).clamp_min(0)
            nearest = distances.topk(effective_k, largest=False, sorted=False).indices
            votes = torch.zeros(len(index), classes, dtype=torch.long, device=device)
            votes.scatter_add_(1, labels_device[nearest], torch.ones_like(nearest))
            correct += int((votes.argmax(1).cpu() == labels[index]).sum())
        output[name] = dict(accuracy=correct / len(evaluation), k=effective_k,
                            query_count=len(evaluation), candidate_count=len(train),
                            dimensions=matrix.shape[1], metric="Euclidean majority vote")
    return output
