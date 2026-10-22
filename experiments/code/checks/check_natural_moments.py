"""Small synthetic checks of fixed moment tasks, OLS, motion bounds and KNN.

No public dataset is loaded, downloaded or trained here.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout import RESULTS

import torch
from torch.nn import functional as F

from natural_moment_eval import (fixed_moment_dictionary, fixed_moment_filters,
                                 fit_readouts, geometry_bound, knn_metrics,
                                 directional_readout_slopes)
from summarize_natural_moments import trajectory_event


def main():
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(20261022)
    assert trajectory_event([(0, .5), (1, .4), (2, .6)], .01)['full_rebound']
    assert not trajectory_event([(0, .5), (1, .4), (2, .45)], .01)['full_rebound']
    assert not trajectory_event([(0, .5), (1, .6), (2, .4)], .01)['full_rebound']
    assert not trajectory_event([(0, .5), (1, .4995), (2, .5005)], .001)['full_rebound']
    images = torch.randn(4, 1, 14, 14, dtype=torch.float64, generator=rng)
    names, values = fixed_moment_dictionary(images, batch_size=2)
    assert len(names) == 11 and values.shape == (4, 11)
    _, filters = fixed_moment_filters()
    patches = F.unfold(images, 11).transpose(1, 2)
    patch_moments = patches.transpose(1, 2) @ patches / patches.shape[1]
    responses = torch.einsum("ci,nij,cj->nc", filters[:, 0].flatten(1), patch_moments,
                              filters[:, 0].flatten(1))
    torch.testing.assert_close(values[:, 0], responses[:, 0])
    torch.testing.assert_close(values[:, 1], patch_moments.diagonal(dim1=1, dim2=2).mean(1))
    torch.testing.assert_close(values[:, 2], values[:, 1] - values[:, 0])
    torch.testing.assert_close(values[:, 3:], responses[:, 1:].reshape(4, 8, 2).sum(-1))
    torch.testing.assert_close(fixed_moment_dictionary(2 * images)[1], 4 * values)
    shifted = fixed_moment_dictionary(images + 2.3)[1]
    torch.testing.assert_close(shifted[:, 2:], values[:, 2:], atol=2e-13, rtol=2e-13)
    assert float(filters[1:].mean((-2, -1)).abs().max()) < 1e-16

    n, channels, outputs, classes = 100, 7, 3, 4
    features = torch.randn(n, channels, dtype=torch.float64, generator=rng)
    head = torch.randn(channels, outputs, dtype=torch.float64, generator=rng)
    targets = torch.randn(n, 5, dtype=torch.float64, generator=rng)
    labels = torch.arange(n) % classes
    train, evaluation = torch.arange(75), torch.arange(75, n)
    result = fit_readouts(features, features @ head, targets, labels, train, evaluation, classes)
    assert all(a <= b + 1e-12 for a, b in zip(result['encoder']['moment_train_mse'],
                                             result['projection']['moment_train_mse']))
    assert result['encoder']['class_train_onehot_mse'] <= result['projection']['class_train_onehot_mse'] + 1e-12
    mixed = torch.randn(channels, channels, dtype=torch.float64, generator=rng) + 3 * torch.eye(channels)
    remixed = fit_readouts(features @ mixed + 1.7, features @ head,
                          targets, labels, train, evaluation, classes)
    torch.testing.assert_close(torch.tensor(remixed['encoder']['moment_evaluation_mse']),
                               torch.tensor(result['encoder']['moment_evaluation_mse']))
    changed_labels = labels.clone()
    changed_labels[evaluation] = (changed_labels[evaluation] + 1) % classes
    changed_targets = targets.clone()
    changed_targets[evaluation] *= 100
    changed = fit_readouts(features, features @ head, changed_targets, changed_labels,
                          train, evaluation, classes)
    assert changed['target_mean'] == result['target_mean']
    assert changed['target_scale'] == result['target_scale']
    for name in ('encoder', 'projection'):
        assert changed[name]['moment_train_mse'] == result[name]['moment_train_mse']
        assert changed[name]['class_train_onehot_mse'] == result[name]['class_train_onehot_mse']
        assert changed[name]['leading_component_moments']['train_r2'] == result[name]['leading_component_moments']['train_r2']
        assert set(result[name]['rank_threshold_checks']) == {'1e-12', '1e-10', '1e-08'}
    # PC1 is computed from unstandardized training features. A task exactly on
    # its uniquely leading direction is read out completely on both splits.
    pc_raw = torch.randn(n, 2, dtype=torch.float64, generator=rng)
    pc_basis, _ = torch.linalg.qr(pc_raw[train] - pc_raw[train].mean(0))
    pc_features = pc_raw.clone()
    pc_features[train] = pc_basis
    pc_features[:, 0] *= 20
    pc_targets = pc_features[:, :1].clone()
    pc_result = fit_readouts(pc_features, pc_features[:, :1], pc_targets, labels, train, evaluation, classes)
    assert abs(pc_result['encoder']['leading_component_moments']['train_r2'][0] - 1) < 1e-12
    assert pc_result['encoder']['leading_component_moments']['evaluation_normalized_mse'][0] < 1e-24
    feature_direction = torch.randn(features.shape, dtype=torch.float64, generator=rng)
    projection_direction = torch.randn(n, outputs, dtype=torch.float64, generator=rng)
    slopes = directional_readout_slopes(features, features @ head, feature_direction,
                                        projection_direction, targets, train)
    epsilon = 1e-5
    plus = fit_readouts(features + epsilon * feature_direction, features @ head + epsilon * projection_direction,
                        targets, labels, train, evaluation, classes)
    minus = fit_readouts(features - epsilon * feature_direction, features @ head - epsilon * projection_direction,
                         targets, labels, train, evaluation, classes)
    for name in ('encoder', 'projection'):
        finite_difference = (torch.tensor(plus[name]['moment_train_mse'], dtype=torch.float64)
                             - torch.tensor(minus[name]['moment_train_mse'], dtype=torch.float64)) / (2 * epsilon)
        torch.testing.assert_close(torch.tensor(slopes[name], dtype=torch.float64), finite_difference,
                                   rtol=1e-7, atol=1e-9)

    # Analytic rank-one rotation: R(theta)=sin(psi-theta)^2.
    raw = torch.randn(80, 2, dtype=torch.float64, generator=rng)
    basis, _ = torch.linalg.qr(raw - raw.mean(0))
    theta, psi = .3, .7
    start = basis[:, :1]
    current = math.cos(theta) * start + math.sin(theta) * basis[:, 1:2]
    task = math.sqrt(len(raw)) * (math.cos(psi) * start + math.sin(psi) * basis[:, 1:2])
    bound = geometry_bound(start, current, task, torch.arange(len(raw)))
    assert abs(bound['distance_frobenius_over_sqrt2'] - math.sin(theta)) < 1e-12
    assert abs(bound['tasks'][0]['a0'] - math.sin(psi) * math.cos(psi)) < 1e-12
    assert abs(bound['tasks'][0]['initial_risk'] - math.sin(psi)**2) < 1e-12
    assert abs(bound['tasks'][0]['current_risk'] - math.sin(psi - theta)**2) < 1e-12
    assert abs(bound['tasks'][0]['delta_risk']) <= bound['tasks'][0]['bound'] + 1e-12
    multi = geometry_bound(features, features + .2 * torch.randn(features.shape, dtype=torch.float64, generator=rng),
                           targets, train)
    assert abs(multi['raw_aggregate']['delta_risk']) <= multi['raw_aggregate']['bound'] + 1e-12
    rank_change = geometry_bound(start, torch.cat((start, basis[:, 1:2]), 1), task, torch.arange(len(raw)))
    assert not rank_change['equal_rank'] and rank_change['tasks'][0]['bound'] is None

    nearest = knn_metrics(features, labels, train, evaluation, classes, k=3, query_batch=4)
    distances = torch.cdist(features[evaluation], features[train])
    neighbours = distances.topk(3, largest=False).indices
    votes = F.one_hot(labels[train][neighbours], classes).sum(1)
    brute_accuracy = float((votes.argmax(1) == labels[evaluation]).double().mean())
    assert nearest['raw']['accuracy'] == brute_accuracy
    remixed_nearest = knn_metrics(features @ mixed, labels, train, evaluation, classes, k=3, query_batch=7)
    assert nearest['whitened']['accuracy'] == remixed_nearest['whitened']['accuracy']
    report = dict(status='passed', data='synthetic only',
                  checks=['moment tasks equal explicit patch-moment linear functionals',
                          'contrast and band tasks invariant to constant intensity shift',
                          'affine OLS inclusion and coordinate invariance',
                          'evaluation targets and labels cannot affect training fits',
                          'unstandardized training PC1 fully predicts its leading moment task',
                          'analytic OLS directional slopes agree with central differences',
                          'analytic rotation risk and Frobenius motion bound',
                          'aggregate bound and rank-change handling',
                          'blocked KNN agrees with brute force; whitened KNN affine invariance'])
    output = RESULTS / 'checks' / 'natural_moment_checks.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
