"""Synthetic checks for refitted OLS derivatives and paired risk-direction branches."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import layout

import copy
import tempfile
from types import SimpleNamespace
import torch
from cifar_local import NaturalSquare
from cifar_risk_direction import (probe, risk_gradient, clean_projection, training_gradient,
                                  choose_branches, run_repeat, summarize, write_json)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(41)
    # A direct feature-space check separates total refitted derivative from the
    # incorrect derivative that freezes the decoder learned at the source state.
    labels = torch.arange(80) % 10
    train, validation = torch.arange(50), torch.arange(50, 80)
    z = torch.randn(80, 3, dtype=torch.float64, requires_grad=True)
    direction = torch.randn_like(z)
    risk, metrics = probe(z, labels, train, validation)
    dz, = torch.autograd.grad(risk, z)
    epsilon = 1e-5
    plus = probe(z.detach()+epsilon*direction, labels, train, validation)[0]
    minus = probe(z.detach()-epsilon*direction, labels, train, validation)[0]
    torch.testing.assert_close((plus-minus)/(2*epsilon), (dz*direction).sum(), rtol=1e-7, atol=1e-10)
    design = torch.cat((torch.ones(80, 1, dtype=z.dtype), z.detach()), 1)
    targets = torch.nn.functional.one_hot(labels, 10).double()
    coef = torch.linalg.lstsq(design[train], targets[train], driver='gelsd').solution
    expected = (design[validation]@coef-targets[validation]).square().mean()
    torch.testing.assert_close(risk.detach(), expected, rtol=1e-12, atol=1e-12)
    frozen_risk = ((torch.cat((torch.ones(80, 1, dtype=z.dtype), z), 1)@coef)[validation]-targets[validation]).square().mean()
    frozen_dz, = torch.autograd.grad(frozen_risk, z)
    assert not torch.allclose(dz, frozen_dz, rtol=1e-3, atol=1e-8)
    try:
        probe(torch.ones(80, 3), labels, train, validation)
        raise AssertionError('Rank deficiency was not detected')
    except ValueError:
        pass

    data = torch.randn(80, 1, 32, 32)
    model = NaturalSquare(5, 3).double()
    with torch.no_grad():
        model.kernel.normal_(0, .04)
        model.weight.normal_(0, .15)
    gradient, initial = risk_gradient(model, data, labels, train, validation, 13)
    direct_risk = probe(model(data.double()), labels, train, validation)[0]
    direct = torch.autograd.grad(direct_risk, (model.kernel, model.weight))
    for a, b in zip(gradient, direct):
        torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-10)
    for batch in [11, 40]:
        torch.testing.assert_close(clean_projection(model, data, batch), model(data.double()).detach(), rtol=1e-12, atol=1e-12)
    g1, loss1 = training_gradient(model, data, train, torch.Generator().manual_seed(90), 12, 2)
    g2, loss2 = training_gradient(copy.deepcopy(model), data, train, torch.Generator().manual_seed(90), 12, 2)
    assert loss1 == loss2
    for a, b in zip(g1, g2):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    checks_root = (layout.RESULTS / 'checks').resolve()
    with tempfile.TemporaryDirectory(prefix='risk_direction_', dir=checks_root) as temporary:
        assert Path(temporary).resolve().is_relative_to(checks_root)
        args = SimpleNamespace(output=Path(temporary), lr=.0002, future_seed=941000,
                               batch_size=12, gradient_batches=2, feature_batch=16,
                               fd_scales=[.1, .05], fd_atol=1e-10, fd_rtol=.02,
                               signal_floor=1e-10, ratio_bounds=[.001, 1000.], horizons=[1, 2])
        threshold, reason, branches = choose_branches(-2., 1., True, args)
        assert threshold == 2. and reason == 'eligible'
        assert [b['ratio'] for b in branches[-2:]] == [1., 4.]
        assert choose_branches(2., 1., True, args)[0] is None
        assert len(choose_branches(-2., 1., False, args)[2]) == 5
        assert len(choose_branches(-1e8, 1., True, args)[2]) == 5
        assert len(choose_branches(-1e-12, 1e-12, True, args)[2]) == 5
        original = {k: v.clone() for k, v in model.state_dict().items()}
        result = run_repeat(args, 0, 0, 0, model, {'common_mobility': 1.}, data, labels,
                            train, validation, gradient, initial)
        assert all(row['passed'] for row in result['prediction']['finite_difference'])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        # Independent one-step reconstruction checks the actual rates, sign and
        # paired first gradient, rather than just checking output structure.
        baseline = copy.deepcopy(model)
        g, _ = training_gradient(baseline, data, train, torch.Generator().manual_seed(args.future_seed), 12, 2)
        with torch.no_grad():
            baseline.kernel.sub_(args.lr*g[0])
            baseline.weight.sub_(args.lr*g[1])
        observed = probe(clean_projection(baseline, data, 16), labels, train, validation)[1]['validation']['onehot_mse']
        row = next(o for o in result['outcomes'] if o['name'] == 'baseline' and o['step'] == 1)
        assert abs(observed-row['metrics']['validation']['onehot_mse']) < 1e-13
        assert row['delta_minus_baseline'] == 0.
        write_json(args.output/'protocol.json', {'arguments': {'seeds': [0, 1], 'source_steps': [0], 'repeats': 1}})
        summary = summarize(args.output)
        assert summary['completed'] == 1 and summary['expected'] == 2 and not summary['complete']
        assert all(c['seed_sample_std'] is None for c in summary['cells'])
    print('PASS: refitted derivative, OLS equivalence, chunk adjoint, finite differences, paired SGD, threshold cases, partial summary; synthetic only')


if __name__ == '__main__':
    main()
