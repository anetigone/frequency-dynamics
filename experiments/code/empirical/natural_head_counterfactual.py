"""Frozen-encoder class head counterfactuals on existing CIFAR checkpoints.

Default is a data-free plan. evaluate extracts clean features and fits only
closed-form supervised readouts; it never changes or trains the source model.
smoke uses synthetic features and images only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS
import natural_moment_experiment as natural


def runtime():
    global torch, F, NaturalSquare, feature_design
    natural.load_runtime()
    torch, F, NaturalSquare = natural.torch, natural.F, natural.NaturalSquare
    from natural_moment_eval import _feature_design
    feature_design = _feature_design


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    natural.write_json(Path(path), value)


def score(scores, labels, rows, classes):
    target = F.one_hot(labels[rows], classes).double()
    return dict(onehot_mse=float((scores[rows] - target).square().mean()),
                accuracy=float((scores[rows].argmax(1) == labels[rows]).double().mean()),
                samples=len(rows))


def fit_affine(features, labels, train, test, classes):
    """Training-only affine unregularized OLS, with no held-out selection."""
    x = features.detach().cpu().double()
    if x.ndim != 2 or not bool(x.isfinite().all()):
        raise ValueError('Features must be a finite two-dimensional matrix')
    design = feature_design(x, train)
    targets = F.one_hot(labels[train], classes).double()
    fit = torch.linalg.lstsq(design[train], targets, driver='gelsd')
    scores = design @ fit.solution
    return dict(train=score(scores, labels, train, classes), test=score(scores, labels, test, classes),
                feature_rank=max(0, int(fit.rank) - 1), design_rank_with_intercept=int(fit.rank),
                singular_values=fit.singular_values.tolist(),
                singular_value_ratio=float(fit.singular_values[-1] / fit.singular_values[0]),
                input_dtype=str(features.dtype), fit_dtype='torch.float64')


def reduced_rank(features, labels, train, test, classes, outputs):
    """Global affine rank-K training least-squares solution via truncated SVD.

    Fitted values are U times the best rank-K approximation of U^T Y_centered.
    The returned head is linear on raw encoder features; the affine intercept
    is assigned to the final decoder, so the projection has exactly K outputs.
    """
    x = features.detach().cpu().double()
    mean = x[train].mean(0)
    centered = x[train] - mean
    scale = centered.square().mean(0).sqrt()
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    z = (x - mean) / scale
    y = F.one_hot(labels[train], classes).double()
    y_mean = y.mean(0)
    u, singular, vh = torch.linalg.svd(z[train], full_matrices=False)
    tolerance = torch.finfo(torch.float64).eps * max(z[train].shape) * singular.max()
    keep = singular > tolerance
    rank = int(keep.sum())
    coefficient = u[:, keep].T @ (y - y_mean)
    left, values, right = torch.linalg.svd(coefficient, full_matrices=False)
    kept = min(outputs, rank, len(values))
    projection = torch.zeros(x.shape[1], outputs, dtype=torch.float64)
    decoder = torch.zeros(outputs, classes, dtype=torch.float64)
    if kept:
        basis = (vh[keep].T / singular[keep]) @ left[:, :kept]
        projection[:, :kept] = basis / scale[:, None]
        decoder[:kept] = values[:kept, None] * right[:kept]
    bias = y_mean - mean @ projection @ decoder
    scores = x @ projection @ decoder + bias
    analytic_mse = float(((y-y_mean).square().sum()-values[:kept].square().sum()) / (len(train)*classes))
    return dict(head=projection.T, decoder=decoder, bias=bias,
                metrics=dict(train=score(scores, labels, train, classes), test=score(scores, labels, test, classes),
                    feature_rank=rank, retained_components=kept,
                    predictive_singular_values=values.tolist(), analytic_train_mse=analytic_mse,
                    interpretation='Global training-MSE optimum at rank K; not test-accuracy optimum'))


def analyze_features(features, heads, labels, train, test, classes, outputs):
    """All heads are multiplied outside the model; source weights stay intact."""
    epoch_keys = list(features)
    full, original, oracle, crossed, artifacts = {}, {}, {}, {}, {}
    for encoder_epoch, matrix in features.items():
        full[encoder_epoch] = fit_affine(matrix, labels, train, test, classes)
        crossed[encoder_epoch] = {}
        for head_epoch, head in heads.items():
            projected = matrix.double() @ head.double().T
            crossed[encoder_epoch][head_epoch] = fit_affine(projected, labels, train, test, classes)
        original[encoder_epoch] = crossed[encoder_epoch][encoder_epoch]
        optimal = reduced_rank(matrix, labels, train, test, classes, outputs)
        projected = matrix.double() @ optimal['head'].T
        refitted = fit_affine(projected, labels, train, test, classes)
        oracle[encoder_epoch] = dict(direct=optimal['metrics'], refitted_ols=refitted,
            direct_minus_refitted_train_mse=optimal['metrics']['train']['onehot_mse']-refitted['train']['onehot_mse'],
            direct_minus_refitted_test_mse=optimal['metrics']['test']['onehot_mse']-refitted['test']['onehot_mse'])
        artifacts[encoder_epoch] = {k:v for k,v in optimal.items() if k != 'metrics'}
    early, late = epoch_keys
    comparisons = dict(same_encoder={}, temporal={})
    for epoch in epoch_keys:
        comparisons['same_encoder'][epoch] = {}
        for split in ['train','test']:
            source = original[epoch][split]['onehot_mse']
            encoder = full[epoch][split]['onehot_mse']
            oracle_risk = oracle[epoch]['refitted_ols'][split]['onehot_mse']
            gap = source - encoder
            comparisons['same_encoder'][epoch][split] = dict(original_minus_full_gap=gap,
                oracle_minus_full_gap=oracle_risk-encoder,
                original_minus_oracle_recovered_gap=source-oracle_risk,
                fraction_gap_removed=(source-oracle_risk)/gap if gap>0 else None,
                early_head_minus_late_head_mse=crossed[epoch][early][split]['onehot_mse']-
                                                   crossed[epoch][late][split]['onehot_mse'])
    for split in ['train','test']:
        risk = lambda fe,he: crossed[fe][he][split]['onehot_mse']
        initial, final = risk(early,early), risk(late,late)
        # Path: change encoder holding early W, then change W holding late F.
        encoder_at_early_head = risk(late,early) - initial
        head_at_late_encoder = final - risk(late,early)
        # Reverse path is also retained; interaction shows path dependence.
        head_at_early_encoder = risk(early,late) - initial
        encoder_at_late_head = final - risk(early,late)
        comparisons['temporal'][split] = dict(total_original_change=final-initial,
            encoder_change_holding_early_head=encoder_at_early_head,
            head_change_holding_late_encoder=head_at_late_encoder,
            head_change_holding_early_encoder=head_at_early_encoder,
            encoder_change_holding_late_head=encoder_at_late_head,
            interaction=head_at_late_encoder-head_at_early_encoder,
            additive_residual=final-initial-encoder_at_early_head-head_at_late_encoder)
    return dict(full_encoder=full, original_head=original, oracle_head=oracle,
                cross_heads=crossed, comparisons=comparisons), artifacts


def suite_protocol(args, source):
    n, batch = source['split']['train_size'], source['arguments']['batch_size']
    return dict(source=str(args.source.resolve()), source_protocol=source,
        seeds=args.seeds, epochs=args.epochs, steps=[natural.epoch_step(e,n,batch) for e in args.epochs],
        outputs=args.rank, task='CIFAR-10 centered one-hot classes; affine unregularized least squares',
        fitting='All ordinary decoders and rank-K oracle heads fit official training labels only',
        evaluation='Official test is reused for fixed comparisons, never for head/rank/checkpoint selection',
        counterfactual='F_early/F_late x W_early/W_late, each with a newly train-fitted OLS decoder',
        interpretation='Supervised rank-K oracle diagnoses available compression, not an unsupervised algorithm')


def extract_states(args, seed, source_protocol, data):
    n, batch = source_protocol['split']['train_size'], source_protocol['arguments']['batch_size']
    features, heads, numerical = {}, {}, {}
    for epoch in args.epochs:
        step = natural.epoch_step(epoch,n,batch)
        path = args.source / f'seed_{seed}' / f'step_{step}.pt'
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['seed'] != seed or state['step'] != step:
            raise ValueError(f'Checkpoint seed/step differs: {path}')
        model = NaturalSquare().to(args.device)
        model.load_state_dict(state['model'])
        if any(not bool(p.isfinite().all()) for p in model.parameters()):
            raise FloatingPointError('Nonfinite source checkpoint')
        model.eval()
        key = f'{epoch:g}'
        features[key], _ = natural.extract(model,data,args.feature_batch)
        heads[key] = model.weight.detach().cpu().clone()
        unchanged = all(torch.equal(v.detach().cpu(),state['model'][name]) for name,v in model.state_dict().items())
        if not unchanged:
            raise AssertionError('Read-only feature extraction changed the source model')
        numerical[key] = dict(checkpoint=str(path.resolve()),step=step,source_model_unchanged=True,
                              feature_shape=list(features[key].shape),feature_dtype=str(features[key].dtype))
    return features,heads,numerical


def evaluate(args):
    source = read(args.source/'protocol.json')
    if (source['arguments']['dataset']!='cifar10' or source['split']['train_size']!=50000
            or source['split']['evaluation_size']!=10000):
        raise ValueError('Use the existing official 50k/10k CIFAR suite')
    args.dataset,args.image_size='cifar10',32
    data,labels,train,test,split = natural.load_images(args)
    if split != source['split']:
        raise ValueError('Current official data preprocessing differs from source protocol')
    protocol=suite_protocol(args,source)
    destination=args.output/'protocol.json'
    if destination.exists() and read(destination)!=protocol:
        raise ValueError('Keep this suite protocol or choose a separate output directory')
    write(destination,protocol)
    failures=[]
    for seed in args.seeds:
        directory=args.output/f'seed_{seed}'
        directory.mkdir(parents=True,exist_ok=True)
        if (directory/'result.json').exists():
            print(f'seed={seed}: existing result retained',flush=True)
            continue
        try:
            features,heads,numerical=extract_states(args,seed,source,data)
            result,artifacts=analyze_features(features,heads,labels,train,test,split['classes'],args.rank)
            result.update(seed=seed,state='complete',epochs=args.epochs,source_checkpoints=numerical)
            natural.save_tensor(directory/'oracle_heads.pt',artifacts)
            write(directory/'result.json',result)
            print(f'seed={seed}: all frozen-encoder/head comparisons complete',flush=True)
        except (OSError,RuntimeError,ValueError) as error:
            write(directory/'failure.json',dict(seed=seed,state='failed',error=str(error)))
            failures.append(seed)
    if failures:
        raise RuntimeError(f'Failed seeds retained: {failures}')


def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8-sig') as stream:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        writer=csv.DictWriter(stream,fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def summarize(args):
    protocol=read(args.output/'protocol.json')
    output=args.output/'summary'
    output.mkdir(parents=True,exist_ok=True)
    coverage,metrics,within,temporal=[],[],[],[]
    for seed in protocol['seeds']:
        directory=args.output/f'seed_{seed}'
        path=directory/'result.json'
        complete=path.exists()
        coverage.append(dict(seed=seed,state='complete' if complete else 'missing',failure=(directory/'failure.json').exists()))
        if not complete:continue
        result=read(path)
        for epoch,readout in result['full_encoder'].items():
            for kind,values in [('full_encoder',readout),('original_head',result['original_head'][epoch]),
                                ('oracle_rank'+str(protocol['outputs']),result['oracle_head'][epoch]['refitted_ols'])]:
                for split in ['train','test']:
                    metrics.append(dict(seed=seed,encoder_epoch=epoch,head_epoch=epoch,readout=kind,split=split,
                                        **values[split],feature_rank=values['feature_rank']))
        for fe,heads in result['cross_heads'].items():
            for he,values in heads.items():
                for split in ['train','test']:
                    metrics.append(dict(seed=seed,encoder_epoch=fe,head_epoch=he,readout='cross_head',split=split,
                                        **values[split],feature_rank=values['feature_rank']))
        for epoch,splits in result['comparisons']['same_encoder'].items():
            for split,values in splits.items():within.append(dict(seed=seed,encoder_epoch=epoch,split=split,**values))
        for split,values in result['comparisons']['temporal'].items():temporal.append(dict(seed=seed,split=split,**values))
    means=[]
    groups={}
    for row in metrics:
        key=tuple(row[k] for k in ['encoder_epoch','head_epoch','readout','split'])
        groups.setdefault(key,[]).append(row)
    for key,rows in groups.items():
        means.append(dict(zip(['encoder_epoch','head_epoch','readout','split'],key),
            observed_seeds=len(rows),planned_seeds=len(protocol['seeds']),
            onehot_mse=sum(r['onehot_mse'] for r in rows)/len(rows),
            accuracy=sum(r['accuracy'] for r in rows)/len(rows)))
    for name,rows in [('coverage',coverage),('metrics',metrics),('mean_metrics',means),
                       ('same_encoder_contrasts',within),('temporal_decomposition',temporal)]:
        write_csv(output/f'{name}.csv',rows)
    write(output/'summary.json',dict(protocol=protocol,coverage=coverage,means=means,
        same_encoder_contrasts=within,temporal_decomposition=temporal))
    lines=['# Frozen encoder / head counterfactuals','',
        'Train-only rank-K least-squares oracle, full encoder OLS, and original/temporal-crossed heads. '
        'Each crossed representation receives a separately refitted training decoder. '
        'The official test is not used to select heads, rank, or checkpoints.','',
        '| Encoder epoch | Readout / head epoch | Split | MSE mean | Accuracy mean (%) | Seeds |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for row in means:
        lines.append(f'| {row["encoder_epoch"]} | {row["readout"]} / {row["head_epoch"]} | {row["split"]} | '
            f'{row["onehot_mse"]:.9f} | {100*row["accuracy"]:.3f} | {row["observed_seeds"]}/{row["planned_seeds"]} |')
    lines+=['','The oracle establishes that a class-dependent compression is available at the same output budget. '
        'It does not show that unlabeled pretraining should find that head, and is optimal for training MSE only. '
        'Training full-encoder <= rank-K oracle <= original-head risks are inclusion comparisons; '
        'test risks and accuracies need not obey this order.','',
        'same_encoder_contrasts.csv isolates head choice at a fixed encoder. '
        'temporal_decomposition.csv gives both orders of encoder/head replacement and their interaction. '
        'It is a paired counterfactual calculation, not a unique causal attribution across all training dynamics. '
        'Missing and failed seeds remain explicit; independent repetition is seed.','']
    (output/'findings.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Summary: {output/"findings.md"}',flush=True)


def smoke(args):
    import tempfile
    generator=torch.Generator().manual_seed(20261023)
    n,d,classes,rank=180,12,10,8
    features={e:torch.randn(n,d,generator=generator,dtype=torch.float64) for e in ['3000','5000']}
    heads={e:torch.randn(rank,d,generator=generator,dtype=torch.float64) for e in features}
    originals={e:w.clone() for e,w in heads.items()}
    labels=torch.arange(n)%classes
    train,test=torch.arange(120),torch.arange(120,n)
    result,artifacts=analyze_features(features,heads,labels,train,test,classes,rank)
    for epoch in features:
        full=result['full_encoder'][epoch]['train']['onehot_mse']
        oracle=result['oracle_head'][epoch]['refitted_ols']['train']['onehot_mse']
        original=result['original_head'][epoch]['train']['onehot_mse']
        assert full<=oracle+1e-12 and oracle<=original+1e-12
        assert abs(result['oracle_head'][epoch]['direct_minus_refitted_train_mse'])<1e-12
        assert abs(result['oracle_head'][epoch]['direct_minus_refitted_test_mse'])<1e-12
        direct=reduced_rank(features[epoch],labels,train,test,classes,rank)
        assert abs(direct['metrics']['analytic_train_mse']-oracle)<1e-12
        for _ in range(4):
            candidate=torch.randn(rank,d,generator=generator,dtype=torch.float64)
            fitted=fit_affine(features[epoch]@candidate.T,labels,train,test,classes)
            assert oracle<=fitted['train']['onehot_mse']+1e-12
        changed=labels.clone()
        changed[test]=(changed[test]+3)%classes
        alternative=reduced_rank(features[epoch],changed,train,test,classes,rank)
        torch.testing.assert_close(direct['head'],alternative['head'],rtol=0,atol=0)
        assert direct['metrics']['train']==alternative['metrics']['train']
        scaled=fit_affine(features[epoch]@(7*heads[epoch]).T,labels,train,test,classes)
        assert abs(scaled['train']['onehot_mse']-original)<1e-12
        assert torch.equal(heads[epoch],originals[epoch])
    for split,value in result['comparisons']['temporal'].items():assert abs(value['additive_residual'])<1e-14
    parent=RESULTS/'checks'
    parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='head_counterfactual_smoke_',dir=parent) as temp:
        directory=Path(temp)
        source=directory/'source'
        source_protocol=dict(arguments=dict(batch_size=4),split=dict(train_size=20,classes=10))
        images=torch.randn(30,1,32,32,generator=generator)
        for epoch in [1.,2.]:
            step=natural.epoch_step(epoch,20,4)
            model=NaturalSquare()
            with torch.no_grad():
                model.kernel.copy_(torch.randn(model.kernel.shape,generator=generator)/11)
                model.weight.copy_(torch.randn(model.weight.shape,generator=generator)/8)
            path=source/'seed_0'/f'step_{step}.pt'
            path.parent.mkdir(parents=True,exist_ok=True)
            natural.save_tensor(path,dict(seed=0,step=step,model=model.state_dict()))
        local=SimpleNamespace(source=source,epochs=[1.,2.],device='cpu',feature_batch=8)
        extracted,loaded,numerical=extract_states(local,0,source_protocol,images)
        assert set(extracted)=={'1','2'} and all(v['source_model_unchanged'] for v in numerical.values())
        for epoch,head in loaded.items():
            cp=torch.load(source/'seed_0'/f'step_{natural.epoch_step(float(epoch),20,4)}.pt',weights_only=True)
            assert torch.equal(head,cp['model']['weight'])
        # Exercise complete per-seed JSON and summary without plotting/NumPy.
        args=SimpleNamespace(output=directory/'output')
        write(args.output/'protocol.json',dict(seeds=[0],outputs=rank))
        write(args.output/'seed_0/result.json',dict(seed=0,**result))
        summarize(args)
        assert (args.output/'summary/temporal_decomposition.csv').exists()
    report=dict(passed=True,data='synthetic only; no downloads or real-data training',
        checks=['rank-8 optimum and OLS inclusion','oracle direct scores equal refitted OLS',
                'held-out labels do not affect fitted head or training metrics',
                'source head scaling does not affect OLS risk','2x2 additive decomposition',
                'checkpoint identity and frozen feature extraction','JSON/CSV/Markdown output'])
    write(parent/'natural_head_counterfactual_smoke.json',report)
    print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',nargs='?',choices=['plan','evaluate','summarize','smoke'],default='plan')
    parser.add_argument('--source',type=Path,default=RESULTS/'empirical/natural_moments_cifar10')
    parser.add_argument('--data-root',type=Path,default=EXPERIMENTS.parent)
    parser.add_argument('--output',type=Path,default=RESULTS/'empirical/natural_head_counterfactual_cifar10')
    parser.add_argument('--seeds',type=int,nargs='+',default=[30,31,32])
    parser.add_argument('--epochs',type=float,nargs=2,default=[3000.,5000.])
    parser.add_argument('--rank',type=int,default=8)
    parser.add_argument('--feature-batch',type=int,default=256)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--threads',type=int,default=1)
    args=parser.parse_args()
    if len(args.seeds)!=len(set(args.seeds)) or not args.seeds:
        parser.error('Seeds must be distinct')
    if not all(math.isfinite(e) and e>0 for e in args.epochs) or not args.epochs[0]<args.epochs[1]:
        parser.error('Choose two finite positive, increasing checkpoint epochs')
    if min(args.rank,args.feature_batch,args.threads)<1:
        parser.error('Rank, feature batch and threads must be positive')
    if args.source.resolve()==args.output.resolve():
        parser.error('Keep the output separate from the source suite')
    if args.phase=='plan':
        print(json.dumps(dict(seeds=args.seeds,epochs=args.epochs,rank=args.rank,
            source=str(args.source),output=str(args.output),source_checkpoints=2*len(args.seeds),
            comparisons=f'Each frozen F: full OLS, original W+OLS, oracle rank{args.rank}+OLS; 2x2 F/W with fresh OLS',
            action='Data-free plan only; no source/data access, download or training'),indent=2))
        return
    if args.phase=='summarize':
        summarize(args)
        return
    runtime()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if args.phase=='smoke':smoke(args)
    else:evaluate(args)


if __name__=='__main__':
    main()
