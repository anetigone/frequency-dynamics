"""Paired CIFAR ordinary/pullback continuations; default is a data-free plan.
Restore the same source RNG in both arms. No h/W rescaling or extra training labels.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from layout import EXPERIMENTS,RESULTS
import natural_moment_experiment as base
ARMS=("sgd","pullback")


def runtime():
    global torch,F,NaturalSquare,augment,empirical_loss,make_audit,audit_model
    import torch
    from torch.nn import functional as F
    from cifar_local import NaturalSquare,augment
    from square_images import empirical_loss
    from cifar_checkpoint_branches import make_audit,audit_model
    base.load_runtime()


def pullback(kernel,gradient):
    """Inverse of J_C*J_C for C=hh^T in the attainable Frobenius tangent."""
    h,g=kernel.detach().flatten(1),gradient.detach().flatten(1)
    q=h.square().sum(1,keepdim=True)
    if not bool((q>0).all()) or not bool(q.isfinite().all()):
        raise FloatingPointError("Pullback undefined for a zero/nonfinite filter")
    return (g/(2*q)-h*(h*g).sum(1,keepdim=True)/(4*q.square())).reshape_as(gradient)


def model_from(state,device):
    k,w=state['kernel'],state['weight']
    if k.ndim!=4 or k.shape[1:]!=(1,11,11) or w.ndim!=2 or w.shape[1]!=len(k):
        raise ValueError("Incompatible square kernel/head shapes")
    if not bool(k.isfinite().all() and w.isfinite().all()):
        raise ValueError("Nonfinite source model")
    model=NaturalSquare(len(k),len(w)).to(device)
    model.load_state_dict(state,strict=True)
    return model


def step_for(epoch,n,batch):return base.epoch_step(epoch,n,batch)


def jobs(args):
    return [dict(seed=s,source_epoch=a,target_epoch=b) for s in args.seeds
            for a,b in zip(args.source_epochs,args.target_epochs)]


def source_directory(args,job):
    return args.output/f"seed_{job['seed']}_from_{job['source_epoch']:g}"


def source_payload(args,job,settings):
    step=step_for(job['source_epoch'],settings['split']['train_size'],settings['batch_size'])
    path=args.source/f"seed_{job['seed']}"/f"step_{step}.pt"
    value=torch.load(path,map_location='cpu',weights_only=True)
    if value['seed']!=job['seed'] or value['step']!=step:
        raise ValueError(f"Wrong source seed/step: {path}")
    if 'generator_state' not in value or 'calibration' not in value:
        raise ValueError("Source requires original training RNG and mobility calibration")
    return value,path


def rates(payload,settings):
    front=settings['lr']*2*payload['calibration']['common_mobility']/(1+settings['ratio'])
    return front,front*settings['ratio']


def estimate_trace(model,data,train,job,args):
    """Paired independent images/signs for both traces; future RNG is untouched."""
    gen=torch.Generator(device=data.device).manual_seed(947000+job['seed']*1000+int(job['source_epoch']*10))
    indices=train.to(data.device)
    original,conditioned=[],[]
    for _ in range(args.trace_batches):
        chosen=indices[torch.randint(len(indices),(args.calibration_size,),generator=gen,device=data.device)]
        z=model(data[chosen]);z=z-z.mean(0)
        for draw in range(args.trace_draws):
            signs=2*torch.randint(2,z.shape,generator=gen,device=data.device)-1
            grad=torch.autograd.grad((z*signs).sum()/math.sqrt(len(chosen)),model.kernel,
                                    retain_graph=draw+1<args.trace_draws)[0]
            moved=pullback(model.kernel,grad)
            original.append(float(grad.double().square().sum()))
            conditioned.append(float((grad.double()*moved.double()).sum()))
    first,second=sum(original)/len(original),sum(conditioned)/len(conditioned)
    if not all(math.isfinite(x) and x>0 for x in (first,second)):
        raise FloatingPointError("Invalid source output-mobility traces")
    alpha=first/second
    return dict(alpha=alpha,ordinary_trace=first,pullback_trace=second,matched_pullback_trace=alpha*second,
                trace_batches=args.trace_batches,draws_per_batch=args.trace_draws,calibration_size=args.calibration_size,
                selection='independent clean training images; centered clean-output VJP trace',
                interpretation='fixed alpha, original head rate, no parameter rescaling')


def source_directions(model,payload,data,train,job,args,settings,alpha):
    gen=torch.Generator(device=data.device).manual_seed(948000+job['seed']*1000+int(job['source_epoch']*10))
    indices=train.to(data.device)
    average=[torch.zeros_like(model.kernel),torch.zeros_like(model.weight)]
    for _ in range(args.local_batches):
        chosen=indices[torch.randint(len(indices),(settings['batch_size'],),generator=gen,device=data.device)]
        first,second=augment(data[chosen],gen),augment(data[chosen],gen)
        loss=empirical_loss(model(first),model(second))
        gradients=torch.autograd.grad(loss,(model.kernel,model.weight))
        for total,gradient in zip(average,gradients):total.add_(gradient.detach()/args.local_batches)
    eta_h,eta_w=rates(payload,settings)
    output={}
    for arm in ARMS:
        dh=-eta_h*average[0] if arm=='sgd' else -eta_h*alpha*pullback(model.kernel,average[0])
        dw=-eta_w*average[1]
        derivative=float((average[0].double()*dh.double()).sum()+(average[1].double()*dw.double()).sum())
        output[arm]=dict(kernel=dh.cpu(),weight=dw.cpu(),mean_objective_derivative=derivative)
    return output


def prepare_source(args,job,payload,data,train,settings):
    folder=source_directory(args,job);path=folder/'calibration.json'
    if path.exists():
        info=base.read_json(path)
        expected=dict(trace_batches=args.trace_batches,draws_per_batch=args.trace_draws,calibration_size=args.calibration_size)
        if any(info[k]!=v for k,v in expected.items()):raise ValueError('Source trace settings differ')
        return info
    model=model_from(payload['model'],data.device)
    info=estimate_trace(model,data,train,job,args)
    directions=source_directions(model,payload,data,train,job,args,settings,info['alpha'])
    folder.mkdir(parents=True,exist_ok=True)
    base.save_tensor(folder/'source_directions.pt',directions)
    base.write_json(path,info)
    return info


def update(model,generator,data,train_device,settings,arm,alpha):
    chosen=train_device[torch.randint(len(train_device),(settings['batch_size'],),generator=generator,device=data.device)]
    first,second=augment(data[chosen],generator),augment(data[chosen],generator)
    loss=empirical_loss(model(first),model(second))
    gradients=torch.autograd.grad(loss,(model.kernel,model.weight))
    if not bool(loss.isfinite()) or not all(bool(g.isfinite().all()) for g in gradients):
        raise FloatingPointError('Nonfinite loss or gradients')
    eta_h,eta_w=settings['rates']
    direction=gradients[0] if arm=='sgd' else alpha*pullback(model.kernel,gradients[0])
    with torch.no_grad():
        model.kernel.add_(direction,alpha=-eta_h)
        model.weight.add_(gradients[1],alpha=-eta_w)
    if not all(bool(p.isfinite().all()) for p in model.parameters()):raise FloatingPointError('Nonfinite updated parameter')
    return float(loss.detach())


def train_branch(args,job,arm,payload,calibration,data,train,audit,settings):
    folder=source_directory(args,job)/arm;folder.mkdir(parents=True,exist_ok=True)
    first_step=payload['step'];end=step_for(job['target_epoch'],len(train),settings['batch_size'])
    requested=end if args.until_epoch is None else min(end,step_for(args.until_epoch,len(train),settings['batch_size']))
    if requested<first_step:return
    model=model_from(payload['model'],data.device)
    gen=torch.Generator(device=data.device);gen.set_state(payload['generator_state'].cpu())
    alpha=calibration['alpha'];latest=folder/'latest.pt'
    fixed={first_step,end,*(step_for(e,len(train),settings['batch_size']) for e in args.eval_epochs
                          if job['source_epoch']<=e<=job['target_epoch'])}
    history,snapshots,crossings,step=[],[],[],first_step
    if latest.exists():
        previous=torch.load(latest,map_location='cpu',weights_only=True)
        if previous['job']!=job or previous['arm']!=arm or previous['alpha']!=alpha:
            raise ValueError('Saved branch identity or source calibration differs')
        model.load_state_dict(previous['model']);gen.set_state(previous['generator_state'])
        step=previous['step'];history,snapshots,crossings=previous['history'],previous['snapshots'],previous['crossings']
    if requested<step:return
    effective=dict(settings,rates=rates(payload,settings))
    source_loss=audit_model(model_from(payload['model'],data.device),audit)['loss']
    def save():
        state=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},generator_state=gen.get_state().cpu(),
                   step=step,job=job,arm=arm,alpha=alpha,history=history,snapshots=snapshots,crossings=crossings)
        base.save_tensor(latest,state)
        if any(row['step']==step for row in snapshots):base.save_tensor(folder/f'step_{step}.pt',state)
        base.write_json(folder/'result.json',dict(job=job,arm=arm,alpha=alpha,source_step=first_step,
            completed_step=step,target_step=end,state='complete' if step==end else 'partial',source_audit_loss=source_loss,
            learning_rates=list(effective['rates']),history=history,snapshots=snapshots,common_progress_crossings=crossings))
    if not history:
        metric=audit_model(model,audit)
        history.append(dict(step=step,epoch=step*settings['batch_size']/len(train),progress_scan=True,**metric))
        snapshots.append(dict(step=step,file=f'step_{step}.pt',fixed=True,audit_drops=[]));save()
    indices=train.to(data.device);start=step;base.synchronize(args.device);begin=time.perf_counter()
    try:
        for step in range(start+1,requested+1):
            loss=update(model,gen,data,indices,effective,arm,alpha);reached=[]
            if step%args.audit_every==0 or step in fixed or step==requested:
                metric=audit_model(model,audit)
                scan=step%args.audit_every==0 or step in fixed
                previous_grid=next((r for r in reversed(history) if r.get('progress_scan',True)),history[-1])
                history.append(dict(step=step,epoch=step*settings['batch_size']/len(train),batch_loss=loss,progress_scan=scan,**metric))
                reached=[d for d in args.audit_drops if scan and metric['loss']<=source_loss-d and not any(c['drop']==d for c in crossings)]
                for d in reached:crossings.append(dict(drop=d,target_loss=source_loss-d,observed_loss=metric['loss'],
                    before_step=previous_grid['step'],before_loss=previous_grid['loss'],
                    overshoot=source_loss-d-metric['loss'],step=step,file=f'step_{step}.pt'))
            if step in fixed or reached:snapshots.append(dict(step=step,file=f'step_{step}.pt',fixed=step in fixed,audit_drops=reached))
            if step%args.save_every==0 or step in fixed or reached or step==requested:
                save();base.synchronize(args.device)
                print(f"seed={job['seed']} from={job['source_epoch']:g} {arm} step={step}/{end} "
                      f"epoch={step*settings['batch_size']/len(train):.3f} "
                      f"seconds/update={(time.perf_counter()-begin)/max(1,step-start):.5f}",flush=True)
    except (FloatingPointError,RuntimeError) as error:
        base.write_json(folder/f'failure_step_{step}.json',dict(job=job,arm=arm,step=step,error=str(error),
                        resume='last successfully saved latest.pt; no automatic retry'))
        raise


def tangent_speed(matrix,direction,train):
    """Sample-space ||dP||_F/sqrt(2), per source-mean update; no N-by-N matrix."""
    x,dx=matrix[train].double(),direction[train].double()
    x,dx=x-x.mean(0),dx-dx.mean(0)
    scale=x.square().mean(0).sqrt();scale=torch.where(scale>0,scale,torch.ones_like(scale))
    x,dx=x/scale,dx/scale
    u,s,vh=torch.linalg.svd(x,full_matrices=False)
    keep=s>s.max()*max(x.shape)*torch.finfo(x.dtype).eps
    u,s,vh=u[:,keep],s[keep],vh[keep]
    normal=dx-u@(u.T@dx);velocity=normal@(vh.T/s)
    return dict(rank=len(s),normal_feature_speed=float(normal.norm()),
                projector_frobenius_speed_over_sqrt2=float(velocity.norm()))


def evaluate_jobs(args,data,labels,train,test,settings):
    from natural_moment_eval import fixed_moment_dictionary,fit_readouts,geometry_bound,directional_readout_slopes
    names,targets=fixed_moment_dictionary(data,args.feature_batch)
    initial_cache={}
    for job in jobs(args):
        seed=job['seed'];payload,_=source_payload(args,job,settings)
        model=model_from(payload['model'],data.device)
        source=base.extract(model,data,args.feature_batch)
        if seed not in initial_cache:
            initial=torch.load(args.source/f'seed_{seed}'/'step_0.pt',map_location='cpu',weights_only=True)
            if initial['seed']!=seed or initial['step']!=0:raise ValueError('Wrong original step-0 reference')
            initial_model=model_from(initial['model'],data.device)
            initial_features=base.extract(initial_model,data,args.feature_batch)
            initial_cache[seed]=fit_readouts(*initial_features,targets,labels,train,test,10)
        initial_readouts=initial_cache[seed]
        source_readouts=fit_readouts(*source,targets,labels,train,test,10)
        parent=source_directory(args,job);local_path=parent/'source_local.json'
        if not local_path.exists():
            directions=torch.load(parent/'source_directions.pt',map_location='cpu',weights_only=True)
            local={}
            for arm in ARMS:
                derivative=base.extract_direction(model,data,directions[arm],args.feature_batch)
                local[arm]=dict(mean_objective_derivative=directions[arm]['mean_objective_derivative'],
                    risk_slopes=directional_readout_slopes(*source,*derivative,targets,train),
                    geometry={rep:tangent_speed(source[i],derivative[i],train)
                              for i,rep in enumerate(('encoder','projection'))})
            base.write_json(local_path,dict(job=job,task_names=names,arms=local,
                interpretation='source-local derivative of independently averaged training-batch gradients, not fixed-audit loss or a finite-window guarantee'))
        for arm in ARMS:
            folder=parent/arm
            if not (folder/'result.json').exists():continue
            result=base.read_json(folder/'result.json')
            for snapshot in result['snapshots']:
                step=snapshot['step'];destination=folder/f'evaluation_{step}.json'
                if destination.exists():continue
                state=torch.load(folder/snapshot['file'],map_location='cpu',weights_only=True)
                model=model_from(state['model'],data.device)
                current=source if step==payload['step'] else base.extract(model,data,args.feature_batch)
                now=source_readouts if step==payload['step'] else fit_readouts(*current,targets,labels,train,test,10)
                primary,index={},names.index('antidiagonal_high_energy')
                for rep in ('encoder','projection'):
                    primary[rep]={}
                    for split in ('train','evaluation'):
                        key=f'moment_{split}_mse'
                        r0,re,rl=(r[rep][key][index] for r in (initial_readouts,source_readouts,now))
                        primary[rep][split]=dict(initial_risk=r0,source_risk=re,current_risk=rl,
                            improvement=r0-re,excess=rl-r0,source_to_current=rl-re,
                            complete_rebound=r0-re>args.risk_threshold and rl-r0>args.risk_threshold)
                base.write_json(destination,dict(job=job,arm=arm,step=step,
                    epoch=step*settings['batch_size']/len(train),fixed=snapshot['fixed'],audit_drops=snapshot['audit_drops'],
                    task_names=names,readouts=now,initial_readouts=initial_readouts,source_readouts=source_readouts,
                    primary_task='antidiagonal_high_energy',primary=primary,
                    primary_encoder_joint=all(primary['encoder'][sp]['complete_rebound'] for sp in ('train','evaluation')),
                    geometry_source={rep:geometry_bound(source[i],current[i],targets,train)
                                     for i,rep in enumerate(('encoder','projection'))}))
                print(f'evaluated {parent.name}/{arm} step={step}',flush=True)


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def summarize(output):
    config=base.read_json(output/'protocol.json')
    coverage,rows,gaps,local_rows,progress=[],[],[],[],[]
    for job in config['jobs']:
        parent=output/f"seed_{job['seed']}_from_{job['source_epoch']:g}"
        if (parent/'source_local.json').exists():
            local=base.read_json(parent/'source_local.json')
            for arm,value in local['arms'].items():
                for rep in ('encoder','projection'):
                    for task,slope in zip(local['task_names'],value['risk_slopes'][rep]):
                        local_rows.append(dict(**job,arm=arm,representation=rep,task=task,slope=slope,
                            mean_objective_derivative=value['mean_objective_derivative'],**value['geometry'][rep]))
        for arm in ARMS:
            folder=parent/arm
            result=base.read_json(folder/'result.json') if (folder/'result.json').exists() else {}
            coverage.append(dict(**job,arm=arm,state=result.get('state','missing'),
                completed_step=result.get('completed_step'),target_step=result.get('target_step'),
                failure_files=' '.join(p.name for p in folder.glob('failure_step_*.json'))))
            for cross in result.get('common_progress_crossings',[]):
                path=folder/f"evaluation_{cross['step']}.json"
                if path.exists():
                    ev=base.read_json(path)
                    for split in ('train','evaluation'):
                        progress.append(dict(**job,arm=arm,split=split,**cross,**ev['primary']['encoder'][split]))
            for path in folder.glob('evaluation_*.json'):
                ev=base.read_json(path)
                for rep in ('encoder','projection'):
                    for split in ('train','evaluation'):
                        for task,risk in zip(ev['task_names'],ev['readouts'][rep][f'moment_{split}_mse']):
                            rows.append(dict(**job,arm=arm,step=ev['step'],epoch=ev['epoch'],
                                             representation=rep,split=split,task=task,risk=risk))
                if ev['step']==result.get('target_step'):
                    for rep in ('encoder','projection'):
                        for split in ('train','evaluation'):
                            gaps.append(dict(**job,arm=arm,representation=rep,split=split,
                                joint_encoder=ev['primary_encoder_joint'],**ev['primary'][rep][split]))
    output.mkdir(parents=True,exist_ok=True)
    for name,value in [('coverage',coverage),('moment_risks',rows),('primary_gaps',gaps),
                       ('source_local',local_rows),('common_progress',progress)]:write_csv(output/f'{name}.csv',value)
    paired=[]
    for job in config['jobs']:
        for rep in ('encoder','projection'):
            for split in ('train','evaluation'):
                cell={a:next((r for r in gaps if r['seed']==job['seed'] and r['source_epoch']==job['source_epoch']
                     and r['arm']==a and r['representation']==rep and r['split']==split),None) for a in ARMS}
                a,b=cell['sgd'],cell['pullback']
                paired.append(dict(**job,representation=rep,split=split,both_available=a is not None and b is not None,
                    sgd_risk=a['current_risk'] if a else None,pullback_risk=b['current_risk'] if b else None,
                    risk_pullback_minus_sgd=b['current_risk']-a['current_risk'] if a and b else None,
                    excess_pullback_minus_sgd=b['excess']-a['excess'] if a and b else None,
                    sgd_joint_encoder=a['joint_encoder'] if a else None,pullback_joint_encoder=b['joint_encoder'] if b else None))
    progress_pairs=[]
    for job in config['jobs']:
        for drop in config['arguments']['audit_drops']:
            for split in ('train','evaluation'):
                cell={a:next((r for r in progress if r['seed']==job['seed'] and r['source_epoch']==job['source_epoch']
                      and r['arm']==a and r['drop']==drop and r['split']==split),None) for a in ARMS}
                a,b=cell['sgd'],cell['pullback']
                progress_pairs.append(dict(**job,drop=drop,split=split,sgd_reached=a is not None,pullback_reached=b is not None,
                    sgd_observed_loss=a['observed_loss'] if a else None,pullback_observed_loss=b['observed_loss'] if b else None,
                    observed_loss_pullback_minus_sgd=b['observed_loss']-a['observed_loss'] if a and b else None,
                    sgd_overshoot=a['overshoot'] if a else None,pullback_overshoot=b['overshoot'] if b else None,
                    sgd_step=a['step'] if a else None,pullback_step=b['step'] if b else None,
                    risk_pullback_minus_sgd=b['current_risk']-a['current_risk'] if a and b else None))
    write_csv(output/'paired_differences.csv',paired)
    write_csv(output/'common_progress_pairs.csv',progress_pairs)
    lines=['# Paired quadratic-factor metric intervention','',
        'Reference risk is original seed step 0, not the branch source.',
        'Primary task: antidiagonal_high_energy. Fixed windows and all sources remain in protocol.json.',
        'The independent units are three seeds; two sources/arms per seed and eleven tasks are correlated measurements.',
        'Train empirical OLS optima and official test errors use training-fitted decoders.','',
        '| Seed | Window | Arm | State | Completed/target update |','| --- | --- | --- | --- | --- |']
    for c in coverage:lines.append(f"| {c['seed']} | {c['source_epoch']:g}->{c['target_epoch']:g} | "
        f"{c['arm']} | {c['state']} | {c.get('completed_step')}/{c.get('target_step')} |")
    lines+=['','primary_gaps.csv retains sources without early improvement.',
        'source_local.csv contains analytic source-mean risk and exterior-motion derivatives; these are not finite forecasts.',
        'common_progress.csv uses first crossings of fixed unlabelled audit-loss drops. Read observed loss and overshoot; states are not exactly equal-loss.',
        'Pullback flattens the Frobenius factor metric only on its attainable tangent. It is not full moment whitening and does not guarantee zero motion.',
        'paired_differences.csv and common_progress_pairs.csv retain missing arms/drops and report pullback-minus-SGD risk and actual audit-loss differences.',
        'Interpret fixed-update and common-audit-progress comparisons together to separate trajectory changes from speed.']
    (output/'findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f"Metric summary: {output/'findings.md'}",flush=True)


def context(args):
    p=base.read_json(args.source/'protocol.json');split=p['split']
    if p['arguments']['dataset']!='cifar10' or split['train_size']!=50000 or split['evaluation_size']!=10000 or not split['test_used']:
        raise ValueError('Expected official 50k/10k natural_moments_cifar10 source')
    settings=dict(split=split,batch_size=p['arguments']['batch_size'],lr=p['arguments']['lr'],ratio=p['arguments']['ratio'])
    data,labels,train,test,actual=base.load_images(SimpleNamespace(dataset='cifar10',data_root=args.data_root,image_size=32,device=args.device))
    for key in ('normalization_mean','normalization_std'):
        if abs(actual[key]-split[key])>1e-7:raise ValueError(f'Source preprocessing mismatch: {key}')
    return data,labels,train,test,settings


def lock(args,settings):
    value=dict(jobs=jobs(args),settings=settings,source=str(args.source.resolve()),
        arguments={k:getattr(args,k) for k in ('trace_batches','trace_draws','calibration_size','local_batches',
            'audit_size','audit_every','save_every','eval_epochs','audit_drops','risk_threshold')},arms=list(ARMS),
        initial_reference='original seed step_0; source is not initialization',
        metric='inverse J_C*J_C in Frobenius coordinates times frozen source trace ratio',
        pairing='both arms restore source generator; calibration uses independent RNG',task='antidiagonal_high_energy')
    path=args.output/'protocol.json'
    if path.exists() and base.read_json(path)!=value:raise ValueError('Existing intervention parameters differ; use a separate output')
    base.write_json(path,value)
    return value


def smoke():
    import tempfile
    runtime();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rng=torch.Generator().manual_seed(20261023)
    h=torch.randn(3,1,11,11,generator=rng,dtype=torch.float64)
    g=torch.randn(h.shape,generator=rng,dtype=torch.float64)
    moved=pullback(h,g).flatten(1);hf=h.flatten(1);q=hf.square().sum(1,keepdim=True)
    check=2*q*moved+2*hf*(hf*moved).sum(1,keepdim=True)
    torch.testing.assert_close(check,g.flatten(1),rtol=1e-12,atol=1e-12)
    assert float((g.flatten(1)*moved).sum())>0
    data=torch.randn(60,1,32,32,generator=rng)
    labels=torch.arange(60)%10;train=torch.arange(40);test=torch.arange(40,60)
    model,calibration=base.initialize(0,4,2,data[:8],2,record_state_hash=False)
    initial_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    gen=torch.Generator().manual_seed(820000)
    settings=dict(split=dict(train_size=40,evaluation_size=20,classes=10,test_used=False),
                  batch_size=4,lr=.002,ratio=1.)
    effective=dict(settings,rates=rates(dict(calibration=calibration),settings))
    # Ordinary-arm update is exactly the historical base runner's update.
    original=copy.deepcopy(model);original_gen=torch.Generator();original_gen.set_state(gen.get_state())
    expected=base.training_step(original,original_gen,data,train,SimpleNamespace(batch_size=4),effective['rates'],augment)
    actual=update(model,gen,data,train,effective,'sgd',1.)
    assert expected==actual and torch.equal(original_gen.get_state(),gen.get_state())
    for key in model.state_dict():torch.testing.assert_close(model.state_dict()[key],original.state_dict()[key],rtol=0,atol=0)
    update(model,gen,data,train,effective,'sgd',1.)
    payload=dict(model={k:v.detach().clone() for k,v in model.state_dict().items()},
                 generator_state=gen.get_state(),step=2,seed=0,calibration=calibration)
    parent=(RESULTS/'checks').resolve();parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='metric_smoke_',dir=parent) as temporary:
        folder=Path(temporary).resolve();assert folder.is_relative_to(parent)
        args=SimpleNamespace(output=folder/'resumed',source=folder/'source',seeds=[0],
            source_epochs=[.2],target_epochs=[.6],device='cpu',feature_batch=16,
            trace_batches=1,trace_draws=2,calibration_size=8,local_batches=2,
            audit_size=8,audit_every=2,save_every=2,eval_epochs=[.2,.6],
            audit_drops=[.000001],risk_threshold=.001,until_epoch=.3)
        sourcefolder=args.source/'seed_0';sourcefolder.mkdir(parents=True)
        base.save_tensor(sourcefolder/'step_2.pt',payload)
        base.save_tensor(sourcefolder/'step_0.pt',dict(payload,step=0,model=initial_state))
        job=jobs(args)[0]
        # Synthetic fixture only: a fixed upcoming batch gives a nonempty loss
        # crossing and lets off-grid pause handling be exercised directly.
        preview=torch.Generator();preview.set_state(payload['generator_state'])
        selected=train[torch.randint(len(train),(4,),generator=preview)]
        clean=data[selected];audit=(clean,augment(clean,preview),augment(clean,preview))
        lock(args,settings);before_rng=payload['generator_state'].clone()
        info=prepare_source(args,job,payload,data,train,settings)
        assert torch.equal(payload['generator_state'],before_rng)
        assert abs(info['ordinary_trace']-info['matched_pullback_trace'])<1e-12
        source_params={k:v.clone() for k,v in payload['model'].items()}
        for arm in ARMS:train_branch(args,job,arm,payload,info,data,train,audit,settings)
        args.until_epoch=.6
        for arm in ARMS:train_branch(args,job,arm,payload,info,data,train,audit,settings)
        resumed={arm:torch.load(source_directory(args,job)/arm/'latest.pt',weights_only=True) for arm in ARMS}
        assert torch.equal(resumed['sgd']['generator_state'],resumed['pullback']['generator_state'])
        assert resumed['sgd']['crossings'] and resumed['pullback']['crossings']
        assert all(c['step']!=3 for a in ARMS for c in resumed[a]['crossings'])
        for k in source_params:assert torch.equal(source_params[k],payload['model'][k])
        evaluate_jobs(args,data,labels,train,test,settings);summarize(args.output)
        with (args.output/'common_progress_pairs.csv').open(encoding='utf-8') as handle:
            pairs=list(csv.DictReader(handle))
        assert any(r['observed_loss_pullback_minus_sgd']!='' for r in pairs)
        from natural_moment_eval import fixed_moment_dictionary,fit_readouts
        _,targets=fixed_moment_dictionary(data,16)
        init_model=model_from(initial_state,'cpu');init_features=base.extract(init_model,data,16)
        init_readouts=fit_readouts(*init_features,targets,labels,train,test,10)
        ev=base.read_json(source_directory(args,job)/'sgd'/'evaluation_6.json')
        j=ev['task_names'].index('antidiagonal_high_energy')
        assert abs(ev['primary']['encoder']['train']['initial_risk']-init_readouts['encoder']['moment_train_mse'][j])<1e-12
        args.output=folder/'direct';args.until_epoch=None;lock(args,settings)
        direct_info=prepare_source(args,job,payload,data,train,settings)
        assert direct_info['alpha']==info['alpha']
        for arm in ARMS:
            train_branch(args,job,arm,payload,direct_info,data,train,audit,settings)
            direct=torch.load(source_directory(args,job)/arm/'latest.pt',weights_only=True)
            for k in direct['model']:torch.testing.assert_close(direct['model'][k],resumed[arm]['model'][k],rtol=0,atol=0)
            assert torch.equal(direct['generator_state'],resumed[arm]['generator_state'])
            assert direct['crossings']==resumed[arm]['crossings']
    report=dict(passed=True,data='synthetic only; no downloads or real-data training',checks=[
        'pullback inverse algebra and positive quadratic form','ordinary update matches historical runner exactly',
        'source RNG and parameters untouched by calibration','paired ordinary/pullback future RNG',
        'exact CPU interrupted/uninterrupted continuation','nonempty progress crossings ignore off-grid pause',
        'paired common-progress CSV with actual audit-loss differences','fixed source alpha',
        'train/evaluate/summarize outputs','original step-0 primary risk reference'])
    base.write_json(parent/'natural_metric_intervention_smoke.json',report)
    print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',nargs='?',choices=['plan','train','evaluate','summarize','smoke'],default='plan')
    parser.add_argument('--source',type=Path,default=RESULTS/'empirical/natural_moments_cifar10')
    parser.add_argument('--output',type=Path,default=RESULTS/'empirical/natural_metric_cifar10')
    parser.add_argument('--data-root',type=Path,default=EXPERIMENTS.parent)
    parser.add_argument('--seeds',type=int,nargs='+',default=[30,31,32])
    parser.add_argument('--source-epochs',type=float,nargs='+',default=[200.,750.])
    parser.add_argument('--target-epochs',type=float,nargs='+',default=[500.,2000.])
    parser.add_argument('--until-epoch',type=float)
    parser.add_argument('--trace-batches',type=int,default=4)
    parser.add_argument('--trace-draws',type=int,default=16)
    parser.add_argument('--calibration-size',type=int,default=256)
    parser.add_argument('--local-batches',type=int,default=16)
    parser.add_argument('--audit-size',type=int,default=256)
    parser.add_argument('--audit-every',type=int,default=100)
    parser.add_argument('--save-every',type=int,default=1000)
    parser.add_argument('--eval-epochs',type=float,nargs='+',default=[200,250,300,400,500,750,1000,1250,1500,1750,2000])
    parser.add_argument('--audit-drops',type=float,nargs='+',default=[.1,.25,.5,1.])
    parser.add_argument('--risk-threshold',type=float,default=.001)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--feature-batch',type=int,default=256)
    args=parser.parse_args()
    numeric=[*args.source_epochs,*args.target_epochs,*args.eval_epochs,*args.audit_drops,args.risk_threshold]
    if args.until_epoch is not None:numeric.append(args.until_epoch)
    if not all(math.isfinite(v) for v in numeric) or min(numeric)<0:parser.error('Finite nonnegative epochs/drops/threshold required')
    if len(args.source_epochs)!=len(args.target_epochs) or not all(a<b for a,b in zip(args.source_epochs,args.target_epochs)):
        parser.error('Each source epoch requires a later target epoch')
    if len(set(args.seeds))!=len(args.seeds) or len(set(args.source_epochs))!=len(args.source_epochs):
        parser.error('Seeds/source epochs must be distinct')
    if min(args.trace_batches,args.trace_draws,args.calibration_size,args.local_batches,args.audit_size,
           args.audit_every,args.save_every,args.threads,args.feature_batch)<1:parser.error('Positive sizes and intervals required')
    if args.phase=='plan':
        updates=sum(step_for(j['target_epoch'],50000,256)-step_for(j['source_epoch'],50000,256) for j in jobs(args))
        print(json.dumps(dict(jobs=jobs(args),arms=list(ARMS),updates_per_arm=updates,total_updates=updates*2,
            primary_task='antidiagonal_high_energy',reference='original step_0, not source',source=str(args.source),
            output=str(args.output),action='plan only; no dataset access, download or training'),indent=2));return
    if args.phase=='summarize':summarize(args.output);return
    if args.phase=='smoke':smoke();return
    runtime();torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    data,labels,train,test,settings=context(args)
    if args.phase=='train':
        lock(args,settings)
        audit,_=make_audit(data,train,min(args.audit_size,len(train)),20261023,record_hashes=False)
        for job in jobs(args):
            payload,_=source_payload(args,job,settings);info=prepare_source(args,job,payload,data,train,settings)
            for arm in ARMS:train_branch(args,job,arm,payload,info,data,train,audit,settings)
    else:
        lock(args,settings);evaluate_jobs(args,data,labels,train,test,settings)


if __name__=='__main__':main()
