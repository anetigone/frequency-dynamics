"""Check low-rank LS optimality, label isolation, exact encoder freeze and end-to-end output."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS
import json
import subprocess
import tempfile
from types import SimpleNamespace

import torch
from frozen_encoder_probe import reduced_rank, run_state
from cifar_capacity import fit_probe, initialize, save_checkpoint, write_json


def main():
    torch.set_num_threads(1)
    generator=torch.Generator().manual_seed(20261020)
    x=torch.randn(80,12,generator=generator).double()
    labels=torch.arange(80)%10
    train,validation=torch.arange(60),torch.arange(60,80)
    errors=[]
    for k in [1,4,8,10]:
        result=reduced_rank(x,labels,train,validation,k)
        errors.append(result['train']['onehot_mse'])
    assert all(a>=b-1e-12 for a,b in zip(errors,errors[1:]))
    full=fit_probe(x,labels,train,validation)
    assert abs(errors[-1]-full['train']['onehot_mse'])<1e-12
    a=reduced_rank(x,labels,train,validation,8)
    changed=labels.clone();changed[validation]=(changed[validation]+1)%10
    b=reduced_rank(x,changed,train,validation,8)
    assert all(torch.equal(a[key],b[key]) for key in ['projection','classifier','bias'])
    zero=reduced_rank(torch.ones_like(x),labels,train,validation,8)
    assert zero['feature_rank']==0 and zero['retained_components']==0
    parent=(RESULTS/'checks').resolve();assert parent.is_relative_to(EXPERIMENTS.resolve())
    with tempfile.TemporaryDirectory(prefix='frozen_check_',dir=parent) as temporary:
        root=Path(temporary).resolve();assert root.is_relative_to(parent)
        data=torch.randn(60,1,32,32,generator=generator)
        y=torch.arange(60)%10;tr,va=torch.arange(40),torch.arange(40,60)
        calibration=data[:12]
        model,_=initialize(0,64,8,calibration,2)
        checkpoint=root/'source.pt'
        source_job=dict(seed=0,architecture='baseline',objective='unsupervised',ratio=1.,channels=64,outputs=8)
        save_checkpoint(checkpoint,model,None,None,source_job,0)
        job=dict(id='seed_0_source_0',seed=0,source_step=0,checkpoint=str(checkpoint))
        args=SimpleNamespace(output=root/'out',source_ratio=1.,device='cpu',feature_batch=16,steps=4,batch_size=4,
                             lr=.0003,classifier_lr=.001,grad_clip=1.)
        write_json(args.output/'protocol.json',dict(arguments=dict(steps=4),jobs=[job]))
        run_state(args,job,data,y,tr,va,calibration)
        result=json.loads((args.output/job['id']/'result.json').read_text())
        assert result['optimization']['encoder_unchanged_exactly'] and result['optimization']['projection_changed']
        saved=torch.load(args.output/job['id']/'head_final.pt',weights_only=True)
        assert torch.equal(saved['model']['kernel'],model.kernel.detach())
        command=[sys.executable,'-B',str(EXPERIMENTS/'code/cifar/summarize_frozen_encoder.py'),str(args.output)]
        process=subprocess.run(command,capture_output=True,encoding='utf-8',errors='replace')
        assert process.returncode==0,process.stderr
        assert len(json.loads((args.output/'summary.json').read_text())['rows'])==5
        # Completed runs are skipped without needing the source checkpoint again.
        checkpoint.rename(root/'source_saved.pt')
        run_state(args,job,data,y,tr,va,calibration)
    report=dict(synthetic_only=True,reduced_rank_training_mse_monotone=True,
                full_rank_matches_unregularized_ols=True,validation_labels_excluded=True,
                degenerate_features_supported=True,encoder_bitwise_unchanged=True,
                projection_trained=True,end_to_end_summary=True,completed_runs_skipped=True)
    write_json(RESULTS/'checks'/'frozen_encoder_checks.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
