"""Dry-run or execute the locked 5-primary + 1-control seed-matched experiment matrix.

Train and validation data must already have passed audit_partitions.py.
--execute is an intentional, explicit opt-in; without it this prints the commands.
"""
import argparse,json,shlex,subprocess,sys,platform
from pathlib import Path
import torch
from models import CONFIGS
from benchmark import sha256

PRIMARY=['B0_flat_ce','B1_flat_balanced','B2_hier_ce','B3_hier_balanced','P_hier_smooth_worst']
SENSITIVITY=['C_flat_matched']
SEEDS=[7,17,29,43,61]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',required=True)
    p.add_argument('--val',required=True)
    p.add_argument('--test',required=True)
    p.add_argument('--out',default='runs')
    p.add_argument('--audit',required=True,help='PASS report from audit_partitions.py')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--seed',type=int,nargs='+',default=SEEDS)
    args=p.parse_args()
    args.train=str(Path(args.train).resolve())
    args.val=str(Path(args.val).resolve())
    args.test=str(Path(args.test).resolve())
    args.audit=str(Path(args.audit).resolve())
    args.out=str(Path(args.out).resolve())
    a=json.loads(Path(args.audit).read_text())
    assert a.get('cross_partition_exact_image_overlap')=='PASS: zero pairs','Audit failed'
    for key,source in [('train',args.train),('validation',args.val),('test',args.test)]:
        if sha256(source)!=a[key]['file_sha256']:
            raise ValueError(f'{key} source file changed since integrity audit; rerun preflight before any training')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    configs=PRIMARY+SENSITIVITY
    env={'python':platform.python_version(),'torch':torch.__version__,
         'device_available':'CUDA' if torch.cuda.is_available() and not args.cpu else 'CPU',
         'configs':configs,'seeds':args.seed,'epochs':args.epochs,'batch_size':args.batch,
         'learning_rate':args.lr,'training_input':args.train,'validation_input':args.val,
         'heldout_input':args.test,'audit_report':args.audit,'test_access_policy':'execute ALL training and validation runs before opening the test dataset for evaluation'}
    plan_file=out/'frozen_execution_plan.json'
    if args.execute and plan_file.exists():
        old=json.loads(plan_file.read_text())
        if old!=env:raise ValueError('Execution differs from earlier dry-run lock. Start a new output folder for protocol changes')
    if args.execute:
        existing={x.name for x in out.iterdir()}
        if existing-{'frozen_execution_plan.json','COMMANDS.txt'}:
            raise ValueError('Output folder contains prior runs. Refusing to overwrite evidence; start a new folder')
    plan_file.write_text(json.dumps(env,indent=2))
    training_commands=[]
    evaluation_commands=[]
    for config in configs:
        for seed in args.seed:
            folder=out/f'{config}_seed{seed}'
            train=[sys.executable,'benchmark.py','train','--train',args.train,'--val',args.val,
                   '--config',config,'--out',str(folder),'--seed',str(seed),'--epochs',str(args.epochs),
                   '--batch',str(args.batch),'--lr',str(args.lr)]
            ev=[sys.executable,'benchmark.py','evaluate','--test',args.test,'--checkpoint',str(folder/'best.pt'),
                '--out',str(folder/'test_results.json'),
                '--save-predictions',str(folder/'predictions.npz')]
            if args.cpu:train.append('--cpu');ev.append('--cpu')
            training_commands.append(train)
            evaluation_commands.append(ev)
    commands=training_commands+evaluation_commands
    (out/'COMMANDS.txt').write_text('\n'.join(' '.join(shlex.quote(s) for s in cmd) for cmd in commands)+'\n')
    print(f'RUN PLAN: {len(configs)} model conditions x {len(args.seed)} seeds = {len(configs)*len(args.seed)} training runs')
    print('Source files are not bundled; this file can only execute after data provenance and leakage gates PASS.')
    if args.execute:
        for cmd in commands:
            print('\nRUN', ' '.join(shlex.quote(s) for s in cmd),flush=True)
            subprocess.run(cmd,cwd=Path(__file__).parent,check=True)
    else:print('DRY RUN ONLY. Saved frozen_execution_plan.json and COMMANDS.txt. Pass --execute when ready.')
if __name__=='__main__':main()
