"""Resumable, source-pinned six-arm/five-seed real-image experiment runner.

The published dataset ZIPs are downloaded by dataset_acquisition.py, and no
holdout predictions are calculated until the entire 30-training-run matrix has
completed. Synthetic tests live separately and never populate this output root.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,time,shutil
from datetime import datetime,timezone
from pathlib import Path

from dataset_acquisition import prepare,sha256,SOURCES,blob_digest,validate_cached_extraction

ROOT_CODE=Path(__file__).resolve().parent/'research_code'
CONFIGS=('B0_flat_ce','B1_flat_balanced','B2_hier_ce','B3_hier_balanced','P_hier_smooth_worst','C_flat_matched')
SEEDS=(7,17,29,43,61)

def timestamp(): return datetime.now(timezone.utc).isoformat()
def atomic_write_json(p,obj):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)

def call(command,log=None):
    print('RUN:', ' '.join(str(s) for s in command),flush=True)
    if log is None:
        return subprocess.run(list(map(str,command)),check=True,cwd=ROOT_CODE)
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as f:
        start=time.perf_counter()
        res=subprocess.run(list(map(str,command)),stdout=f,stderr=subprocess.STDOUT,cwd=ROOT_CODE)
        f.write(f'\n# subprocess exit code: {res.returncode}; wall_seconds: {time.perf_counter()-start:.2f}\n')
    if res.returncode:raise RuntimeError(f'Experiment failure; see {log} (returncode {res.returncode})')
    print(f'PASS {log} ({time.perf_counter()-start:.1f}s)')

def code_hashes():
    paths=list(sorted(ROOT_CODE.glob('*.py')))+[Path(__file__).resolve(),
        Path(__file__).resolve().parent/'dataset_acquisition.py',
        Path(__file__).resolve().parent/'publication_outputs.py']
    return {str(f.relative_to(Path(__file__).resolve().parent)):sha256(f) for f in paths}

def ensure_data(root, *, acquire):
    prepared=root/'prepared_canonical'
    if acquire or not prepared.exists():
        prepare(root,ROOT_CODE)
    upstream=root/'upstream_source_receipts.json'
    if not upstream.exists():raise FileNotFoundError('Upstream original data provenance record missing')
    orig=json.loads(upstream.read_text())
    for family,info in SOURCES.items():
        f=Path(orig[family]['archive'])
        if not f.exists() or blob_digest(f)!=info['sha1_git_blob']:
            raise ValueError(f'Original {family} archive has changed or is missing; do not continue')
    if not (root/'PRETRAINING_POLARITY_AUDIT.json').exists():
        raise FileNotFoundError('Missing polarity evidence; prepare datasets using Stage4 source-pinned acquisition')
    for p in ('train.npz','val.npz','untouched_test.npz','source_and_split_manifest.json'):
        if not (prepared/p).exists():raise FileNotFoundError(prepared/p)
    audit=root/'partition_integrity_audit.json'
    command=[sys.executable,ROOT_CODE/'audit_partitions.py', '--train',prepared/'train.npz',
             '--val',prepared/'val.npz','--test',prepared/'untouched_test.npz',
             '--manifest',prepared/'source_and_split_manifest.json','--out',audit]
    call(command)
    record=json.loads(audit.read_text())
    if record['cross_partition_exact_image_overlap']!='PASS: zero pairs':raise ValueError('Integrity gate failed')
    return prepared,record

def manifest(args,prepared,audit):
    return {'protocol_version':'stage4-public-source-retrieval-real-image-experiment-v1',
      'source':'MADBase + AHCD public archives; immutable archive receipts',
      'labels':'38 combined categories: digits 0-9; letters 10-37',
      'train_sha256':sha256(prepared/'train.npz'),
      'validation_sha256':sha256(prepared/'val.npz'),
      'test_sha256':sha256(prepared/'untouched_test.npz'),
      'partition_audit_sha256':sha256(args.root/'partition_integrity_audit.json'),
      'polarity_audit_sha256':sha256(args.root/'PRETRAINING_POLARITY_AUDIT.json'),
      'upstream_source_receipts_sha256':sha256(args.root/'upstream_source_receipts.json'),
      'code_sha256':code_hashes(),'configs':list(CONFIGS),'seeds':list(SEEDS),
      'epochs':args.epochs,'batch':args.batch,'lr':args.lr,'lam':args.lam,'tau':args.tau,
      'fixed_corruption_seed':1234,
      'checkpoint_selection':'max validation worst-family macro-F1, first epoch wins ties',
      'integrity_gate':'exact cross-partition image fingerprints; no overlapping preprocessed images',
      'test_policy':'no test evaluation until 30 complete training and validation records',
      'inference_settings':['clean','noise σ=0.05','noise σ=0.10','Gaussian blur σ=0.8','Gaussian blur σ=1.3','contrast=.75','contrast=.50'],
      'description':'observational evaluation of controlled model variants; not a rerun of the archived 98.86% experiment'}

def ensure_manifest(root,proposed):
    fp=root/'EXPERIMENT_LOCK.json'
    if fp.exists():
        old=json.loads(fp.read_text())
        if old!=proposed:
            diff=[key for key in set(old)|set(proposed) if old.get(key)!=proposed.get(key)]
            raise ValueError(f'FROZEN plan differs for keys {diff}; preserve existing runs and start a new output directory')
    else:atomic_write_json(fp,proposed)
    return fp

def completed_training(root,plan,config,seed):
    folder=root/'runs'/f'{config}_seed{seed}'
    fp=folder/'training_record.json';cp=folder/'best.pt'
    if not (fp.exists() and cp.exists()):return False
    data=json.loads(fp.read_text())
    if data['config']!=config or data['seed']!=seed or data['epochs']!=plan['epochs']:
        raise ValueError(f'Cannot trust completed run {folder}: training metadata does not match frozen settings')
    if data['train_file_sha256']!=plan['train_sha256'] or data['val_file_sha256']!=plan['validation_sha256']:
        raise ValueError('Frozen-data mismatch in prior completed run')
    return True

def train(root,prepared,plan,args):
    n_done=0;n_started=0
    for config in CONFIGS:
        for seed in SEEDS:
            folder=root/'runs'/f'{config}_seed{seed}'
            if completed_training(root,plan,config,seed):
                n_done+=1;continue
            if n_started>=args.max_new_train_runs:
                print(f'PAUSED after {n_started} newly completed runs; {n_done}/30 complete; rerun this notebook to resume.')
                return n_done
            if folder.exists() and any(folder.iterdir()):
                # A prior crashed run is preserved unchanged for inspection.
                raise ValueError(f'Incomplete or corrupted run directory {folder}. Preserve audit trail; move to a separate quarantine folder before deliberate restart.')
            folder.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,ROOT_CODE/'benchmark.py','train','--train',prepared/'train.npz','--val',prepared/'val.npz',
                 '--config',config,'--out',folder,'--seed',seed,'--epochs',args.epochs,'--batch',args.batch,'--lr',args.lr,
                 '--lam',args.lam,'--tau',args.tau]
            if args.cpu:cmd.append('--cpu')
            start=timestamp()
            call(cmd,root/'logs'/f'train_{config}_seed{seed}.log')
            if not completed_training(root,plan,config,seed):
                raise RuntimeError('Training exited without producing valid model and metadata')
            n_done+=1;n_started+=1
            atomic_write_json(root/'RUN_PROGRESS.json',{'completed_training_runs':n_done,'total_training_runs':30,
                                                       'last_completed':f'{config}_seed{seed}','run_started':start,'updated':timestamp()})
            print(f'TRAINING PROGRESS {n_done}/30')
    return n_done

def evaluate(root,prepared,plan,args):
    for config in CONFIGS:
        for seed in SEEDS:
            if not completed_training(root,plan,config,seed):
                raise ValueError('REAL TEST SET LOCKED until ALL 30 training runs complete')
    n_done=0;n_started=0
    for config in CONFIGS:
        for seed in SEEDS:
            folder=root/'runs'/f'{config}_seed{seed}'
            pred=folder/'predictions.npz';res=folder/'test_results.json'
            if pred.exists() and res.exists():
                r=json.loads(res.read_text())
                if r.get('predictions_sha256')!=sha256(pred) or r.get('test_sha256')!=plan['test_sha256']:
                    raise ValueError(f'Mismatched prediction evidence in {folder}')
                n_done+=1;continue
            if n_started>=args.max_new_eval_runs:
                print(f'PAUSED with {n_done}/30 evaluated model runs. Rerun notebook to resume.')
                return n_done
            if pred.exists() or res.exists():
                raise ValueError(f'Incomplete evaluation in {folder}, preserve before retry')
            cmd=[sys.executable,ROOT_CODE/'benchmark.py','evaluate','--test',prepared/'untouched_test.npz',
                 '--checkpoint',folder/'best.pt','--out',res,'--save-predictions',pred]
            if args.cpu:cmd.append('--cpu')
            call(cmd,root/'logs'/f'test_{config}_seed{seed}.log')
            n_done+=1;n_started+=1
            atomic_write_json(root/'EVALUATION_PROGRESS.json',{'completed_evaluation_runs':n_done,'total_evaluation_runs':30,'updated':timestamp()})
    if n_done==30:
        stats=root/'PAIRED_HELDOUT_STATISTICS.json'
        call([sys.executable,ROOT_CODE/'analyze_predictions.py','--runs',root/'runs','--out',stats,'--bootstrap',args.bootstrap],root/'logs'/'paired_statistical_analysis.log')
        make_summary(root)
        # Release plots and tables only after a complete 30-run held-out evaluation.
        call([sys.executable,Path(__file__).parent/'publication_outputs.py','--root',root],
             root/'logs'/'generate_publication_outputs.log')
    return n_done

def make_summary(root):
    stats=json.loads((root/'PAIRED_HELDOUT_STATISTICS.json').read_text())
    contrasts=stats['contrasts']
    lines=['# Real-data hypothesis contrasts','',
           'These values are computed from held-out predictions for the **full 30-run** matrix.','',
           '| Hypothesis contrast | Mean paired Δ worst-family macro-F1 | Across-seed 95% interval |',
           '|---|---:|---|']
    for name,data in contrasts.items():
        clean=data['clean_worst_family_F1_difference'];lo,hi=clean['ci_95']
        lines.append(f'| {name} | {clean["mean"]:+.4f} | [{lo:+.4f}, {hi:+.4f}] |')
    lines.append('')
    (root/'RESULTS_SUMMARY_REAL_DATA.md').write_text('\n'.join(lines)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',required=True,type=Path)
    ap.add_argument('--phase',choices=['prepare','train','evaluate','all'],default='all')
    ap.add_argument('--epochs',type=int,default=20)
    ap.add_argument('--batch',type=int,default=128)
    ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--lam',type=float,default=.5)
    ap.add_argument('--tau',type=float,default=.2)
    ap.add_argument('--cpu',action='store_true')
    ap.add_argument('--bootstrap',type=int,default=500)
    ap.add_argument('--max-new-train-runs',type=int,default=30)
    ap.add_argument('--max-new-eval-runs',type=int,default=30)
    args=ap.parse_args();args.root=args.root.resolve();args.root.mkdir(parents=True,exist_ok=True)
    if not 1<=args.max_new_train_runs<=30 or not 1<=args.max_new_eval_runs<=30:raise ValueError('Run caps must be 1..30')
    if args.epochs<1:raise ValueError('epochs must be positive')
    # IMPORTANT: Download sources is allowed even in eval mode only when no prepared source exists.
    prepared,audit=ensure_data(args.root,acquire=args.phase=='prepare')
    plan=manifest(args,prepared,audit)
    if args.phase=='prepare':
        print('DATA PREPARED AND AUDITED. Inspect orientation preview before full execution.')
        return
    ensure_manifest(args.root,plan)
    if args.phase in ('train','all'):
        n=train(args.root,prepared,plan,args)
        if n<30:
            print('FULL HELDOUT EVALUATION IS SEALED until 30/30 runs pass.')
            return
    if args.phase in ('evaluate','all'):
        ne=evaluate(args.root,prepared,plan,args)
        if ne==30:print('REAL DATA RESULTS COMPLETE:',args.root/'RESULTS_SUMMARY_REAL_DATA.md')
if __name__=='__main__':main()
