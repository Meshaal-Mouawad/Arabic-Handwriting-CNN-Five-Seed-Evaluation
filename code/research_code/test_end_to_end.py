"""Small synthetic end-to-end test with a tiny stand-in model (NOT the paper CNN)."""
import tempfile
from pathlib import Path
from argparse import Namespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import benchmark as engine
from analyze_predictions import run,CONTRASTS

class TinyStandin(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=nn.Linear(64*64,38)
    def forward(self,x):
        logp=F.log_softmax(self.layer(x.flatten(1)),dim=1)
        return logp,torch.stack((torch.logsumexp(logp[:,:10],dim=1),
                                 torch.logsumexp(logp[:,10:],dim=1)),dim=1)

def sample(seed):
    rng=np.random.default_rng(seed)
    y=np.repeat(np.arange(38,dtype=np.int64),2)
    x=rng.uniform(0,1,(len(y),1,64,64)).astype('float32')
    return x,y

def execute():
    real_configs=engine.CONFIGS
    rules={name:real_configs[name][1] for pair in CONTRASTS.values() for name in pair}
    engine.CONFIGS={name:(TinyStandin,rule) for name,rule in rules.items()}
    with tempfile.TemporaryDirectory() as tmp:
        td=Path(tmp)
        for name,seed in [('train',11),('val',13),('test',17)]:
            x,y=sample(seed)
            np.savez_compressed(td/f'{name}.npz',images=x,labels=y)
        for config in rules:
            folder=td/f'{config}_seed7'
            engine.run_train(Namespace(train=str(td/'train.npz'),val=str(td/'val.npz'),
                config=config,out=str(folder),seed=7,epochs=1,batch=38,
                lr=.001,lam=.5,tau=.2,cpu=True))
            engine.run_evaluate(Namespace(test=str(td/'test.npz'),checkpoint=str(folder/'best.pt'),
                out=str(folder/'test_results.json'),save_predictions=str(folder/'predictions.npz'),
                corruption_seed=1234,cpu=True))
        out=run(td,[7],10)
        assert set(out['contrasts'])==set(CONTRASTS)
        for key in out['contrasts']:
            assert len(out['contrasts'][key]['condition_F1_differences'])==7
    print('PASS SYNTHETIC PIPELINE: 6 tiny stand-in training runs, 42 corruption evaluations, paired analysis; no empirical recognition results retained')
if __name__=='__main__':
    torch.set_num_threads(2)
    execute()
