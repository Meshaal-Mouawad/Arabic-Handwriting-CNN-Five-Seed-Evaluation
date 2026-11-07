"""Synthetic *software sanity tests* only; NOT scientific evidence."""
import torch
from models import FlatCNN,HierarchicalCNN,family_risk,CONFIGS

torch.set_num_threads(2)
torch.manual_seed(7)
x=torch.randn(4,1,64,64)
y=torch.tensor([1,9,15,37])
for label,(klass,rule) in CONFIGS.items():
    model=klass(); model.train()
    p,fg=model(x)
    assert p.shape==(4,38) and fg.shape==(4,2)
    assert torch.allclose(torch.exp(p).sum(1),torch.ones(4),atol=1e-5),label
    loss,_=family_risk(p,y,rule,lam=0.5,tau=0.2)
    assert torch.isfinite(loss),label
    loss.backward()
    assert model.encoder[0].weight.grad is not None,label
    print(label,"trainable_params",sum(z.numel() for z in model.parameters()),"synthetic loss",round(loss.item(),4))
print("PASS: all configured models; shapes, normalized posterior, finite loss and gradients")
