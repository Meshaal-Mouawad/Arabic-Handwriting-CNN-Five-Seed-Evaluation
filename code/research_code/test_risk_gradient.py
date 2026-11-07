"""Check analytic derivatives of the proposed smooth worst-family objective."""
import torch

def test_gradients():
    ld=torch.tensor(0.7,dtype=torch.double,requires_grad=True)
    ll=torch.tensor(1.4,dtype=torch.double,requires_grad=True)
    lam=.5;tau=.2
    mean=(ld+ll)/2
    soft=tau*(torch.logsumexp(torch.stack([ld,ll])/tau,0)-torch.log(torch.tensor(2.,dtype=torch.double)))
    objective=(1-lam)*mean+lam*soft
    g1,g2=torch.autograd.grad(objective,(ld,ll))
    w=torch.softmax(torch.stack([ld.detach(),ll.detach()])/tau,0)
    assert torch.allclose(g1,(1-lam)/2+lam*w[0],atol=1e-12)
    assert torch.allclose(g2,(1-lam)/2+lam*w[1],atol=1e-12)
    assert abs((g1+g2).item()-1.)<1e-12
    assert g2>g1
    print('PASS: smooth-group-risk gradient matches analytic expression; higher-loss family has greater marginal weight')
if __name__=='__main__':test_gradients()
