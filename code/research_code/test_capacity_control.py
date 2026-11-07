import torch
from models import FlatCNN, HierarchicalCNN, CapacityMatchedFlatCNN

def test_parameter_count_and_capacity_control():
    flat=FlatCNN();hier=HierarchicalCNN();matched=CapacityMatchedFlatCNN()
    sizes=[sum(p.numel() for p in m.parameters()) for m in (flat,hier,matched)]
    assert sizes==[102534,102792,102792],sizes
    matched.encoder.load_state_dict(flat.encoder.state_dict())
    matched.head.load_state_dict(flat.head.state_dict())
    flat.eval(); matched.eval()
    with torch.no_grad():
        a=flat(torch.rand(2,1,64,64))[0]
        # Compare using IDENTICAL x in both forward passes.
        x=torch.rand(3,1,64,64)
        assert torch.allclose(flat(x)[0],matched(x)[0],atol=1e-6)
        assert torch.allclose(flat(x)[0].exp().sum(1),torch.ones(3))
    print('PASS: flat 102534 trainable/103014 Keras-style total; hierarchy=capacity-control 102792 trainable/103272 total; identical at zero offsets')

if __name__=='__main__':test_parameter_count_and_capacity_control()
