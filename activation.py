
import torch

def torchify(f):
    def g(x):
        acceptible_typ = (torch.Tensor, torch.FloatTensor, torch.nn.Parameter)
        if type(x) not in acceptible_typ: x = torch.tensor(x)
        return f(x)
    return g

Softplus = torch.nn.Softplus(beta=3.0, threshold=5)

@torchify
def activation(x):
    raise NotImplementedError('Please use one of x.relu() or Softplus(x).')
    # return x.relu()
    # return Softplus(x)
