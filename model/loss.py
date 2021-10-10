import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def mse_loss(output, target):
    print("output shape: {}".format(output.shape))
    print("target shape: {}".format(target.shape))
    return F.mse_loss(output, target)
