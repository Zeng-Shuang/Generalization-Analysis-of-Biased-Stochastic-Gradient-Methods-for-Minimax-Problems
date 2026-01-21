import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.1)
        # torch.nn.init.constant_(m.weight, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.1)
        # torch.nn.init.constant_(m.weight, 0.0)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.1)
        # torch.nn.init.constant_(m.weight, 0.0)
        torch.nn.init.zeros_(m.bias)

def compute_gan_loss(output, label, loss='gan'):
    if loss == 'bce':
        gan_loss = nn.BCELoss()(output, label)
    elif loss == 'gan':
        gan_loss = - torch.mean(label * torch.log(output + 1e-8) + (1 - label) * torch.log(1 - output + 1e-8))
    elif loss == 'wgan':
        label = 2 * label - 1  # convert to 1, -1
        gan_loss = torch.mean(label * output)
    else:
        raise NotImplementedError()

    return gan_loss