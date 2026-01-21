import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from tqdm import tqdm
import random

from utils import weights_init, compute_gan_loss


def train_clip(dataset, manual_seed, options):  
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    model = options['model']
    loss = options['loss']
    data = options['data']
    lr = options['learning_rate']
    nz = options['nz']
    batch_size = 2000
    num_epochs = options['num_epochs']
    device = options['device']
    clip_value = options['clip_value']  # Clipping parameter

    # Define GAN networks
    if model == 'vgan':
        from vgan import VanillaDiscriminator, VanillaGenerator

        if data == 'mnist':
            generator = VanillaGenerator(nz).to(device)
            discriminator = VanillaDiscriminator().to(device)
        elif data == 'cifar10':
            generator = VanillaGenerator(nz, n_c=3).to(device)
            discriminator = VanillaDiscriminator(n_c=3).to(device)
    elif model == 'dcgan':
        from dcgan import DCGANDiscriminator, DCGANGenerator

        if data == 'mnist':
            generator = DCGANGenerator(nz, n_out=1).to(device)
            discriminator = DCGANDiscriminator(n_in=1).to(device)
        elif data == 'cifar10':
            generator = DCGANGenerator(nz).to(device)
            discriminator = DCGANDiscriminator().to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Optimizers
    optim_g = optim.SGD(generator.parameters(), lr=lr)
    optim_d = optim.SGD(discriminator.parameters(), lr=lr)

    # Initialize parameter saving
    gen_param = []
    dis_param = []

    print('Training with Gradient Clipping......')

    for epoch in range(num_epochs):
        sampler = RandomSampler(dataset, replacement=True, num_samples=len(dataset))
        train_loader = DataLoader(dataset, batch_sampler=BatchSampler(sampler, batch_size=batch_size, drop_last=False))

        epoch_gen_param = []
        epoch_dis_param = []

        losses_g = 0.0
        losses_d = 0.0

        for i, (images, _) in tqdm(enumerate(train_loader, 0), total=int(len(dataset)/batch_size)):
            ############################
            # (1) Update Discriminator (D) network with gradient clipping
            ###########################
            discriminator.zero_grad()
            images = images.to(device)
            b_size = images.size()[0]
            label = torch.full((b_size,), 1, dtype=images.dtype, device=device)
            output = discriminator(images)
            loss_real = compute_gan_loss(output, label, loss=loss)
            loss_real.backward()
            D_x = output.mean().item()

            # Train with fake images
            noises = torch.randn(b_size, nz, device=device)
            images_fake = generator(noises)
            label.fill_(0)
            output = discriminator(images_fake.detach())
            loss_fake = compute_gan_loss(output, label, loss=loss)
            loss_fake.backward()
            D_G_z1 = output.mean().item()
            loss_d = loss_real + loss_fake

            # Calculate the gradient norm of the discriminator and clip it (scale if the norm exceeds clip_value)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip_value)

            optim_d.step()

            ############################
            # (2) Update Generator (G) network with gradient clipping
            ###########################
            generator.zero_grad()
            label.fill_(1)
            output = discriminator(images_fake)
            loss_g = compute_gan_loss(output, label, loss=loss)
            loss_g.backward()
            D_G_z2 = output.mean().item()

            # Calculate the gradient norm of the generator and clip it
            torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_value)

            optim_g.step()

            losses_d += loss_d.item()
            losses_g += loss_g.item()

        print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, num_epochs, losses_d / i, losses_g / i, D_x, D_G_z1, D_G_z2))

        # Save model parameters
        for param in generator.parameters():
            epoch_gen_param.append(param.data.clone())
        for param in discriminator.parameters():
            epoch_dis_param.append(param.data.clone())

        gen_param.append(epoch_gen_param)
        dis_param.append(epoch_dis_param)

    return gen_param, dis_param

if __name__ == '__main__':
    manual_seed = 123
    options = dict()
    options['model'] = 'dcgan'
    options['loss'] = 'wgan'
    options['data'] = 'mnist'
    options['metric'] = 'frobenius'
    options['learning_rate'] = 0.0002
    options['nz'] = 8
    options['batch_size'] = 500
    options['num_epochs'] = 2
    options['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    options['clip_value'] = 1.0  # Clipping parameter

    if options['data'] == 'mnist':
        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.MNIST(root='./data/', download=True, transform=transform)
    elif options['data'] == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.CIFAR10(root='./data/', download=True, transform=transform)
    train_clip(dataset, manual_seed, options)