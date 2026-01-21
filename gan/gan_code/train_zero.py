import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from tqdm import tqdm
import random

from utils import weights_init, compute_gan_loss


def sample_from_unit_sphere(shape, device):
    """
    Sample uniformly from the unit sphere
    Args:
        shape: Shape of the sampling vector
        device: Device (cpu/gpu)
    Returns:
        u: Random vector on the unit sphere (L2 norm = 1)
    """
    # Generate Gaussian random vector
    u = torch.randn(shape, device=device)
    # Calculate L2 norm
    norm = torch.norm(u.view(-1), p=2)
    # Normalize to unit sphere
    if norm > 1e-10:  # Avoid division by zero
        u = u / norm
    else:
        # Resample if norm is too small
        u = torch.randn(shape, device=device)
        norm = torch.norm(u.view(-1), p=2)
        u = u / norm
    return u

def approx_gradient(model, loss_fn, inputs, original_loss, mu, K, device):
    """
    Basic version of zeroth-order gradient estimation (Gaussian random sampling)
    Args:
        model: Model for gradient estimation (generator or discriminator)
        loss_fn: Loss function (used to calculate loss after perturbation)
        inputs: Input data required for loss calculation (varies by model type)
        original_loss: Loss under original parameters
        mu: Perturbation step size (μ1 or μ2)
        K: Number of sampling iterations (average to reduce variance)
        device: Computing device (cpu/gpu)
    Returns:
        approx_grads: List of approximate gradients matching the model parameter structure
    """
    # Initialize approximate gradients (same shape and device as model parameters)
    approx_grads = [torch.zeros_like(param, device=device) for param in model.parameters()]
    
    for _ in range(K):
        # 1. Sample independent standard Gaussian random vectors for each parameter
        # (Core difference: no need to normalize to unit sphere)
        u_list = [torch.randn_like(param, device=device) for param in model.parameters()]
        
        # 2. Apply perturbation to model parameters: w = w + μ*u
        for param, u in zip(model.parameters(), u_list):
            param.data.add_(mu * u)  # Temporarily modify parameters
        
        # 3. Calculate loss after perturbation
        perturbed_loss = loss_fn(*inputs)
        
        # 4. Restore original parameters (avoid perturbation affecting subsequent iterations)
        for param, u in zip(model.parameters(), u_list):
            param.data.sub_(mu * u)  # Restore parameters
        
        # 5. Accumulate gradient estimation: (f(w+μu) - f(w))/μ * u
        # Gaussian sampling version does not need to multiply by parameter dimension
        # (no unbiased correction for unit sphere)
        loss_diff = (perturbed_loss - original_loss) / mu
        for i, (u, grad) in enumerate(zip(u_list, approx_grads)):
            approx_grads[i] += loss_diff * u
    
    # 6. Average results over K sampling iterations (reduce variance)
    for grad in approx_grads:
        grad.div_(K)
    
    return approx_grads


def approx_gradient_sphere(model, loss_fn, inputs, original_loss, mu, K, device):
    """
    Approximate gradient of model parameters (Zeroth-order gradient estimation) - using unit sphere sampling
    Args:
        model: Model for gradient estimation (generator or discriminator)
        loss_fn: Loss function (used to calculate loss after perturbation)
        inputs: Input data required for loss calculation (varies by model type)
        original_loss: Loss under original parameters
        mu: Perturbation step size (μ1 or μ2)
        K: Number of sampling iterations (used to reduce variance by averaging)
        device: Device (cpu/gpu)
    Returns:
        approx_grads: List of approximate gradients matching the model parameter structure
    """
    # Initialize approximate gradients (same structure as model parameters)
    approx_grads = [torch.zeros_like(param) for param in model.parameters()]
    
    for _ in range(K):
        u_list = []
        # 1. Sample from unit sphere for each parameter
        for param in model.parameters():
            u = sample_from_unit_sphere(param.shape, device)
            u_list.append(u)
        
        # 2. Apply perturbation to model parameters: w + μ*u
        for param, u in zip(model.parameters(), u_list):
            param.data.add_(mu * u)  # Temporarily modify parameters
        
        # 3. Calculate loss after perturbation
        perturbed_loss = loss_fn(*inputs)
        
        # 4. Restore original parameters (avoid perturbation affecting subsequent calculations)
        for param, u in zip(model.parameters(), u_list):
            param.data.sub_(mu * u)  # Restore parameters
        
        # 5. Accumulate gradient approximation: (f(w+μu) - f(w))/μ * u * d
        # where d is the parameter dimension for unbiased estimation
        loss_diff = (perturbed_loss - original_loss) / mu
        for i, (u, grad) in enumerate(zip(u_list, approx_grads)):
            # Multiply by parameter dimension to get unbiased estimation
            param_dim = u.numel()
            approx_grads[i] += loss_diff * u * param_dim
        
    # 6. Average results over K sampling iterations
    for grad in approx_grads:
        grad.div_(K)
    
    return approx_grads


def approx_gradient_sphere_efficient(model, loss_fn, inputs, original_loss, mu, K, device):
    """
    Efficient version: Sample unit vectors for all parameters at once
    """
    # Initialize approximate gradients
    approx_grads = [torch.zeros_like(param) for param in model.parameters()]
    
    for _ in range(K):
        u_list = []
        # Sample unit vectors for each parameter
        for param in model.parameters():
            # Method 1: Direct sampling and normalization
            u = torch.randn_like(param, device=device)
            t = torch.rand(1, device=device) ** (1./param.numel())
            norm = torch.norm(u.view(-1), p=2)
            if norm > 1e-10:
                u = u / norm * t
            else:
                u = torch.ones_like(param, device=device) / torch.sqrt(torch.tensor(param.numel(), device=device)) 
            u_list.append(u)
            
            # Method 2: Use torch.nn.functional.normalize
            # u = torch.randn_like(param, device=device)
            # u = nn.functional.normalize(u.view(-1), p=2, dim=0).view_as(param)
            # u_list.append(u)
        
        # Apply perturbation
        for param, u in zip(model.parameters(), u_list):
            param.data.add_(mu * u)
        
        # Calculate perturbed loss
        perturbed_loss = loss_fn(*inputs)
        
        # Restore parameters
        for param, u in zip(model.parameters(), u_list):
            param.data.sub_(mu * u)
        
        # Accumulate gradients
        loss_diff = (perturbed_loss - original_loss) / mu
        for i, (u, grad) in enumerate(zip(u_list, approx_grads)):
            approx_grads[i] += loss_diff * u
    
    # Average
    for grad in approx_grads:
        grad.div_(K)
    
    return approx_grads


def train_zero(dataset, manual_seed, options):
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # Parse parameters
    model = options['model']
    loss_type = options['loss']
    data = options['data']
    lr = options['learning_rate']
    nz = options['nz']
    batch_size = options['batch_size']
    num_epochs = options['num_epochs']
    device = options['device']
    mu1 = options['mu1']
    mu2 = options['mu2']
    K = options['K']
    
    # Select gradient estimation method
    use_sphere_sampling = options.get('use_sphere_sampling', True)
    if use_sphere_sampling:
        approx_gradient_func = approx_gradient_sphere_efficient
    else:
        approx_gradient_func = approx_gradient  # Use original Gaussian sampling

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

    # Initialize parameter saving lists
    gen_param = []
    dis_param = []

    print('Training Zeroth-order SGDA with Sphere Sampling......')

    for epoch in range(num_epochs):
        sampler = RandomSampler(dataset, replacement=True, num_samples=len(dataset))
        train_loader = DataLoader(dataset, batch_sampler=BatchSampler(sampler, batch_size=batch_size, drop_last=False))

        epoch_gen_param = []
        epoch_dis_param = []
        losses_g = 0.0
        losses_d = 0.0

        for i, (images, _) in tqdm(enumerate(train_loader, 0), total=int(len(dataset)/batch_size)):
            images = images.to(device)
            b_size = images.size()[0]
            noises = torch.randn(b_size, nz, device=device)
            images_fake = generator(noises)

            ############################
            # (1) Update Discriminator
            ###########################
            def dis_loss_fn(real, fake, label_real, label_fake):
                output_real = discriminator(real)
                loss_real = compute_gan_loss(output_real, label_real, loss=loss_type)
                output_fake = discriminator(fake.detach())
                loss_fake = compute_gan_loss(output_fake, label_fake, loss=loss_type)
                return loss_real + loss_fake

            label_real = torch.full((b_size,), 1, dtype=images.dtype, device=device)
            label_fake = torch.full((b_size,), 0, dtype=images.dtype, device=device)
            original_loss_d = dis_loss_fn(images, images_fake, label_real, label_fake)

            dis_inputs = (images, images_fake, label_real, label_fake)
            g_v = approx_gradient_func(
                model=discriminator,
                loss_fn=dis_loss_fn,
                inputs=dis_inputs,
                original_loss=original_loss_d,
                mu=mu2,
                K=K,
                device=device
            )

            for param, grad in zip(discriminator.parameters(), g_v):
                param.data.sub_(lr * grad)

            ############################
            # (2) Update Generator
            ###########################
            def gen_loss_fn(noises, label):
                fake = generator(noises)
                output = discriminator(fake)
                return compute_gan_loss(output, label, loss=loss_type)

            label_gen = torch.full((b_size,), 1, dtype=images.dtype, device=device)
            original_loss_g = gen_loss_fn(noises, label_gen)

            gen_inputs = (noises, label_gen)
            g_w = approx_gradient_func(
                model=generator,
                loss_fn=gen_loss_fn,
                inputs=gen_inputs,
                original_loss=original_loss_g,
                mu=mu1,
                K=K,
                device=device
            )

            for param, grad in zip(generator.parameters(), g_w):
                param.data.sub_(lr * grad)

            # Record losses
            losses_d += original_loss_d.item()
            losses_g += original_loss_g.item()
            
            with torch.no_grad():
                D_x = discriminator(images).mean().item()
                D_G_z1 = discriminator(images_fake.detach()).mean().item()
                D_G_z2 = discriminator(generator(noises)).mean().item()

        print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, num_epochs, losses_d / i, losses_g / i, D_x, D_G_z1, D_G_z2))

        # Save parameters
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
    options['mu1'] = 0.01
    options['mu2'] = 0.01
    options['K'] = 10
    options['use_sphere_sampling'] = True  # Whether to use unit sphere sampling
    
    # Data loading
    if options['data'] == 'mnist':
        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.MNIST(root='./data/', download=True, transform=transform)
    elif options['data'] == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.CIFAR10(root='./data/', download=True, transform=transform)
    
    # Call training function
    train_zero(dataset, manual_seed, options)