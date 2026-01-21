import pickle as pkl
from matplotlib import pyplot as plt
import torch


with open('res/dcgan_wgan_mnist_fro_clip.pkl', 'rb') as f:
    results_clip = pkl.load(f)


with open('res/dcgan_wgan_mnist_fro_zero.pkl', 'rb') as f:
    results_zero = pkl.load(f)

with open('res/dcgan_wgan_mnist_fro_sgda.pkl', 'rb') as f:
    results_sgda = pkl.load(f)


tex_fonts = {
        "text.usetex": False,
        "font.family": "serif",
        "axes.labelsize": 16,
        "font.size": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    }
plt.rcParams.update(tex_fonts)
    

# 创建图形，设置1行2列的子图布局
fig, axes = plt.subplots(1, 2, figsize=(20, 7))

# 获取epochs数据
epochs = torch.arange(results_clip['options']['num_epochs']) + 1

# 第一个子图：Generator结果
ax1 = axes[0]

ax1.errorbar(epochs, torch.sum(results_sgda['gen']['mean'], 0) / 4, 
             yerr=torch.sum(results_sgda['gen']['std'], 0) / 4, 
             color='navy', linestyle='--', linewidth=1.5, capsize=4,
             markeredgewidth=1, elinewidth=2, marker='o', label='SGDA')
ax1.errorbar(epochs, torch.sum(results_clip['gen']['mean'], 0) / 4, 
             yerr=torch.sum(results_clip['gen']['std'], 0) / 4, 
             color='maroon', linestyle='-', linewidth=1.5, capsize=4,
             markeredgewidth=1, elinewidth=2, marker='s', label='Cliped SGDA')

ax1.errorbar(epochs, torch.sum(results_zero['gen']['mean'], 0) / 4 , 
             yerr=torch.sum(results_zero['gen']['std'], 0) / 4, 
             color='orange', linestyle='-.', linewidth=1.5, capsize=4,
             markeredgewidth=1, elinewidth=2, marker='*', label='Zeroth-Order SGDA')


ax1.legend(loc='lower right', frameon=True, fontsize=18)
ax1.set_xlabel('Number of Passes', fontsize=20)
ax1.set_ylabel('Euclidean Distance', fontsize=20)
ax1.set_title('Generator', fontsize=20)
ax1.grid(linestyle='--', alpha=0.5)


# 第二个子图：Discriminator结果
ax2 = axes[1]

ax2.errorbar(epochs, torch.sum(results_sgda['dis']['mean'], 0) / 4, 
             yerr=torch.sum(results_sgda['dis']['std'], 0) / 4, 
             color='navy', linestyle='--', linewidth=1.5, capsize=4,
             markeredgewidth=1, elinewidth=2, marker='o', label='SGDA')
ax2.errorbar(epochs, torch.sum(results_clip['dis']['mean'], 0) / 4, 
             yerr=torch.sum(results_clip['dis']['std'], 0) / 4, 
             color='maroon', linestyle='-', linewidth=1.5, capsize=4,
             markeredgewidth=1, elinewidth=2, marker='s', label='Clipped SGDA')
ax2.errorbar(epochs, torch.sum(results_zero['dis']['mean'], 0) / 4, 
             yerr=torch.sum(results_zero['dis']['std'], 0) / 4, 
             color='orange', linestyle='-.', linewidth=1.5, capsize=4,
             markeredgewidth=1, elinewidth=2, marker='*', label='Zeroth-Order SGDA')



ax2.legend(loc='lower right', frameon=True, fontsize=18)
ax2.set_xlabel('Number of Passes', fontsize=20)
#ax2.set_ylabel('Euclidean Distance', fontsize=20)
ax2.set_title('Discriminator', fontsize=20)

ax1.grid(linestyle='--', alpha=0.5)
ax2.grid(linestyle='--', alpha=0.5)

ax1.semilogy()
ax2.semilogy()


plt.savefig('stability_gan.png', bbox_inches='tight', dpi=300)