import torch
from stability import exp_stability
import os
from itertools import product
import pickle as pkl

import argparse

parser = argparse.ArgumentParser(description='Training setting')
parser.add_argument('--model', default='dcgan', choices={'vgan', 'dcgan'}, help='GAN structure (default: vgan)')
# nsgan is non-saturating gan
parser.add_argument('--loss', default='wgan', choices={'bce', 'gan', 'wgan'}, help='GAN loss (default: gan)')
parser.add_argument('--data', default='mnist', choices={'mnist', 'cifar10'}, help='Dataset (default: mnist)')
parser.add_argument('--metric', default='fro', choices={'fro', 'ned'}, help='Distance metric')
parser.add_argument('--method', default='sgda', choices={'sgda', 'clip', 'zero'}, help='Algorithm type')  # 改动1
args = parser.parse_args()

data_path = os.path.join(os.getcwd(), 'data')
if not os.path.exists(data_path):
    os.mkdir(data_path)

# learning parameters
options = dict()
options['model'] = args.model
options['loss'] = args.loss
options['data'] = args.data
options['metric'] = args.metric
options['learning_rate'] = 0.0002
options['batch_size'] = 2000 # large batch size for GPU can speed up
options['num_epochs'] = 3
options['nz'] = 128 # latent vector size
options['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
options['method'] = args.method

# Zeroth-order SGDA additional parameters
options['mu1'] = 0.0001
options['mu2'] = 0.0001 
options['K'] = 784 

# Clipping parameter for gradient clipping method
options['clip_value'] = 1


print('device: %s' % options['device'])

# data parameters
manual_seed_list = [1, 123, 58317, 962548, 20457563]
remove_index_list = [7, 48, 321, 4592, 26195]
# generator_seed_list = [3, 864, 77777, 9345634, 2145483647]
# manual_seed_list = manual_seed_list[:2]
# remove_index_list = remove_index_list[:2]
# generator_seed_list = generator_seed_list[:2]

# initialize saving
num_cv = len(manual_seed_list) * len(remove_index_list)
gen_diff_cv = torch.zeros(num_cv, 4, options['num_epochs']) # 4 is the number of layers
dis_diff_cv = torch.zeros(num_cv, 4, options['num_epochs'])

# cross validation
for i, (manual_seed, remove_index) in enumerate(product(manual_seed_list, remove_index_list)):
    print('Current cross validation number: %d / Total: %d' %(i+1, num_cv))
    gen_diff, dis_diff = exp_stability(remove_index, manual_seed, options)
    gen_diff_cv[i] = gen_diff
    dis_diff_cv[i] = dis_diff

results_gen = dict()
results_dis = dict()
results_gen['mean'] = torch.mean(gen_diff_cv,0)
results_gen['std'] = torch.std(gen_diff_cv,0)
results_dis['mean'] = torch.mean(dis_diff_cv,0)
results_dis['std'] = torch.std(dis_diff_cv,0)
results = dict()
results['gen'] = results_gen
results['dis'] = results_dis
results['options'] = options


# saving
res_path = os.path.join(os.getcwd(), 'res')
if not os.path.exists(res_path):
    os.mkdir(res_path)
f = open(os.path.join(res_path, '%s_%s_%s_%s_%s.pkl'
                      % (args.model, args.loss, args.data, args.metric, args.method)), 'wb')  # 改动4
pkl.dump(results, f)
f.close()