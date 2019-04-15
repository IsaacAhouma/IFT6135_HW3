import argparse
import time
import collections
import os
import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from mnist_loader import get_data_loader
from vae import VAE

parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--lr', type=float, default=0.0003,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs')
parser.add_argument('--batch_size', type=int, default=64,
                    help='size of one minibatch')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

args = parser.parse_args()
argsdict = args.__dict__
argsdict['code_file'] = sys.argv[0]

# Use the model, optimizer, and the flags passed to the script to make the
# name for the experimental dir
print("\n########## Setting Up Experiment ######################")
flags = [flag.lstrip('--').replace('/', '').replace('\\', '') for flag in sys.argv[1:]]
experiment_path = os.path.join(args.save_dir + '_'.join([argsdict['model'],
                                                         argsdict['optimizer']]
                                                        + flags))

i = 0
while os.path.exists(experiment_path + "_" + str(i)):
    i += 1
experiment_path = experiment_path + "_" + str(i)

os.mkdir(experiment_path)
print("\nPutting log in %s" % experiment_path)
argsdict['save_dir'] = experiment_path
with open(os.path.join(experiment_path, 'exp_config.txt'), 'w') as f:
    for key in sorted(argsdict):
        f.write(key + '    ' + str(argsdict[key]) + '\n')

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")

train_loader, valid_loader, test_loader = get_data_loader("binarized_mnist", 64)

model = VAE()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
epochs = args.epochs

reconstruction_error = nn.MSELoss(size_average=False)

def loss_fn(x_tilde, x, mu, log_variance):
    error = reconstruction_error(x_tilde, x)  # E[log p(x|z)]
    D_KL = 0.5 * torch.sum(1 + log_variance - mu.pow(2) - log_variance.exp())
    ELBO = error - D_KL

    return ELBO


