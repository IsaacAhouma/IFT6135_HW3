import argparse
import sys
import torch.nn.functional as F
from torchvision.datasets import utils
import torch.utils.data as data_utils
import torch
import os
import numpy as np
from vae import VAE


parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--lr', type=float, default=0.0003,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=5,
                    help='number of epochs')
parser.add_argument('--batch_size', type=int, default=64,
                    help='size of one minibatch')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--model', type=str, default='VAE',
                    help='VAE')
parser.add_argument('--optimizer', type=str, default='ADAM',
                    help='optimization algo to use; ADAM')

parser.add_argument('--save_dir', type=str, default='',
                    help='path to save the experimental config, logs, model \
                    This is automatically generated based on the command line \
                    arguments you pass and only needs to be set if you want a \
                    custom dir name')

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


# DATA ####


def get_data_loader(dataset_location, batch_size):
    URL = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
    # start processing
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])
    splitdata = []
    for splitname in ["train", "valid", "test"]:
        filename = "binarized_mnist_%s.amat" % splitname
        filepath = os.path.join(dataset_location, filename)
        utils.download_url(URL + filename, dataset_location)
        with open(filepath) as f:
            lines = f.readlines()
        x = lines_to_np_array(lines).astype('float32')
        x = x.reshape(x.shape[0], 1, 28, 28)
        # pytorch data loader
        dataset = data_utils.TensorDataset(torch.from_numpy(x))
        dataset_loader = data_utils.DataLoader(x, batch_size=batch_size, shuffle=splitname == "train")
        splitdata.append(dataset_loader)
    return splitdata


train, valid, test = get_data_loader("binarized_mnist", 64)

#### MODEL ####

model = VAE()

# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")

# train_loader, valid_loader, test_loader = get_data_loader("binarized_mnist", 64)

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
epochs = args.epochs

# see Appendix B from VAE paper:
# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
# https://arxiv.org/abs/1312.6114

if not os.path.isdir('/results'):
    os.mkdir('/results')


def loss_fn(x_tilde, x, mu, log_variance):
    x = x.reshape(x.shape[0], -1)
    x_tilde = x_tilde.reshape(x_tilde.shape[0], -1)
    reconstruction_error = -F.binary_cross_entropy_with_logits(x_tilde, x, reduction='none').sum(dim=-1)  # E[log p(x|z)]
    # reconstruction_error = (x * torch.log(x_tilde) + (1 - x) * torch.log(1 - x_tilde)).sum(dim=-1)
    # print('reconstruction error:', reconstruction_error)
    # D_KL = -0.5 * torch.sum(1 + log_variance - mu.pow(2) - log_variance.exp())
    D_KL = -0.5 * (1 + log_variance - mu.pow(2) - log_variance.exp()).sum(dim=-1)
    # print('KL Divergence:', D_KL)
    ELBO = (reconstruction_error - D_KL).mean()
    # print('elbo:', ELBO)
    loss = -ELBO
    return loss


########################
def train_elbo(epoch):
    model.train()
    train_elbo = 0
    num_minibatches = 0
    for values in enumerate(train_loader):
        num_minibatches += 1
        batch_idx = values[0]
        data = values[1]
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_fn(recon_batch, data, mu, logvar)
        loss.backward()
        train_elbo += -loss.item()
        optimizer.step()
        # if batch_idx % 5 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader),
        #         loss.item() / len(data)))

    # print('====> Epoch: {} Average loss: {:.4f}'.format(
    #       epoch, train_loss / len(train_loader.dataset)))
    print('====> Epoch: {} TRAINING ELBO: {:.4f}'.format(
          epoch, train_elbo / num_minibatches))


def valid_elbo(epoch):
    model.eval()
    test_elbo = 0
    num_minibatches = 0
    with torch.no_grad():
        for values in enumerate(valid_loader):
            i = values[0]
            num_minibatches += 1
            data = values[1]
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_elbo += -loss_fn(recon_batch, data, mu, logvar).item()
            # if i == 0:
            #     n = min(data.size(0), 8)
            #     comparison = torch.cat([data[:n],
            #                           recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            #     save_image(comparison.cpu(),
            #              'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    # test_loss /= len(valid_loader.dataset)
    # print('====> Test set loss: {:.4f}'.format(test_loss))
    test_elbo /= num_minibatches
    print('====> Validation set ELBO: {:.4f}'.format(test_elbo))


for epoch in range(1, args.epochs + 1):
    train_elbo(epoch)
    valid_elbo(epoch)
# if __name__ == "__main__":
#     for epoch in range(1, args.epochs + 1):
#         train_elbo(epoch)
#         valid_elbo(epoch)

