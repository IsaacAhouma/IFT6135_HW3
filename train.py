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
parser.add_argument('--epochs', type=int, default=20,
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
    data = []
    for splitname in ["train", "valid", "test"]:
        filename = "binarized_mnist_%s.amat" % splitname
        filepath = os.path.join(dataset_location, filename)
        utils.download_url(URL + filename, dataset_location)
        with open(filepath) as f:
            lines = f.readlines()
        #f.close()
        x = lines_to_np_array(lines).astype('float32')
        x = x.reshape(x.shape[0], 1, 28, 28)
        y = np.zeros((x.shape[0], 1)).astype('float32')
        # pytorch data loader
        data.append(torch.from_numpy(x))
        dataset = data_utils.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        dataset_loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=splitname == "train")
        splitdata.append(dataset_loader)
    return splitdata, data


(train_loader, valid_loader, test_loader), (train_data, valid_data, test_data) = get_data_loader("binarized_mnist", 64)

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

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
epochs = args.epochs

if not os.path.isdir('results'):
    os.mkdir('results')

# Training ###########


def loss_fn(x_tilde, x, mu, log_variance):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    x = x.reshape(x.shape[0], -1)
    x_tilde = x_tilde.reshape(x_tilde.shape[0], -1)
    reconstruction_error = -F.binary_cross_entropy_with_logits(x_tilde, x, reduction='none').sum(dim=-1)  # E[log p(x|z)]
    D_KL = -0.5 * (1 + log_variance - mu.pow(2) - log_variance.exp()).sum(dim=-1)
    ELBO = (reconstruction_error - D_KL).mean()
    loss = -ELBO
    return loss


def train(epoch):
    model.train()
    train_elbo = 0
    num_minibatches = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        num_minibatches += 1
        # data = values[1]
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_fn(recon_batch, data, mu, logvar)
        loss.backward()
        train_elbo += -loss.item()
        optimizer.step()

    print('====> Epoch: {} TRAINING ELBO: {:.4f}'.format(
          epoch, train_elbo / num_minibatches))


def evaluate_elbo(data_name='valid'):
    model.eval()
    elbo = 0
    num_minibatches = 0

    if data_name == 'test':
        data_loader = test_loader
    else:
        data_loader = valid_loader

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            num_minibatches += 1
            # data = values[1]
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            elbo += -loss_fn(recon_batch, data, mu, logvar).item()

    elbo /= num_minibatches
    if data_name == 'test':
        print('====> Test set ELBO: {:.4f}'.format(elbo))
    else:
        print('====> Validation set ELBO: {:.4f}'.format(elbo))


def importance_sampling(model, X, Z):
    """
    :param model: A nn.Module object
    :param X: an array of shape (M, D)
    :param Z: a tensor of shape (M, K, L)
    :return: a vector of length M
    """
    model.eval()
    X = X.to(device)
    Z = Z.to(device)
    (M, D) = X.size()
    (_, K, L) = Z.size()
    x = X.view(M, 1, 28, 28)
    mu, logvar = model.encode(x)
    sigma = torch.exp(logvar * 0.5)
    recon_x = model.decode(Z.view(-1, L))
    recon_x = recon_x.view(M, K, D)
    x = X
    x = x.unsqueeze(1).expand(M, K, D)
    mu = mu.unsqueeze(1).expand(M, K, L)
    sigma = sigma.unsqueeze(1).expand(M, K, L)
    log_p_x_given_z = -F.binary_cross_entropy_with_logits(recon_x, x, reduction='none').sum(dim=-1)
    log_p_z = - 0.5 * L * np.log(2 * np.pi) - torch.norm(Z, dim=-1)**2
    log_q_z_given_x = (-0.5 * L * np.log(2 * np.pi)) + (-0.5 * torch.log(sigma**2).sum(dim=-1)) \
                  + (-0.5 * torch.norm((Z - mu) / sigma, dim=-1)**2)
    log_p = log_p_x_given_z + log_p_z - log_q_z_given_x
    log_likelihood = np.log(1 / K) + torch.logsumexp(log_p, dim=1)

    return log_likelihood

def evaluate_importance_sampling(data_name='valid'):
    model.eval()
    log_likelihood = 0
    num_minibatches = 0

    if data_name == 'test':
        data_loader = test_loader
    else:
        data_loader = valid_loader

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            num_minibatches += 1
            data = data.to(device)
            z = generate_z(data)
            x = data.view(data.size(0), -1)
            log_likelihood += importance_sampling(model, x, z).mean(dim=0)

    log_likelihood /= num_minibatches
    if data_name == 'test':
        print('====> Test set Log-Likelihood: {:.4f}'.format(log_likelihood))
    else:
        print('====> Validation set Log-Likelihood: {:.4f}'.format(log_likelihood))


def generate_z(x, k=200, latent_dim=100, data_name='valid'):
    K = k
    (M, C, H, W) = x.size()
    D = H * W
    # z = torch.empty(M, k, latent_dim)
    mu, logvar = model.encode(x)
    mu = mu.unsqueeze(1).expand(M, K, latent_dim)
    logvar = logvar.unsqueeze(1).expand(M, K, latent_dim)
    z = model.reparameterize(mu, logvar)

    return z

# def generate_z(_model, k=200, latent_dim=100, data_name='valid'):
#     if data_name == 'test':
#         data = test_data
#     else:
#         data = valid_data
#
#     (N, _, _, _) = data.size()
#     z = torch.empty(N, k, latent_dim)
#     for j in range(N):
#         mu, logvar = _model.encode(data[j].view(1, -1))
#         mu = mu.unsqueeze(-1).expand(200)
#         logvar = logvar.unsqueeze(-1).expand(200)
#         z[j] = _model.reparameterize(mu, logvar)
#
#     return z

if __name__ == "__main__":
    for epoch in range(1, args.epochs):
        train(epoch)
        evaluate_elbo()
    # Z_valid = generate_z(model)
    # log_likelihood_valid = importance_sampling(model, valid_data.view(-1, 784), Z_valid).mean(dim=0)
    # print('====> Valid set log likelihood: {:.4f}'.format(log_likelihood_valid))
    evaluate_importance_sampling()
    evaluate_elbo()

    # Z_test = generate_z(model, data_name='test')
    # log_likelihood_test = importance_sampling(model, test_data.view(-1, 784), Z_test).mean(dim=0)
    # print('====> Test set log likelihood: {:.4f}'.format(log_likelihood_test))
    evaluate_importance_sampling(data_name='test')
    evaluate_elbo(data_name='test')
