import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.optim import SGD

from Problem1.samplers import *
from Problem1.mlp import MLP
from Problem1.loss import *

# Example of usage
dist = iter(distribution1(0, 100))
samples = next(dist)


def train_jsd(mlp, loss_fn, optimizer, p_iterator, q_iterator, steps):
    with tqdm(total=steps) as t:
        for step in range(steps):
            # Sample from distributions and convert to tensor
            p = torch.from_numpy(next(p_iterator)).float()
            q = torch.from_numpy(next(q_iterator)).float()

            # Convert to torch Variables
            p = Variable(p)
            q = Variable(q)

            # Feed input to network
            p_output = mlp(p)
            q_output = mlp(q)

            # Train over distributions
            optimizer.zero_grad()
            loss = loss_fn(p_output, q_output)
            loss.backward()
            optimizer.step()

            t.set_postfix(loss='{:05.3f}'.format(loss.data))
            t.update()

    print(p_output)
    print(q_output)
    return loss.data


def train_wd(mlp, loss_fn, optimizer, p_iterator, q_iterator, steps):
    with tqdm(total=steps) as t:
        for step in range(steps):
            # Sample from distributions and convert to tensor
            p = torch.from_numpy(next(p_iterator)).float()
            q = torch.from_numpy(next(q_iterator)).float()

            # Convert to torch Variables
            p = Variable(p)
            q = Variable(q)

            # Feed input to network
            p_output = mlp(p)
            q_output = mlp(q)

            # Generate z
            a = np.random.uniform(0, 1)
            z = Variable(a * p + (1 - a) * q, requires_grad=True)

            # Feed z
            z_output = mlp(z)

            # Train over distributions
            optimizer.zero_grad()
            loss = loss_fn(p_output, q_output, z, z_output)
            loss.backward()
            optimizer.step()

            t.set_postfix(loss='{:05.3f}'.format(loss.data))
            t.update()

    return loss.data


def plot(x, y, title, save_name):
    fig, ax = plt.subplots()
    ax.plot(x, y, color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=12)

    ax.set(xlabel='$\phi$', ylabel='Estimate',
           title=title)
    ax.grid()

    fig.savefig(save_name)
    plt.show()


def density_estimation(model):
    # plot p0 and p1
    plt.figure()

    # empirical
    xx = torch.randn(10000)
    f = lambda x: torch.tanh(x * 2 + 1) + x * 0.75
    d = lambda x: (1 - torch.tanh(x * 2 + 1) ** 2) * 2 + 0.75
    plt.hist(f(xx), 100, alpha=0.5, density=1)
    plt.hist(xx, 100, alpha=0.5, density=1)
    plt.xlim(-5, 5)
    # exact
    xx = np.linspace(-5, 5, 1000)
    N = lambda x: np.exp(-x ** 2 / 2.) / ((2 * np.pi) ** 0.5)
    plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy() ** (-1) * N(xx))
    plt.savefig("histogram.png")
    plt.plot(xx, N(xx))

    r = model(torch.from_numpy(np.float32(xx)).unsqueeze(-1))
    r = np.squeeze(r.detach().cpu().numpy())
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(xx, r)
    plt.title(r'$D(x)$')

    estimate = N(xx) * r / (1 - r)

    plt.subplot(1, 2, 2)
    plt.plot(xx, estimate)
    plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy() ** (-1) * N(xx))
    plt.legend(['Estimated', 'True'])
    plt.title('Estimated vs True')
    plt.savefig("density_estimation.png")
    plt.show()


if __name__ == '__main__':
    # Code for problem 1.3 Jensen Shannon Divergence
    jsd_results = []

    # Loop over phi
    for phi in np.linspace(-1, 1, 21):
        mlp = MLP(2, 1, [32, 32, 32], 'relu', 'sigmoid')
        loss_fn = JensenShannonDivergence()
        optimizer = SGD(mlp.parameters(), lr=1e-3)
        p_iterator = iter(distribution1(0.0, 512))
        q_iterator = iter(distribution1(phi, 512))

        # Loop and save result
        loss = train_jsd(mlp, loss_fn, optimizer, p_iterator, q_iterator, 80000)
        jsd_results.append(-loss)

    plot(np.linspace(-1, 1, 21), jsd_results, 'Estimate for Jensen Shannon Divergence', "jsd.png")

    # Code for problem 1.3 Wassertein Distance
    wd_results = []

    # Loop over phi
    for phi in np.linspace(-1, 1, 21):
        mlp = MLP(2, 1, [32, 32, 32], 'relu', None)
        loss_fn = WasserteinDistance()
        optimizer = SGD(mlp.parameters(), lr=1e-3)
        p_iterator = iter(distribution1(0.0, 512))
        q_iterator = iter(distribution1(phi, 512))

        # Loop and save result
        loss = train_wd(mlp, loss_fn, optimizer, p_iterator, q_iterator, 10000)
        wd_results.append(-loss)

    plot(np.linspace(-1, 1, 21), wd_results, 'Estimate for Wassertein Distance', "wd.png")

    # Code for problem 1.4

    mlp = MLP(1, 1, [32, 32, 32], 'relu', 'sigmoid')
    loss_fn = JensenShannonDivergence(bias=1.0, cost=1.0)
    optimizer = SGD(mlp.parameters(), lr=1e-3)
    p_iterator = iter(distribution4(512))
    q_iterator = iter(distribution3(512))

    # Loop and save result
    loss = train_jsd(mlp, loss_fn, optimizer, p_iterator, q_iterator, 80000)
    density_estimation(mlp)



