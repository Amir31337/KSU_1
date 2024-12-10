"""Global configuration for the experiments"""
import torch
import numpy as np
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet, ConditionNode
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

######################
#  General settings  #
######################

lambda_mse = 0.1
grad_clamp = 15
eval_test = 10 # crystal small data issue, we have to use smaller one

# Compute device to perform the training on, 'cuda' or 'cpu'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#######################
#  Training schedule  #
#######################

# Initial learning rate
lr_init = 1.0e-3
# Batch size
batch_size = 500
# Total number of epochs to train for
n_epochs = 1000
# End the epoch after this many iterations (or when the train loader is exhausted)
pre_low_lr = 0
# Decay exponentially each epoch, to final_decay*lr_init at the last epoch.
final_decay = 0.02
# L2 weight regularization of model parameters
l2_weight_reg = 1e-5
# Parameters beta1, beta2 of the Adam optimizer
adam_betas = (0.9, 0.95)

#####################
#  Data dimensions  #
#####################

ndim_x = 7
ndim_pad_x = 0

ndim_y = 1
ndim_z = 6
ndim_pad_zy = 0

train_forward_mmd = True
train_backward_mmd = True
train_reconstruction = False
train_max_likelihood = False

lambd_fit_forw = 1
lambd_mmd_forw = 1
lambd_reconstruct = 1
lambd_mmd_back = 1
lambd_max_likelihood = 1

# Both for fitting, and for the reconstruction, perturb y with Gaussian
# noise of this sigma
add_y_noise = 5e-3
# For reconstruction, perturb z
add_z_noise = 2e-3
# In all cases, perturb the zero padding
add_pad_noise = 1e-3
# MLE loss
zeros_noise_scale = 5e-3

# For noisy forward processes, the sigma on y (assumed equal in all dimensions).
# This is only used if mmd_back_weighted of train_max_likelihoiod are True.
y_uncertainty_sigma = 0.12 * 4

mmd_forw_kernels = [(0.2, 2), (1.5, 2), (3.0, 2)]
mmd_back_kernels = [(0.2, 0.1), (0.2, 0.5), (0.2, 2)]
mmd_back_weighted = False

###########
#  Model  #
###########

# Initialize the model parameters from a normal distribution with this sigma
init_scale = 0.10
#
N_blocks = 6
#
exponent_clamping = 2.0
#
hidden_layer_sizes = 256
#
use_permutation = True
#
verbose_construction = False

def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, hidden_layer_sizes), nn.ReLU(),
                         nn.Linear(hidden_layer_sizes, c_out))

# Set up the conditional node (y)
cond_node = ConditionNode(ndim_y)
# Start from input layer
nodes = [InputNode(ndim_x, name='input')]

for k in range(N_blocks):
    nodes.append(Node(nodes[-1],
                      GLOWCouplingBlock,
                      {'subnet_constructor': subnet_fc, 'clamp': 2.0},
                      conditions=cond_node,
                      name=F'coupling_{k}'))
    nodes.append(Node(nodes[-1],
                      PermuteRandom,
                      {'seed': k},
                      name=F'permute_{k}'))

nodes.append(OutputNode(nodes[-1], name='output'))
nodes.append(cond_node)
model = ReversibleGraphNet(nodes, verbose=False)
model.to(device)

params_trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
for p in params_trainable:
    p.data = init_scale * torch.randn(p.data.shape).to(device)

gamma = (final_decay) ** (1. / n_epochs)
optim = torch.optim.Adam(params_trainable, lr=lr_init, betas=adam_betas, eps=1e-6, weight_decay=l2_weight_reg)

weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=gamma)

def scheduler_step():
    weight_scheduler.step()

def MMD_matrix_multiscale(x, y, widths_exponents):
    xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = torch.clamp(rx.t() + rx - 2. * xx, 0, np.inf)
    dyy = torch.clamp(ry.t() + ry - 2. * yy, 0, np.inf)
    dxy = torch.clamp(rx.t() + ry - 2. * xy, 0, np.inf)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    for C, a in widths_exponents:
        XX += C ** a * ((C + dxx) / a) ** -a
        YY += C ** a * ((C + dyy) / a) ** -a
        XY += C ** a * ((C + dxy) / a) ** -a

    return XX + YY - 2. * XY

def l2_dist_matrix(x, y):
    xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    return torch.clamp(rx.t() + ry - 2. * xy, 0, np.inf)

def forward_mmd(y0, y1):
    return MMD_matrix_multiscale(y0, y1, mmd_forw_kernels)

def backward_mmd(x0, x1):
    return MMD_matrix_multiscale(x0, x1, mmd_back_kernels)

def l2_fit(input, target):
    return torch.sum((input - target) ** 2) / batch_size

## specific loss terms
def noise_batch(ndim):
    return torch.randn(batch_size, ndim).to(device)

def loss_max_likelihood(out, y):
    jac = model.jacobian(run_forward=False)

    neg_log_likeli = (0.5 / y_uncertainty_sigma ** 2 * torch.sum((out[:, -ndim_y:] - y[:, -ndim_y:]) ** 2, 1)
                      + 0.5 / zeros_noise_scale ** 2 * torch.sum((out[:, ndim_z:-ndim_y] - y[:, ndim_z:-ndim_y]) ** 2, 1)
                      + 0.5 * torch.sum(out[:, :ndim_z] ** 2, 1)
                      - jac)

    return lambd_max_likelihood * torch.mean(neg_log_likeli)

def loss_forward_fit_mmd(out, y):
    # Shorten output, and remove gradients wrt y, for latent loss
    output_block_grad = torch.cat((out[:, :ndim_z], out[:, -ndim_y:].data), dim=1)
    y_short = torch.cat((y[:, :ndim_z], y[:, -ndim_y:]), dim=1)

    l_forw_fit = lambd_fit_forw * l2_fit(out[:, ndim_z:], y[:, ndim_z:])
    l_forw_mmd = lambd_mmd_forw * torch.mean(forward_mmd(output_block_grad, y_short))

    return l_forw_fit, l_forw_mmd

def loss_backward_mmd(x, y):
    x_samples = model(y, rev=True)
    MMD = backward_mmd(x, x_samples)
    if mmd_back_weighted:
        MMD *= torch.exp(- 0.5 / y_uncertainty_sigma ** 2 * l2_dist_matrix(y, y))
    return lambd_mmd_back * torch.mean(MMD)

def loss_reconstruction(out_y, y, x):
    cat_inputs = [out_y[:, :ndim_z] + add_z_noise * noise_batch(ndim_z)]
    if ndim_pad_zy:
        cat_inputs.append(out_y[:, ndim_z:-ndim_y] + add_pad_noise * noise_batch(ndim_pad_zy))
    cat_inputs.append(out_y[:, -ndim_y:] + add_y_noise * noise_batch(ndim_y))

    x_reconstructed = model(torch.cat(cat_inputs, 1), rev=True)
    return lambd_reconstruct * l2_fit(x_reconstructed, x)

cINN_method = 1

rng = 0
data_x = pd.read_csv('../Simulated_DataSets/MoS2/data_x.csv', header=None).values
data_y = pd.read_csv('../Simulated_DataSets/MoS2/data_y.csv', header=None).values

xtrain, xtest, ytrain, ytest = train_test_split(data_x, data_y, test_size=0.2, random_state=rng)

x_train = torch.tensor(xtrain, dtype=torch.float)
y_train = torch.tensor(ytrain, dtype=torch.float)
x_test = torch.tensor(xtest, dtype=torch.float)
y_test = torch.tensor(ytest, dtype=torch.float)

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test, y_test),
    batch_size=batch_size, shuffle=False, drop_last=True)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=batch_size, shuffle=True, drop_last=True)

# training function
def train_epoch():
    """
    The major training function. This would start the training using information given in the flags
    :return: None
    """
    print(model)
    print("Starting training now")

    for epoch in range(n_epochs):
        # Set to Training Mode
        train_loss = 0
        train_mse_y_loss = 0
        train_mmd_x_loss = 0

        model.train()
        loss_factor = min(1., 2. * 0.002 ** (1. - (float(epoch) / n_epochs)))

        train_loss_history = []
        for j, (x, y) in enumerate(train_loader):

            batch_losses = []

            x, y = Variable(x).to(device), Variable(y).to(device)

            ######################
            #  Forward step      #
            ######################
            optim.zero_grad()

            if cINN_method:
                z = model(x, y)
                zz = torch.sum(z ** 2, dim=1)
                jac = model.log_jacobian(run_forward=False)  # get the log jacobian
                neg_log_likeli = 0.5 * zz - jac
                loss_total = torch.mean(neg_log_likeli)
                loss_total.backward()

            ######################
            #  Gradient Clipping #
            ######################
            for parameter in model.parameters():
                parameter.grad.data.clamp_(-grad_clamp, grad_clamp)

            #########################
            # Descent your gradient #
            #########################
            optim.step()  # Move one step the optimizer

            # MLE training
            train_loss += loss_total

        # Calculate the avg loss of training
        train_avg_loss = train_loss.cpu().data.numpy() / (j + 1)

        if epoch % eval_test == 0:
            model.eval()
            print("Doing Testing evaluation on the model now")

            test_loss = 0
            test_mse_y_loss = 0
            test_mmd_x_loss = 0

            test_loss_history = []
            for j, (x, y) in enumerate(test_loader):

                batch_losses = []

                x, y = Variable(x).to(device), Variable(y).to(device)

                # ######################
                # #  Forward step      #
                # ######################
                optim.zero_grad()
                #
                z = model(x, y)
                zz = torch.sum(z ** 2, dim=1)
                jac = model.log_jacobian(run_forward=False)  # get the log jacobian
                neg_log_likeli = 0.5 * zz - jac
                loss_total = torch.mean(neg_log_likeli)

                # MLE training
                # print(loss_total)
                test_loss += loss_total

            # Calculate the avg loss of training
            test_avg_loss = test_loss.cpu().data.numpy() / (j + 1)
            print("This is Epoch %d, training loss %.5f, testing loss %.5f" % (epoch, train_avg_loss, test_avg_loss))

        scheduler_step()

def generate_samples(filename, n_samps=1000, y0=1.0):
    model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

    y_fix = np.zeros((n_samps, 1)) + y0
    y_fix = torch.tensor(y_fix, dtype=torch.float)
    y_fix = y_fix.to(device)

    dim_z = ndim_z
    z_fix = torch.randn(n_samps, dim_z, device=device)
    rev_x0 = model(z_fix, y_fix, rev=True).cpu().data.numpy()

    fname = 'gen_samps.csv'
    np.savetxt(fname, rev_x0, fmt='%.6f', delimiter=',')

if __name__ == "__main__":
    train_epoch()
    torch.save(model.state_dict(), 'MoS2_cinn.pkl')

    generate_samples('MoS2_cinn.pkl')