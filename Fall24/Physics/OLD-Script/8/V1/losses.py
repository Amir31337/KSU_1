import torch
import config_and_data as c

def loss_max_likelihood(z, jacobian):
    """
    Computes the negative log-likelihood loss.

    Args:
        z (torch.Tensor): Latent variable after transformation.
        jacobian (torch.Tensor): Log-determinant of the Jacobian.

    Returns:
        torch.Tensor: Computed loss.
    """
    zz = torch.sum(z ** 2, dim=1)
    neg_log_likeli = 0.5 * zz - jacobian
    loss = c.lambd_max_likelihood * torch.mean(neg_log_likeli)
    return loss