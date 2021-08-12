import torch
import numpy as np

def kl_divergence(mu, std):#log_var):

    var = std ** 2
    log_var = torch.log(var)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - torch.exp(log_var), dim = 1 )
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - var, dim = 1 )
    return KLD.mean()

def log_prob_gauss(y_hat, mu, var, eps=1e-6):
    eps = torch.ones_like(var) * eps
    var = torch.maximum(eps, var)
    loss = 0.5 *  torch.sum(torch.log(var) + ((y_hat - mu)**2) / var, dim= 1) 
    return loss.mean()

def log_prob_gauss(y_hat, mu, var, eps=1e-6):
    eps = torch.ones_like(var) * eps
    var = torch.maximum(eps, var)
    loss = 0.5 *  torch.sum(torch.log(var) + ((y_hat - mu)**2) / var, dim= 1) 
    return loss.mean()

