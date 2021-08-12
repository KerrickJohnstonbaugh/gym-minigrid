from models.autoencoder import Autoencoder, ConvAutoencoder
import torch.nn as nn

def get_model(args, input_dim, model):
    if model == 'ae':
        return Autoencoder(input_dim = input_dim,
            z_dim=args.z_dim, hiddens=args.hiddens,
            use_lin=args.use_lin, lr=args.lr)
    elif model == 'convae':
        return ConvAutoencoder(input_dim, z_dim=args.z_dim, lr = args.lr)
    

def get_model_class(model):
    if model == 'ae':
        return Autoencoder
    elif model == 'convae':
        return ConvAutoencoder

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=1e-3)
        m.bias.data.fill_(1e-3)
