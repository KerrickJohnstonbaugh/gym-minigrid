import numpy as np
import torch
import torch.nn as nn
from models.networks import MLP
import pytorch_lightning as pl


class Autoencoder(pl.LightningModule):
    def __init__(self, input_dim, z_dim=16,
                         hiddens=[128, 128],
                         act = 'tanh',
                         use_lin='regular',
                         lr=1e-6):
        super().__init__()
        self.save_hyperparameters()

        # create the encoder and decoder networks
        self.encoder = MLP(inputs = input_dim, hiddens=hiddens, out = z_dim , activation= act, lin_type='regular') 
        self.decoder = MLP(inputs = z_dim, hiddens=hiddens, out = input_dim, activation=act, lin_type=use_lin)

        self.z_dim = z_dim
        self.inputs = input_dim

        self.steps = 0.0
        self.lr = lr


    def encode(self, x):
        z = self.encoder(x)
        return z 

    def decode(self, z):
        inp = z
        return self.decoder(inp)


    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def calculate_loss(self, y, preds):
        #log_prob = log_prob_gauss(preds, y, torch.ones_like(preds))
        loss_fn = torch.nn.MSELoss()
        #loss = log_prob  
        loss = loss_fn(preds, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor =0.99)
        return {'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'train_loss'
                    }
                }

    def training_step(self, train_batch, batch_idx):
        inputs = train_batch

        preds, z = self.forward(inputs)

        loss = self.calculate_loss(inputs, preds)

        self.steps += 1
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs = val_batch

        preds, z = self.forward(inputs)
        loss = self.calculate_loss(inputs, preds)

        self.log('val_loss', loss)

        return loss


class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim, input_dim = [3, 256, 256]):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(input_dim[0], 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(self.lin_inp_dims(input_dim), 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )

    def lin_inp_dims(self, input_dim):
        return (input_dim[1]//8) * (input_dim[2]//8) * 32
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim,input_dim = [3, 256, 256]):
        super().__init__()
        self.i_dim = [input_dim[1]//8, input_dim[2]//8] # dimenion at interface between conv and linear sections
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, self.i_dim[0] * self.i_dim[1] * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, self.i_dim[0], self.i_dim[1]))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=2, 
            padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

    
class ConvAutoencoder(pl.LightningModule):
    def __init__(self, input_dim, z_dim=16,
                         lr=1e-6):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(encoded_space_dim = z_dim)
        self.decoder = Decoder(encoded_space_dim = z_dim)
        self.steps = 0.0

        self.input_dim = input_dim
        self.lr = lr

    def encode(self, x):
        z = self.encoder(x)
        return z 

    def decode(self, z):
        inp = z
        return self.decoder(inp)


    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def calculate_loss(self, preds, y):
        #log_prob = log_prob_gauss(preds, y, torch.ones_like(preds))
        loss_fn = torch.nn.MSELoss()
        #loss = log_prob  
        loss = loss_fn(preds, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor =0.99)
        return {'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'train_loss'
                    }
                }

    def training_step(self, train_batch, batch_idx):
        inputs = train_batch

        preds, z = self.forward(inputs)

        loss = self.calculate_loss(preds, inputs)

        self.steps += 1
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs = val_batch

        preds, z = self.forward(inputs)
        loss = self.calculate_loss(preds, inputs)

        self.log('val_loss', loss)

        return loss
