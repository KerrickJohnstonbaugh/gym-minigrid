import argparse
from models.utils import get_model, init_weights
from image_utils import ImageDataset
from torch.utils.data import DataLoader,random_split
import time 
import pytorch_lightning as pl
from pytorch_lightning import Trainer 
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import yaml

def main(args):
    dataset = ImageDataset(datapath=args.train_pth)

    m = len(dataset)
    train_data, val_data = random_split(dataset, [4096-256, 256]) # make auto

    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size = len(val_data), shuffle=False)

    if args.model == 'ae':
        inp_dim = np.prod(train_data[0].shape)
    elif args.model == 'convae':
        inp_dim = train_data[0].shape
    model = get_model(args = args, input_dim = inp_dim, model=args.model)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=None,
        filename='model-{epoch:02d}-{val_loss:.6f}',
        save_top_k=3,
        mode='min'
        )
    trainer = Trainer.from_argparse_args(args,
        logger=True,
        callbacks=[checkpoint_callback] )
    trainer.fit(model, train_loader, val_loader)
    with open(trainer.checkpoint_callback.dirpath + '/args.yaml', 'w') as f:
        print(trainer.checkpoint_callback.dirpath)
        arg_dict = vars(args)
        yaml.dump(arg_dict, f)

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")

    parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--train_pth", default="data/5x5-random.h5",
            help="train data set json file")
    parser.add_argument("--val_pth", default='data/5x5-random.h5',
            help="validation dataset")
    parser.add_argument('--activation')
    parser.add_argument('--hiddens', default=[256,256], type= lambda x: [int(x) for x in x.split(',')],
            help="hidden layers in encoders and decoders")
    parser.add_argument('--z_dim', default=16, type=int,
            help="latent action dimensions")
    parser.add_argument('--lr', default=1e-3, type=float,
            help="learning rate for optimizer")
    parser.add_argument('--min_lr', default=1e-5, type=float,
            help="minimal learning rate")
    parser.add_argument("--use_lin", default='regular', type=str,
            help="linear layer type")
    parser.add_argument('--model', default='ae', type=str,
           help="model to run: ae|convae")
    parser.add_argument('-n', '--num-epochs', default=501, type=int, help='number of training epochs')
    parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda')
    parser.add_argument('--jit', action='store_true', default=False, help='whether to use PyTorch jit')
    parser.add_argument('--batch_size', type=int, default=256, help='training batch size')

    args = parser.parse_args()
    print(args)
    model = main(args)
