import argparse
from models.utils import get_model, get_model_class
from image_utils import ImageDataset
from torch.utils.data import DataLoader,random_split
from torch import unsqueeze
import time 
import pytorch_lightning as pl
from pytorch_lightning import Trainer 
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import glob
import yaml

def main(args):

    folder = args.exp_folder
    print("Running results in following folder:")
    print(folder)


    #choose best checkpointed model for creating video
    models = glob.glob(folder + '/*.ckpt')
    best = sorted(models, key= lambda x: float(x.split('val_loss=')[1].split('.ckpt')[0]), reverse=False)[0]
    #TODO: for future experiments, get from arg.yaml file
    model = get_model_class('convae').load_from_checkpoint(checkpoint_path=best)

    dataset = ImageDataset(datapath=args.train_pth)

    m = len(dataset)
    train_data, val_data = random_split(dataset, [4096-256, 256])

    import matplotlib.pyplot as plt


    i = 0
    
    fig, axs = plt.subplots(nrows=2,ncols=5)
    for i in range(5):
        input = val_data[i]
        output, _ = model(unsqueeze(input,0))
        axs[0,i].imshow(input.numpy().transpose((1,2,0)))
        axs[1,i].imshow(output.detach().numpy().squeeze().transpose((1,2,0)))

        axs[0,i].xaxis.set_ticks([])
        axs[0,i].yaxis.set_ticks([])

        axs[1,i].xaxis.set_ticks([])
        axs[1,i].yaxis.set_ticks([])

        i += 1

    axs[0,2].set_title('Input', rotation=0, size='large')
    axs[1,2].set_title('Reconstruction', rotation=0, size='large')
    fig.tight_layout()
    plt.show()


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="display reconstructed images")
    parser.add_argument('--exp_folder', type=str,
            help="directory where exp is, format: <path>/lightning_logs/version_<number>/checkpoints")
    parser.add_argument("--train_pth", default="data/8x8-random.h5",
            help="train data set json file")
    args = parser.parse_args()
    main(args)
