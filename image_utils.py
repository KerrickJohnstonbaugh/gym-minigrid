# code from https://realpython.com/storing-images-in-python/

import h5py
import numpy as np
import os
from pathlib import Path

hdf5_dir = Path("data/hdf5/")
hdf5_dir.mkdir(parents=True, exist_ok=True)

def store_single_hdf5(image, image_id, label):
    """ Stores a single image to an HDF5 file.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "image", np.shape(image), h5py.h5t.STD_U8BE, data=image
    )
    meta_set = file.create_dataset(
        "meta", np.shape(label), h5py.h5t.STD_U8BE, data=label
    )
    file.close()


def read_single_hdf5(image_id):
    """ Stores a single image to HDF5.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "r+")

    image = np.array(file["/image"]).astype("uint8")
    label = int(np.array(file["/meta"]).astype("uint8"))

    return image, label


def store_many_hdf5(file_path, images):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """

    # Create a new HDF5 file
    file = h5py.File(file_path, "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    file.close()

def read_many_hdf5(file_path):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images = []

    # Open the HDF5 file
    file = h5py.File(file_path, "r+")

    images = np.array(file["/images"]).astype("uint8")

    file.close()
    return images


from torch.utils.data import Dataset
import torch

class ImageDataset(Dataset):
    def __init__(self, datapath = './data/5x5-random.h5'):
        data = read_many_hdf5(datapath)
        self.examples = data
        '''self.examples = [None]*len(data)
        for i in range(len(data)):
            self.examples[i] = torch.tensor(data[i])'''
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        #return torch.tensor(self.examples[idx].flatten()).float()
        img = torch.tensor(self.examples[idx]).float()
        return img.permute(2,0,1)/255.0

#im = None

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = ImageDataset(datapath='./data/5x5-random.h5')

    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size = 10, shuffle=True, num_workers=8)
    import matplotlib.pyplot as plt
    fig = plt.figure()

    from timeit import timeit
    figimgtimes = []
    canvastimes = []
    pausetimes = []
    i = 0

    def figim_fct(img):
        fig.figimage(img)
    def can_fct():
        fig.canvas.draw()
    def pause_fct():
        plt.pause(0.01)

    img = np.zeros((160,160))
    img_artist = plt.imshow(img)
    for b in dataloader:
        for img in b:
            #figimgtimes.append(timeit("figim_fct(image)","from __main__ import figim_fct; image=img",number=1,globals=globals()))
            #canvastimes.append(timeit("can_fct()","from __main__ import can_fct",number=1))
            #pausetimes.append(timeit("pause_fct()","from __main__ import pause_fct",number=1))
            #figim_fct(img)
            #can_fct()
            #pause_fct()
            img_artist.set_data(img)
            plt.pause(0.001)
        i+=1
        if i>10:
            break

    plt.figure()
    plt.plot(figimgtimes,label='fig img')
    plt.plot(canvastimes,label='canvas')
    plt.plot(pausetimes,label='pause')
    plt.legend()
    plt.show()
    