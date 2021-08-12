import gym_minigrid.envs.doorkey as dk
import matplotlib.pyplot as plt
from gym_minigrid.wrappers import RGBImgObsWrapper
from gym_minigrid.simpleminigrid import SimpleDoorKey
import gym
import time
import numpy as np
from image_utils import read_many_hdf5, read_single_hdf5, store_many_hdf5, store_single_hdf5

def step_and_render(env, action):
    obs = env.step(action)
    plt.imshow(obs[0]['image'])
    plt.show()

def step_and_gymrender(env, action):
    env.step(action)
    env.render()

def main():
    my_env = gym.make('MiniGrid-Simple-DoorKey-8x8-v0')
    seed = 4
    my_env.seed(seed)
    my_env.reset()

    img_set = []
    for i in range(4096):
        action = np.random.randint(0,3)
        obs, reward, done, _ = my_env.step(action)
        my_env.render()
        img = my_env.render('rgb_array')
        img_set.append(img)
        if done:
            # seed 2 for 5x5 dataset
            my_env.seed(seed)
            my_env.reset()

    img_set = np.array(img_set)
    store_many_hdf5('./data/8x8-random.h5',img_set)


if __name__ == '__main__':
    #my_imgs = read_many_hdf5('./data/5x5-random.h5')
    main()