import gym_minigrid.envs.doorkey as dk
import matplotlib.pyplot as plt
from gym_minigrid.wrappers import RGBImgObsWrapper
from gym_minigrid.simpleminigrid import SimpleDoorKey
import gym
import time

def step_and_render(env, action):
    obs = env.step(action)
    plt.imshow(obs[0]['image'])
    plt.show()

def step_and_gymrender(env, action):
    env.step(action)
    env.render()

def main():
    my_env = gym.make('MiniGrid-Simple-DoorKey-8x8-v0')
    my_env.seed(1)
    my_env.reset()
    my_env.render()
    time.sleep(1)
    #my_env = RGBImgObsWrapper(SimpleDoorKey())
    #obs = my_env.reset()
    #plt.imshow(obs['image'])
    #plt.show()
    #plt.show()

    step_and_gymrender(my_env, my_env.actions.right)
    time.sleep(1)
    step_and_gymrender(my_env, my_env.actions.forward)
    time.sleep(1)
    step_and_gymrender(my_env, my_env.actions.forward)
    time.sleep(1)
    step_and_gymrender(my_env, my_env.actions.right)
    time.sleep(1)
    step_and_gymrender(my_env, my_env.actions.forward)
    time.sleep(1)
    step_and_gymrender(my_env, my_env.actions.forward)
    time.sleep(1)
    step_and_gymrender(my_env, my_env.actions.left)
    time.sleep(1)
    step_and_gymrender(my_env, my_env.actions.left)
    time.sleep(1)
    step_and_gymrender(my_env, my_env.actions.left)
    time.sleep(1)
    step_and_gymrender(my_env, my_env.actions.forward)
    time.sleep(1)
    step_and_gymrender(my_env, my_env.actions.forward)
    time.sleep(1)
    step_and_gymrender(my_env, my_env.actions.forward)
    time.sleep(1)
    step_and_gymrender(my_env, my_env.actions.forward)
    time.sleep(1)
    '''
    step_and_gymrender(my_env, my_env.actions.forward)
    step_and_gymrender(my_env, my_env.actions.forward)
    step_and_gymrender(my_env, my_env.actions.forward)
    step_and_gymrender(my_env, my_env.actions.forward)
    step_and_gymrender(my_env, my_env.actions.forward)
    step_and_gymrender(my_env, my_env.actions.right)
    step_and_gymrender(my_env, my_env.actions.forward)
    step_and_gymrender(my_env, my_env.actions.forward)
    step_and_gymrender(my_env, my_env.actions.forward)
    step_and_gymrender(my_env, my_env.actions.right)
    step_and_gymrender(my_env, my_env.actions.forward)
    step_and_gymrender(my_env, my_env.actions.forward)
    step_and_gymrender(my_env, my_env.actions.forward)
    step_and_gymrender(my_env, my_env.actions.forward)
    '''

if __name__ == '__main__':
    main()