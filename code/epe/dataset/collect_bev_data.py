# File: collect_bev_data.py
import os
from pathlib import Path

import carla_gym
import gym
from PIL import Image

import datetime
import re
import numpy as np
import gc
import time

from carla_gym.sensors.rgb_sensor import RGBCameraSensor
from carla_gym.sensors.semantic_sensor import SemanticSensor

class random_policy():
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def step(self, repeat=4):
        while True:
            action = self.action_space.sample()
            if action[0] > 0:
                break
        for i in range(repeat):
            obs, r, done, info = self.env.step(action)

        return (obs, r, done, info), action

def write_to_file(PATH, data):
    rgb_file = open(PATH, 'a')
    rgb_file.write(data)
    rgb_file.close()
    return 1

def main(episode_folder='data4', town='Town01', start_ep='0'):
    # save path
    os.umask(000)
    episodes_folder = episode_folder
    time = re.sub("[^0-9^.]", "", str(datetime.datetime.now()))
    PATH = f'CARLA/{town}/{str(time)}/'
    Path(PATH + 'rgb/').mkdir(parents=True, exist_ok=True)
    Path(PATH + 'semantic/').mkdir(parents=True, exist_ok=True)

    # gym & sensors
    camera_settings = {"image_dim" : (540, 960)}

    semantic = SemanticSensor(camera_settings)
    rgb = RGBCameraSensor(camera_settings)

    params = {
        'sensors': [rgb, semantic],
        'max_episode_length': 800,
        'carla_host': 'localhost',
        # 'delta_seconds': 1.0 / 15,
        'world': town,
        'controller': carla_gym.controllers.AutopilotController()
    }

    env = gym.make('carla-v0', params=params)

    time.sleep(10)

    # 3000 episodes of 80 images each
    episodeLength = 80
    maxEpisodes = 3000

    # breakpoint()

    episode = start_ep
    while episode < maxEpisodes:
        percentage = round(float(episode) / float(maxEpisodes), 3)
        print(f'Episode: {episode}/{maxEpisodes} ({percentage})')

        obs = env.reset()

        # get images for episode
        step = 0
        rgb_paths = ''
        sem_paths = ''
        while step < episodeLength:
            obs, r, done, _= env.step(None)

            rgb_im = Image.fromarray(obs[0])
            sem_im = obs[1]

            # semantic labels are only red channel of image
            sem_im = sem_im[:,:,0]*11
            #sem_im = (np.arange(sem_im.max()+1) == sem_im[...,None]).astype(int)
            sem_im = Image.fromarray(sem_im, 'L')

            rgb_path = f'{PATH}rgb/rgb_{episode}_{step}.png'
            rgb_im.save(rgb_path)
            rgb_paths += f'{rgb_path}\n'

            sem_path = f'{PATH}semantic/sem_{episode}_{step}.png'
            sem_im.save(sem_path)
            sem_paths += f'{sem_path}\n'
            
            if done:
                break

            step += 1
            
        write_to_file(f'{PATH}rgb/rgb_paths.txt', rgb_paths)
        write_to_file(f'{PATH}semantic/sem_paths.txt', sem_path)

        episode += 1

        del rgb_paths, sem_paths
        del rgb_im, sem_im
        gc.collect()




if __name__ == '__main__':
    main(episode_folder='data_town_1_testUE4', town='Town01', start_ep=0)
