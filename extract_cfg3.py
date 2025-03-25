import argparse
from logging import Logger
import os
import queue
from alive_progress import alive_bar
import cv2
from matplotlib import pyplot as plt
import yaml
import time
import kinpy as kp
import open3d as o3d
import numpy as np
from rh20t_api.configurations import load_conf, tcp_as_q, Configuration
from rh20t_api.convert import timestamp_to_datetime_str
from rh20t_api.online import aligned_tcp_glob_mat, zeroed_force_torque_base
from rh20t_api.scene import RH20TScene
from utils.keyboard_listener import KeyboardListener
from utils.logger import logger_begin
from utils.point_cloud import create_point_cloud_manager
from utils.stopwatch import Stopwatch
from utils.robot import RobotModel
import librosa
import librosa.display
from typing import Dict, Any
import traceback
from pathlib import Path
from extract_rgbd import *
import copy
import multiprocessing
import time

def list_folders(directory):
    try:
        folder_list = [folder.name for folder in Path(directory).iterdir() if ( folder.is_dir() and (not 'human' in folder.name) and ('task' in folder.name)) ]
        folder_list = sorted( folder_list )
        return folder_list
    except FileNotFoundError:
        return "Directory not found"
    except PermissionError:
        return "Permission denied"

def get_description():
    task_dict = {}
    with open("tasks.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()  # Reads all lines into a list
        for line in lines:
            # print(line.strip())  # Removes trailing newline characters
            split_space_idx = line.find(' ')
            number = int( line[ : split_space_idx - 1] )
            description = line[split_space_idx+1 : -1] # "remove \n "
            task_dict[number] = description
            # print(number)
            # print(description)
    return task_dict


def worker(num, save_data_dir, folders, tasks_lang, goal_cams, vis_cfg):
    """Worker function that simulates work by sleeping for 2 seconds."""
    print(f"Worker {num} started")

    eval_ration = 0.2
    current_folders = folders[ num*100 : (num+1)*100 ]

    for idx, folder in enumerate( current_folders ):
        
        train = np.random.uniform() > eval_ration
        
        ep_idx = num*100 + idx

        task_num = int(folder[5:9])
        task_lang = tasks_lang[task_num]
        # data_preprocess( os.path.join(directory_path,folder), save_data_dir, idx, task_num, task_lang, goal_cams, vis_cfg_dict, train)
        try:
            data_preprocess( os.path.join(directory_path,folder), save_data_dir, ep_idx, task_num, task_lang, goal_cams, vis_cfg, train)
        except:
            pass
    print(f"Worker {num} finished")

if __name__ == "__main__":

    vis_cfg_dict = None
    try:
        with open(os.path.join('configs', 'default.yaml'), 'r') as settings_file: vis_cfg_dict = yaml.load(settings_file, Loader = yaml.FullLoader)
    except: 
        vis_logger.error("No configuration file `./configs/default.yaml` existing!")
        exit(1)

    # task_0001_user_0016_scene_0008_cfg_0003

    # Example usage
    directory_path = "/media/jiahe/data/RH20T_rgb_resized/RH20T_cfg3"  # Change this to your desired path
    folders = list_folders(directory_path)
    tasks = set()
    for folder in folders:
        # print("folder: ", folder)
        task_num = int(folder[5:9])
        if(task_num not in tasks):
            tasks.add(task_num)
    # print("tasks: ", tasks)
    tasks_lang = get_description()
    # print("task_lang: ", tasks_lang)
    goal_cams = ['036422060909', '038522062288', '045322071843']
    # robot_demo_folders = []
    save_data_dir = '/media/jiahe/data/RH20T_rgb_resized/processed/'
    OUTPUT_DIR = Path(save_data_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sub_dirs = ['train', 'eval']
    for sub_dir in sub_dirs:
        save_data_sub_dir = '/media/jiahe/data/RH20T_rgb_resized/processed/' + sub_dir
        OUTPUT_DIR = Path(save_data_sub_dir)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    task_number_map = {}
    for idx, folder in enumerate( folders ):

        task_num = int(folder[5:9])
        if(task_num not in task_number_map):
            task_number_map[task_num] = 1
        else:
            task_number_map[task_num] += 1
    print("task_number_map: ", task_number_map)


    num = 3
    print(f"Worker {num} started")
    eval_ration = 0.2
    current_folders = folders[ num*100 : (num+1)*100 ]
    np.random.seed(0)
    for idx, folder in enumerate( current_folders ):
        # if(idx < 67):
        #   continue
        train = np.random.uniform() > eval_ration
        
        ep_idx = num*100 + idx

        task_num = int(folder[5:9])
        task_lang = tasks_lang[task_num]
        # data_preprocess( os.path.join(directory_path,folder), save_data_dir, idx, task_num, task_lang, goal_cams, vis_cfg_dict, train)
        try:
            data_preprocess( os.path.join(directory_path,folder), save_data_dir, ep_idx, task_num, task_lang, goal_cams, vis_cfg_dict, train)
        except:
            pass
    print(f"Worker {num} finished")


    # num_processes = 8 # Number of processes to spawn
    # processes = []

    # # Create and start multiple processes
    # for i in range(num_processes):
    #     p = multiprocessing.Process(target=worker, args=(i, save_data_dir, folders, tasks_lang, goal_cams, vis_cfg_dict))
    #     processes.append(p)
    #     p.start()

    # # Wait for all processes to complete
    # for p in processes:
    #     p.join()

    print("All processes completed")