import json
import os
import pickle

from numcodecs import Blosc
import zarr
import numpy as np
from PIL import Image
from tqdm import tqdm
import open3d as o3d
import copy
import cv2
from numpy.linalg import inv
import torch
# from data_processing.rlbench_utils import (
#     keypoint_discovery,
#     image_to_float_array,
#     store_instructions
# )


ROOT = '/media/jiahe/data/RH20T_rgb_resized/processed_test'
STORE_PATH = '/media/jiahe/data/RH20T_rgb_resized/processed_zarr/'
STORE_EVERY = 5  # in keyposes
TRAJ_LENGTH = 15
NCAM = 3
IM_SIZE = 256
DEPTH_SCALE = 2**24 - 1

def visualize_pcd(pcd, traj_lists = None, curr_pose = None, drawlines = False):

    coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    coor_frame.scale(0.1, center=(0., 0., 0.))
    vis.add_geometry(coor_frame)
    vis.get_render_option().background_color = np.asarray([255, 255, 255])

    view_ctl = vis.get_view_control()

    vis.add_geometry(pcd)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(0.1, center=(0., 0., 0.) )
    # if(use_arrow):
    #     mesh = o3d.geometry.TriangleMesh.create_arrow( cylinder_radius=0.01, cone_radius=0.01, cylinder_height=0.005, cone_height=0.01, resolution=20, cylinder_split=4, cone_split=1 )
    # print("curr_pose: ", curr_pose)
    if(traj_lists is not None):

        for traj_idx ,traj in enumerate( traj_lists, 0 ):
            points = [ [0,0,0] ]
            lines = []
            colors = []
            if(curr_pose is not None):
                points.append( curr_pose[traj_idx][0:3,3] )
                lines.append( [ len(points) - 1 , len(points) - 2])
                colors.append( [1,0,0] )
            for node_idx, point in enumerate( traj , 0 ):
                new_mesh = copy.deepcopy(mesh).transform(point)
                vis.add_geometry(new_mesh)
                if drawlines:
                    points.append(point[0:3,3])
                    lines.append( [ len(points) - 1 , len(points) - 2])
                    colors.append( [1,0,0] )


            
            if drawlines:
                # lines.append( [ 0, len(points)])
                # colors.append( [1,0,0] )
                # print("points: ", len(points))
                # print("lines: ", lines)
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(points),
                    lines=o3d.utility.Vector2iVector(lines),
                )
                line_set.colors = o3d.utility.Vector3dVector(colors)
                vis.add_geometry(line_set)
                # o3d.visualization.draw_geometries([line_set])

    if(curr_pose is not None):
        for pose in curr_pose:
            curr_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            curr_mesh.scale(0.05, center=(0., 0., 0.) )
            curr_mesh = curr_mesh.transform(pose)
            vis.add_geometry(curr_mesh)

    view_ctl.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction
    view_ctl.set_front((-0.3, 0.0, 0.2))  # set the positive direction of the x-axis toward you
    view_ctl.set_lookat((0.0, 0.0, 0.3))  # set the original point as the center point of the window
    vis.run()
    vis.destroy_window()

def xyz_from_depth(depth_image, depth_intrinsic, depth_extrinsic, depth_scale=1000.):
    # Return X, Y, Z coordinates from a depth map.
    # This mimics OpenCV cv2.rgbd.depthTo3d() function
    fx = depth_intrinsic[0, 0]
    fy = depth_intrinsic[1, 1]
    cx = depth_intrinsic[0, 2]
    cy = depth_intrinsic[1, 2]
    # Construct (y, x) array with pixel coordinates
    y, x = np.meshgrid(range(depth_image.shape[0]), range(depth_image.shape[1]), sparse=False, indexing='ij')

    X = (x - cx) * depth_image / (fx * depth_scale)
    Y = (y - cy) * depth_image / (fy * depth_scale)
    ones = np.ones( ( depth_image.shape[0], depth_image.shape[1], 1) )
    xyz = np.stack([X, Y, depth_image / depth_scale], axis=2)
    xyz[depth_image == 0] = 0.0

    # print("xyz: ", xyz.shape)
    # print("ones: ", ones.shape)
    # print("depth_extrinsic: ", depth_extrinsic.shape)
    xyz = np.concatenate([xyz, ones], axis=2)
    xyz =  xyz @ np.transpose( depth_extrinsic)
    xyz = xyz[:,:,0:3]
    return xyz

def depth_to_world_xyz(depth, intrinsics, extrinsics):
    """
    Convert batched depth images to 3D world XYZ coordinates.

    Args:
        depth: Tensor of shape (B, H, W) representing depth maps.
        intrinsics: Tensor of shape (B, 3, 3) representing batched camera intrinsic matrices.
        extrinsics: Tensor of shape (B, 4, 4) representing batched camera extrinsic matrices.

    Returns:
        xyz_world: Tensor of shape (B, 3, H, W) representing 3D world coordinates.
    """
    B, H, W = depth.shape
    depth = depth/1000.
    # Create meshgrid of pixel coordinates
    u = torch.linspace(0, W - 1, W, device=depth.device).repeat(H, 1)  # (H, W)
    v = torch.linspace(0, H - 1, H, device=depth.device).view(H, 1).repeat(1, W)  # (H, W)

    # Stack and add batch dimension
    uv1 = torch.stack([u, v, torch.ones_like(u)], dim=0).unsqueeze(0)  # Shape (1, 3, H, W)
    uv1 = uv1.repeat(B, 1, 1, 1)  # Shape (B, 3, H, W)

    # Invert intrinsics to get normalized camera rays
    K_inv = torch.inverse(intrinsics)  # Shape (B, 3, 3)

    # Compute camera coordinates
    cam_coords = torch.einsum('bij, bjhw -> bihw', K_inv.float(), uv1.float())  # (B, 3, H, W)
    xyz_camera = cam_coords * depth.unsqueeze(1)  # Scale by depth, Shape (B, 3, H, W)

    # Convert to homogeneous coordinates (add a row of ones)
    ones = torch.ones((B, 1, H, W), device=depth.device)  # (B, 1, H, W)
    xyz_homogeneous = torch.cat([xyz_camera, ones], dim=1)  # (B, 4, H, W)
    inv_extrinsics = torch.inverse( extrinsics )
    # Apply extrinsics (transform from camera to world)
    xyz_world = torch.einsum('bij, bjhw -> bihw', inv_extrinsics[:, :3, :], xyz_homogeneous)  # (B, 3, H, W)
    # print("xyz_world: ", xyz_world.shape)
    return xyz_world
    # return xyz_camera

def get_filtered_data( ep, cam_list):
    intrinsics = ep['intrinsics']
    extrinsics = ep['extrinsics']
    # print("data: ")
    length  = len( ep['rgbds'] )

    resized_img_size = ( 256, 256 )
    
    rgb_list = [ [] for _ in cam_list]
    depth_list = [ [] for _ in cam_list]

    intrinsic_list = []
    extrinsic_list = []

    intrinsic_dict = copy.deepcopy(intrinsics)
    extrinsic_dict = copy.deepcopy(extrinsics)
    state_list = []
    # print("intrinsics: ", intrinsics)
    # ep['ee_pose'] = ee_poses
    # ep['cmds'] = gripper_cmds
    # ep['rgbds'] = rgbds
    # ep['task_idx'] =  task_num
    # ep['language'] = task_lang
    for idx in range(length):
        # print("step: ", idx)
        pcds = []
        rgbd_data = ep['rgbds'][idx]
        # print("ee_pose: ", ep['ee_pose'][idx])
        # print("cmds: ", ep['cmds'][idx])

        openness = int( ep['cmds'][idx] > 50. )
        state = np.append( ep['ee_pose'][idx],  openness) 
        state_list.append( state )
        # print("state: ", state)

        for idx, cam in enumerate( cam_list ):
            if( cam not in rgbd_data.keys() or cam not in intrinsics.keys() or cam not in extrinsics.keys()):
                rgb_list[idx].append( np.zeros( (resized_img_size[0], resized_img_size[1], 3) ) ) 
                depth_list[idx].append( np.zeros( resized_img_size ) )
                if(cam not in intrinsics.keys()):
                    intrinsic_dict[cam] = np.eye( 3 ) 
                if(cam not in extrinsics.keys()):
                    extrinsic_dict[cam] = np.eye( 4 ) 
                continue
            # print("cam: ", cam)
            height, width = rgbd_data[cam]['rgb'].shape[0], rgbd_data[cam]['rgb'].shape[1]
            # print("intrinsics: ", intrinsics)
            # print("rgbd_data.keys(): ", rgbd_data.keys())
            intrinsic = intrinsics[cam]
            extrinsic = extrinsics[cam]

            rgb = rgbd_data[cam]['rgb']
            depth = rgbd_data[cam]['depth']
            # print("depth: ", depth )
            # print("depth: ", np.max(depth), " ", np.min(depth))
            rgb_list[idx].append(rgb)
            depth_list[idx].append(depth)
            # print("intrinsic: ", intrinsic)
            
            xyz = xyz_from_depth(depth, intrinsic, inv(extrinsic) )
            # xyz = xyz_from_depth(depth, intrinsic, np.eye(4) )
            xyz = np.transpose( xyz, (2,0,1))

            intrinsics_tensor = torch.from_numpy(intrinsic)
            intrinsics_tensor = intrinsics_tensor.unsqueeze(0)
            depth_tensor = torch.from_numpy(depth)
            # print("depth_tensor: ", depth_tensor.shape)
            depth_tensor = depth_tensor.unsqueeze(0)

            extrinsics_tensor = torch.from_numpy(extrinsic)
            extrinsics_tensor = extrinsics_tensor.unsqueeze(0)

            # xyz_tensor = depth_to_world_xyz(depth_tensor, intrinsics_tensor, extrinsics_tensor)
            # diff = xyz - xyz_tensor.numpy()
            # print("diff: ", diff)
            # print("diff: ", np.max(diff), " ", np.min(diff) )
            # xyz_2d = xyz.reshape( -1, 3)
            # rgb_2d = rgb.reshape( -1, 3) / 255.0

            # pcd = o3d.geometry.PointCloud()
            # # Convert NumPy array to Open3D format using Vector3dVector
            # pcd.points = o3d.utility.Vector3dVector(xyz_2d)
            # pcd.colors = o3d.utility.Vector3dVector(rgb_2d)
            # visualize_pcd( pcd )
    for idx, cam in enumerate( cam_list ):
        intrinsic_list.append( intrinsic_dict[cam] )
        extrinsic_list.append( extrinsic_dict[cam] )
    return rgb_list , depth_list , state_list, intrinsic_list, extrinsic_list





def all_tasks_main(split='train'):
    cameras = [
        "shoulder_left", "front", "hand"
    ]
    cam_list = ['036422060909', '038522062288', '045322071843']
    cam_dict = {
        '036422060909': 'shoulder_left',
        '038522062288': 'front',
        '045322071843': 'hand',
    }
    # task2id = {task: t for t, task in enumerate(tasks)}
    # Initialize zarr
    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
    with zarr.open_group(f"{STORE_PATH}{split}.zarr", mode="w") as zarr_file:
        zarr_file.create_dataset(
            "rgb",
            shape=(0, NCAM, 3, IM_SIZE, IM_SIZE),
            chunks=(STORE_EVERY, NCAM, 3, IM_SIZE, IM_SIZE),
            compressor=compressor,
            dtype="uint8"
        )
        zarr_file.create_dataset(
            "depth",
            shape=(0, NCAM, IM_SIZE, IM_SIZE),
            chunks=(STORE_EVERY, NCAM, IM_SIZE, IM_SIZE),
            compressor=compressor,
            dtype="float16"
        )
        zarr_file.create_dataset(
            "proprioception",
            shape=(0, 3, 1, 8),
            chunks=(STORE_EVERY, 3, 1, 8),
            compressor=compressor,
            dtype="float32"
        )
        zarr_file.create_dataset(
            "action",
            shape=(0, TRAJ_LENGTH, 1, 8),
            chunks=(STORE_EVERY, TRAJ_LENGTH, 1, 8),
            compressor=compressor,
            dtype="float32"
        )
        zarr_file.create_dataset(
            "extrinsics",
            shape=(0, NCAM, 4, 4),
            chunks=(STORE_EVERY, NCAM, 4, 4),
            compressor=compressor,
            dtype="float16"
        )
        zarr_file.create_dataset(
            "intrinsics",
            shape=(0, NCAM, 3, 3),
            chunks=(STORE_EVERY, NCAM, 3, 3),
            compressor=compressor,
            dtype="float16"
        )
        zarr_file.create_dataset(
            "task_id", shape=(0,), chunks=(STORE_EVERY,),
            compressor=compressor,
            dtype="uint8"
        )
        zarr_file.create_dataset(
            "variation", shape=(0,), chunks=(STORE_EVERY,),
            compressor=compressor,
            dtype="uint8"
        )

        # Loop through episodes

        task_folder = os.path.join( ROOT, split) 
        episodes = sorted(os.listdir(task_folder))
        episodes = episodes[0:30]
        for files in tqdm(episodes):
            
            data = np.load(os.path.join(ROOT, split, files), allow_pickle=True)
            ep = data.item()

            # Keypose discovery
            length  = len( ep['rgbds'] ) // STORE_EVERY
            key_frames = [ _ for _ in range(length)]
            key_frames *= STORE_EVERY

            rgb_list , depth_list , state_list, intrinsic_list, extrinsic_list = get_filtered_data( ep , cam_list)

            # Loop through keyposes and store:
            # RGB (keyframes, cameras, 3, 256, 256)
            rgb = np.stack([
                np.stack(
                [
                    np.array( rgb_list[cam_idx][k] ) for cam_idx in range( len(cam_list) )
                ])
                for k in key_frames[:-1]
            ])
            rgb = rgb.transpose(0, 1, 4, 2, 3)
            rgb = rgb.astype(np.uint8)

            depth = np.stack([
                np.stack(
                [
                    np.array( depth_list[cam_idx][k] ) for cam_idx in range( len(cam_list) )
                ])
                for k in key_frames[:-1]
            ])
            depth = depth.astype(np.float16)
            # print("key_frames: ", len(key_frames) )
            # print("depth: ", depth.shape)

            # Proprioception (keyframes, 3, 2, 8)
            states = np.stack([state_list[k] for k in key_frames]).astype(np.float32)
            # print("states: ", states.shape)
            # Store current eef pose as well as two previous ones
            prop = states[:-1]
            prop_1 = np.concatenate([prop[:1], prop[:-1]])
            prop_2 = np.concatenate([prop_1[:1], prop_1[:-1]])
            prop = np.concatenate([prop_2, prop_1, prop], 1)
            prop = prop.reshape(len(prop), 3, 1, 8)
            # print("prop: ", prop.shape)
            
            # Action (keyframes, 15, 2, 8)
            # actions:  (129, 1, 1, 8)
            actions = [] #states[1:].reshape(len(states[1:]), 1, 1, 8)
            for k in key_frames[:-1]:
                actions.append( states[k: k + TRAJ_LENGTH ].reshape(-1, 1, 8) )
            actions = np.stack(actions).astype(np.float32)
            # print("actions: ", actions.shape)


            # Extrinsics (keyframes, cameras, 4, 4)
            extrinsics = np.stack([
                np.stack([
                    
                    np.array( extrinsic_list[cam_idx].astype(np.float16) ) for cam_idx in range( len(cam_list) )
                    # for cam in cameras
                ])
                for k in key_frames[:-1]
            ])
            # print("extrinsics: ", extrinsics.shape)
            # Intrinsics (keyframes, cameras, 3, 3)
            intrinsics = np.stack([
                np.stack([
                    
                    np.array( intrinsic_list[cam_idx].astype(np.float16) ) for cam_idx in range( len(cam_list) )
                    # for cam in cameras
                ])
                for k in key_frames[:-1]
            ])
            # print("intrinsics: ", intrinsics.shape)
            # Task id (keyframes,)
            task_id = np.array([ ep['task_idx'] ] * len(key_frames[:-1]))
            task_id = task_id.astype(np.uint8)

            # Variation (keyframes,)
            # with open(f"{task_folder}/{ep}/variation_number.pkl", 'rb') as f:
            #     var_ = pickle.load(f)
            var_ = np.array([ 0 ] * len(key_frames[:-1]))
            var_ = var_.astype(np.uint8)

            # # Write
            zarr_file['rgb'].append(rgb)
            zarr_file['depth'].append(depth)
            zarr_file['proprioception'].append(prop)
            zarr_file['action'].append(actions)
            zarr_file['extrinsics'].append(extrinsics)
            zarr_file['intrinsics'].append(intrinsics)
            zarr_file['task_id'].append(task_id)
            zarr_file['variation'].append(var_)


def _num2id(int_):
    str_ = str(int_)
    return '0' * (4 - len(str_)) + str_


def filter_tasks():
    task_folder = ROOT
    episodes = sorted(os.listdir(task_folder))
    task_num = {}

    episodes = episodes[0:30]

    for files in tqdm(episodes):
        data = np.load(os.path.join(ROOT,files), allow_pickle=True)
        ep = data.item()
        if ep['task_idx'] not in task_num.keys() :
            task_num[ ep['task_idx'] ] = 0
        
        task_num[ ep['task_idx'] ] += 1

    for task in task_num.keys():       
        print("demo num: ", task, " ", task_num[ task ])

if __name__ == "__main__":

    # filter_tasks()
    for split in ['train', 'eval']:
        all_tasks_main( split )
    
    # Store instructions as json (can be run independently)
    # os.makedirs('instructions/peract', exist_ok=True)
    # instr_dict = store_instructions(ROOT, tasks)
    # with open('instructions/peract/instructions.json', 'w') as fid:
    #     json.dump(instr_dict, fid)
