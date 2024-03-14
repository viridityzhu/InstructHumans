import json
import torch
import numpy as np

from ..utils.camera import *

def seg_verts(smpl_V, seg_name: str, seg_map_path='data/smplx_vert_segmentation.json'):
    with open(seg_map_path) as f:
        seg = json.load(f)
    return smpl_V[seg[seg_name]]

def random_camera_rays(vertices, n_views:int = 10, width=1024, region = 'face'):
    '''Get camera rays for rendering. More randomness.
    Args:
        n_views (int): the number of views.
        fov (float): the field of view.
        width (int): the width of the image.
        region (str): region, should be "face", "head_back", "front", "back", or "side"
    Returns:
        ray_o_images : the origin of the rays of n_views*height*width*3
        ray_d_images : the direction of the rays of n_views*height*width*3
    '''

    # Calculate mesh bbox and center
    min_bounds, max_bounds = torch.min(vertices, dim=0)[0], torch.max(vertices, dim=0)[0]
    mesh_size = torch.norm(max_bounds - min_bounds)
    mesh_center = (max_bounds + min_bounds) / 2

    # Set variations
    if region in ["face", "head_back"]:
        min_distance = 0.28 # mesh_size / 2 * 1.2  # e.g., 1.2
        max_distance = 0.42 # mesh_size * 2. # e.g., 3
    else:
        min_distance = 1.75 # mesh_size / 2 * 1.2  # e.g., 1.2
        max_distance = 1.95 # mesh_size * 2. # e.g., 3
    look_at_pos_variation = mesh_size * 0.005  # e.g., 0.05

    angle_variation = 0.01
    angle_offset = (torch.rand(1) - 0.5) * np.pi / n_views
    cam_y_offset = (torch.rand(n_views, device=torch.device('cuda')) - 0.5) * 0.002 * mesh_size
    fov_range = [19.999, 20.001]

    # Randomly generate params
    random_fov = torch.rand(n_views) * (fov_range[1] - fov_range[0]) + fov_range[0]
    random_camera_distance = torch.rand(n_views, device=torch.device('cuda')) * (max_distance - min_distance) + min_distance
    random_look_at = mesh_center + torch.randn(n_views, 3).cuda() * look_at_pos_variation
    camera_up_direction = torch.tensor( [[0, 1, 0]], dtype=torch.float32, device=torch.device('cuda')).repeat(n_views, 1,)

    # * angle of each region
    if region == 'face':
        angle = torch.linspace(-np.pi/2, np.pi/2, n_views) # only front view
    elif region == 'head_back':
        angle = torch.linspace(np.pi/2, np.pi * 3/2, n_views)
    elif region == 'front':
        angle = torch.linspace(-3/8 * np.pi, 3/8 * np.pi, n_views)
    elif region == 'back':
        angle = torch.linspace(5/8 * np.pi, 11/8*np.pi, n_views) # 360 degree view
    elif region == 'side':
        angle_right = torch.linspace(3/8 * np.pi, 5/8 * np.pi, n_views // 2) # pi / 4, 3/4 pi
        angle_left = torch.linspace(11/8 * np.pi, 13/8 * np.pi, n_views - n_views // 2) # 5/4 pi, 7/4 pi
        angle = torch.cat((angle_right, angle_left))

    angle = (angle + torch.randn(n_views) * angle_variation + angle_offset).cuda()

    random_camera_position = torch.stack((random_camera_distance*torch.sin(angle), 
                                    mesh_center[1] + cam_y_offset, 
                                    random_camera_distance*torch.cos(angle)), dim=1).cuda()
    # print(f'mesh_center: {mesh_center}, mesh size: {mesh_size}\ncam dist: {random_camera_distance}\nfov: {random_fov}\nlook at: {random_look_at}\nangle: {angle}\ncam pos: {random_camera_position}\n\n')

    ray_o_images = []
    ray_d_images = []
    for i in range(n_views):
        camera = Camera.from_args(eye=random_camera_position[i],
                                    at=random_look_at[i],
                                    up=camera_up_direction[i],
                                    fov=random_fov[i],
                                    width=width,
                                    height=width,
                                    dtype=torch.float32)

        ray_grid = generate_centered_pixel_coords(camera.width, camera.height,
                                                camera.width, camera.height, device=torch.device('cuda'))

        ray_orig, ray_dir = \
            generate_pinhole_rays(camera.to(ray_grid[0].device), ray_grid)

        ray_o_images.append(ray_orig.reshape(camera.height, camera.width, -1))
        ray_d_images.append(ray_dir.reshape(camera.height, camera.width, -1))

    return torch.stack(ray_o_images, dim=0), torch.stack(ray_d_images, dim=0)
