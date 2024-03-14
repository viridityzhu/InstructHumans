
import argparse
import os
import random
import sys
import time
import copy
import pickle
import torch
import torch.nn as nn
import h5py
import logging as log


import numpy as np
import logging as log
from tqdm import tqdm
import yaml

from lib.models.tracer import SDFTracer
from lib.utils.camera import *
from lib.models.trainer import Trainer
from lib.utils.eval_util import seg_verts, random_camera_rays
from lib.utils.config import *


RANDOM_VIEW_BATCH_NUM = 20
REGION_NUM = 5
views_per_batch = 10
class TracingPrepare(nn.Module):

    def __init__(self, config, output_dir='traced_points'):
        super().__init__()

        self.cfg = config
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.sdf_field = None

        self.tracer = SDFTracer(self.cfg)


    def init_models(self, trainer: Trainer):
        '''Initialize the models for evaluation.
        Args:
            sdf_field (SDFNet): the sdf field model from trainer.
            rgb_field (RGBNet): the rgb field model from trainer.
        '''

        self.sdf_field = copy.deepcopy(trainer.sdf_field)

    def _get_camera_rays_focus_on_head(self, n_views=4, fov=20, width=1024, camera_distance=0.4, head_position=torch.tensor([0, 0.62, 0])):
        '''Get camera rays for rendering.
        Args:
            n_views (int): the number of views.
            fov (float): the field of view.
            width (int): the width of the image.
        Returns:
            ray_o_images : the origin of the rays of n_views*height*width*3
            ray_d_images : the direction of the rays of n_views*height*width*3
        '''
            
        if not isinstance(head_position, torch.Tensor):
            head_position = torch.tensor(head_position, dtype=torch.float32)
        look_at = head_position.repeat(n_views, 1).to(torch.device('cuda'))
        camera_up_direction = torch.tensor( [[0, 1, 0]], dtype=torch.float32, device=torch.device('cuda')).repeat(n_views, 1,)
        angle = torch.linspace(-np.pi/2, np.pi/2, n_views+1)[:-1]
        # camera_position = torch.stack( (2*torch.sin(angle), torch.zeros_like(angle), 2*torch.cos(angle)), dim=1).cuda()
        camera_position = torch.stack((camera_distance*torch.sin(angle), head_position[1] + torch.zeros_like(angle), camera_distance*torch.cos(angle)), dim=1).cuda()

        ray_o_images = []
        ray_d_images = []
        for i in range(n_views):
            camera = Camera.from_args(eye=camera_position[i],
                                      at=look_at[i],
                                      up=camera_up_direction[i],
                                      fov=fov,
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
    
    def _get_camera_rays(self, n_views=4, fov=20, width=1024):
        '''Get camera rays for rendering.
        Args:
            n_views (int): the number of views.
            fov (float): the field of view.
            width (int): the width of the image.
        Returns:
            ray_o_images : the origin of the rays of n_views*height*width*3
            ray_d_images : the direction of the rays of n_views*height*width*3
        '''
            
        look_at = torch.zeros( (n_views, 3), dtype=torch.float32, device=torch.device('cuda'))
        camera_up_direction = torch.tensor( [[0, 1, 0]], dtype=torch.float32, device=torch.device('cuda')).repeat(n_views, 1,)
        angle = torch.linspace(0, 2*np.pi, n_views+1)[:-1]
        camera_position = torch.stack( (2*torch.sin(angle), torch.zeros_like(angle), 2*torch.cos(angle)), dim=1).cuda()

        ray_o_images = []
        ray_d_images = []
        for i in range(n_views):
            camera = Camera.from_args(eye=camera_position[i],
                                      at=look_at[i],
                                      up=camera_up_direction[i],
                                      fov=fov,
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



    def prepare_tracing(self):
        """
        Prepare camera views and their ray tracing results of each subject. This is to save time when editing.
        Camera views are divided into 5 regions. They can be weighted sampled when editing.
        """

        torch.cuda.empty_cache()
        start = time.time()
        
        # ! ------- Preparation --------
        # * Camera view number
        fix_n_views_face = self.cfg.n_views // 2

        # * create dataset file
        outfile = h5py.File(os.path.join(self.cfg.output_path), 'w')
        subject_list  = eval(self.cfg.subject_list) # [32, 34,]
        print(subject_list)
        subject_list = [int(i) for i in subject_list]
        print(subject_list)
        num_subjects = len(subject_list)
        outfile.create_dataset( 'num_subjects', data=num_subjects, dtype=np.int32)
        # for rendering
        dataset_fix_render_xs = outfile.create_dataset( 'fix_render_xs', shape=(num_subjects, self.cfg.n_views + fix_n_views_face, self.cfg.width*self.cfg.width, 3),
                                    chunks=True, dtype=np.float32)
        dataset_fix_render_hits = outfile.create_dataset( 'fix_render_hits',shape=(num_subjects, self.cfg.n_views + fix_n_views_face, self.cfg.width*self.cfg.width, 3),
                                    chunks=True, dtype=np.float32)
        # for training
        dataset_xs = outfile.create_dataset( 'xs', shape=(num_subjects, REGION_NUM, RANDOM_VIEW_BATCH_NUM, views_per_batch, self.cfg.width*self.cfg.width, 3),
                                    chunks=True, dtype=np.float32)
        dataset_hits = outfile.create_dataset( 'hits',shape=(num_subjects, REGION_NUM, RANDOM_VIEW_BATCH_NUM, views_per_batch, self.cfg.width*self.cfg.width, 3),
                                    chunks=True, dtype=np.float32)

        region_index_mapping = {'face': 0, 'head_back': 1, 'front': 2, 'back': 3, 'side': 4}
        # ! ----------------- prepare each subject -----------------
        for s, code_idx in enumerate(tqdm(subject_list)):

            # * prepare fixed views
            with torch.no_grad():
                log.info(f"Rendering fix {self.cfg.n_views + fix_n_views_face} views for {code_idx}th subject...")
                start = time.time()
                smpl_V = self.sdf_field.smpl_V[code_idx]
                head_V = seg_verts(smpl_V, 'head')
                head_min_bounds, head_max_bounds = torch.min(head_V, dim=0)[0], torch.max(head_V, dim=0)[0]
                self.head_pos = ((head_min_bounds + head_max_bounds) / 2).cpu()
                ray_o_images, ray_d_images = self._get_camera_rays(n_views=self.cfg.n_views, fov=self.cfg.fov, width=self.cfg.width)
                ray_o_images_face, ray_d_images_face = self._get_camera_rays_focus_on_head(n_views=fix_n_views_face, fov=self.cfg.fov, width=self.cfg.width, head_position=self.head_pos)
                ray_o_images = torch.cat((ray_o_images, ray_o_images_face), dim=0)
                ray_d_images = torch.cat((ray_d_images, ray_d_images_face), dim=0)

                _idx = torch.tensor([code_idx], dtype=torch.long, device = torch.device('cuda')).repeat(self.cfg.n_views + fix_n_views_face).detach()
                x, hit = self.tracer(self.sdf_field.forward, _idx,
                                ray_o_images.view(self.cfg.n_views + fix_n_views_face, -1, 3),
                                ray_d_images.view(self.cfg.n_views + fix_n_views_face, -1, 3))
                log.info(f"Ray tracing finished in {time.time()-start} seconds.")

                dataset_fix_render_xs[s] = x.detach().cpu().numpy()
                dataset_fix_render_hits[s] = hit.detach().cpu().numpy()


            # ! ----------------- prepare each random view batch -----------------
            for region in [ 'face', 'head_back', 'front', 'back', 'side']:
                log.info(f"Preparing region {region}: {RANDOM_VIEW_BATCH_NUM} batches, each has {views_per_batch} views.")
                x_batches = []
                hit_batches = []
                for b in tqdm(range(RANDOM_VIEW_BATCH_NUM)):
                    start = time.time()
                    # * Random camera views
                    with torch.no_grad():
                        if region in ['face', 'head_back']:
                            region_V = seg_verts(smpl_V, 'head')
                        else:
                            region_V = smpl_V
                        ray_o_images, ray_d_images = random_camera_rays(region_V, n_views=views_per_batch, width=self.cfg.width, region=region)

                        ray_o_images = ray_o_images.view(views_per_batch, -1, 3) # n_views, 400, 400, 3 -> n_views, 400 * 400, 3
                        ray_d_images = ray_d_images.view(views_per_batch, -1, 3)

                        _idx = torch.tensor([code_idx], dtype=torch.long, device = torch.device('cuda')).repeat(views_per_batch).detach() # [id]

                        # * trace rays
                        x, hit = self.tracer(self.sdf_field.forward, _idx,
                                        ray_o_images,
                                        ray_d_images)

                    # shuffle views
                    shuffled_indices = torch.randperm(views_per_batch)
                    x = x[shuffled_indices]
                    hit = hit[shuffled_indices]

                    x_batches.append(x)
                    hit_batches.append(hit)

                    end = time.time()
                    log.info(f"Ray tracing finished in {end-start} seconds.") # 4.8

                # store data into dataset
                dataset_xs[s, region_index_mapping[region]] = torch.stack(x_batches).detach().cpu().numpy()
                dataset_hits[s, region_index_mapping[region]] = torch.stack(hit_batches).detach().cpu().numpy()


        outfile.close()

def main(config):
    # Set random seed.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    with open('data/smpl_mesh.pkl', 'rb') as f:
        smpl_mesh = pickle.load(f)

    trainer = Trainer(config, smpl_mesh['smpl_V'], smpl_mesh['smpl_F'], 'temp')

    trainer.load_checkpoint(os.path.join(config.pretrained_root, config.model_name))

    evaluator =  TracingPrepare(config)
    evaluator.init_models(trainer)

    evaluator.prepare_tracing()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process dataset to H5 file')
    parser = parse_options()

    parser.add_argument('--subject_list', type=str, default='[32]', help='list of subjects to be processed')
    parser.add_argument('--output-path', type=str, default='prepared_dataset.h5', help='output path to H5 file')
    parser.add_argument('--pretrained-root', type=str, default='checkpoints/demo', help='pretrained model path')
    parser.add_argument('--model-name', type=str, default='model-1000.pth', help='load model name')
    

    args, args_str = argparse_to_str(parser)
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=args.log_level,
                        format='%(asctime)s|%(levelname)8s| %(message)s',
                        handlers=handlers)
    logging.info(f'Info: \n{args_str}')


    main(args)
