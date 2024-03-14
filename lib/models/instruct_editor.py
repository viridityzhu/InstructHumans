import json
import os
import random
import time
import copy
import torch
import torch.nn as nn
import trimesh
import lpips

import numpy as np
import logging as log
from tqdm import tqdm
from PIL import Image
import yaml

from .tracer import SDFTracer
from ..ops.mesh import load_obj, point_sample, closest_tex
from ..utils.camera import *
from lib.models.trainer import Trainer
from .ip2p import InstructPix2Pix
from ..utils.image import crop_to_mask_tensor
from ..utils.eval_util import seg_verts, random_camera_rays
from .losses import SmoothnessLoss
from ..utils.visualize import Visualizer
from ..utils.metrics import ClipSimilarity
from ..datasets.traced_points_dataset import TracedPointsDataset

from kaolin.ops.conversions import voxelgrids_to_trianglemeshes
from kaolin.ops.mesh import subdivide_trianglemesh

import wandb


class InstructEditor(nn.Module):

    def __init__(self, config, log_dir):
        super().__init__()


        self.init_log_dir(config, log_dir)

        self.sdf_field = None
        self.rgb_field = None

        self.tracer = SDFTracer(self.cfg)
        self.subdivide = self.cfg.subdivide
        self.res = self.cfg.grid_size

        # select device for InstructPix2Pix
        self.ip2p_device = (
            torch.device('cuda')
            if self.cfg.ip2p_device is None
            else torch.device(self.cfg.ip2p_device)
        )

        self.ip2p = InstructPix2Pix(self.ip2p_device, ip2p_use_full_precision=self.cfg.ip2p_use_full_precision)

        self.n_views = self.cfg.n_views

        self.clip_metrics = ClipSimilarity()
        self.fix_render_x = None
        self.sample_indices = []

    def init_log_dir(self, config, log_dir):
        self.cfg = config
        self.log_dir = log_dir
        self.mesh_dir = os.path.join(log_dir, 'meshes')
        os.makedirs(self.mesh_dir, exist_ok=True)
        self.image_dir = os.path.join(log_dir, 'images')
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(os.path.join(self.image_dir, 'pred'), exist_ok=True)
        os.makedirs(os.path.join(self.image_dir, 'ori'), exist_ok=True)
        os.makedirs(os.path.join(self.image_dir, 'edited'), exist_ok=True)
        os.makedirs(os.path.join(self.image_dir, 'render'), exist_ok=True)
        os.makedirs(os.path.join(self.image_dir, 'one_step_edit'), exist_ok=True)
        os.makedirs(os.path.join(self.image_dir, 'vis'), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'cp'), exist_ok=True)

    def init_models(self, trainer: Trainer):
        '''Initialize the models for evaluation.
        Args:
            sdf_field (SDFNet): the sdf field model from trainer.
            rgb_field (RGBNet): the rgb field model from trainer.
        '''

        self.sdf_field = copy.deepcopy(trainer.sdf_field)
        self.rgb_field = copy.deepcopy(trainer.rgb_field)
        self.smpl_F = trainer.smpl_F.clone().detach().cpu()
        self.smoothness_loss = SmoothnessLoss(trainer.smpl_V[0].size(0), self.smpl_F, device=self.ip2p_device)

    def editing_by_instruction(self, code_idx: int, instruction: str, target_smpl_obj=None, num_steps=200):
        """Fitting the color latent code to the rendered images
           Store the optimzed code in the code_idx-th entry of the codebook.
           Given target images of n_views (target_dict contains rgb, coord, and mask),
           this function takes out texture features of given subject, and train it, and then put it back.
        """

        torch.cuda.empty_cache()

        start = time.time()
        # * backup config file
        config_dict = vars(self.cfg)
        yaml_file_path = os.path.join(self.log_dir, 'config.yaml')
        with open(yaml_file_path, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False)
        
        # ! ------- Preparation --------
        # * Text embedding
        # load base text embedding using classifier free guidance
        text_embedding = self.ip2p.pipe._encode_prompt(
            instruction, device=self.ip2p_device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=""
        )

        # * texture latent codes
        # get tex_feats of idx-th subject from the codebook
        tex_feats= self.rgb_field.get_feature_by_idx(code_idx).clone().unsqueeze(0).detach().data
        tex_feats.requires_grad = True
        _idx = torch.tensor([code_idx], dtype=torch.long, device = torch.device('cuda')).detach() # [id]

        params = []
        params.append({'params': tex_feats, 'lr': self.cfg.lr_tex_code})

        # * optimizer
        optimizer = torch.optim.Adam(params, betas=(0.9, 0.999))
        step_start = 0
        # Initialize LPIPS loss function
        lpips_loss_fn = lpips.LPIPS(net='vgg').cuda()  # Use VGG network for LPIPS

        # * load checkpoint
        if self.cfg.load_edit_checkpoint:
            step_start, tex_feats_checkpoint, optimizer_state_checkpoint, rgb_checkpoint = self.load_edit_checkpoint(self.cfg.edit_checkpoint_file)
            if tex_feats_checkpoint is not None:
                tex_feats = tex_feats_checkpoint
                tex_feats.requires_grad = True
            # fix: Reinitialize the optimizer with the updated tensors
            params = []
            params.append({'params': tex_feats, 'lr': self.cfg.lr_tex_code})
            if self.cfg.train_nerf:
                import copy
                # copy ori rgb field because we need ori images
                ori_rgb_field = copy.deepcopy(self.rgb_field).to(self.ip2p_device)
                # load nerf params
                if rgb_checkpoint is not None:
                    self.rgb_field.load_state_dict(rgb_checkpoint)
                decoder_params = list(self.rgb_field.decoder.parameters())
                params.append({'params': decoder_params,
                                'lr': self.cfg.lr_nerf,
                                "weight_decay": self.cfg.weight_decay_nerf})
            optimizer = torch.optim.Adam(params, betas=(0.9, 0.999))
            if optimizer_state_checkpoint is not None:
                optimizer_state_current = optimizer.state_dict()
                optimizer_state_current['state'].update(optimizer_state_checkpoint['state'])
                optimizer_state_current['param_groups'][0].update(optimizer_state_checkpoint['param_groups'][0])
                optimizer.load_state_dict(optimizer_state_current)
            if step_start is None:
                step_start = 0

        # * SMPL mesh vertices
        if target_smpl_obj is not None:
            smpl_V, _ = load_obj(target_smpl_obj, load_materials=False)
            smpl_V = smpl_V.cuda()
        else:
            smpl_V = self.rgb_field.get_smpl_vertices_by_idx(code_idx)

        # * Camera view number
        n_views_body = self.cfg.random_views
        if self.cfg.zoom_in_face: # additional 20% views for face
            n_views_face = int(self.cfg.random_views * 0.2)
            random_views_total = n_views_body + n_views_face
        if self.cfg.grad_aware_cam_sampl:
            random_views_total = self.cfg.random_views

        if self.cfg.use_traced_point_dataset:
            self.tracedPointDataset = TracedPointsDataset(code_idx)
            self.tracedPointDataset.init_from_h5(self.cfg.traced_points_data_root)

        # * sampling strategy choice. dynamic t sampling, sds, or original
        self.ip2p.init_sampling_strategy(self.cfg.sampl_strategy, total_it=step_start + num_steps, M=self.cfg.mode_change_step, M2=self.cfg.mode_change_step2, min_step_r=self.cfg.min_step_rate, max_step_r=self.cfg.max_step_rate)

        end = time.time()
        log.info(f"Preparation finished in {end-start} seconds.") # 1.9
        # * prepare original rendered images for evaluation
        self.render_2D(code_idx, epoch=-1, fix=True, write_to_wandb=True)

        # !--------------- Edit loop start ------------
        loop = tqdm(range(step_start + 1, step_start + 1 + num_steps))
        tex_feats_old = None
        vis = Visualizer()
        log.info(f"Edit {num_steps * random_views_total} iterations in total.")

        for i in loop:

            # * gradient aware camera sampling
            if self.cfg.grad_aware_cam_sampl:
                if i == 1: # initialize accumulated grad magnitude
                    acc_tex_grad_magn = torch.zeros(tex_feats.shape[1], device=self.ip2p_device)
                elif i == 2: # after that, compute weights of regions
                    self.cal_view_weight(acc_tex_grad_magn, random_views_total)
                    
            # ! deprecated since preparing the traced point dataset speed up a lot!
            if not self.cfg.use_traced_point_dataset:
                _idx = torch.tensor([code_idx], dtype=torch.long, device = torch.device('cuda')).repeat(random_views_total).detach() # [id]
                start = time.time()
                # * Random camera views
                # Randomly generate n_views images
                with torch.no_grad():

                    ray_o_images, ray_d_images = random_camera_rays(smpl_V, n_views=n_views_body, width=self.cfg.width)

                    if self.cfg.zoom_in_face:
                        head_V = seg_verts(smpl_V, 'head')
                        ray_o_images_face, ray_d_images_face = random_camera_rays(head_V, n_views=n_views_face, width=self.cfg.width, is_head=True)
                        ray_o_images = torch.cat((ray_o_images, ray_o_images_face), dim=0)
                        ray_d_images = torch.cat((ray_d_images, ray_d_images_face), dim=0)

                    ray_o_images = ray_o_images.view(random_views_total, -1, 3) # n_views, 400, 400, 3 -> n_views, 400 * 400, 3
                    ray_d_images = ray_d_images.view(random_views_total, -1, 3)


                end = time.time()
                log.info(f"Random view generation for all views finished in {end-start} seconds.") # 0.04

                start = time.time()
                # * trace rays and render the original image
                with torch.no_grad():
                    xs, hits = self.tracer(self.sdf_field.forward, _idx,
                                    ray_o_images,
                                    ray_d_images)
                    if self.cfg.train_nerf:
                        rgb_2ds = ori_rgb_field.forward(xs.detach(), _idx) * hits
                    else:
                        rgb_2ds = self.rgb_field.forward(xs.detach(), _idx) * hits

                end = time.time()
                log.info(f"Ray tracing for current view finished in {end-start} seconds.") # 4.8

            # !--------------- n views loop start ------------
            if self.cfg.use_traced_point_dataset:
                self.tracedPointDataset.resample_batch()

            if self.cfg.grad_aware_cam_sampl:
                self.weighted_cam_sampl(calc_weight = i==1)

            optimizer.zero_grad()
            for no_view, region_and_idx in enumerate(self.sample_indices):

                start = time.time()
                if self.cfg.use_traced_point_dataset:
                    self.tracedPointDataset.set_fix(False)
                    x, hit = self.tracedPointDataset[region_and_idx] # actually the views are already randomed in the dataset. But we random again here to make sure each time they are different
                    x, hit = x.to(self.ip2p_device), hit.to(self.ip2p_device)

                    if self.cfg.train_nerf:
                        rgb_2d = ori_rgb_field.forward(x.detach(), _idx) * hit
                    else:
                        rgb_2d = self.rgb_field.forward(x.unsqueeze(0).detach(), _idx) * hit.unsqueeze(0)
                    pred_rgb = self.rgb_field.forward_fitting(x.unsqueeze(0).detach(), tex_feats, smpl_V.unsqueeze(0)) * hit.unsqueeze(0)
                else:
                    raise NotImplementedError("This feature has not been implemented yet.")

                # * images process
                # reshape images
                # B x w*w x 3 -> B x w x w x 3
                rgb_2d = rgb_2d.view(1, self.cfg.width, self.cfg.width, 3).permute(0, 3, 1, 2)
                pred_rgb = pred_rgb.view(1, self.cfg.width, self.cfg.width, 3).permute(0, 3, 1, 2)

                # crop background
                mask = hit.view(1, self.cfg.width, self.cfg.width)
                rgb_2d = crop_to_mask_tensor(rgb_2d, mask)
                pred_rgb = crop_to_mask_tensor(pred_rgb, mask)

                # convert to half precision
                if not self.cfg.ip2p_use_full_precision:
                    rgb_2d = rgb_2d.half()
                    pred_rgb = pred_rgb.half()

                # * save images
                if not self.cfg.most_efficient and not no_view % self.cfg.show_edited_img_freq:
                    Image.fromarray((rgb_2d.permute(0, 2, 3, 1).squeeze().cpu().detach().numpy() * 255).astype(np.uint8)).save(
                            os.path.join(self.image_dir, 'ori', 'step-%03d_view-%03d_render_ori-%03d_.png' % (i, no_view, code_idx)) )
                if not no_view % self.cfg.show_edited_img_freq:
                    Image.fromarray((pred_rgb.permute(0, 2, 3, 1).squeeze().cpu().detach().numpy() * 255).astype(np.uint8)).save(
                            os.path.join(self.image_dir, 'pred', 'step-%03d_view-%03d_render_pred-%03d_.png' % (i, no_view, code_idx)) )

                end = time.time()
                log.debug(f"Render finished in {end-start} seconds.") # 0.08
                # !------------- Calculate SDS and other losses ----------

                start = time.time()
                # * get sds loss
                sds_dict = self.ip2p.compute_sds(
                                text_embedding,
                                pred_rgb,
                                rgb_2d,
                                current_it = i,
                                guidance_scale=self.cfg.guidance_scale,
                                image_guidance_scale=self.cfg.image_guidance_scale,
                                lower_bound=self.cfg.lower_bound,
                                upper_bound=self.cfg.upper_bound,
                            )
                
                end = time.time()
                log.debug(f"SDS computation finished in {end-start} seconds.") # 0.6-1
                start = time.time()
                loss_ls = []
                total_loss = sds_dict['loss_sds']
                loss_ls.append(('sds', sds_dict['loss_sds'].item()))

                # save image
                if not no_view % self.cfg.show_edited_img_freq:
                    prev_img = sds_dict['prev_img'] # edited image through the random denoising step
                    Image.fromarray((prev_img.permute(0, 2, 3, 1).squeeze().cpu().detach().numpy() * 255).astype(np.uint8)).save(
                            os.path.join(self.image_dir, 'one_step_edit', 'step-%03d_view-%03d_one_step_edit-%03d_.png' % (i, no_view, code_idx)) )

                # generate the edited image of the total diffusion steps (20) for visualization
                if not self.cfg.most_efficient and self.cfg.show_edited_img and not no_view % self.cfg.show_edited_img_freq:
                    edited_img = self.ip2p.edit_image( # by default, diffusion steps = 20
                                text_embedding.to(self.ip2p_device),
                                pred_rgb.to(self.ip2p_device),
                                rgb_2d.to(self.ip2p_device),
                                guidance_scale=self.cfg.guidance_scale,
                                image_guidance_scale=self.cfg.image_guidance_scale,
                                lower_bound=self.cfg.lower_bound,
                                upper_bound=self.cfg.upper_bound,
                    )
                    Image.fromarray((edited_img.permute(0, 2, 3, 1).squeeze().cpu().detach().numpy() * 255).astype(np.uint8)).save(
                            os.path.join(self.image_dir, 'edited', 'step-%03d_view-%03d_edit_img-%03d_.png' % (i, no_view, code_idx)) )

                # * get regu loss
                if i > 1 and self.cfg.loss_reg_weight > 0.:
                    loss_reg = self.cfg.loss_reg_weight * (tex_feats**2).mean()
                    loss_ls.append(('regu', loss_reg.item()))
                else:
                    loss_reg = 0.
                total_loss += loss_reg

                # * get smoothness loss
                if i > 1 and self.cfg.loss_smooth_weight > 0. and tex_feats_old is not None:
                    delta_tex_feats = tex_feats - tex_feats_old
                    smoothness_loss = self.smoothness_loss.cal(delta_tex_feats, self.cfg.loss_smooth_weight)
                    total_loss += smoothness_loss
                    loss_ls.append(('smooth', smoothness_loss.item()))
                else:
                    smoothness_loss = 0.

                if not self.cfg.most_efficient and self.cfg.visualize_more and not no_view % self.cfg.show_edited_img_freq:
                    pred_rgb.retain_grad() # for visualization

                end = time.time()
                log.debug(f"loss caculation finished in {end-start} seconds.") # 0.006

                start = time.time()
                total_loss = total_loss / 5
                total_loss.backward()

                # * generate visualization plot
                if not self.cfg.most_efficient and self.cfg.visualize_more and not no_view % self.cfg.show_edited_img_freq:
                    grad_sds = pred_rgb.grad.detach().clone()
                    tex_grad_sds = tex_feats.grad.detach().clone()
                    vis(i, grad_sds, rgb_2d, pred_rgb, prev_img, edited_img,
                        tex_grad_sds, smpl_V.cpu(),
                        save_path=os.path.join(self.image_dir, 'vis', 'step-%03d_view-%03d_vis-%03d_.png' % (i, no_view, code_idx)))
                
                # * accumulate vertex gradients in the 1st iteration
                if self.cfg.grad_aware_cam_sampl and i == 1:
                    tex_grad_sds = tex_feats.grad.detach().clone()
                    acc_tex_grad_magn += torch.sqrt(torch.sum(tex_grad_sds**2, dim=-1)).squeeze()

                # * store old texture latents for the smoothness loss in next iteration
                if self.cfg.loss_smooth_weight > 0.:
                    tex_feats_old = tex_feats.clone().detach()

                end = time.time()
                log.debug(f"loss backward finished in {end-start} seconds.") # 0.04

                if no_view % 5 == 0:
                    start = time.time()
                    optimizer.step()

                    if not self.cfg.most_efficient and self.cfg.visualize_more and not no_view % self.cfg.show_edited_img_freq:
                        # Clear the gradients of 'pred_rgb'
                        if pred_rgb.grad is not None:
                            pred_rgb.grad = None

                    end = time.time()
                    optimizer.zero_grad()
                    log.debug(f"optimizer step finished in {end-start} seconds.") # 0.04

                # * log results
                self.log(loop, loss_ls, i, step_start + num_steps, region_and_idx, no_view, len(self.sample_indices))


            # * save checkpoint
            if i % self.cfg.save_edit_freq == 0:
                self.save_edit_checkpoint(i, tex_feats, optimizer, filename=self.cfg.edit_checkpoint_file)
                with torch.no_grad():
                    ori_tex_feats= self.rgb_field.get_feature_by_idx(code_idx).clone().unsqueeze(0).detach().data
                    self.rgb_field.replace_feature_by_idx(code_idx, tex_feats)
                self.render_2D(code_idx, epoch=i, fix=True, write_to_wandb=True)
                with torch.no_grad(): # back to original features
                    self.rgb_field.replace_feature_by_idx(code_idx, ori_tex_feats)


        # * replace the original latent codes by the edited one. This can be used later for rendering
        with torch.no_grad():
            self.rgb_field.replace_feature_by_idx(code_idx, tex_feats)


    def _marching_cubes (self, geo_idx=0, tex_idx=None, subdivide=True, res=300) -> trimesh.Trimesh:
        '''Marching cubes to generate mesh.
        Args:
            geo_idx (int): the index of geometry to be generated.
            tex_idx (int): the index of texture to be generated.
            subdivide (bool): whether to subdivide the mesh.
            res (int): the resolution of the marching cubes.
        Returns:
            mesh (trimesh): the generated mesh.
        '''

        width = res
        window_x = torch.linspace(-1., 1., steps=width, device='cuda')
        window_y = torch.linspace(-1., 1., steps=width, device='cuda')
        window_z = torch.linspace(-1., 1., steps=width, device='cuda')

        coord = torch.stack(torch.meshgrid(window_x, window_y, window_z, indexing='ij')).permute(1, 2, 3, 0).reshape(1, -1, 3).contiguous()

        
        # Debug smpl grid
        #smpl_vertice = self.sdf_field.get_smpl_vertices_by_idx(geo_idx)
        #d = trimesh.Trimesh(vertices=smpl_vertice.cpu().detach().numpy(), 
        #            faces=self.smpl_F.cpu().detach().numpy())
        #d.export(os.path.join(self.log_dir, 'smpl_sub_%03d.obj' % (geo_idx)) )
        
        if tex_idx is None:
            tex_idx = geo_idx
        geo_idx = torch.tensor([geo_idx], dtype=torch.long, device = torch.device('cuda')).view(1).detach()
        tex_idx = torch.tensor([tex_idx], dtype=torch.long, device = torch.device('cuda')).view(1).detach()

        _points = torch.split(coord, int(2*1e6), dim=1)
        voxels = []
        for _p in _points:
            pred_sdf = self.sdf_field(_p, geo_idx)
            voxels.append(pred_sdf)

        voxels = torch.cat(voxels, dim=1)
        voxels = voxels.reshape(1, width, width, width)
        
        vertices, faces = voxelgrids_to_trianglemeshes(voxels, iso_value=0.)
        vertices = ((vertices[0].reshape(1, -1, 3) - 0.5) / (width/2)) - 1.0
        faces = faces[0]

        if subdivide:
            vertices, faces = subdivide_trianglemesh(vertices, faces, iterations=1)

        pred_rgb = self.rgb_field(vertices, tex_idx, pose_idx=geo_idx)            
        
        h = trimesh.Trimesh(vertices=vertices[0].cpu().detach().numpy(), 
                faces=faces.cpu().detach().numpy(), 
                vertex_colors=pred_rgb[0].cpu().detach().numpy())

        # remove disconnect par of mesh
        connected_comp = h.split(only_watertight=False)
        max_area = 0
        max_comp = None
        for comp in connected_comp:
            if comp.area > max_area:
                max_area = comp.area
                max_comp = comp
        h = max_comp
    
        trimesh.repair.fix_inversion(h)

        return h

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

    def reconstruction(self, idx, epoch=None):
        '''
        Reconstruct the mesh the idx-th subject.
        Export the Trimesh as an .ojb file
        '''
        if epoch is None:
            epoch = 0
        log.info(f"Reconstructing {idx}th mesh at epoch {epoch}...")
        start = time.time()
        
        with torch.no_grad():
            h = self._marching_cubes (geo_idx=idx, subdivide=self.subdivide, res=self.res)
        
        h.export(os.path.join(self.mesh_dir, '%03d_reco_src-%03d.obj' % (epoch, idx)) )
        end = time.time()
        log.info(f"Reconstruction finished in {end-start} seconds.")

    def render_2D(self, idx, epoch=None, fix=False, write_to_wandb=False, transparent = True):
        '''
        Render the 2D images of the idx-th subject.
        '''
        torch.cuda.empty_cache()


        with torch.no_grad():

            if self.cfg.use_traced_point_dataset:
                start = time.time()
                self.tracedPointDataset.set_fix(True)
                x, hit = self.tracedPointDataset[0]
                x, hit = x.to(self.ip2p_device), hit.to(self.ip2p_device)
                _idx = torch.tensor([idx], dtype=torch.long, device = torch.device('cuda')).repeat(x.shape[0]).detach()
                rgb_2d = self.rgb_field.forward(x, _idx) * hit
            else:
                # old ways
                if fix and self.fix_render_x is not None:
                    start = time.time()
                    hit = self.fix_render_hit
                    rgb_2d = self.rgb_field.forward(self.fix_render_x, self.fix_render_idx) * self.fix_render_hit
                else:
                    log.info(f"Rendering {idx}th subject at epoch {epoch}...")
                    start = time.time()
                    n_views_face = self.cfg.n_views // 2
                    smpl_V = self.rgb_field.smpl_V[idx]
                    head_V = seg_verts(smpl_V, 'head')
                    head_min_bounds, head_max_bounds = torch.min(head_V, dim=0)[0], torch.max(head_V, dim=0)[0]
                    self.head_pos = ((head_min_bounds + head_max_bounds) / 2).cpu()
                    if fix:
                        ray_o_images, ray_d_images = self._get_camera_rays(n_views=self.cfg.n_views, fov=self.cfg.fov, width=self.cfg.width)
                        ray_o_images_face, ray_d_images_face = self._get_camera_rays_focus_on_head(n_views=n_views_face, fov=self.cfg.fov, width=self.cfg.width, head_position=self.head_pos)
                    else:
                        ray_o_images, ray_d_images = random_camera_rays(smpl_V, n_views=self.cfg.n_views, width=self.cfg.width)
                        ray_o_images_face, ray_d_images_face = random_camera_rays(head_V, n_views=n_views_face, width=self.cfg.width, is_head=True)
                    ray_o_images = torch.cat((ray_o_images, ray_o_images_face), dim=0)
                    ray_d_images = torch.cat((ray_d_images, ray_d_images_face), dim=0)

                    _idx = torch.tensor([idx], dtype=torch.long, device = torch.device('cuda')).repeat(self.cfg.n_views).detach()
                    x, hit = self.tracer(self.sdf_field.forward, _idx,
                                    ray_o_images.view(self.cfg.n_views, -1, 3),
                                    ray_d_images.view(self.cfg.n_views, -1, 3))
                    log.info(f"Rat tracing finished in {time.time()-start} seconds.")
                    if fix and self.fix_render_x is None:
                        self.fix_render_x = x.detach()
                        self.fix_render_hit = hit
                        self.fix_render_idx = _idx
                    start = time.time()
                    rgb_2d = self.rgb_field.forward(x.detach(), _idx) * hit

        n_views_face = self.cfg.n_views // 2
        if transparent: # add alpha channel
            rgba_2d = torch.cat((rgb_2d, hit[:,:,0].unsqueeze(-1)),dim=2)
        else:
            rgba_2d = rgb_2d
        rgb_img = rgba_2d.reshape(self.cfg.n_views + n_views_face, self.cfg.width, self.cfg.width, -1).cpu().detach().numpy() * 255
        if epoch == -1: # original images
            self.ori_imgs = rgb_2d.reshape(self.cfg.n_views + n_views_face, self.cfg.width, self.cfg.width, 3).permute(0, 3, 1, 2) # N, 3, W, H
            self.clip_metrics.init_ori_imgs(self.ori_imgs, self.cfg.caption_ori)
        else: # target images, compute evaluation metrics
            self.cur_imgs = rgb_2d.reshape(self.cfg.n_views + n_views_face, self.cfg.width, self.cfg.width, 3).permute(0, 3, 1, 2) # N, 3, W, H
            self.eval_edit(epoch, write_to_wandb)
        if write_to_wandb:
            step = epoch if epoch >=0 else 0
            wandb.log({"Rendered Images": [wandb.Image(rgb_img[i]) for i in range(self.cfg.n_views + n_views_face)]}, step=step)

        for i in range(self.cfg.n_views + n_views_face):
            Image.fromarray(rgb_img[i].astype(np.uint8)).save(
                os.path.join(self.image_dir, 'render', '%03d_render_src-%03d_view-%03d.png' % (epoch, idx, i)) )

        log.info(f"Rendering finished in {time.time()-start} seconds.")
        # render_dict = {'coord': self.fix_render_x.cpu(), 'rgb': rgb_2d.cpu().detach(), 'mask': self.fix_render_hit.cpu().detach()}
        # with open(os.path.join(self.image_dir, 'render', 'render_dict.pkl'), 'wb') as f:
        #     pickle.dump(render_dict, f)

        # return render_dict

    def eval_edit(self, step, write_to_wandb=True):
        sim_ori, sim_tgt, clip_direc_sim, sim_image, sim_view, lpips_image = self.clip_metrics(self.cur_imgs, self.cfg.caption_tgt)
        if write_to_wandb:
            wandb.log({'metrics/sim_ori': sim_ori, 
                    'metrics/sim_tgt': sim_tgt, 
                    'metrics/clip_direc_sim': clip_direc_sim, 
                    'metrics/sim_image': sim_image,
                    'metrics/clip_view_consis': sim_view, 
                    'metrics/lpips_image': lpips_image, 
                    }, step=step)
        else:
            log.info(f"sim_ori: {sim_ori}, sim_tgt: {sim_tgt}, clip_direc_sim: {clip_direc_sim}, clip_view_consis: {sim_view}, sim_img: {sim_image}, lpips_img: {lpips_image}")

    def reposing(self, idx, target_smpl_obj, epoch=None):
        '''
        Reconstruct the mesh the idx-th subject. given the target smpl obj.
        '''
        
        if epoch is None:
            epoch = 0
        smpl_V, _ = load_obj(target_smpl_obj, load_materials=False)
        log.info(f"Reposing {idx}th mesh at epoch {epoch}...")
        start = time.time()

        with torch.no_grad():

            tmp_smpl_V = self.sdf_field.get_smpl_vertices_by_idx(idx)

            # just replace a whole mesh obj
            self.sdf_field.replace_smpl_vertices_by_idx(idx, smpl_V)
            self.rgb_field.replace_smpl_vertices_by_idx(idx, smpl_V)

            h = self._marching_cubes (geo_idx=idx, subdivide=self.subdivide, res=self.res)

            # replace back, so that the new pose is not stored
            self.sdf_field.replace_smpl_vertices_by_idx(idx, tmp_smpl_V)
            self.rgb_field.replace_smpl_vertices_by_idx(idx, tmp_smpl_V)

        h.export(os.path.join(self.mesh_dir, '%03d_repose_src-%03d.obj' % (epoch, idx)) )
        end = time.time()
        log.info(f"Reposing finished in {end-start} seconds.")

    def drive_by_motion(self, code_idx: int, save_path, motion_folder):
        """
        Drive the edited human by a series of smplx objects
        - name: name to be used as the save path name
        """

        torch.cuda.empty_cache()

        # ! ------- Preparation --------

        # * load checkpoint
        _, tex_feats, _, _ = self.load_edit_checkpoint(self.cfg.edit_checkpoint_file)
        with torch.no_grad():
            # ori_tex_feats= self.rgb_field.get_feature_by_idx(code_idx).clone().unsqueeze(0).detach().data
            self.rgb_field.replace_feature_by_idx(code_idx, tex_feats)


        for _, _, files in os.walk(motion_folder):
            for file_name in tqdm(sorted(files)):
                if file_name[-4:] != '.obj':
                    print(f'skip {file_name}')
                    continue

                log.info(f"Reposing {file_name}th mesh...")
                smpl_V, _ = load_obj(os.path.join(motion_folder, file_name), load_materials=False)
                start = time.time()

                with torch.no_grad():
                    # just replace a whole mesh obj
                    self.sdf_field.replace_smpl_vertices_by_idx(code_idx, smpl_V)
                    self.rgb_field.replace_smpl_vertices_by_idx(code_idx, smpl_V)
                    # h = self._marching_cubes (geo_idx=code_idx, subdivide=self.subdivide, res=self.res)
                # h.export(os.path.join(save_path, 'mesh', f'edited_{file_name}') )

                # * render
                with torch.no_grad():
                    log.info(f"Rendering {file_name} subject...")
                    start = time.time()
                    ray_o_images, ray_d_images = self._get_camera_rays(n_views=self.cfg.n_views, fov=self.cfg.fov, width=self.cfg.width)

                    _idx = torch.tensor([code_idx], dtype=torch.long, device = torch.device('cuda')).repeat(self.cfg.n_views).detach()
                    x, hit = self.tracer(self.sdf_field.forward, _idx,
                                    ray_o_images.view(self.cfg.n_views, -1, 3),
                                    ray_d_images.view(self.cfg.n_views, -1, 3))
                    log.info(f"Ray tracing finished in {time.time()-start} seconds.")
                    rgb_2d = self.rgb_field.forward(x.detach(), _idx) * hit

                rgb_2d = torch.cat((rgb_2d, hit[:,:,0].unsqueeze(-1)),dim=2)
                rgb_img = rgb_2d.reshape(self.cfg.n_views, self.cfg.width, self.cfg.width, -1).cpu().detach().numpy() * 255

                for i in range(self.cfg.n_views):
                    Image.fromarray(rgb_img[i].astype(np.uint8)).save(
                        os.path.join(save_path, 'render', f'render_{file_name[:-4]}_view-{i:03d}.png') )

                end = time.time()
                log.info(f"Reposing finished in {end-start} seconds.")

    def transfer_features(self, src_idx, tar_idx, vert_idx=None):
        '''
        Copy the features from src_idx to tar_idx at vert_idx.
        '''
        with torch.no_grad():
            src_geo = self.sdf_field.get_feature_by_idx(src_idx, vert_idx=vert_idx).clone()
            src_tex = self.rgb_field.get_feature_by_idx(src_idx, vert_idx=vert_idx).clone()
            self.sdf_field.replace_feature_by_idx(tar_idx, src_geo, vert_idx=vert_idx)
            self.rgb_field.replace_feature_by_idx(tar_idx, src_tex, vert_idx=vert_idx)


    def fitting_3D(self, code_idx, target_mesh, target_smpl_obj, num_steps=300, fit_nrm=False, fit_rgb=False):
        """Fitting the latent code to the target mesh.
           Store the optimzed code in the code_idx-th entry of the codebook.
        """

        torch.cuda.empty_cache()

        geo_code = self.sdf_field.get_mean_feature().clone().unsqueeze(0).detach().data
        tex_code = self.rgb_field.get_mean_feature().clone().unsqueeze(0).detach().data

        geo_code.requires_grad = True
        tex_code.requires_grad = True

        V, F, texv, texf, mats = load_obj(target_mesh, load_materials=True)
        smpl_V, _ = load_obj(target_smpl_obj, load_materials=False)
        smpl_V = smpl_V.cuda()

        params = []
        params.append({'params': geo_code, 'lr': 0.005})
        params.append({'params': tex_code, 'lr': 0.01})

        optimizer = torch.optim.Adam(params, betas=(0.9, 0.999))
        loop = tqdm(range(num_steps))
        log.info(f"Start fitting latent code to the target mesh...")
        for i in loop:
            coord_1 = point_sample(V.cuda(), F.cuda(), ['near', 'trace', 'rand'], 20000, 0.01)
            coord_2 = point_sample(smpl_V, self.smpl_F.cuda(), ['near', 'trace'], 50000, 0.2)
            coord = torch.cat((coord_1, coord_2), dim=0)
            rgb, nrm, sdf = closest_tex(V.cuda(), F.cuda(), texv.cuda(), texf.cuda(), mats, coord.cuda())
            coord = coord.unsqueeze(0)
            sdf = sdf.unsqueeze(0)
            rgb = rgb.unsqueeze(0)
            nrm = nrm.unsqueeze(0)

            sdf_loss = torch.tensor(0.0).cuda()
            nrm_loss = torch.tensor(0.0).cuda()
            rgb_loss = torch.tensor(0.0).cuda()

            optimizer.zero_grad()

            pred_sdf = self.sdf_field.forward_fitting(coord, geo_code, smpl_V.unsqueeze(0))
            sdf_loss += torch.abs(pred_sdf - sdf).mean()

            if fit_rgb:
                pred_rgb = self.rgb_field.forward_fitting(coord, tex_code, smpl_V.unsqueeze(0))
                rgb_loss += torch.abs(pred_rgb - rgb).mean()

            if fit_nrm:
                pred_nrm = self.sdf_field.normal_fitting(coord, tex_code, smpl_V.unsqueeze(0))
                nrm_loss += torch.abs(pred_nrm - nrm).mean()
            

            loss = 10*sdf_loss + rgb_loss + nrm_loss
            loss.backward()
            optimizer.step()
            loop.set_description('Step [{}/{}] Total Loss: {:.4f} - L1:{:.4f} - RGB:{:.4f} - NRM:{:.4f}'
                           .format(i, num_steps, loss.item(), sdf_loss.item(), rgb_loss.item(), nrm_loss.item()))

        log.info(f"Fitting finished. Store the optimized code and the new SMPL pose in the codebook.")

        with torch.no_grad():
            self.sdf_field.replace_feature_by_idx(code_idx, geo_code)
            self.rgb_field.replace_feature_by_idx(code_idx, tex_code)
            self.sdf_field.replace_smpl_vertices_by_idx(code_idx, smpl_V)
            self.rgb_field.replace_smpl_vertices_by_idx(code_idx, smpl_V)


    def fitting_2D(self, code_idx, target_dict, target_smpl_obj=None, num_steps=500):
        """Fitting the color latent code to the rendered images
           Store the optimzed code in the code_idx-th entry of the codebook.
           Given target images of n_views (target_dict contains rgb, coord, and mask),
           this function takes out texture features of given subject, and train it, and then put it back.
        """

        torch.cuda.empty_cache()

        tex_code = self.rgb_field.get_feature_by_idx(code_idx).clone().unsqueeze(0).detach().data
        tex_code.requires_grad = True

        rgb = target_dict['rgb'].cuda()
        coord = target_dict['coord'].cuda()
        mask = target_dict['mask'].cuda()

        b_size = rgb.shape[0] # b_size = n_views


        inputs = []
        targets = []
        for i in range(b_size):
            _xyz = coord[i]
            _rgb = rgb[i]
            _mask = mask[i, :, 0]
            inputs.append(_xyz[_mask].view(1,-1,3))
            targets.append(_rgb[_mask].view(1,-1,3))

        inputs = torch.cat(inputs, dim=1)
        targets = torch.cat(targets, dim=1)

        if target_smpl_obj is not None:
            smpl_V, _ = load_obj(target_smpl_obj, load_materials=False)
            smpl_V = smpl_V.cuda()
        else:
            smpl_V = self.rgb_field.get_smpl_vertices_by_idx(code_idx)

        params = []
        params.append({'params': tex_code, 'lr': 0.005})

        optimizer = torch.optim.Adam(params, betas=(0.9, 0.999))
        loop = tqdm(range(num_steps))


        for i in loop:

            rgb_loss = torch.tensor(0.0).cuda()

            optimizer.zero_grad()

            pred_rgb = self.rgb_field.forward_fitting(inputs, tex_code, smpl_V.unsqueeze(0))
            rgb_loss += torch.abs(pred_rgb - targets).mean()

            rgb_loss.backward()
            optimizer.step()
            loop.set_description('Step [{}/{}] Total Loss: {:.4f}'.format(i, num_steps, rgb_loss.item()))

        with torch.no_grad():
            self.rgb_field.replace_feature_by_idx(code_idx, tex_code)
            #self.rgb_field.replace_smpl_vertices_by_idx(code_idx, smpl_V)


    def weighted_cam_sampl(self, calc_weight:bool):
        """
        use self.region_weights to generate random sample indices
        generate self.sample_indices: a list of length random_views_total; each element in the list is a tuple of (region_idx, view_idx). region_idx in range [0,5), view_idx in range [0,10)
        self.region_weights: tensor of shape 5. Each value is the weight of a region. All 5 weights sum up to 1.
        input:
            random_views_total: total number of random views
        """

        if calc_weight:
            regions = torch.tensor([2, 3, 4], device=self.ip2p_device)
            views_per_region = [6, 6, 4]

            region_idx_list = []
            view_idx_list = []
            for region_idx, views_count in zip(regions, views_per_region):
                region_idx_repeated = region_idx.repeat(views_count)
                view_indices = torch.randperm(10, device=self.ip2p_device)[:views_count]
                region_idx_list.append(region_idx_repeated)
                view_idx_list.append(view_indices)
            region_idx_tensor = torch.cat(region_idx_list)
            view_idx_tensor = torch.cat(view_idx_list)
            self.sample_indices = torch.stack((region_idx_tensor, view_idx_tensor), dim=1)
            perm = torch.randperm(self.sample_indices.size(0))
            self.sample_indices = self.sample_indices[perm]
            return


        indices_list = []

        for region_idx, num_views in enumerate(self.views_per_region):
            full_cycles, remainder = divmod(num_views.item(), 10)
            
            # If num_views <= 10, use torch.randperm for sampling without replacement
            if full_cycles == 0:
                view_indices = torch.randperm(10, device=self.ip2p_device)[:num_views]
            else:
                # For num_views > 10, repeat range [0, 10) and add random remainder
                repeated_range = torch.tile(torch.arange(10, device=self.ip2p_device), (full_cycles,))
                if remainder > 0:
                    additional_indices = torch.randperm(10, device=self.ip2p_device)[:remainder]
                    view_indices = torch.cat((repeated_range, additional_indices), 0)
                else:
                    view_indices = repeated_range
            
            region_indices = torch.full_like(view_indices, region_idx)
            indices_list.append(torch.stack((region_indices, view_indices), dim=1))
            
        self.sample_indices = torch.cat(indices_list, dim=0)
        perm = torch.randperm(self.sample_indices.size(0))
        self.sample_indices = self.sample_indices[perm]
    
    def cal_view_weight(self, acc_tex_grad_magn, random_views_total):
        with open('./data/vertex_regions.json', 'r') as f:
            j = json.load(f)
            face_indices = torch.tensor(j['face']) # 3015
            back_head_indices = torch.tensor(j['back_head']) # 617
            side_body_indices = torch.tensor(j['side_body']) # 3466
            front_body_indices = torch.tensor(j['front_body']) # 1726
            back_body_indices = torch.tensor(j['back_body']) # 1721
        region_face_w = acc_tex_grad_magn[face_indices].mean()
        region_back_head_w = acc_tex_grad_magn[back_head_indices].mean()
        region_side_body_w = acc_tex_grad_magn[side_body_indices].mean()
        region_front_body_w = acc_tex_grad_magn[front_body_indices].mean()
        region_back_body_w = acc_tex_grad_magn[back_body_indices].mean()

        # 'face', 'head_back', 'front', 'back', 'side'
        magnitudes = torch.tensor([region_face_w, region_back_head_w, region_front_body_w, region_back_body_w, region_side_body_w])
        magn_var = torch.tensor([acc_tex_grad_magn[face_indices].var(), 
                                    acc_tex_grad_magn[back_head_indices].var(), 
                                    acc_tex_grad_magn[side_body_indices].var(), 
                                    acc_tex_grad_magn[front_body_indices].var(), 
                                    acc_tex_grad_magn[back_body_indices].var()])

        w = magnitudes
        self.region_weights = w / w.sum()
        views_per_region = (self.region_weights * random_views_total).int()
        
        while views_per_region.sum() < random_views_total:
            views_per_region[torch.argmin(views_per_region)] += 1
        while views_per_region.sum() > random_views_total:
            views_per_region[torch.argmax(views_per_region)] -= 1
        self.views_per_region = views_per_region
        log.info(f'magn: {magnitudes}\nmagn var: {magn_var}\nregion weights: {self.region_weights}\nviews per region: {self.views_per_region}')
        wandb.log({'region_weight/magnitudes': magnitudes}, step=1)
        wandb.log({'region_weight/region_weights': self.region_weights}, step=1)
        wandb.log({'region_weight/views': self.views_per_region}, step=1)


    
    

    def log(self, loop, loss_ls, step, epoch, region_and_idx, curr_view_no, ttl_view_no):
        """Log the editing iteration information.
        """

        loss_str = ''
        for loss_item in loss_ls:
            key, value = loss_item[0], loss_item[1]
            loss_str += f', {key}: {value:.4f}'
            wandb.log({f'loss/{key}': value}, step=step)

        loop.set_description('Step [{}/{}], region-{}-{} [{}/{}]{}'.format(
                step, 
                epoch, 
                region_and_idx[0],
                region_and_idx[1],
                curr_view_no,
                ttl_view_no,
                loss_str
            ))

    def write_images(self, epoch, imgs):
        """Write images to wandb.
        """
        wandb.log({"Rendered Images": [wandb.Image(imgs[i]) for i in range(imgs.shape[0])]}, step=epoch)


    def save_edit_checkpoint(self, step, tex_feats, optimizer, filename="edit_checkpoint.pth.tar", replace=False):
        """Save checkpoint."""
        if not replace:
            filename = filename.replace('.pth.tar', f'_step{step:04d}.pth.tar')
        if '/' not in filename:
            model_fname = os.path.join(self.log_dir, 'cp', filename)
        else:
            model_fname = os.path.join(self.log_dir, 'cp', filename.split('/')[-1])

        state = {
            'step': step,
            'tex_feats': tex_feats.cpu().detach(),
            'optimizer': optimizer.state_dict(),
        }
        if self.cfg.train_nerf:
            state['rgb'] = self.rgb_field.state_dict()
        torch.save(state, model_fname)

    def load_edit_checkpoint(self, filename="edit_checkpoint.pth.tar"):
        """Load checkpoint."""
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            step = checkpoint['step']
            tex_feats = checkpoint['tex_feats'].to(self.ip2p_device).requires_grad_()
            optimizer_state = checkpoint['optimizer']
            if self.cfg.train_nerf and 'rgb' in checkpoint:
                    rgb_state = checkpoint['rgb']
            else: # this may happen when 1. we just start the finetuning 2. we don't train nerf
                rgb_state = None
            
            print(f"=> Loaded checkpoint at step {step}")
            return step, tex_feats, optimizer_state, rgb_state
        else:
            print(f"=> No checkpoint found at '{filename}'")
            return None, None, None, None

