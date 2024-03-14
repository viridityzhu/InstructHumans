"""usage: 
python test_cp.py --edit_checkpoint_file checkpoints/jap32_decom1_weightcam/cp/checkpoint_step1000.pth.tar \
--instruction "Make the person into wearing traditional Japanese kimono" \
--id 32 --caption_ori "A photo of a person" --caption_tgt "A person wearing traditional Japanese kimono" \
--traced_points_data_root prepared_region_22_32_34.h5

"""
import glob
import os, sys
import logging as log
import time
import cv2
import numpy as np
import torch
import pickle
import random
import json
from PIL import Image
import re
from lib.models.instruct_editor import InstructEditor
from lib.datasets.traced_points_dataset import TracedPointsDataset
from lib.models.trainer import Trainer

from lib.utils.config import *
from tools.convert_video import create_video_from_images

def main(config):
    
    # Set random seed.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # auto determine epoch num and save path
    # log_dir = config.pretrained_root if config.save_path is None else config.save_path
    log_dir = '/'.join(config.edit_checkpoint_file.split('/')[:2]) # must use relative path

    epoch_num = int(re.findall(r'\d+', config.edit_checkpoint_file.split('_')[-1])[0])
    print(f'log dir: {log_dir}, epoch_num: {epoch_num}')

    with open('data/smpl_mesh.pkl', 'rb') as f:
        smpl_mesh = pickle.load(f)

    trainer = Trainer(config, smpl_mesh['smpl_V'], smpl_mesh['smpl_F'], log_dir)

    trainer.load_checkpoint(os.path.join(config.pretrained_root, config.model_name))


    evaluator = InstructEditor(config, log_dir)

    evaluator.init_models(trainer)

    # load
    _, tex_feats_checkpoint, _, _ = evaluator.load_edit_checkpoint(config.edit_checkpoint_file)
    if tex_feats_checkpoint is not None:
        tex_feats = tex_feats_checkpoint
    else:
        raise Exception("Please check your checkpoint path")

    if config.use_traced_point_dataset:
        evaluator.tracedPointDataset = TracedPointsDataset(config.id)
        evaluator.tracedPointDataset.init_from_h5(config.traced_points_data_root)

    if config.cal_metric:
        assert config.caption_ori is not None, 'plz provide captions for eval'
        rendered = evaluator.render_2D(config.id, epoch=-1, fix=True) # prepare ori render for metrics calculation

    # replace texture latents
    with torch.no_grad():
        evaluator.rgb_field.replace_feature_by_idx(config.id, tex_feats)

    # recon
    # evaluator.reconstruction(config.id, epoch=config.edit_num_steps)

    if config.cal_metric:
        # Render the 3D mesh to 2D images
        rendered = evaluator.render_2D(config.id, epoch=epoch_num, fix=True)

    if config.render_more:
        # Render around
        os.makedirs(os.path.join(evaluator.image_dir, 'render_more'), exist_ok=True)
        render_more(config.render_views, config.render_width, config.id, evaluator)
        
        # create video
        image_folder=os.path.join(evaluator.image_dir, 'render_more')
        output_video_file = os.path.join(evaluator.image_dir, f'{config.id}_views{config.render_views}_wid{config.render_width}.mp4')
        fps = 30
        # Get all the image files from the folder
        image_files = sorted(glob.glob(os.path.join(image_folder, f'*.png')), key=lambda x: int(os.path.split(x)[-1].split('.')[0].split('-')[-1]))
        # image_files = sorted(glob.glob(os.path.join(image_folder, '*_view-000.png')))
        
        # Read the first image to determine the video size
        frame = cv2.imread(image_files[0])
        height, width, layers = frame.shape

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
        # fourcc = 0x00000021 # cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

        for image_file in image_files:
            frame = cv2.imread(image_file)
            out.write(frame)  # Write out frame to video

        out.release()
        # os.system(f"ffmpeg -i {os.path.abspath(output_video_file)} -vcodec libx264 -f mp4 {output_video_file}")
        log.info(f"Video saved as {output_video_file}")

def render_more(n_views, width, idx, evaluator):
    '''
    Render much more 2D images of the idx-th subject.
    '''
        

    # def render_2D(evaluator, idx, epoch=None, fix=False, write_to_wandb=False, transparent = True):
    torch.cuda.empty_cache()


    with torch.no_grad():

        log.info(f"Rendering {idx}th subject for {n_views} views...")
        # no head
        # n_views_face = n_views // 2
        # head_V = seg_verts(smpl_V, 'head')
        # head_min_bounds, head_max_bounds = torch.min(head_V, dim=0)[0], torch.max(head_V, dim=0)[0]
        # evaluator.head_pos = ((head_min_bounds + head_max_bounds) / 2).cpu()

        ray_o_images, ray_d_images = evaluator._get_camera_rays(n_views=n_views, fov=evaluator.cfg.fov, width=width)
        # no head
        # ray_o_images_face, ray_d_images_face = evaluator._get_camera_rays_focus_on_head(n_views=n_views_face, fov=evaluator.cfg.fov, width=width, head_position=evaluator.head_pos)
        # ray_o_images = torch.cat((ray_o_images, ray_o_images_face), dim=0)
        # ray_d_images = torch.cat((ray_d_images, ray_d_images_face), dim=0)

        _idx = torch.tensor([idx], dtype=torch.long, device = torch.device('cuda')).repeat(n_views).detach()
        # divide into small patches
        patchSize = 15
        xLs, hitLs = [], []
        full_cycles, remainder = divmod(n_views, patchSize)
        log.info(f'divide into {full_cycles} batches + remained {remainder}')
        for b in range(full_cycles):
            start = time.time()
            _idx_b = _idx[b*patchSize : b*patchSize + patchSize]
            ray_o_images_b = ray_o_images[b*patchSize : b*patchSize + patchSize]
            ray_d_images_b = ray_d_images[b*patchSize : b*patchSize + patchSize]
            x, hit = evaluator.tracer(evaluator.sdf_field.forward, _idx_b,
                            ray_o_images_b.view(patchSize, -1, 3),
                            ray_d_images_b.view(patchSize, -1, 3))
            xLs.append(x)
            hitLs.append(hit)
            log.info(f"Rat tracing for {b*patchSize}:{b*patchSize+patchSize} finished in {time.time()-start} seconds.")
            
        start = time.time()
        _idx_b = _idx[-remainder:]
        ray_o_images_b = ray_o_images[-remainder:]
        ray_d_images_b = ray_d_images[-remainder:]
        x, hit = evaluator.tracer(evaluator.sdf_field.forward, _idx_b,
                        ray_o_images_b.view(remainder, -1, 3),
                        ray_d_images_b.view(remainder, -1, 3))
        xLs.append(x)
        hitLs.append(hit)
        log.info(f"Ray tracing for last {-remainder} finished in {time.time()-start} seconds.")
        x = torch.cat(xLs)
        hit = torch.cat(hitLs)
        start = time.time()
        rgb_2d = evaluator.rgb_field.forward(x.detach(), _idx) * hit
        log.info(f"render finished in {time.time()-start} seconds.")

    rgba_2d = torch.cat((rgb_2d, hit[:,:,0].unsqueeze(-1)),dim=2)
    rgb_img = rgba_2d.reshape(n_views, width, width, -1).cpu().detach().numpy() * 255

    for i in range(n_views):
        Image.fromarray(rgb_img[i].astype(np.uint8)).save(
            os.path.join(evaluator.image_dir, 'render_more', 'render_src-%03d_view-%03d.png' % (idx, i)) )
    log.info(f"images saved.")

if __name__ == "__main__":

    parser = parse_options()
    parser.add_argument('--pretrained-root', type=str, default='checkpoints/demo', help='pretrained model path')
    parser.add_argument('--save-path', type=str, default=None, required=False, help='path to save the outputs')
    parser.add_argument('--model-name', type=str, default='model-1000.pth', help='load model name')
    parser.add_argument('--id', type=int, default=32, help='id of the human')

    parser.add_argument('--render-views', type=int, default=50, help='id of the human')
    parser.add_argument('--render-width', type=int, default=512, help='id of the human')

    # whether to calculate metrics. if 1, need to provide instruction, caption_ori and caption_tgt.
    parser.add_argument('--cal-metric', type=int, default=1, help='id of the human')
    parser.add_argument('--render-more', type=int, default=0, help='id of the human')
    parser.add_argument('--instruction', type=str, required=False, help='load model name')

    # usage 1: provide task name [and cpNum]
    parser.add_argument('--task-name', type=str, required=False, help='load model name')
    parser.add_argument('--cpNum', type=int, default=1000, help='id of the human')
    # usage 2: provide edit_checkpoint_file and id
    # example:
    # python -m test.test_cp --edit_checkpoint_file checkpoints/suit32_ver_no_diseng/cp/checkpoint_step2000.pth.tar --instruction "Put the person in a suit" --caption_ori "A photo of a person." --caption_tgt "A person wearing a suit" --id 32 --traced_points_data_root prepared_region_22_32_34.h5  --use_traced_point_dataset True --cal-metric 1



    args, args_str = argparse_to_str(parser)
    if args.task_name is not None:
        theId = int(re.findall(r'\d+', args.task_name)[0])
        args.id = theId
        args.load_edit_checkpoint = True
        args.edit_checkpoint_file = f'checkpoints/{args.task_name}/cp/checkpoint_step{args.cpNum}.pth.tar'
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=args.log_level,
                        format='%(asctime)s|%(levelname)8s| %(message)s',
                        handlers=handlers)
    logging.info(f'Info: \n{args_str}')
    main(args)