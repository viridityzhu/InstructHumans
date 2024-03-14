"""
usage:
python drive_motion.py  
    --id 9 # subject id (only affect the geometry)
    --load_edit_checkpoint True 
    --edit_checkpoint_file checkpoints/joker9_decom1_weightcam/cp/checkpoint_step1000.pth.tar  # texture checkpoint
    --motion-folder game_motion/subset_0001/Dance_Back # many obj files defining the motion
    --output-name joker9 # output folder's name
    --n_views 4 # rendered views per frame
"""
import os, sys
import logging as log
import numpy as np
import torch
import pickle
import random
import json
from lib.models.instruct_editor import InstructEditor
from lib.models.trainer import Trainer

from lib.utils.config import *
from tools.convert_video import create_video_from_images


def main(config):
    
    # Set random seed.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    log_dir = config.pretrained_root # if config.save_path is None else config.save_path
    print(f'log dir: {log_dir}')

    with open('data/smpl_mesh.pkl', 'rb') as f:
        smpl_mesh = pickle.load(f)

    trainer = Trainer(config, smpl_mesh['smpl_V'], smpl_mesh['smpl_F'], log_dir)
    trainer.load_checkpoint(os.path.join(config.pretrained_root, config.model_name))

    instruct_editor = InstructEditor(config, log_dir)
    instruct_editor.init_models(trainer)
    
    if config.output_name is not None:
        output_name = config.output_name
    else:
        output_name = config.edit_checkpoint_file.split('/')[-3] # <name>/cp/checkpoint_step1000.pth.tar
    motion_name = config.motion_folder.split('/')[-1]
    save_path = os.path.join(config.output_path, f'{motion_name}_{output_name}')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'mesh'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'render'), exist_ok=True)
    print(f'Generating motion {motion_name} of {output_name}, saving to {save_path}...')
    instruct_editor.drive_by_motion(config.id, save_path=save_path, motion_folder=config.motion_folder)

    # create video
    output_video_file = os.path.join(save_path, f'{motion_name}_{output_name}.mp4')
    print(f'Creating video {output_video_file}...')
    create_video_from_images(os.path.join(save_path, 'render'), output_video_file, fps=30, view_id=0)

if __name__ == "__main__":

    parser = parse_options()
    parser.add_argument('--pretrained-root', type=str, default='checkpoints/demo', help='pretrained model path')
    parser.add_argument('--model-name', type=str, default='model-1000.pth', help='load model name')
    parser.add_argument('--output-path', type=str, default='test/outputs', required=False, help='path to save the outputs')
    parser.add_argument('--output-name', type=str, required=False, help='path to save the outputs')

    parser.add_argument('--id', type=int, default=32, help='id of the human')
    # parser.add_argument('--motion-folder', type=str, default='game_motion/subset_0001/Dance_Dance', help='path to the folder containing smplx obj files of a series of poses')
    parser.add_argument('--motion-folder', type=str, default='test/motion_data/HAA500/subset_0002/Battle-rope_Jumping-jack_clip_3', help='path to the folder containing smplx obj files of a series of poses')

    args, args_str = argparse_to_str(parser)
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=args.log_level,
                        format='%(asctime)s|%(levelname)8s| %(message)s',
                        handlers=handlers)
    logging.info(f'Info: \n{args_str}')
    main(args)