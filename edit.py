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

import wandb

def main(config):
    
    # Set random seed.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    log_dir = config.pretrained_root if config.save_path is None else config.save_path
    print(f'log dir: {log_dir}')

    with open('data/smpl_mesh.pkl', 'rb') as f:
        smpl_mesh = pickle.load(f)

    trainer = Trainer(config, smpl_mesh['smpl_V'], smpl_mesh['smpl_F'], log_dir)

    trainer.load_checkpoint(os.path.join(config.pretrained_root, config.model_name))

    instruct_editor = InstructEditor(config, log_dir)

    # * wandb config
    if config.wandb_id is not None:
        wandb_id = config.wandb_id
    else:
        wandb_id = wandb.util.generate_id()
        with open(os.path.join(log_dir, 'wandb_id.txt'), 'w+') as f:
            f.write(wandb_id)

    wandb_mode = "disabled" if (not config.wandb) else "online"
    wandb.init(id=wandb_id,
               project=config.wandb_name,
               config=config,
               name=os.path.basename(log_dir),
               resume="allow",
               settings=wandb.Settings(start_method="fork"),
               mode=wandb_mode,
               dir=log_dir)
    wandb.watch(instruct_editor)

    instruct_editor.init_models(trainer)
    instruct_editor.editing_by_instruction(config.id, config.instruction, num_steps=config.edit_num_steps)
    # Generate the 3D mesh using marching cube
    instruct_editor.reconstruction(config.id, epoch=config.edit_num_steps)

    # Render the 3D mesh to 2D images
    rendered = instruct_editor.render_2D(config.id, epoch=config.edit_num_steps, fix=True)

    wandb.finish()

if __name__ == "__main__":

    parser = parse_options()
    parser.add_argument('--pretrained-root', type=str, default='checkpoints/demo', help='pretrained model path')
    parser.add_argument('--save-path', type=str, default='checkpoints/clown32', required=False, help='path to save the outputs')
    parser.add_argument('--model-name', type=str, default='model-1000.pth', help='load model name')
    parser.add_argument('--instruction', type=str, default='Turn him into a clown', required=True, help='load model name')
    parser.add_argument('--id', type=int, default=32, help='id of the human')

    args, args_str = argparse_to_str(parser)
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=args.log_level,
                        format='%(asctime)s|%(levelname)8s| %(message)s',
                        handlers=handlers)
    logging.info(f'Info: \n{args_str}')
    main(args)