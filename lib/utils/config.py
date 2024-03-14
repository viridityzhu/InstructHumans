import argparse
import pprint
import yaml
import logging

def int_or_none(value):
    if value.lower() == 'none':
        return None
    return int(value)

def list_or_none(value):
    if value.lower() == 'none':
        return None
    return list(value)

def parse_options():

    parser = argparse.ArgumentParser(description='Custom Humans Code')


    ###################
    # Global arguments
    ###################
    global_group = parser.add_argument_group('global')
    global_group.add_argument('--config', type=str, default='config.yaml', 
                               help='Path to config file to replace defaults')
    global_group.add_argument('--save-root', type=str, default='./checkpoints/', 
                               help="outputs path")
    global_group.add_argument('--exp-name', type=str, default='test',
                               help="Experiment name.")
    global_group.add_argument('--seed', type=int, default=123)
    global_group.add_argument('--resume', type=str, default=None,
                                help='Resume from the checkpoint.')
    global_group.add_argument(
        '--log_level', action='store', type=int, default=logging.INFO,
        help='Logging level to use globally, DEBUG: 10, INFO: 20, WARN: 30, ERROR: 40.')
        
    ###################
    # Arguments for dataset
    ###################
    data_group = parser.add_argument_group('dataset')
    data_group.add_argument('--data-root', type=str, default='CustomHumans.h5',
                            help='Path to dataset')
    data_group.add_argument('--num-samples', type=int, default=20480,
                            help='Number of samples to use for each subject during training')
    data_group.add_argument('--repeat-times', type=int, default=8,
                            help='Number of times to repeat each subject during training')


    ###################
    # Arguments for optimizer
    ###################
    optim_group = parser.add_argument_group('optimizer')
    optim_group.add_argument('--lr-codebook', type=float, default=0.001, 
                             help='Learning rate for the codebook.')
    optim_group.add_argument('--lr-decoder', type=float, default=0.001, 
                             help='Learning rate for the decoder.')
    optim_group.add_argument('--lr-dis', type=float, default=0.004,
                                help='Learning rate for the discriminator.')
    optim_group.add_argument('--beta1', type=float, default=0.5,
                                help='Beta1.')
    optim_group.add_argument('--beta2', type=float, default=0.999,
                                help='Beta2.')
    optim_group.add_argument('--weight-decay', type=float, default=0, 
                             help='Weight decay.')


    ###################
    # Arguments for training
    ###################
    train_group = parser.add_argument_group('train')
    train_group.add_argument('--epochs', type=int, default=800, 
                             help='Number of epochs to run the training.')
    train_group.add_argument('--batch-size', type=int, default=2, 
                             help='Batch size for the training.')
    train_group.add_argument('--workers', type=int, default=0,
                             help='Number of workers for the data loader. 0 means single process.')
    train_group.add_argument('--save-every', type=int, default=50, 
                             help='Save the model at every N epoch.')
    train_group.add_argument('--log-every', type=int, default=100,
                             help='write logs to wandb at every N iters')
    train_group.add_argument('--use-2d-from-epoch', type=int, default=-1,
                             help='Adding 2D loss from this epoch. -1 indicates not using 2D loss.')
    train_group.add_argument('--train-2d-every-iter', type=int, default=1,
                             help='Train 2D loss every N iterations.')
    train_group.add_argument('--use-nrm-dis', action='store_true',
                             help='train with normal loss discriminator.')
    train_group.add_argument('--use-cached-pts', action='store_true',
                             help='Use cached point coordinates instead of online raytracing during training.')

    ###################
    # Arguments for Feature Dictionary
    ###################
    sample_group = parser.add_argument_group('dictionary')
    sample_group.add_argument('--shape-dim', type=int, default=32,
                                help='Dimension of the shape feature code.')
    sample_group.add_argument('--color-dim', type=int, default=32,
                                help='Dimension of the color feature code.')
    sample_group.add_argument('--feature-std', type=float, default=0.1,
                                help='Standard deviation for initializing the feature code.')
    sample_group.add_argument('--feature-bias', type=float, default=0.1,
                                help='Bias for initializing the feature code.')
    sample_group.add_argument('--shape-pca-dim', type=int, default=8,
                                help='Dimension of the shape pca code.')
    sample_group.add_argument('--color-pca-dim', type=int, default=16,
                                help='Dimension of the color pca code.')
    
    ###################
    # Arguments for Network
    ###################
    net_group = parser.add_argument_group('network')
    net_group.add_argument('--pos-dim', type=int, default=3,
                          help='input position dimension')
    net_group.add_argument('--c-dim', type=int, default=0,
                          help='conditional input dimension, if 0, no conditional input')
    net_group.add_argument('--num-layers', type=int, default=4, 
                             help='Number of layers for the MLPs.')
    net_group.add_argument('--hidden-dim', type=int, default=128,
                          help='Network width')
    net_group.add_argument('--activation', type=str, default='relu',
                            choices=['relu', 'sin', 'softplus', 'lrelu'])
    net_group.add_argument('--layer-type', type=str, default='none',
                            choices=['none', 'spectral_norm', 'frobenius_norm', 'l_1_norm', 'l_inf_norm'])
    net_group.add_argument('--skip', type=int, nargs='*', default=[2],
                          help='Layer to have skip connection.')

    ###################
    # Embedder arguments
    ###################
    embedder_group = parser.add_argument_group('embedder')
    embedder_group.add_argument('--shape-freq', type=int, default=5,
                                help='log2 of max freq')
    embedder_group.add_argument('--color-freq', type=int, default=10,
                                help='log2 of max freq')


    ###################
    # Losses arguments
    ###################
    embedder_group = parser.add_argument_group('losses')
    embedder_group.add_argument('--lambda-sdf', type=float, default=1000,
                                help='lambda for sdf loss')
    embedder_group.add_argument('--lambda-rgb', type=float, default=150,
                                help='lambda for rgb loss')
    embedder_group.add_argument('--lambda-nrm', type=float, default=10,
                                help='lambda for normal loss')
    embedder_group.add_argument('--lambda-reg', type=float, default=1,
                                help='lambda for regularization loss')
    embedder_group.add_argument('--gan-loss-type', type=str, default='logistic',
                                choices=['logistic', 'hinge'],
                                help='loss type for gan loss')
    embedder_group.add_argument('--lambda-gan', type=float, default=1,  
                                help='lambda for gan loss')
    embedder_group.add_argument('--lambda-grad', type=float, default=10,
                                help='lambda for gradient penalty')

   ###################
    # Arguments for validation
    ###################
    valid_group = parser.add_argument_group('validation')
    valid_group.add_argument('--valid-every', type=int, default=10,
                             help='Frequency of running validation.')
    valid_group.add_argument('--subdivide', type=bool, default=True, 
                            help='Subdivide the mesh before marching cubes')
    valid_group.add_argument('--grid-size', type=int, default=300, 
                            help='Grid size for marching cubes')
    valid_group.add_argument('--width', type=int, default=1024, 
                            help='Image width (height) for rendering')
    valid_group.add_argument('--fov', type=float, default=20.0, 
                            help='Field of view for rendering')
    valid_group.add_argument('--n_views', type=int, default=4, 
                            help='Number of views for rendering')

    ###################
    # Arguments for wandb
    ###################
    wandb_group = parser.add_argument_group('wandb')
    
    wandb_group.add_argument('--wandb-id', type=str, default=None,
                             help='wandb id')
    wandb_group.add_argument('--wandb', action='store_true',
                             help='Use wandb')
    wandb_group.add_argument('--wandb-name', default='default', type=str,
                             help='wandb_name')

    ###################
    # Arguments for instruction-based editing
    ###################
    instruct_humans_group = parser.add_argument_group('instruct_humans')
    
    instruct_humans_group.add_argument('--ip2p_device', type=int_or_none, default=None,
                            help='ip2p device')
    instruct_humans_group.add_argument('--ip2p_use_full_precision', type=bool, default=False, 
                            help='Use ip2p_full_precision')
    instruct_humans_group.add_argument('--image_guidance_scale', type=float, default=1.5, 
                            help='image_guidance_scale')
    instruct_humans_group.add_argument('--guidance_scale', type=float, default=7.5, 
                            help='guidance_scale')
    instruct_humans_group.add_argument('--lower_bound', type=float, default=0.02, 
                            help='lower_bound')
    instruct_humans_group.add_argument('--upper_bound', type=float, default=0.98, 
                            help='upper_bound')
    instruct_humans_group.add_argument('--edit_num_steps', type=int, default=1000, 
                            help='Number of steps to edit the obj')
    instruct_humans_group.add_argument('--show_edited_img', type=bool, default=False, 
                            help='Show edited image')
    instruct_humans_group.add_argument('--show_edited_img_freq', type=int, default=10, 
                            help='Frequency to show edited image')
    instruct_humans_group.add_argument('--zoom_in_face', type=bool, default=True, 
                            help='zoom_in_face')
    instruct_humans_group.add_argument('--load_edit_checkpoint', type=bool, default=False, 
                            help='load_edit_checkpoint')
    instruct_humans_group.add_argument('--save_edit_freq', type=int, default=100, 
                            help='Frequency to save edit checkpoint')
    instruct_humans_group.add_argument('--random_views', type=int, default=50, 
                            help='no. random views in each edit iteration')
    instruct_humans_group.add_argument('--edit_checkpoint_file', type=str, default=None,
                            help='edit_checkpoint_file')
    instruct_humans_group.add_argument('--most_efficient', type=bool, default=False, 
                            help='omit all intermediate outputs')
    instruct_humans_group.add_argument('--loss_reg_weight', type=float, default=4., 
                            help='loss_reg_weight')
    instruct_humans_group.add_argument('--lr_tex_code', type=float, default=0.005, 
                            help='Learning rate for texture latents')
    instruct_humans_group.add_argument('--loss_smooth_weight', type=float, default=300, 
                            help='loss_smooth_weight')
    instruct_humans_group.add_argument('--loss_mse_weight', type=float, default=-1., 
                            help='loss_mse_weight')
    instruct_humans_group.add_argument('--loss_lpips_weight', type=float, default=-1., 
                            help='loss_lpips_weight')
    instruct_humans_group.add_argument('--visualize_more', type=bool, default=False, 
                            help='visualize grads plots. slower.')

    instruct_humans_group.add_argument('--train_nerf', type=bool, default=False, 
                            help='Train nerf parameters, instead of fixing it.')
    instruct_humans_group.add_argument('--weight_decay_nerf', type=float, default=0, 
                            help='Weight decay.')
    instruct_humans_group.add_argument('--lr_nerf', type=float, default=0.0025, 
                            help='Learning rate for rgb nerf')
    instruct_humans_group.add_argument('--sampl_strategy', type=str, default='dt+ds3',
                            help='dynamic t sampling, sds, or not. Should be either dt+ds3, dt+ds2, dt+ds, dt, ds, or ori.')
    instruct_humans_group.add_argument('--mode_change_step', type=int, default=800, 
                            help='Select the mode-disengaging term when t > this, and mode-seeking term otherwise')
    instruct_humans_group.add_argument('--mode_change_step2', type=int, default=150, 
                            help='Select the mode-disengaging term when t > this, and mode-seeking term otherwise')
    instruct_humans_group.add_argument('--min_step_rate', type=float, default=0.02, 
                            help='the rate for the min timestep to be sampled from for denoising')
    instruct_humans_group.add_argument('--max_step_rate', type=float, default=0.80, 
                            help='the rate for the max timestep to be sampled from for denoising')
    instruct_humans_group.add_argument('--caption_ori', type=str, default="A photo of a person",
                            help='caption text to describe the original human.')
    instruct_humans_group.add_argument('--caption_tgt', type=str, default="A photo of a clown",
                            help='caption text to describe the target human.')
    instruct_humans_group.add_argument('--traced_points_data_root', type=str, default='prepared_tracing.h5',
                            help='the prepared traced points data file.')
    instruct_humans_group.add_argument('--use_traced_point_dataset', type=bool, default=False, 
                            help='use the prepared traced points to speed up.')
    instruct_humans_group.add_argument('--grad_aware_cam_sampl', type=bool, default=True, 
                            help='gradient aware camera sampling')

    return parser


def parse_yaml_config(config_path, parser):
    """Parses and sets the parser defaults with a yaml config file.

    Args:
        config_path : path to the yaml config file.
        parser : The parser for which the defaults will be set.
        parent : True if parsing the parent yaml. Should never be set to True by the user.
    """
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    list_of_valid_fields = []
    for group in parser._action_groups:
        group_dict = {list_of_valid_fields.append(a.dest) for a in group._group_actions}
    list_of_valid_fields = set(list_of_valid_fields)
    
    defaults_dict = {}

    # Loads child parent and overwrite the parent configs
    # The yaml files assumes the argument groups, which aren't actually nested.
    for key in config_dict:
        for field in config_dict[key]:
            if field not in list_of_valid_fields:
                raise ValueError(
                    f"ERROR: {field} is not a valid option. Check for typos in the config."
                )
            defaults_dict[field] = config_dict[key][field]


    parser.set_defaults(**defaults_dict)

def argparse_to_str(parser, args=None):
    """Convert parser to string representation for Tensorboard logging.

    Args:
        parser (argparse.parser): Parser object. Needed for the argument groups.
        args : The parsed arguments. Will compute from the parser if None.
    
    Returns:
        args    : The parsed arguments.
        arg_str : The string to be printed.
    """
    
    if args is None:
        args = parser.parse_args()

    if args.config is not None:
        parse_yaml_config(args.config, parser)

    args = parser.parse_args()

    args_dict = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        args_dict[group.title] = vars(argparse.Namespace(**group_dict))

    pp = pprint.PrettyPrinter(indent=2)
    args_str = pp.pformat(args_dict)
    args_str = f'```{args_str}```'

    return args, args_str