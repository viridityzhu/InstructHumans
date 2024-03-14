# InstructHumans
## Installation
Our code has been tested with PyTorch 2.0.1, CUDA 11.7.

- Create conda environment:
	```bash
	conda env create -f environment.yml
	```

	Or update your existing env via:
	```bash
	conda activate InstructHumans
	conda env update -f environment.yml
	```

- `kaolin` may require to be installed separately. Check their docs - [Installation](https://kaolin.readthedocs.io/en/latest/notes/installation.html). They provided prebuilt wheels for some older versions of CUDA and pytorch.

	```bash
	TORCH_VER="2.0.1"
	CUDA_VER="117"
	pip install kaolin==0.14.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-$TORCH_VER\_cu$CUDA_VER\.html
	```

	Note the latest CUDA version supported by kaolin is 11.7. If your CUDA version (see `nvcc -V`) is unluckily too new and incompatible, you may install an older version in the conda environment:

	```bash
	conda install -c conda-forge cudatoolkit=11.7
	```

	Alternatively, you can install the compatible versions all together:
	```bash
	conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
	```

- Other missing packages
	```bash
	pip install trimesh tinyobjloader==2.0.0rc8 wandb rich jaxtyping
	pip install nerfstudio[gen]
	```

## Data preparation

### CustomHumans and SMPL-X

Apply for the CustomHumans dataset in the link: [here](https://forms.gle/oY4PKUyhH6Qqd5YA9). We use pretrained human models from their paper.

- Preparation:

1. Download the checkpoint file from CustomHumans. Make sure the model is put into the `checkpoints` folder in the following structure:

```sh
checkpoints/demo/model-1000.pth
```

2. Download [SMPL-X](https://smpl-x.is.tue.mpg.de/) models and move them to the `smplx` folder.
   You should have the following data structure:

```sh
	smplx
	├── SMPLX_NEUTRAL.pkl
	├── SMPLX_NEUTRAL.npz
	├── SMPLX_MALE.pkl
	├── SMPLX_MALE.npz
	├── SMPLX_FEMALE.pkl
	└── SMPLX_FEMALE.npz
```

## 3D Human Editing

- We provide an *optional* pre-processing to speed up the editing.

```bash
python traced_points_dataset.py
```

We presample traced points per human subject and cache them in a h5 file. This way, we can avoid repeating ray tracing and extremely speed up the editing procedure.

If you have prepared the traced points, simply add the argument `--use_traced_point_dataset True --traced_points_data_root path/to/prepared_dataset.h5` to use it in editing.

### Edit human with instruction

Simply run the below command:

```sh
python edit.py --instruction "Turn him into a clown" --id 32
```
Here are some configuration flags you can use; otherwise you can find full default settings in `config.yaml` and descriptions in `lig/utils/config.py`:
* `--instruction`: textual editing instruction.
* `--id`: human subject index. Use this to indicate the original human to be edited. They should be included in the pretrained checkpoints of CustomHumans.
* `--save-path `: path to the folder to save the checkpoints.
* `--config`: path to the config file. Default is `config.yaml`
* `--wandb`: we use wandb for monitoring the training. Activate this flag if you want to use it.
* `--caption_ori` and `--caption_tgt`: these do not affect the editing at all, but helps calculate evaluation metrics. They are captions describing the original or target images. 
* `--sampl_strategy `: to select SDS-E / SDS-E' / SDS to use, set "dt+ds" / "dt+ds2" / "ori", respectively.

### Test a pre-trained checkpoint

- We provide `test/test_cp.py` for testing a pre-trained checkpoint. Usage:

```sh
python test_cp.py --edit_checkpoint_file path/to/checkpoint_step1000.pth.tar \
    --instruction "Make the person into wearing traditional Japanese kimono" \
    --id 32 --caption_ori "A photo of a person" --caption_tgt "A person wearing traditional Japanese kimono"
```

Basically, the supported arguments are the same as `edit.py`.

- We save the evaluation metrics via *wandb*. But if wandb is disabled, the metrics should be printed instead.

## Demo for animating an edited human

1. Prepare SMPL-X models with desired pose. For example, you can download MotionX dataset, and use `tools/load_motionx_smplx.py` to convert its SMPL-X json data into `.obj` files.
  
	Example usage: `python tools/load_motionx_smplx.py -i test/motion_data/selected_motions`
2. Reposing and rendering, usage:
    ```sh
	python -m test.drive_motion \
		--id 9 # subject id (only affect the geometry) \
		--load_edit_checkpoint True \
		--edit_checkpoint_file checkpoints/joker9_decom1_weightcam/cp/checkpoint_step1000.pth.tar  # texture checkpoint \
		--motion-folder game_motion/subset_0001/Dance_Back # many obj files defining the motion \
		--output-name joker9 # output folder's name \
		--n_views 4 # rendered views per frame \
	```
	```sh
	python -m test.drive_motion \
		--id 33 \
		--load_edit_checkpoint True \
		--edit_checkpoint_file checkpoints/suit33_decom1_sm10/cp/suit33_1000.pth.tar \
		--motion-folder test/motion_data/selected_motions/Bai_Jingting_Said_It_Looks_Good_And_Then_I_Posted_It_Clip1 \
		--output-name suit33_motion1 \
		--n_views 1 \
	```
	Once down, you'll get generated rendered per frame images as well as an mp4 file in `test/outputs/`.
