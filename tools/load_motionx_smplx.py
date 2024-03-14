import os
from smplx import SMPLX
import numpy as np
import torch
import json
import trimesh
import argparse
from tqdm import tqdm

SMPL_PATH = './smplx/'

class MotionxSmplxLoader:
    def __init__(self, *args, **kwargs):
        super( MotionxSmplxLoader, self).__init__(*args, **kwargs)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # body_model = SMPLX(model_path=SMPL_PATH, gender='male', use_pca=True, num_pca_comps=12, flat_hand_mean=True).to(device)
        # smplx_model = SMPLX(smplx_model_path, num_betas=10, use_pca=False, use_face_contour=True, batch_size=1).cuda()
    
    def load_motion_data(self, motion_file):
        # read motion and save as smplx representation
        motion = np.load(motion_file)
        motion = torch.tensor(motion).float().to(self.device)
        param = {
                    'root_orient': motion[:, :3],  # controls the global root orientation
                    'pose_body': motion[:, 3:3+63],  # controls the body
                    'pose_hand': motion[:, 66:66+90],  # controls the finger articulation
                    'pose_jaw': motion[:, 66+90:66+93],  # controls the yaw pose
                    'face_expr': motion[:, 159:159+50],  # controls the face expression
                    'face_shape': motion[:, 209:209+100],  # controls the face shape
                    'trans': motion[:, 309:309+3],  # controls the global body position
                    'betas': motion[:, 312:],  # controls the body shape. Body shape is static
                }

        batch_size = param['face_expr'].shape[0]
        print('batchsize', batch_size)
        self.body_model = SMPLX(model_path=SMPL_PATH, gender='male', num_betas=10, use_pca=False, use_face_contour=True, batch_size=batch_size).to(self.device)
        # J_0 = self.body_model(body_pose = param['pose_body'], betas=param['betas']).joints.contiguous().detach()
        Jtrans = self.body_model(body_pose = param['pose_body'], betas=param['betas'], transl=param['trans']).joints.contiguous().detach()


        zero_pose = torch.zeros((batch_size, 3)).float().to(self.device)

        smplx_output = self.body_model(betas=param['betas'], 
                                        body_pose=param['pose_body'],
                                        # pose2rot=True, 
                                        jaw_pose=zero_pose, 
                                        leye_pose=zero_pose, 
                                        reye_pose=zero_pose,
                                        left_hand_pose=param['pose_hand'][:, :45], 
                                        right_hand_pose=param['pose_hand'][:, 45:],
                                        # expression=param['face_expr'][:, :10],
                                        global_orient=param['root_orient'],  # try to get global transformation
                                        transl=param['trans']-Jtrans[batch_size//2,0,:])# -J_0[0,0,:].repeat(batch_size,1)) #-J_0[:,0,:],)
        
        return smplx_output


    def process(self, args):

        for _, _, files in os.walk(args.input_folder):
            for file_name in tqdm(files):
                if file_name[-4:] != '.npy':
                    print(f'skip {file_name}')
                    continue

                name = file_name[:-4]
                save_path = os.path.join(args.input_folder, name)
                os.makedirs(save_path, exist_ok=True)

                smplx_output = self.load_motion_data(os.path.join(args.input_folder, file_name))
                for idx, mesh_frame in enumerate(smplx_output.vertices.detach().cpu().numpy()):
                    d = trimesh.Trimesh(vertices=mesh_frame, faces=self.body_model.faces)
                    d.export(os.path.join(save_path, f'smplx_{file_name.replace(".npy", "")}_frame-{idx}.obj'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate SMPL-X mesh from motionx')

    # parser.add_argument("-i", "--input-folder", default='./game_motion/subset_0001', type=str, help="Input motion x folder")
    parser.add_argument("-i", "--input-folder", default='./test/motion_data/HAA500/subset_0002', type=str, help="Input motion x folder")

    loader = MotionxSmplxLoader()
    loader.process(parser.parse_args())
