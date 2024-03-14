'''
    Codes are borrowed from Instruct NeRF2NeRF
'''
import argparse
import os
import pprint
import re
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import lpips
from einops import rearrange

from lib.utils.config import argparse_to_str, parse_options

class ClipSimilarity(nn.Module):
    def __init__(self, name: str = "ViT-L/14"):
        super().__init__()
        assert name in ("RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px")  # fmt: skip
        self.size = {"RN50x4": 288, "RN50x16": 384, "RN50x64": 448, "ViT-L/14@336px": 336}.get(name, 224)

        self.model, _ = clip.load(name, device="cuda", download_root="./")
        self.model.eval().requires_grad_(False)
        # Initialize LPIPS loss function
        self.lpips_loss_fn = lpips.LPIPS(net='vgg').cuda()  # Use VGG network for LPIPS

        self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073), device='cuda'))
        self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711), device='cuda'))

    def encode_text(self, text):
        text = clip.tokenize(text, truncate=True).to(next(self.parameters()).device)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def encode_image(self, image):  # Input images in range [0, 1].
        image = F.interpolate(image.float(), size=self.size, mode="bicubic", align_corners=False)
        image = image - rearrange(self.mean, "c -> 1 c 1 1")
        image = image / rearrange(self.std, "c -> 1 c 1 1")
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def init_ori_imgs(self, image, text):
        self.image_0 = image
        self.image_features_0 = self.encode_image(image) # 15, 768
        self.text_features_0 = self.encode_text(text).repeat(image.size(0), 1)  # [N, feature_dim]
        sim_0 = F.cosine_similarity(self.image_features_0, self.text_features_0, dim=1)
        self.sim_0_mean = torch.mean(sim_0)
        shifted_image_features_0 = torch.cat((self.image_features_0[1:], self.image_features_0[0].unsqueeze(0)), dim=0)
        self.view_direction_0 = shifted_image_features_0 - self.image_features_0

    def forward(self, image_1, text_1):
        """
        Inputs:
            - image_0 and image_1 should have shape [N, 3, H, W]. They are the original images and edited images, respectively.
            - text_0 and text_1 are two strings. They are the captions for the original human and the target human.
        Returns:
            - sim_0: mean similarity of the original images and the original caption.
            - sim_1: mean similarity of the edited images and the target caption.
            - sim_direction: mean similarity of the image editing direction and the desired text direction.
            - sim_image: mean similarity of the edited and original images.
            - sim_view_consistency: (C(oi+1) - C(oi)) * (C(ei+1) - C(ei))
        """
        image_features_1 = self.encode_image(image_1)
        text_features_1 = self.encode_text(text_1).repeat(image_1.size(0), 1)  # [N, feature_dim]
        sim_1 = F.cosine_similarity(image_features_1, text_features_1, dim=1)
        sim_direction = F.cosine_similarity(image_features_1 - self.image_features_0, text_features_1 - self.text_features_0, dim=1)
        sim_image = F.cosine_similarity(self.image_features_0, image_features_1, dim=1)

        shifted_image_features_1 = torch.cat((image_features_1[1:], image_features_1[0].unsqueeze(0)), dim=0)
        view_direction_1 = shifted_image_features_1 - image_features_1

        # Compute cosine similarities between the circular differences
        sim_view_consistency = F.cosine_similarity(self.view_direction_0, view_direction_1, dim=1)

        lpips_image = self.lpips_loss_fn(image_1, self.image_0)

        return self.sim_0_mean, torch.mean(sim_1), torch.mean(sim_direction), torch.mean(sim_image), torch.mean(sim_view_consistency), torch.mean(lpips_image)

def load_img_from_path(path):
    file_paths = [os.path.join(path, fname) for fname in os.listdir(path)]
    file_paths.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    images = [transform(Image.open(file_path)) for file_path in file_paths]
    images_tensor = torch.stack(images, dim=0).cuda()

    return images_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test IN2N')
    # parser = parse_options()
    parser.add_argument('--test-name', type=str, default='9modi', help='load model name')

    # parser.add_argument('--cam-path', type=str, default='final15_9', help='load model name')
    # parser.add_argument('--ori-image-name', type=str, default='9nerf', help='load model name')
    parser.add_argument('--cam-path', type=str, default='final15_32', help='load model name')
    parser.add_argument('--ori-image-name', type=str, default='32nerf', help='load model name')

    parser.add_argument('--image-path', type=str, default='./renders/', help='load model name')
    parser.add_argument('--instruction', type=str, required=False, help='load model name')

    parser.add_argument('--caption-ori', type=str, default='A photo of a person', help='load model name')
    # parser.add_argument('--caption-tgt', type=str, default='A photo of Batman', help='load model name')
    parser.add_argument('--caption-tgt', type=str, default= 'a Modigliani painting', help='load model name')
    


    args = parser.parse_args()
    args_dict = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        args_dict[group.title] = vars(argparse.Namespace(**group_dict))

    pp = pprint.PrettyPrinter(indent=2)
    args_str = pp.pformat(args_dict)
    args_str = f'```{args_str}```'

    print(args_str)
    
    evaluator = ClipSimilarity()

    ori_img_path = os.path.join(args.image_path, args.ori_image_name, args.cam_path)
    edited_img_path = os.path.join(args.image_path, args.test_name, args.cam_path)

    ori_images = load_img_from_path(ori_img_path)
    edited_images = load_img_from_path(edited_img_path)

    evaluator.init_ori_imgs(ori_images, args.caption_ori)
    sim_ori, sim_tgt, clip_direc_sim, sim_image, sim_view, lpips_image = evaluator.forward(edited_images, args.caption_tgt)

    print(f'Evaluating {args.test_name}')
    print(f"sim_ori: {sim_ori}, sim_tgt: {sim_tgt}, clip_direc_sim: {clip_direc_sim}, clip_view_consis: {sim_view}, sim_img: {sim_image}, lpips_img: {lpips_image}")