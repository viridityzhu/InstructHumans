import os
import pickle
import cv2
import torch

import torchvision.transforms as transforms

def update_edited_images(image_path, pickle_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    img_list = [ os.path.join(image_path, f) for f in sorted(os.listdir(image_path)) if f.endswith('.png') ]
    transform = transforms.Compose([
                transforms.ToTensor()
                ])
    for i, img in enumerate(img_list):
        rgb_img = cv2.imread(img)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb = transform(rgb_img).permute(1,2,0).view(-1, 3)
        data['rgb'][i] = rgb

    return data

def crop_to_mask_tensor(image_tensor, mask):
    """
    Crop a 3D image tensor to the region specified by the mask.
    ! ensure the dimensions of the cropped area are multiples of 16.

    Parameters:
    - image_tensor : torch.Tensor
        A 3D tensor representing the image with shape (B, C, H, W).
    - mask : torch.Tensor
        A 2D tensor representing the mask with shape (1, H, W), containing True for
        pixels to keep and False for the background.

    Returns:
    - torch.Tensor
        Cropped image tensor.
    """
    if image_tensor.size(2) != mask.size(1) or image_tensor.size(3) != mask.size(2):
        raise ValueError("The size of the mask must match the height and width of the image tensor")

    non_background_indices = torch.nonzero(mask.squeeze(0), as_tuple=False)
    # Find the bounding box of the non-background pixels
    min_height = torch.min(non_background_indices[:, 0]).item()
    max_height = torch.max(non_background_indices[:, 0]).item() + 1
    min_width = torch.min(non_background_indices[:, 1]).item()
    max_width = torch.max(non_background_indices[:, 1]).item() + 1

    # Adjust the bounding box to make its dimensions multiples of 16
    def adjust_dimension(min_dim, max_dim, max_allowed):
        dim_length = max_dim - min_dim
        adjustment = (16 - dim_length % 16) % 16
        min_dim -= adjustment // 2
        max_dim += adjustment - adjustment // 2

        # Ensure dimensions are within image bounds
        min_dim = max(min_dim, 0)
        max_dim = min(max_dim, max_allowed)

        # Readjust if the adjustment pushed dimensions out of bounds
        if max_dim - min_dim != dim_length + adjustment:
            adjustment = (16 - (max_dim - min_dim) % 16) % 16
            if min_dim == 0:
                max_dim += adjustment
            elif max_dim == max_allowed:
                min_dim -= adjustment
            else:
                max_dim += adjustment // 2
                min_dim -= adjustment - adjustment // 2

        return max(min_dim, 0), min(max_dim, max_allowed)

    min_height, max_height = adjust_dimension(min_height, max_height, mask.size(1))
    min_width, max_width = adjust_dimension(min_width, max_width, mask.size(2))

    # Crop the image tensor to the adjusted bounding box
    cropped_tensor = image_tensor[..., min_height:max_height, min_width:max_width]

    return cropped_tensor