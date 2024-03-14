import matplotlib.pyplot as plt
plt.switch_backend('agg') # avoid using GUI, which caused an odd error.
from typing import Any
import numpy as np
import torch
import torchvision.transforms.functional as TF
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap, Normalize

import wandb

class Visualizer:
    def __init__(self):
        super().__init__()
    
    def __call__(self, step, grad_sds: torch.Tensor, rgb_2d: torch.Tensor, pred_rgb: torch.Tensor, prev_img: torch.Tensor, edit_img: torch.Tensor, tex_grad: torch.Tensor, V, save_path) -> Any:
        grad_np = grad_sds.squeeze().cpu().numpy()


        fig, axs = plt.subplots(3, 4, figsize=(15, 15))

        colormaps = ['viridis', 'viridis', 'viridis']

        rgb_2d_np = rgb_2d.detach().squeeze().permute(1,2,0).cpu().numpy().astype(np.float32)
        pred_rgb_np = pred_rgb.detach().squeeze().permute(1,2,0).cpu().numpy().astype(np.float32)
        prev_img_np = prev_img.detach().squeeze().permute(1,2,0).cpu().numpy().astype(np.float32)
        edit_img_np = edit_img.detach().squeeze().permute(1,2,0).cpu().numpy().astype(np.float32)

        img_list = [rgb_2d_np, pred_rgb_np, prev_img_np, edit_img_np]
        title_list = ['Original Rendered', 'Current Rendered', 'Guidance Image (One denoising step)', '20 denoising steps']
        # 1 row: Display the original images
        for i in range(4):
            axs[0, i].imshow(img_list[i])
            axs[0, i].set_title(title_list[i])
            axs[0, i].axis('off')  # Turn off axis for these images

        # Resize the image to match the target size
        resized_pred_np = TF.resize(pred_rgb, prev_img.shape[-2:]).detach().squeeze().permute(1,2,0).cpu().numpy().astype(np.float32)

        cur_ori = self.scale_difference(image1=pred_rgb_np, image2=rgb_2d_np)
        prev_pred = self.scale_difference(image1=prev_img_np, image2=resized_pred_np)

        img_list = [cur_ori, prev_pred]
        title_list = ['Difference: Current - Original', 'Difference: Guidance - Current', 'Normed grad per Vertex']
        # 2 row: Display the original images
        for i in range(2):
            im = axs[1, i].imshow(img_list[i])
            axs[1, i].set_title(title_list[i])
            axs[1, i].axis('off')  # Turn off axis for these images

        axs[1, 2].remove()  # Remove the existing 2D axis
        axs[1, 2] = fig.add_subplot(3, 3, 6, projection='3d')
        self.cal_grad_per_vert(axs[1, 2], title_list[2], tex_grad, V)

        axs[1, 3].axis('off')

        title_list = ['Gradient of image - Red Channel', 'grad of image - Green Channel', 'grad of image - Blue Channel']
        # 3 row: Plot each channel's gradient
        for i in range(3):
            # gradi = self.scale_difference(diff=grad_np[i]) # no need to scale as the colormap is auto determined by the min and max values
            im = axs[2, i].imshow(grad_np[i], cmap=colormaps[i])
            axs[2, i].axis('off')  # Turn off axis
            axs[2, i].set_title(title_list[i])
            fig.colorbar(im, ax=axs[2, i], fraction=0.046, pad=0.04)  # Colorbar for each channel

        axs[2, 3].axis('off')

        # Adjust layout and save the figure
        plt.tight_layout()
        wandb.log({"Visualization": plt}, step=step)
        # plt.savefig(save_path)
        plt.close(fig)

    def cal_grad_per_vert(self, ax, title, tex_grad, V, color_min=(0, 0, 1), color_max=(1, 0, 0)):
        """
        Visualize the gradient map on a 3D mesh and save as an image file using matplotlib.
        Uses a linear gradient of colors between color_min and color_max.
        Parameters:
        ax (matplotlib axis): The axis to plot on.
        title (str): Title of the plot.
        tex_grad (torch.Tensor): Tensor of gradient vectors at mesh vertices.
        V (numpy.array): Vertices of the mesh.
        color_min (tuple): RGB color for the minimum gradient value.
        color_max (tuple): RGB color for the maximum gradient value.
        """
        # calculate L2 norm for gradient vectors at each vertex
        if tex_grad.ndim > 1:
            tex_grad = torch.sqrt(torch.sum(tex_grad**2, dim=-1)).squeeze()

        if isinstance(tex_grad, torch.Tensor):
            tex_grad = tex_grad.detach().cpu().numpy()

        norm = Normalize(vmin=tex_grad.min(), vmax=tex_grad.max())

        # linear segmented colormap
        color_map = LinearSegmentedColormap.from_list("grad_colormap", [color_min, color_max])

        colors = color_map(norm(tex_grad))

        # plotting the mesh with vertex colors
        ax.scatter(-V[:, 1], V[:, 0], V[:, 2], c=colors, marker='o', s=4) # Adjust 's' to change point size

        ax.set_title(title)
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.view_init(elev=90, azim=0)
        # adjust the axis limits to fit the data
        ax.auto_scale_xyz(-V[:, 1], V[:, 0], V[:, 2])
        ax.axis('on')

    def scale_difference(self, image1=None, image2=None, diff=None):
        """
        Provide either two images or difference of the two images
        """
        if diff is None:
            diff = image1 - image2
        
        max_diff = np.max(np.abs(diff))

        # Scale the differences to be in the range [-0.5, 0.5] and shift to [0, 1]
        # This will make 0 difference map to 0.5 (neutral gray)
        if max_diff != 0:
            scaled_diff = 0.5 + (diff / (2 * max_diff))
        else:
            scaled_diff = np.zeros_like(diff) + 0.5

        return scaled_diff

