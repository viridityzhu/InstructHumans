# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""InstructPix2Pix module"""

# Modified from https://github.com/ashawkey/stable-dreamfusion/blob/main/nerf/sd.py

import matplotlib.pyplot as plt
plt.switch_backend('agg') # avoid using GUI, which caused an odd error.

import sys
from dataclasses import dataclass
from typing import Union

import torch
from rich.console import Console
from torch import Tensor, nn
from jaxtyping import Float
import torch.nn.functional as F

import wandb

CONSOLE = Console(width=120)

from diffusers import (
    DDIMScheduler,
    StableDiffusionInstructPix2PixPipeline,
)
from transformers import logging


logging.set_verbosity_error()
IMG_DIM = 512
CONST_SCALE = 0.18215

DDIM_SOURCE = "CompVis/stable-diffusion-v1-4"
SD_SOURCE = "runwayml/stable-diffusion-v1-5"
CLIP_SOURCE = "openai/clip-vit-large-patch14"
IP2P_SOURCE = "timbrooks/instruct-pix2pix"


@dataclass
class UNet2DConditionOutput:
    sample: torch.FloatTensor

class InstructPix2Pix(nn.Module):
    """InstructPix2Pix implementation
    Args:
        device: device to use
        num_train_timesteps: number of training timesteps
    """

    def __init__(self, device: Union[torch.device, str], num_train_timesteps: int = 1000, ip2p_use_full_precision=False) -> None:
        super().__init__()

        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.ip2p_use_full_precision = ip2p_use_full_precision

        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(IP2P_SOURCE, torch_dtype=torch.float16, safety_checker=None)
        pipe.scheduler = DDIMScheduler.from_pretrained(DDIM_SOURCE, subfolder="scheduler")
        pipe.scheduler.set_timesteps(100)
        assert pipe is not None
        pipe = pipe.to(self.device)

        self.pipe = pipe

        # improve memory performance
        pipe.enable_attention_slicing()

        self.scheduler = pipe.scheduler
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # type: ignore

        pipe.unet.eval()
        pipe.vae.eval()

        # use for improved quality at cost of higher memory
        if self.ip2p_use_full_precision:
            pipe.unet.float()
            pipe.vae.float()
        else:
            if self.device.index:
                pipe.enable_model_cpu_offload(self.device.index)
            else:
                pipe.enable_model_cpu_offload(0)

        self.unet = pipe.unet
        self.auto_encoder = pipe.vae
        self.dynamic_sds = False
        self.decom2 = 0
        self.dynamic_t = False

        CONSOLE.print("InstructPix2Pix loaded!")

    def init_sampling_strategy(self, strategy:str, total_it = None, M = 800, M2 = 150, min_step_r=0.02, max_step_r=0.80):
        '''
        strategy: 
        - 'dt+ds': non-increasing timestep sampling + SDS-E
        - 'dt+ds2': non-increasing timestep sampling + SDS-E'
        - 'dt+ds3': alternative version of SDS-E
        - 'ori': original SDS
        - other: 'dt', 'ds'
        '''
        if 'dt' in strategy:
            self.dynamic_t = True
            self.dynamic_t_sampling_init(total_it, min_step_r, max_step_r)
            self.plot_t_choice()
        if 'ds' in strategy:
            self.dynamic_sds = True
            self.M = M
            if 'ds2' in strategy:
                self.M2 = M2
                self.decom2 = 1
            elif 'ds3' in strategy:
                self.M2 = M2
                self.decom2 = 2
        if strategy != 'ori' and 'dt' not in strategy and 'ds' not in strategy:
            raise Exception("unsupported strategy. should be 'dt+ds', 'dt', 'ds', or 'ori'.")

    def plot_t_choice(self):
        t_choice_values = [t.item() for t in self.t_choice]
        plt.figure(figsize=(10, 6))
        plt.plot(t_choice_values, label='Timestep Choices')
        plt.title('Timestep Choices Over Iterations')
        plt.xlabel('Iteration Index')
        plt.ylabel('Timestep Choice')
        plt.minorticks_on()  # Enable minor ticks
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')  # Minor grid lines

        wandb.log({"t_choice_plot": plt}, step=0)
        plt.close()

    def dynamic_t_sampling_init(self, total_it, min_step_r=0.02, max_step_r=0.80):
        # --------- non-increasing t sampling ---------
        self.min_step = int(min_step_r * self.num_train_timesteps)
        self.max_step = int(max_step_r * self.num_train_timesteps)
        self.time_prior = [800, 500, 300, 100]
        r1, r2, s1, s2 = self.time_prior # r1, r2 for range, s1 s2 for exponents
        weights = torch.cat(
            (
                torch.exp( # 800-900
                    -(torch.arange(self.max_step, r1, -1) - r1)
                        / (2 * s1)
                    ),
                torch.ones(r1 - r2 + 1), # 500-800
                torch.exp( # 20-500
                        -(torch.arange(r2 - 1, self.min_step, -1) - r2) / (2 * s2)
                    ),
            )
        )
        weights = weights / torch.sum(weights)
        self.cumulative_density = torch.cumsum(weights, dim=0)
        self.iters = total_it
        self.t_choice = self.t_choice_nonlinear(self.iters)
    
    def t_choice_nonlinear(self, max_it: int):
        total_num_steps = self.max_step - self.min_step
        t_choice = []
        for i in range(0, max_it + 1):
            current_it_ratio = i / (max_it + 1)
            time_index = torch.where(
                        (self.cumulative_density - current_it_ratio) > 0
                    )[0][0]
            if time_index == 0 or torch.abs(
                self.cumulative_density[time_index] - current_it_ratio
            ) < torch.abs(
                self.cumulative_density[time_index - 1] - current_it_ratio
            ):
                t = total_num_steps - time_index
            else:
                t = total_num_steps - time_index + 1
            t = torch.clip(t, self.min_step, self.max_step + 1)
            t = torch.full((1,), t, dtype=torch.long, device=self.device)
            t_choice.append(t)
        return t_choice


    def edit_image(
        self,
        text_embeddings: Float[Tensor, "N max_length embed_dim"],
        image: Float[Tensor, "BS 3 H W"],
        image_cond: Float[Tensor, "BS 3 H W"],
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        diffusion_steps: int = 20,
        lower_bound: float = 0.70,
        upper_bound: float = 0.98
    ) -> torch.Tensor:
        """Edit an image using InstructPix2Pix. Code is from Instruct-NeRF2NeRF
        Args:
            text_embeddings: Text embeddings
            image: rendered image to edit
            image_cond: corresponding training image to condition on
            guidance_scale: text-guidance scale
            image_guidance_scale: image-guidance scale
            diffusion_steps: number of diffusion steps
            lower_bound: lower bound for diffusion timesteps to use for image editing
            upper_bound: upper bound for diffusion timesteps to use for image editing
        Returns:
            edited image
        """

        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)

        # select t, set multi-step diffusion
        T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        
        self.scheduler.config.num_train_timesteps = T.item()
        self.scheduler.set_timesteps(diffusion_steps)

        with torch.no_grad():
            # prepare image and image_cond latents
            latents = self.imgs_to_latent(image)
            image_cond_latents = self.prepare_image_latents(image_cond)

        # add noise
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[0])  # type: ignore

        # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
        for i, t in enumerate(self.scheduler.timesteps):

            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # pred noise
                latent_model_input = torch.cat([latents] * 3)
                latent_model_input = torch.cat([latent_model_input, image_cond_latents], dim=1)

                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform classifier-free guidance
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + guidance_scale * (noise_pred_text - noise_pred_image)
                + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )

            # get previous sample, continue loop
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # decode latents to get edited image
        with torch.no_grad():
            decoded_img = self.latents_to_img(latents)

        return decoded_img

    def compute_sds(
        self,
        text_embeddings:  Float[Tensor, "N max_length embed_dim"],
        image: Float[Tensor, "BS 3 H W"],
        image_cond: Float[Tensor, "BS 3 H W"],
        current_it: int,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        lower_bound: float = 0.70,
        upper_bound: float = 0.98,
        grad_clamp=1
    ) -> dict:
        """Compute sds (Score distillation sampling) using InstructPix2Pix
        Args:
            text_embeddings: Text embeddings
            image: rendered image to edit
            image_cond: corresponding training image to condition on
            guidance_scale: text-guidance scale
            image_guidance_scale: image-guidance scale
            lower_bound: lower bound for diffusion timesteps to use for image editing
            upper_bound: upper bound for diffusion timesteps to use for image editing
        Returns:
            a dict containing sds loss and others
            prev_img: unlike edit_image() which provides an edited img after whole timesteps, this is just an one step previous image
        """
        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)

        # select t, set multi-step diffusion
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        batch_size = image.shape[0]
        if self.dynamic_t:
            t = self.t_choice[current_it]
        else:
            t = torch.randint(min_step, max_step + 1, [batch_size], dtype=torch.long, device=self.device)

        # prepare image and image_cond latents
        latents = self.imgs_to_latent(image) # [1, 4, 44, 50]
        image_cond_latents = self.prepare_image_latents(image_cond) # [3, 4, 44, 50]

        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 3)
            latent_model_input = torch.cat(
                [latent_model_input, image_cond_latents], dim=1
            ) # [3, 4, 44, 50] + [3, 4, 44, 50] --> [3, 8, 44, 50]

            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
        mode_disengaging = (image_guidance_scale - 1) * (noise_pred_image - noise_pred_uncond)
        if self.decom2 == 1 or self.decom2 == 2:
            mode_disengaging_img = (guidance_scale - 1) * (noise_pred_text - noise_pred_image)
            mode_seeking_txt = noise_pred_text
            noise_pred = mode_disengaging + mode_disengaging_img + mode_seeking_txt
        elif self.decom2 == 0:
            mode_seeking = guidance_scale * (noise_pred_text - noise_pred_image) + noise_pred_image
            noise_pred = mode_disengaging + mode_seeking
        else:
            raise Exception("Unsupported sds decomposition strategy")
        # noise_pred = (
        #     noise_pred_uncond # unconditioned
        #     + guidance_scale * (noise_pred_text - noise_pred_image) # text guidance
        #     + image_guidance_scale * (noise_pred_image - noise_pred_uncond) # image guidance
        # )

        # select current term
        if self.dynamic_sds:
            if self.decom2 == 1:
                if t > self.M: # 800-900
                    # disengaging
                    curr_term_edit = mode_disengaging
                    curr_term = mode_disengaging
                elif t > self.M2:
                    curr_term_edit  = mode_disengaging_img
                    curr_term = mode_disengaging_img
                else: # 20-800
                    # seeking
                    curr_term_edit  = mode_seeking_txt
                    curr_term = mode_seeking_txt - noise
            elif self.decom2 == 2:
                if t > self.M: # 800-900
                    # disengaging
                    curr_term_edit = mode_disengaging
                    curr_term = mode_disengaging
                elif t > self.M2:
                    curr_term_edit  = mode_disengaging_img + mode_seeking_txt
                    curr_term = mode_disengaging_img + mode_seeking_txt
                else: # 20-800
                    # seeking
                    curr_term_edit  = mode_seeking_txt
                    curr_term = mode_seeking_txt - noise
            else:
                if t > self.M: # 800-900
                    # disengaging
                    curr_term_edit  = mode_disengaging
                    curr_term = mode_disengaging
                else: # 20-800
                    # seeking
                    curr_term_edit  = mode_seeking
                    curr_term = mode_seeking - noise
        else:
            curr_term_edit  = noise_pred
            curr_term = noise_pred - noise

        # decode latents to get edited image
        with torch.no_grad():
            # get previous sample
            latents_prev = self.scheduler.step(curr_term_edit.cpu(), t.cpu(), latents.cpu()).prev_sample

            if noise_pred.dtype == torch.float16:
                latents_prev = latents_prev.half()
            decoded_prev_img = self.latents_to_img(latents_prev.to(self.device))

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1) # w = 1 - alpha_t
        if noise_pred.dtype == torch.float16:
            w = w.half()

        grad = w * curr_term

        # return grad

        grad = torch.nan_to_num(grad)
        if grad_clamp > 0:
            grad = grad.clamp(-grad_clamp, grad_clamp)
        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        return {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": min_step,
            "max_step": max_step,
            "prev_img": decoded_prev_img, # unlike edit_image() which provides an edited img after many timesteps, this is just an one step previous image
        }

    def latents_to_img(self, latents: Float[Tensor, "BS 4 H W"]) -> Float[Tensor, "BS 3 H W"]:
        """Convert latents to images
        Args:
            latents: Latents to convert
        Returns:
            Images
        """

        latents = 1 / CONST_SCALE * latents

        with torch.no_grad():
            imgs = self.auto_encoder.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def imgs_to_latent(self, imgs: Float[Tensor, "BS 3 H W"]) -> Float[Tensor, "BS 4 H W"]:
        """Convert images to latents
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        posterior = self.auto_encoder.encode(imgs).latent_dist
        latents = posterior.sample() * CONST_SCALE

        return latents

    def prepare_image_latents(self, imgs: Float[Tensor, "BS 3 H W"]) -> Float[Tensor, "BS 4 H W"]:
        """Convert conditioning image to latents used for classifier-free guidance
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        image_latents = self.auto_encoder.encode(imgs).latent_dist.mode()

        uncond_image_latents = torch.zeros_like(image_latents)
        image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        return image_latents

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
