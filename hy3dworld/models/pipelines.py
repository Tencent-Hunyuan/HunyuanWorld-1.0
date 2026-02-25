# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
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

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FluxIPAdapterMixin, FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL

from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DiffusionPipeline
from diffusers.pipelines.flux import FluxPipelineOutput

# try to import DecoderOutput from diffusers.models
try:
    from diffusers.models.autoencoders.vae import DecoderOutput
except Exception:
    from diffusers.models.vae import DecoderOutput

# Check if PyTorch XLA (for TPU support) is available
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

# Initialize logger for the module
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    r"""   
    Calculate the shift value for the image sequence length based on the base and maximum sequence lengths.
    Args:
        image_seq_len (`int`):
            The sequence length of the image.
        base_seq_len (`int`, *optional*, defaults to 256):
            The base sequence length.
        max_seq_len (`int`, *optional*, defaults to 4096):
            The maximum sequence length.
        base_shift (`float`, *optional*, defaults to 0.5):
            The base shift value.
        max_shift (`float`, *optional*, defaults to 1.16):
            The maximum shift value.
    Returns:
        `float`: The calculated shift value for the image sequence length.
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(
            scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    r"""
    Retrieves the latents from the encoder output based on the sample mode.
    Args:
        encoder_output (`torch.Tensor` or `FluxPipelineOutput`):
            The output from the encoder, which can be a tensor or a custom output object.
        generator (`torch.Generator`, *optional*):
            A random number generator for sampling. If `None`, the default generator is used.
        sample_mode (`str`, *optional*, defaults to `"sample"`):
            The mode for sampling latents. Can be either `"sample"` or `"argmax"`.
    Returns:
        `torch.Tensor`: The sampled or argmax latents from the encoder output.
    Raises:
        `AttributeError`: If the encoder output does not have the expected attributes for latents.
    """
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError(
            "Could not access latents of provided encoder_output")

class FluxBasePipeline(DiffusionPipeline):
    """Base class for Flux pipelines containing shared functionality."""
    
    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]
    
    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        **kwargs
    ):
        super().__init__()
        
        # Register core components
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
        )
        
        # Calculate scale factors
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) 
            if hasattr(self, "vae") and self.vae is not None else 8
        )
        
        # Initialize processors
        self._init_processors(**kwargs)
        
        # Default configuration
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length 
            if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 128
        
    def _init_processors(self, **kwargs):
        """Initialize image and mask processors."""
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        
        # Only initialize mask processor for inpainting pipeline
        if hasattr(self, 'mask_processor'):
            self.mask_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor * 2,
                vae_latent_channels=self.vae.config.latent_channels,
                do_normalize=False,
                do_binarize=True,
                do_convert_grayscale=True,
            )
    
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Generate prompt embeddings using T5 text encoder."""
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        # Convert single prompt to list
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        # Handle textual inversion if applicable
        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

        # Tokenize input
        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        
        # Check for truncation
        untruncated_ids = self.tokenizer_2(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids
        
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(
                untruncated_ids[:, self.tokenizer_max_length - 1: -1]
            )
            logger.warning(
                f"Truncated input (max_length={max_sequence_length}): {removed_text}"
            )

        # Get embeddings from T5 encoder
        prompt_embeds = self.text_encoder_2(
            text_input_ids.to(device), output_hidden_states=False
        )[0].to(dtype=dtype, device=device)

        # Expand for multiple images per prompt
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        """Generate pooled prompt embeddings using CLIP text encoder."""
        device = device or self._execution_device

        # Convert single prompt to list
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        # Handle textual inversion if applicable
        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        # Tokenize input
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        
        # Check for truncation
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids
        
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer_max_length - 1: -1]
            )
            logger.warning(
                f"CLIP truncated input (max_length={self.tokenizer_max_length}): {removed_text}"
            )
        
        # Get pooled embeddings from CLIP
        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), output_hidden_states=False
        ).pooler_output.to(dtype=self.text_encoder.dtype, device=device)

        # Expand for multiple images per prompt
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, -1
        )

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        """Main method to encode prompts using both text encoders."""
        # Handle LoRA scaling if applicable
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        # Process prompts if embeddings not provided
        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        # Reset LoRA scaling if applied
        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        # Prepare text IDs tensor
        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(
            device=device, dtype=dtype
        )

        return prompt_embeds, pooled_prompt_embeds, text_ids

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        """Create coordinate-based latent image IDs."""
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        return latent_image_ids.reshape(height * width, 3).to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        """Pack latents into sequence format."""
        latents = latents.view(
            batch_size, num_channels_latents, height // 2, 2, width // 2, 2
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(
            batch_size, (height // 2) * (width // 2), num_channels_latents * 4
        )

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        """Unpack latents from sequence format back to spatial format."""
        batch_size, num_patches, channels = latents.shape

        # Adjust dimensions for VAE scaling
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(batch_size, channels // (2 * 2), height, width)

    def blend_v(self, a, b, blend_extent):
        """Vertical blending between two tensors."""
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = (
                a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + 
                b[:, :, y, :] * (y / blend_extent)
            )
        return b

    def blend_h(self, a, b, blend_extent):
        """Horizontal blending between two tensors."""
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = (
                a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + 
                b[:, :, :, x] * (x / blend_extent)
            )
        return b

    def enable_vae_slicing(self):
        """Enable sliced VAE decoding."""
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        """Disable sliced VAE decoding."""
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        """Enable tiled VAE decoding."""
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        """Disable tiled VAE decoding."""
        self.vae.disable_tiling()

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        """Prepare initial noise latents for generation."""
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(
                batch_size, height // 2, width // 2, device, dtype
            )
            return latents.to(device=device, dtype=dtype), latent_image_ids

        # Validate generator list length
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"Generator list length {len(generator)} != batch size {batch_size}"
            )

        # Generate random noise
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        
        # Apply blending extension
        latents = torch.cat([latents, latents[:, :, :, :self.blend_extend]], dim=-1)
        width_new_blended = latents.shape[-1]

        # Pack latents and prepare IDs
        latents = self._pack_latents(
            latents, batch_size, num_channels_latents, height, width_new_blended
        )
        latent_image_ids = self._prepare_latent_image_ids(
            batch_size, height // 2, width_new_blended // 2, device, dtype
        )

        return latents, latent_image_ids, width_new_blended

    def prepare_inpainting_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        is_strength_max=True,
        timestep=None,
    ):
        """Prepare latents for inpainting pipeline."""
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        
        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(
                batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # Check if generation strength is at its maximum
        if not is_strength_max:
            image = image.to(device=device, dtype=dtype)
            image_latents = self._encode_vae_image(
                image=image, generator=generator)

        # Generate noise latents
        noise = randn_tensor(shape, generator=generator,
                             device=device, dtype=dtype)
        latents = noise if is_strength_max else self.scheduler.scale_noise(
            image_latents, timestep, noise)
        width_new_blended = latents.shape[-1]

        # Organize the latents into proper batch structure with specific shape
        latents = self._pack_latents(
            latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(
            batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids, width_new_blended

    def _predict_noise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        guidance: Optional[torch.Tensor],
        pooled_prompt_embeds: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        is_inpainting: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Predict noise using transformer with proper conditioning."""
        # Prepare transformer inputs
        transformer_inputs = {
            "hidden_states": torch.cat([latents, kwargs.get('masked_image_latents', latents)], dim=2) 
                            if is_inpainting else latents,
            "timestep": timestep / 1000,
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": self._joint_attention_kwargs,
            "return_dict": False,
        }
        
        return self.transformer(**transformer_inputs)[0]

    def _apply_blending(
        self,
        latents: torch.Tensor,
        height: int,
        width_new_blended: int,
        num_channels_latents: int,
        batch_size: int,
        **kwargs
    ) -> torch.Tensor:
        """Apply horizontal blending to latents."""
        # Unpack latents for processing
        latents_unpack = self._unpack_latents(
            latents, height, width_new_blended, self.vae_scale_factor
        )
        
        # Apply blending
        latents_unpack = self.blend_h(
            latents_unpack, latents_unpack, self.blend_extend
        )
        
        # Repack latents after blending
        return self._pack_latents(
            latents_unpack,
            batch_size,
            num_channels_latents,
            height // 8,
            width_new_blended // 8
        )
    
    def _apply_blending_mask(
        self,
        latents: torch.Tensor,
        height: int,
        width_new_blended: int,
        num_channels_latents: int,
        batch_size: int,
        **kwargs
    ) -> torch.Tensor:
        return self._apply_blending(
            latents, height, width_new_blended, 
            num_channels_latents + self.vae_scale_factor * self.vae_scale_factor, 
            batch_size, **kwargs
        )

    def _final_process_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width_new_blended: int,
        target_width: int
    ) -> torch.Tensor:
        """Final processing of latents before decoding."""
        # Unpack and crop to target width
        latents_unpack = self._unpack_latents(
            latents, height, width_new_blended, self.vae_scale_factor
        )
        latents_unpack = self.blend_h(
            latents_unpack, latents_unpack, self.blend_extend
        )
        latents_unpack = latents_unpack[:, :, :, :target_width // self.vae_scale_factor]
        
        # Repack for final output
        return self._pack_latents(
            latents_unpack,
            latents.shape[0],  # batch size
            latents.shape[2] // 4,  # num_channels_latents
            height // 8,
            target_width // 8
        )

    def _check_inputs(
        self,
        prompt: Optional[Union[str, List[str]]],
        prompt_2: Optional[Union[str, List[str]]],
        height: int,
        width: int,
        negative_prompt: Optional[Union[str, List[str]]],
        negative_prompt_2: Optional[Union[str, List[str]]],
        prompt_embeds: Optional[torch.FloatTensor],
        negative_prompt_embeds: Optional[torch.FloatTensor],
        pooled_prompt_embeds: Optional[torch.FloatTensor],
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor],
        callback_on_step_end_tensor_inputs: List[str],
        max_sequence_length: int,
        is_inpainting: bool,
        **kwargs
    ):
        """Validate all pipeline inputs."""
        # Check dimensions
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"Input dimensions should be divisible by {self.vae_scale_factor * 2}. "
                f"Got height={height}, width={width}. Will be resized automatically."
            )

        # Check callback inputs
        if callback_on_step_end_tensor_inputs is not None:
            invalid_inputs = [k for k in callback_on_step_end_tensor_inputs 
                            if k not in self._callback_tensor_inputs]
            if invalid_inputs:
                raise ValueError(
                    f"Invalid callback tensor inputs: {invalid_inputs}. "
                    f"Allowed inputs: {self._callback_tensor_inputs}"
                )

        # Check prompt vs prompt_embeds
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                "Cannot provide both prompt and prompt_embeds. Please use only one."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                "Cannot provide both prompt_2 and prompt_embeds. Please use only one."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Must provide either prompt or prompt_embeds."
            )
        elif prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError(
                f"prompt must be string or list, got {type(prompt)}"
            )
        elif prompt_2 is not None and not isinstance(prompt_2, (str, list)):
            raise ValueError(
                f"prompt_2 must be string or list, got {type(prompt_2)}"
            )

        # Check negative prompts
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                "Cannot provide both negative_prompt and negative_prompt_embeds."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                "Cannot provide both negative_prompt_2 and negative_prompt_embeds."
            )

        # Check embeddings shapes
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "prompt_embeds and negative_prompt_embeds must have same shape."
                )

        # Check pooled embeddings
        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "Must provide pooled_prompt_embeds with prompt_embeds."
            )
        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "Must provide negative_pooled_prompt_embeds with negative_prompt_embeds."
            )

        # Check sequence length
        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(
                f"max_sequence_length cannot exceed 512, got {max_sequence_length}"
            )

        # Inpainting specific checks
        if is_inpainting:
            if kwargs.get('image') is not None and kwargs.get('mask_image') is None:
                raise ValueError(
                    "Must provide mask_image when using inpainting."
                )
            if kwargs.get('image') is not None and kwargs.get('masked_image_latents') is not None:
                raise ValueError(
                    "Cannot provide both image and masked_image_latents."
                )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt
    
    def get_batch_size(self, prompt):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        return batch_size

    def prepare_timesteps(self, 
        num_inference_steps: int,
        height: int,
        width: int,
        device: Union[str, torch.device],
        sigmas: Optional[np.ndarray] = None,
    ):
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = (int(height) // self.vae_scale_factor //
                         2) * (int(width) // self.vae_scale_factor // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        return timesteps, num_inference_steps

    def prepare_blending_latent(
        self, latents, height, width, batch_size, num_channels_latents, width_new_blended=None
    ):
        # Unpack and process latents for blending
        latents_unpack = self._unpack_latents(
            latents, height, width, self.vae_scale_factor)
        latents_unpack = torch.cat(
            [latents_unpack, latents_unpack[:, :, :, :self.blend_extend]], dim=-1)
        width_new_blended = latents_unpack.shape[-1] * 8

        # Repack the processed latents
        latents = self._pack_latents(
            latents_unpack, 
            batch_size, 
            num_channels_latents, 
            height // 8, 
            width_new_blended // 8
        )
        return latents, width_new_blended

    @torch.no_grad()
    def _call_shared(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        blend_extend: int = 6,
        is_inpainting: bool = False,
        helper: Optional[Callable] = None,
        **kwargs,
    ):
        """Shared implementation between generation and inpainting pipelines."""
        # Enable tiled decoding
        self.vae.enable_tiling()
        
        def _decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
            if self.use_tiling:
                return self.tiled_decode(z, return_dict=return_dict)
            if self.post_quant_conv is not None:
                z = self.post_quant_conv(z)
            dec = self.decoder(z)
            if not return_dict:
                return (dec,)
            return DecoderOutput(sample=dec)

        def tiled_decode(
            self,
            z: torch.FloatTensor,
            return_dict: bool = True
        ) -> Union[DecoderOutput, torch.FloatTensor]:
            r"""Decode a batch of images using a tiled decoder.

            Args:
            When this option is enabled, the VAE will split the input tensor into tiles to compute decoding in several
            steps. This is useful to keep memory use constant regardless of image size.
            The end result of tiled decoding is: different from non-tiled decoding due to each tile using a different
            decoder. To avoid tiling artifacts, the tiles overlap and are blended together to form a smooth output.
            You may still see tile-sized changes in the look of the output, but they should be much less noticeable.
                z (`torch.FloatTensor`): Input batch of latent vectors. return_dict (`bool`, *optional*, defaults to
                `True`):
                    Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
            """
            overlap_size = int(self.tile_latent_min_size *
                               (1 - self.tile_overlap_factor))
            blend_extent = int(self.tile_sample_min_size *
                               self.tile_overlap_factor)
            row_limit = self.tile_sample_min_size - blend_extent

            w = z.shape[3]

            z = torch.cat([z, z[:, :, :, :2]], dim=-1)  # [1, 16, 64, 160]

            # Split z into overlapping 64x64 tiles and decode them separately.
            # The tiles have an overlap to avoid seams between tiles.
            rows = []
            for i in range(0, z.shape[2], overlap_size):
                row = []
                tile = z[:, :, i:i + self.tile_latent_min_size, :]
                if self.config.use_post_quant_conv:
                    tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                vae_scale_factor = decoded.shape[-1] // tile.shape[-1]
                row.append(decoded)
                rows.append(row)
            result_rows = []
            for i, row in enumerate(rows):
                result_row = []
                for j, tile in enumerate(row):
                    # blend the above tile and the left tile
                    # to the current tile and add the current tile to the result row
                    if i > 0:
                        tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                    if j > 0:
                        tile = self.blend_h(row[j - 1], tile, blend_extent)
                    result_row.append(
                        self.blend_h(
                            tile[:, :, :row_limit, w * vae_scale_factor:],
                            tile[:, :, :row_limit, :w * vae_scale_factor],
                            tile.shape[-1] - w * vae_scale_factor))
                result_rows.append(torch.cat(result_row, dim=3))

            dec = torch.cat(result_rows, dim=2)
            if not return_dict:
                return (dec, )
            return DecoderOutput(sample=dec)

        self.vae.tiled_decode = tiled_decode.__get__(self.vae, AutoencoderKL)
        self.vae._decode = _decode.__get__(self.vae, AutoencoderKL)
        
        self.blend_extend = blend_extend

        # Set default dimensions
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # Check inputs (handles both pipelines)
        self._check_inputs(
            prompt, prompt_2, height, width, 
            negative_prompt, negative_prompt_2,
            prompt_embeds, negative_prompt_embeds,
            pooled_prompt_embeds, negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            max_sequence_length,
            is_inpainting,
            **kwargs
        )

        # Set class variables
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs or {}
        self._interrupt = False

        # Determine if the strength is at its maximum
        if is_inpainting:
            strength = kwargs.get('strength', 1.0)
            is_strength_max = strength == 1.0

        # Determine batch size
        batch_size = self.get_batch_size(prompt)

        device = self._execution_device

        # Prepare timesteps
        timesteps, num_inference_steps = self.prepare_timesteps(
            num_inference_steps, height, width, device
        )

        # Adjust timesteps based on strength parameter
        if kwargs.get('is_inpainting', False):
            timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, kwargs['strength'], device)

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Encode prompts
        lora_scale = self._joint_attention_kwargs.get("scale", None)
        do_true_cfg = true_cfg_scale > 1 and (negative_prompt is not None or 
                                            (negative_prompt_embeds is not None and 
                                            negative_pooled_prompt_embeds is not None))
        
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        
        if do_true_cfg:
            negative_prompt_embeds, negative_pooled_prompt_embeds, _ = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # Prepare latents
        if is_inpainting:
            image = kwargs.get('image', None)

            # Create latent timestep tensor
            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
            # Get number of latent channels from VAE config
            num_channels_latents = self.vae.config.latent_channels

            latents, latent_image_ids, width_new_blended = self.prepare_inpainting_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
                self.image_processor.preprocess(
                    image, height=height, width=width).to(dtype=torch.float32),
                is_strength_max,
                latent_timestep
            )

            # if needed
            latents, width_new_blended = self.prepare_blending_latent(
                latents, height, width, batch_size, num_channels_latents, width_new_blended
            )

            # Prepare latent image IDs for the new blended width
            if not kwargs.get('blend_extra_chanel', False):
                latent_image_ids = self._prepare_latent_image_ids(
                    batch_size * num_images_per_prompt,
                    height // 16, 
                    width_new_blended // 16,
                    latents.device,
                    latents.dtype
                )
            
            # Prepare mask and masked image latents
            masked_image_latents =  kwargs.get('masked_image_latents', None)

            if masked_image_latents is not None:
                masked_image_latents = masked_image_latents.to(latents.device)
            else:
                mask_image = kwargs.get('mask_image', None)
                # Preprocess input image and mask
                image = self.image_processor.preprocess(image, height=height, width=width)
                mask_image = self.mask_processor.preprocess(mask_image, height=height, width=width)

                # Create masked image
                masked_image = image * (1 - mask_image)
                masked_image = masked_image.to(device=device, dtype=prompt_embeds.dtype)
                
                # Prepare mask and masked image latents
                height, width = image.shape[-2:]
                mask, masked_image_latents = self.prepare_mask_latents(
                    mask_image,
                    masked_image,
                    batch_size,
                    num_channels_latents,
                    num_images_per_prompt,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    kwargs.get('blend_extra_chanel', False)
                )

                # Combine mask and masked image latents
                masked_image_latents = torch.cat(
                    (masked_image_latents, mask), dim=-1)

                # if needed
                masked_image_latents, masked_width_new_blended = self.prepare_blending_latent(
                    masked_image_latents, height, width, batch_size, 
                    num_channels_latents + self.vae_scale_factor * self.vae_scale_factor,
                    width_new_blended
                )
                # update masked_image_latents
                kwargs["masked_image_latents"] = masked_image_latents
        else:
            num_channels_latents = self.transformer.config.in_channels // 4
            latents, latent_image_ids, width_new_blended = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
            width_new_blended = width_new_blended * self.vae_scale_factor

        # Handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                
                if helper != None:
                    helper.cur_timestep = i
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # Predict noise
                noise_pred = self._predict_noise(
                    latents, timestep, guidance, pooled_prompt_embeds,
                    prompt_embeds, text_ids, latent_image_ids,
                    is_inpainting, **kwargs
                )

                # Apply true CFG if enabled
                if do_true_cfg:
                    neg_noise_pred = self._predict_noise(
                        latents, timestep, guidance, negative_pooled_prompt_embeds,
                        negative_prompt_embeds, text_ids, latent_image_ids,
                        is_inpainting, **kwargs
                    )
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # Step with scheduler
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # Apply blending in early steps
                if i <= kwargs.get('early_steps', 4):
                    latents = self._apply_blending(
                        latents, height, width_new_blended, num_channels_latents, batch_size, **kwargs
                    )
                    if is_inpainting:
                        masked_image_latents = self._apply_blending_mask(
                            masked_image_latents, height, 
                            masked_width_new_blended, 
                            num_channels_latents, batch_size,
                            **kwargs
                        )

                # Fix dtype issues
                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave 
                    # due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

                # Handle callbacks
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # Update progress
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

            # Final processing
            latents = self._final_process_latents(latents, height, width_new_blended, width)

        # Decode latents
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Clean up
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)


class FluxPipeline(
    FluxBasePipeline, 
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
    FluxIPAdapterMixin,
):
    """Main Flux generation pipeline"""
    _optional_components = ["image_encoder", "feature_extractor"]
    
    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
    ):
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
        )
        
        # Register optional components
        self.register_modules(
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )

    def encode_image(self, image, device, num_images_per_prompt):
        """Encode input image into embeddings."""
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder(image).image_embeds
        return image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    
    @torch.no_grad()
    def __call__(self, **kwargs):
        """Main generation call"""
        return self._call_shared(is_inpainting=False, **kwargs)


class FluxFillPipeline(
    FluxBasePipeline, 
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
):
    """Flux inpainting pipeline."""
    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = []

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
    ):
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
        )
        # Initialize mask processor
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2,
            vae_latent_channels=self.vae.config.latent_channels,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        r"""
        Encodes the input image using the VAE and returns the encoded latents.
        Args:
            image (`torch.Tensor`):
                The input image tensor to be encoded.
            generator (`torch.Generator`):
                A random number generator for sampling.
        Returns:
            `torch.Tensor`: The encoded image latents.
        """
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(
                    image[i: i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(
                self.vae.encode(image), generator=generator)

        image_latents = (
            image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        return image_latents
    
    def get_timesteps(
        self, 
        num_inference_steps,
        strength,
        device
    ):
        timesteps = timesteps[int((1 - strength) * num_inference_steps):]
        return timesteps, num_inference_steps

    def prepare_mask_latents(
        self,
        mask,
        masked_image,
        batch_size,
        num_channels_latents,
        num_images_per_prompt,
        height,
        width,
        dtype,
        device,
        generator,
        blend_extra_chanel=False
    ):
        r""" Prepares the mask and masked image latents for the FluxFillpipeline.
        Args:
            mask (`torch.Tensor`):
                The mask tensor to be processed.
            masked_image (`torch.Tensor`):
                The masked image tensor to be processed.
            batch_size (`int`):
                The batch size for the input data.
            num_channels_latents (`int`):
                The number of channels in the latents.
            num_images_per_prompt (`int`):
                The number of images to generate per prompt.
            height (`int`):
                The height of the input image.
            width (`int`):
                The width of the input image.
            dtype (`torch.dtype`):
                The data type for the latents and mask.
            device (`torch.device`):
                The device to run the model on.
            generator (`torch.Generator`, *optional*):
                A random number generator for sampling.
        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`: A tuple containing the processed mask and masked image latents.
        """
        # 1. calculate the height and width of the latents
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        # 2. encode the masked image
        if masked_image.shape[1] == num_channels_latents:
            masked_image_latents = masked_image
        else:
            masked_image_latents = retrieve_latents(
                self.vae.encode(masked_image), generator=generator)

        masked_image_latents = (
            masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        masked_image_latents = masked_image_latents.to(
            device=device, dtype=dtype)

        # 3. duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        batch_size = batch_size * num_images_per_prompt
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(
                batch_size // masked_image_latents.shape[0], 1, 1, 1)

        # 4. pack the masked_image_latents
        # batch_size, num_channels_latents, height, width -> batch_size, height//2 * width//2 , num_channels_latents*4
        if blend_extra_chanel:
            masked_image_latents = torch.cat(
                [masked_image_latents, masked_image_latents[:, :, :, :self.blend_extend]], dim=-1)
        
        width_new_blended = masked_image_latents.shape[-1]
        masked_image_latents = self._pack_latents(
            masked_image_latents,
            batch_size,
            num_channels_latents,
            height,
            width_new_blended if blend_extra_chanel else width,
        )
        # latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        # 5.resize mask to latents shape we we concatenate the mask to the latents
        # batch_size, 8 * height, 8 * width (mask has not been 8x compressed)
        mask = mask[:, 0, :, :]
        mask = mask.view(
            batch_size, height, self.vae_scale_factor, width, self.vae_scale_factor
        )  # batch_size, height, 8, width, 8
        mask = mask.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
        mask = mask.reshape(
            batch_size, self.vae_scale_factor * self.vae_scale_factor, height, width
        )  # batch_size, 8*8, height, width
        if blend_extra_chanel:
            mask = torch.cat([mask, mask[:, :, :, :self.blend_extend]], dim=-1)
        
        # 6. pack the mask:
        # batch_size, 64, height, width -> batch_size, height//2 * width//2 , 64*2*2
        mask = self._pack_latents(
            mask,
            batch_size,
            self.vae_scale_factor * self.vae_scale_factor,
            height,
            width_new_blended if blend_extra_chanel else width,
        )
        mask = mask.to(device=device, dtype=dtype)

        return mask, masked_image_latents

    @torch.no_grad()
    def __call__(self, **kwargs):
        """Main inpainting call."""
        return self._call_shared(is_inpainting=True, **kwargs)
