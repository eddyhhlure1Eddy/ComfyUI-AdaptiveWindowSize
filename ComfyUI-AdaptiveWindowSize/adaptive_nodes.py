import os
import gc
import math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy
from PIL import Image

try:
    from comfy import model_management as mm
    from comfy.utils import ProgressBar, common_upscale, load_torch_file
    from comfy.clip_vision import clip_preprocess, ClipVisionModel
    from comfy.cli_args import args, LatentPreviewMethod
    import folder_paths

    # Try to import from WanVideoWrapper if available
    import sys
    import importlib.util

    # Add WanVideoWrapper to path if it exists
    wan_video_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ComfyUI-WanVideoWrapper')
    if os.path.exists(wan_video_path):
        sys.path.insert(0, wan_video_path)

        # Import necessary modules from WanVideoWrapper
        try:
            from utils import log, print_memory, fourier_filter, optimized_scale, dict_to_device
        except ImportError:
            # Fallback logging function
            class SimpleLogger:
                @staticmethod
                def info(msg):
                    print(f"[INFO] {msg}")
            log = SimpleLogger()

            def print_memory(device):
                if torch.cuda.is_available():
                    print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f}GB")

            def fourier_filter(*args, **kwargs):
                return args[0] if args else None

            def optimized_scale(*args, **kwargs):
                return args[0] if args else None

            def dict_to_device(d, device):
                return {k: v.to(device) if hasattr(v, 'to') else v for k, v in d.items()}

except ImportError as e:
    print(f"Warning: Could not import ComfyUI modules: {e}")
    # Fallback implementations
    class SimpleLogger:
        @staticmethod
        def info(msg):
            print(f"[INFO] {msg}")
    log = SimpleLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
offload_device = torch.device("cpu")

class AdaptiveWanVideoAnimateEmbeds:
    """
    Adaptive WanVideo Animate Embeds node that automatically adjusts window size
    to minimize waste frames and improve video alignment.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("WANVAE",),
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "force_offload": ("BOOLEAN", {"default": True}),
            "frame_window_size": ("INT", {"default": 77, "min": 1, "max": 1000, "step": 1, "tooltip": "Base number of frames to use for temporal attention window"}),
            "adaptive_window_mode": (
            [
                'disabled',
                'adaptive',
                'optimal_fit'
            ], {
               "default": 'optimal_fit', "tooltip": "Adaptive window size mode: disabled (fixed size), adaptive (adjust for remainder), optimal_fit (minimize waste frames)"
            },),
            "colormatch": (
            [
                'disabled',
                'mkl',
                'hm',
                'reinhard',
                'mvgd',
                'hm-mvgd-hm',
                'hm-mkl-hm',
            ], {
               "default": 'disabled', "tooltip": "Color matching method to use between the windows"
            },),
            "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional multiplier for the pose"}),
            "face_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional multiplier for the face"}),
            },
            "optional": {
                "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS", {"tooltip": "Clip vision encoded image"}),
                "ref_images": ("IMAGE", {"tooltip": "Image to encode"}),
                "pose_images": ("IMAGE", {"tooltip": "end frame"}),
                "face_images": ("IMAGE", {"tooltip": "end frame"}),
                "bg_images": ("IMAGE", {"tooltip": "background images"}),
                "mask": ("MASK", {"tooltip": "mask"}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "AdaptiveWanVideo"

    def calculate_adaptive_window_size(self, total_frames, base_window_size, mode):
        """
        Calculate adaptive window size to minimize waste frames
        Args:
            total_frames: Total number of frames
            base_window_size: Base window size
            mode: Adaptive mode ('adaptive' or 'optimal_fit')
        """
        if mode == "adaptive":
            # Calculate remainder frames
            remainder = total_frames % base_window_size
            if remainder > 0 and remainder < base_window_size * 0.5:
                # If remainder is less than half window size, adjust window size
                num_windows = total_frames // base_window_size
                if num_windows > 0:
                    optimal_size = total_frames // num_windows
                    # Ensure window size is at least half of base size and aligned to 4
                    optimal_size = max(optimal_size, base_window_size // 2)
                    optimal_size = ((optimal_size - 1) // 4) * 4 + 1  # Align to 4 + 1 pattern
                    return optimal_size
            return base_window_size

        elif mode == "optimal_fit":
            # Find optimal window size that minimizes waste frames
            min_waste = float('inf')
            optimal_size = base_window_size

            # Search in range of ±25% of base window size, aligned to 4
            min_size = max(32, int(base_window_size * 0.75))
            max_size = int(base_window_size * 1.25)

            for size in range(min_size, max_size + 1):
                # Align to pattern (n-1)//4*4+1
                aligned_size = ((size - 1) // 4) * 4 + 1
                waste = total_frames % aligned_size
                if waste == 0:
                    return aligned_size
                if waste < min_waste:
                    min_waste = waste
                    optimal_size = aligned_size

            return optimal_size

        return base_window_size

    def process(self, vae, width, height, num_frames, force_offload, frame_window_size, adaptive_window_mode, colormatch, pose_strength, face_strength,
                ref_images=None, pose_images=None, face_images=None, clip_embeds=None, tiled_vae=False, bg_images=None, mask=None):

        H = height
        W = width

        lat_h = H // vae.upsampling_factor
        lat_w = W // vae.upsampling_factor

        num_refs = ref_images.shape[0] if ref_images is not None else 0
        num_frames = ((num_frames - 1) // 4) * 4 + 1

        # Apply adaptive window size calculation
        original_frame_window_size = frame_window_size
        if adaptive_window_mode != 'disabled':
            frame_window_size = self.calculate_adaptive_window_size(num_frames, frame_window_size, adaptive_window_mode)
            if frame_window_size != original_frame_window_size:
                log.info(f"AdaptiveWanAnimate: Window size changed from {original_frame_window_size} to {frame_window_size} for {num_frames} frames")
                waste_frames_before = num_frames % original_frame_window_size
                waste_frames_after = num_frames % frame_window_size
                log.info(f"AdaptiveWanAnimate: Waste frames reduced from {waste_frames_before} to {waste_frames_after}")

        looping = num_frames > frame_window_size

        if num_frames < frame_window_size:
            frame_window_size = num_frames

        target_shape = (16, (num_frames - 1) // 4 + 1 + num_refs, lat_h, lat_w)
        latent_window_size = ((frame_window_size - 1) // 4)

        if not looping:
            num_frames = num_frames + num_refs * 4
        else:
            latent_window_size = latent_window_size + 1

        vae.to(device)

        # Process pose images
        pose_latents = ref_latents = ref_latent = None
        if pose_images is not None:
            pose_images = pose_images[..., :3]
            if pose_images.shape[1] != H or pose_images.shape[2] != W:
                resized_pose_images = common_upscale(pose_images.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_pose_images = pose_images.permute(3, 0, 1, 2) # C, T, H, W
            resized_pose_images = resized_pose_images * 2 - 1
            pose_latents = vae.encode([resized_pose_images.to(device, vae.dtype)], device, tiled=tiled_vae)
            pose_latents = pose_latents.to(offload_device)
            vae.model.clear_cache()
            if not looping and pose_latents.shape[2] < latent_window_size:
                log.info(f"AdaptiveWanAnimate: Padding pose latents from {pose_latents.shape} to length {latent_window_size}")
                pad_len = latent_window_size - pose_latents.shape[2]
                pad = torch.zeros(pose_latents.shape[0], pose_latents.shape[1], pad_len, pose_latents.shape[3], pose_latents.shape[4], device=pose_latents.device, dtype=pose_latents.dtype)
                pose_latents = torch.cat([pose_latents, pad], dim=2)
            del resized_pose_images

        # Process background images
        bg_latents = None
        if bg_images is not None:
            if bg_images.shape[1] != H or bg_images.shape[2] != W:
                resized_bg_images = common_upscale(bg_images.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_bg_images = bg_images.permute(3, 0, 1, 2) # C, T, H, W
            resized_bg_images = resized_bg_images[:3] * 2 - 1
            bg_latents = vae.encode([resized_bg_images.to(device, vae.dtype)], device, tiled=tiled_vae)
            bg_latents = bg_latents.to(offload_device)
            vae.model.clear_cache()

        # Process reference images
        if ref_images is not None:
            ref_images = ref_images[..., :3]
            if ref_images.shape[1] != H or ref_images.shape[2] != W:
                resized_ref_images = common_upscale(ref_images.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_ref_images = ref_images.permute(3, 0, 1, 2) # C, T, H, W
            resized_ref_images = resized_ref_images * 2 - 1
            ref_latents = vae.encode([resized_ref_images.to(device, vae.dtype)], device, tiled=tiled_vae)
            ref_latents = ref_latents.to(offload_device)
            vae.model.clear_cache()
            ref_latent_masked = torch.zeros(4, ref_latents.shape[2], lat_h, lat_w, device=ref_latents.device, dtype=ref_latents.dtype)
            ref_latent_masked[:, :ref_latents.shape[2]] = ref_latents[0]

        # Process mask
        if mask is not None:
            ref_mask = mask.clone()
            if ref_mask.shape[-2] != lat_h or ref_mask.shape[-1] != lat_w:
                ref_mask = common_upscale(ref_mask.unsqueeze(0), lat_w, lat_h, "nearest-exact", "disabled").squeeze(0)
            ref_mask = ref_mask.to(vae.dtype).to(offload_device)
            ref_mask = ref_mask.unsqueeze(-1).permute(3, 0, 1, 2) # C, T, H, W

            if bg_images is None:
                ref_mask[:, :num_refs] = 1
            ref_mask_mask_repeated = torch.repeat_interleave(ref_mask[:, 0:1], repeats=4, dim=1) # T, C, H, W
            ref_mask = torch.cat([ref_mask_mask_repeated, ref_mask[:, 1:]], dim=1)
            ref_mask = ref_mask.view(1, ref_mask.shape[1] // 4, 4, lat_h, lat_w) # 1, T, C, H, W
            ref_mask = ref_mask.movedim(1, 2)[0]# C, T, H, W

            if not looping:
                if bg_images is not None:
                    bg_latents_masked = torch.cat([ref_mask[:, :bg_latents.shape[1]], bg_latents], dim=0)
                    ref_latent = torch.cat([ref_latent_masked, bg_latents_masked], dim=1)
                else:
                    ref_latent = torch.cat([ref_mask, ref_latent], dim=0)
            else:
                ref_latent = ref_latent_masked

        # Process face images
        if face_images is not None:
            face_images = face_images[..., :3]
            if face_images.shape[1] != 512 or face_images.shape[2] != 512:
                resized_face_images = common_upscale(face_images.movedim(-1, 1), 512, 512, "lanczos", "center").movedim(0, 1)
            else:
                resized_face_images = face_images.permute(3, 0, 1, 2) # B, C, T, H, W
            resized_face_images = (resized_face_images * 2 - 1).unsqueeze(0)
            resized_face_images = resized_face_images.to(offload_device, dtype=vae.dtype)

        vae.model.clear_cache()

        seq_len = math.ceil((target_shape[2] * target_shape[3]) / 4 * target_shape[1])

        if force_offload:
            vae.model.to(offload_device)
            if hasattr(mm, 'soft_empty_cache'):
                mm.soft_empty_cache()
            gc.collect()

        image_embeds = {
            "clip_context": clip_embeds.get("clip_embeds", None) if clip_embeds is not None else None,
            "negative_clip_context": clip_embeds.get("negative_clip_embeds", None) if clip_embeds is not None else None,
            "max_seq_len": seq_len,
            "pose_latents": pose_latents,
            "bg_images": resized_bg_images if bg_images is not None and looping else None,
            "ref_masks": ref_mask if mask is not None and looping else None,
            "ref_latent": ref_latent,
            "ref_image": resized_ref_images if ref_images is not None else None,
            "face_pixels": resized_face_images if face_images is not None else None,
            "num_frames": num_frames,
            "target_shape": target_shape,
            "frame_window_size": frame_window_size,
            "adaptive_window_mode": adaptive_window_mode,
            "original_frame_window_size": original_frame_window_size,
            "lat_h": lat_h,
            "lat_w": lat_w,
            "vae": vae,
            "colormatch": colormatch,
            "looping": looping,
            "pose_strength": pose_strength,
            "face_strength": face_strength,
        }

        return (image_embeds,)

# Node mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AdaptiveWanVideoAnimateEmbeds": AdaptiveWanVideoAnimateEmbeds,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdaptiveWanVideoAnimateEmbeds": "Adaptive WanVideo Animate Embeds",
}