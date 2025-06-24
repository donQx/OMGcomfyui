
import os
from typing import List, Tuple
from PIL import Image, ImageOps

import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
from accelerate import Accelerator

from .omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from .omnigen2.utils.img_util import resize_image # This might not be used directly, but kept for completeness.


import os
from typing import List, Tuple
from PIL import Image, ImageOps

import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
from accelerate import Accelerator

from .omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from .omnigen2.utils.img_util import resize_image


def load_pipeline(model_path, accelerator, weight_dtype, scheduler_type, enable_sequential_cpu_offload, enable_model_cpu_offload):
    print(f"--- OmniGen2 Model Loading ---")
    print(f"Attempting to load model from: '{model_path}'")
    print(f"Requested dtype: {weight_dtype}")

    pipeline = None
    try:
        pipeline = OmniGen2Pipeline.from_pretrained(
            model_path,
            torch_dtype=weight_dtype,
            trust_remote_code=True,
            local_files_only=True, # <--- IMPORTANT: This forces local loading
        )
        print(f"SUCCESS: Model '{model_path}' loaded from local files.")
    except Exception as e:
        print(f"ERROR: Failed to load model locally from '{model_path}'.")
        print(f"Reason: {e}")
        print("This likely means the model files are not found or not structured correctly at the specified path.")
        print("Please ensure 'OmniGen2/OmniGen2' points to the root directory of a complete Hugging Face-style model download (e.g., containing config.json, model.safetensors, etc.).")
        print("Attempting a normal load (which might trigger a re-download if not in HF cache)...")
        # Fallback to normal loading if local_files_only failed, this might redownload
        pipeline = OmniGen2Pipeline.from_pretrained(
            model_path,
            torch_dtype=weight_dtype,
            trust_remote_code=True,
            # local_files_only=False is default
        )
        print(f"SUCCESS (Fallback): Model loaded (might have downloaded/cached).")


    if scheduler_type == "dpmsolver":
        from .omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
        scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            solver_order=2,
            prediction_type="flow_prediction",
        )
        pipeline.scheduler = scheduler

    if enable_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    elif enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(accelerator.device)
    
    print(f"--- Model Loading Complete ---")
    return pipeline

# preprocess function: Corrected return type hint to reflect actual return (List[Image.Image])
def preprocess(input_image_path: List[str] = []) -> List[Image.Image]:
    """Preprocess the input images."""

    input_images = [] # Initialize as empty list

    if input_image_path:
        if isinstance(input_image_path, str):
            input_image_path = [input_image_path]

        if len(input_image_path) == 1 and os.path.isdir(input_image_path[0]):
            # Filter for common image file extensions
            input_images = [Image.open(os.path.join(input_image_path[0], f)).convert("RGB")
                            for f in os.listdir(input_image_path[0]) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))]
        else:
            input_images = [Image.open(path).convert("RGB") for path in input_image_path]

        input_images = [ImageOps.exif_transpose(img) for img in input_images]

    return input_images


# run function: Needs 'seed' passed as an argument
def run(pipeline, input_images, width, height, num_inference_step, text_guidance_scale, image_guidance_scale, cfg_range_start, cfg_range_end,
        num_images_per_prompt, accelerator, instruction, negative_prompt, seed): # Added 'seed' parameter
    """Run the image generation pipeline with the given parameters."""

    # ComfyUI's IMAGE type is torch.Tensor (B, H, W, C).
    # The pipeline expects a list of PIL Images for input_images.
    processed_input_images = []
    if input_images is not None and input_images.numel() > 0: # Check for non-empty tensor
        for img_tensor in input_images:
            if img_tensor.min() < 0: # If range is [-1,1], convert to [0,1]
                pil_img = to_pil_image((img_tensor.permute(2,0,1) * 0.5 + 0.5).float())
            else: # If range is [0,1], no normalization needed
                pil_img = to_pil_image(img_tensor.permute(2,0,1).float())
            processed_input_images.append(pil_img)
    elif isinstance(input_images, list) and all(isinstance(i, Image.Image) for i in input_images): # Fallback if already PIL list
        processed_input_images = input_images

    generator = torch.Generator(device=accelerator.device).manual_seed(seed) # Replaced args.seed

    results = pipeline(
        prompt=instruction,
        input_images=processed_input_images if processed_input_images else None,
        width=width,
        height=height,
        num_inference_steps=num_inference_step,
        max_sequence_length=1024,
        text_guidance_scale=text_guidance_scale,
        image_guidance_scale=image_guidance_scale,
        cfg_range=(cfg_range_start, cfg_range_end),
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
        output_type="pil",
    )

    return results


def create_collage(images: List[torch.Tensor]) -> Image.Image:
    """Create a horizontal collage from a list of images."""

    if not images:
        return Image.new('RGB', (64, 64), (0, 0, 0)) # Return a small black image if no images

    normalized_images = []
    for img_tensor in images:
        if img_tensor.min() < 0: # If range is [-1,1], convert to [0,1]
            normalized_images.append((img_tensor * 0.5 + 0.5).float())
        else: # If already [0,1], ensure float
            normalized_images.append(img_tensor.float())

    max_height = max(img.shape[-2] for img in normalized_images)
    total_width = sum(img.shape[-1] for img in normalized_images)
    canvas = torch.zeros((3, max_height, total_width), device=normalized_images[0].device)

    current_x = 0
    for img in normalized_images:
        h, w = img.shape[-2:]
        canvas[:, :h, current_x:current_x+w] = img
        current_x += w

    return to_pil_image(canvas)


class LoadOmniGen2Image:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_path": ("STRING", {"default": "assets/demo.png"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    # Corrected function name to match the method
    FUNCTION = "input_image"
    CATEGORY = "OmniGen2"

    def input_image(self, image_path):
        input_images_pil = preprocess(image_path)
        if input_images_pil:
            # Convert list of PIL Images to a single batched ComfyUI IMAGE tensor (B, H, W, C)
            tensors = [to_tensor(img).permute(1, 2, 0) for img in input_images_pil] # Convert to HWC
            return (torch.stack(tensors),)
        # Return an empty (B,H,W,C) tensor if no images, to prevent errors downstream
        return (torch.empty(0, 1, 1, 3, dtype=torch.float32),)


class LoadOmniGen2Model:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "OmniGen2/OmniGen2"}),
                "dtype": (["fp32", "fp16", "bf16"], {"default": "bf16"}),
                # Added scheduler_type, offload options, and seed to INPUT_TYPES
                "scheduler_type": (["dpmsolver", "default"], {"default": "default"}),
                "enable_sequential_cpu_offload": ("BOOLEAN", {"default": False}),
                "enable_model_cpu_offload": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    # Now return accelerator and dtype, as OmniGen2 node will need them
    RETURN_TYPES = ("MODEL", "ACCELERATOR", "DTYPE_MODEL")
    RETURN_NAMES = ("pipeline", "accelerator", "model_dtype")
    FUNCTION = "load_model"
    CATEGORY = "OmniGen2"

    def load_model(self, model_path, dtype, scheduler_type, enable_sequential_cpu_offload, enable_model_cpu_offload, seed):
        # Initialize accelerator
        accelerator = Accelerator(mixed_precision=dtype if dtype != 'fp32' else 'no')

        # Set weight dtype
        weight_dtype = torch.float32
        if dtype == 'fp16':
            weight_dtype = torch.float16
        elif dtype == 'bf16':
            weight_dtype = torch.bfloat16

        # Pass new parameters to load_pipeline
        pipeline = load_pipeline(model_path, accelerator, weight_dtype, scheduler_type, enable_sequential_cpu_offload, enable_model_cpu_offload)

        # Return accelerator and weight_dtype for the OmniGen2 node
        return (pipeline, accelerator, weight_dtype)


class OmniGen2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("MODEL",),
                "input_images": ("IMAGE",),
                # Add accelerator and model_dtype as inputs from LoadOmniGen2Model
                "accelerator": ("ACCELERATOR",),
                "model_dtype": ("DTYPE_MODEL",), # Kept as per your original structure
                "width": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 8}),
                "num_inference_step": ("INT", {"default": 50, "min": 1, "max": 200, "step": 1}),
                "text_guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "image_guidance_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "cfg_range_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cfg_range_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "num_images_per_prompt": ("INT", {"default": 1, "min": 1, "max": 10}),
                "instruction": ("STRING", {"default": "A dog running in the park", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar", "multiline": True}),
                # Add seed input here as well
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("results",)
    FUNCTION = "generate"
    CATEGORY = "OmniGen2"

    def generate(self, pipeline, input_images, accelerator, model_dtype, # Order is crucial here! Matches INPUT_TYPES
                 width, height, num_inference_step, text_guidance_scale, image_guidance_scale, cfg_range_start, cfg_range_end,
                 num_images_per_prompt, instruction, negative_prompt, seed): # Added seed here

        # The accelerator should be initialized once in LoadOmniGen2Model and passed.
        # Removing this line: accelerator = Accelerator(mixed_precision=dtype if dtype != 'fp32' else 'no')
        # And also remove 'dtype' as a parameter to generate, as model_dtype is passed.

        results = run(pipeline, input_images, width, height, num_inference_step, text_guidance_scale, image_guidance_scale, cfg_range_start, cfg_range_end,
                      num_images_per_prompt, accelerator, instruction, negative_prompt, seed) # Pass seed to run

        # Convert PIL Images from results.images to torch.Tensor for ComfyUI output (B, H, W, C)
        output_tensors = [to_tensor(img).permute(1, 2, 0) for img in results.images] # Convert to HWC format
        if len(output_tensors) > 1:
            return (torch.stack(output_tensors),)
        elif output_tensors:
            return (output_tensors[0].unsqueeze(0),) # Add batch dimension if only one image
        else:
            return (torch.empty(0, 1, 1, 3, dtype=torch.float32),)


class SaveOmniGen2Image:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_image_path": ("STRING", {"default": "output.png"}),
                "results": ("IMAGE",), # ComfyUI passes image tensors here (B, H, W, C)
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save"
    CATEGORY = "OmniGen2"

    def save(self, output_image_path, results):

        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

        pil_images = []
        if results.numel() > 0: # Check if the tensor is not empty
            for img_tensor in results: # Iterate through the batch (B, H, W, C)
                # Convert HWC tensor to PIL, handling potential [-1, 1] range
                if img_tensor.min() < 0:
                    pil_img = to_pil_image((img_tensor.permute(2, 0, 1) * 0.5 + 0.5).float())
                else:
                    pil_img = to_pil_image(img_tensor.permute(2, 0, 1).float())
                pil_images.append(pil_img)
        
        if len(pil_images) > 1:
            for i, image in enumerate(pil_images):
                image_name, ext = os.path.splitext(output_image_path)
                image.save(f"{image_name}_{i}{ext}")
        
        if pil_images: # Create collage only if there are images
            # Prepare images for create_collage: list of CHW tensors, ideally in [-1,1] range
            collage_tensors = []
            if results.numel() > 0:
                for img_tensor in results:
                    chw_tensor = img_tensor.permute(2, 0, 1)
                    if chw_tensor.min() >= 0 and chw_tensor.max() <= 1:
                        collage_tensors.append(chw_tensor * 2 - 1)
                    else:
                        collage_tensors.append(chw_tensor)

            output_image = create_collage(collage_tensors)
            output_image.save(output_image_path)
            print(f"Image saved to {output_image_path}")
        else:
            print(f"No images to save to {output_image_path}")
            
        return ()


NODE_CLASS_MAPPINGS = {
    "LoadOmniGen2Image": LoadOmniGen2Image,
    "LoadOmniGen2Model": LoadOmniGen2Model,
    "OmniGen2": OmniGen2,
    "SaveOmniGen2Image": SaveOmniGen2Image,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadOmniGen2Image": "Load OmniGen2 Image",
    "LoadOmniGen2Model": "Load OmniGen2 Model",
    "OmniGen2": "OmniGen2 Generator",
    "SaveOmniGen2Image": "Save OmniGen2 Image",
}
