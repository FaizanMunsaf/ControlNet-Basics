from diffusers import DiffusionPipeline
import torch
from PIL import Image

# Load base pipeline
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
base.to("cuda")

# Enable memory-efficient settings
base.enable_xformers_memory_efficient_attention()
base.enable_attention_slicing()

# Load refiner pipeline (shares text encoder & VAE with base)
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
refiner.to("cuda")

# Enable memory-efficient settings for refiner too
refiner.enable_xformers_memory_efficient_attention()
refiner.enable_attention_slicing()

# Settings
n_steps = 40
high_noise_frac = 0.8
prompt = "A majestic lion jumping from a big stone at night"

# Generate with base
base_output = base(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent"  # needed to pass latents to refiner
)
latents = base_output.images

# Refine image
refined_output = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=latents  # latent input from base
)
final_image = refined_output.images[0]

# Save output
final_image.save("output_image.png")
print("✅ Image saved as output_image.png")
