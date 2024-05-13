from functools import partial

import torch
from torchmetrics.functional.multimodal import clip_score
from diffusers import StableDiffusionPipeline


sd_pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
).to("cuda")
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

prompts = [
    "a photo of an astronaut riding a horse on mars",
    "A high tech solarpunk utopia in the Amazon rainforest",
    "A pikachu fine dining with a view to the Eiffel Tower",
    "A mecha robot in a favela in expressionist style",
    "an insect robot preparing a delicious meal",
    "A small cabin on top of a snowy mountain in the style of Disney, artstation",
]
images = sd_pipeline(prompts, num_images_per_prompt=1, output_type="np").images


def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(
        torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts
    ).detach()
    return round(float(clip_score), 4)


sd_clip_score = calculate_clip_score(images, prompts)
print(f"CLIP score: {sd_clip_score}")
