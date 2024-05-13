import asyncio
import os
from functools import partial
from typing import Dict

import numpy as np

import torch
from torchmetrics.functional.multimodal import clip_score
from diffusers import StableDiffusionPipeline
from diffusers.utils.loading_utils import load_image

import wandb
import weave
from weave import Evaluation


class PipelineInferer:

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

        self.generated_images_dir = os.path.join(os.getcwd(), "generated_images")
        os.makedirs(self.generated_images_dir, exist_ok=True)

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
        ).to("cuda")
        self.pipeline.set_progress_bar_config(leave=False, desc="Generating Image")

        self.clip_score_fn = partial(
            clip_score, model_name_or_path="openai/clip-vit-base-patch16"
        )

        self.inference_counter = 1
        self.wandb_table = wandb.Table(columns=["prompt", "image", "clip_score"])

    @weave.op()
    async def infer(self, prompt: str):
        image_url = os.path.join(
            self.generated_images_dir, f"{self.inference_counter}.png"
        )
        self.inference_counter += 1
        self.pipeline(
            prompt,
            num_images_per_prompt=1,
            generator=torch.Generator(device="cuda").manual_seed(self.seed),
        ).images[0].save(image_url)
        return {"image_url": image_url}

    @weave.op()
    async def calculate_clip_score(self, prompt: str, model_output: Dict):
        pil_image = load_image(model_output["image_url"])
        images = np.expand_dims(np.array(pil_image), axis=0)
        clip_score = float(
            self.clip_score_fn(
                torch.from_numpy(images).permute(0, 3, 1, 2), prompt
            ).detach()
        )
        self.wandb_table.add_data(prompt, wandb.Image(pil_image), clip_score)
        return {"clip_score": clip_score}
    
    def log_table(self):
        if wandb.run is not None:
            wandb.log({"evaluation_table": self.wandb_table})


if __name__ == "__main__":
    weave.init(project_name="t2i_eval")
    wandb.init(project="t2i_eval", entity="geekyrakshit", job_type="test/evaluation")

    dataset = [
        {"prompt": "a photo of an astronaut riding a horse on mars"},
        {"prompt": "A high tech solarpunk utopia in the Amazon rainforest"},
        {"prompt": "A pikachu fine dining with a view to the Eiffel Tower"},
        {"prompt": "A mecha robot in a favela in expressionist style"},
        {"prompt": "an insect robot preparing a delicious meal"},
        {
            "prompt": "A small cabin on top of a snowy mountain in the style of Disney, artstation"
        },
    ]

    inferer = PipelineInferer()
    evaluation = Evaluation(dataset=dataset, scorers=[inferer.calculate_clip_score])
    asyncio.run(evaluation.evaluate(inferer.infer))
    inferer.log_table()
