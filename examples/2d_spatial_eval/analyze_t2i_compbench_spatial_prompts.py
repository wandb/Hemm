from typing import Optional

import fire
import weave

import wandb
from hemm.metrics.spatial_relationship import SpatialPromptAnalyzer


def main(
    openai_model: str = "gpt-3.5-turbo-0125",
    openai_seed: Optional[int] = None,
    project_name: str = "diffusion_leaderboard",
):
    wandb.init(project=project_name, job_type="analyze_spatial_prompts")
    weave.init(project_name=project_name)
    analyzer = SpatialPromptAnalyzer(
        openai_model=openai_model, openai_seed=openai_seed, project_name=project_name
    )
    analyzer()


if __name__ == "__main__":
    fire.Fire(main)
