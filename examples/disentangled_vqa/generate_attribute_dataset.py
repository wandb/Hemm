from typing import Optional

import fire
from hemm.metrics.attribute_binding import AttributeBindingDatasetGenerator


def main(
    openai_model: Optional[str] = "gpt-3.5-turbo",
    openai_seed: Optional[int] = 42,
    num_prompts_in_single_call: Optional[int] = 20,
    num_api_calls: Optional[int] = 50,
    project_name: Optional[str] = "diffusion_leaderboard",
    dump_dir: Optional[str] = "./dump",
):
    AttributeBindingDatasetGenerator(
        openai_model=openai_model,
        openai_seed=openai_seed,
        num_prompts_in_single_call=num_prompts_in_single_call,
        num_api_calls=num_api_calls,
        project_name=project_name,
    )(dump_dir=dump_dir)


if __name__ == "__main__":
    fire.Fire(main)
