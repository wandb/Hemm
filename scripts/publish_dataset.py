import weave
from fire import Fire
from t2i_diffusion_benchmark.utils import publish_prompt_dataset_to_weave


def publish(
    project_name: str,
    dataset_path: str,
    dataset_name: str,
    prompt_column: str,
    split: str,
    data_limit: int,
):
    weave.init(project_name=project_name)
    publish_prompt_dataset_to_weave(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        prompt_column=prompt_column,
        split=split,
        data_limit=data_limit,
    )


if __name__ == "__main__":
    Fire(publish)
