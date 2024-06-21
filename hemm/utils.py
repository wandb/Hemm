import base64
import io
import os
from PIL import Image
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import jsonlines
import wandb
import weave
from datasets import load_dataset
from tqdm.auto import tqdm
from weave.trace.refs import ObjectRef


EXT_TO_MIMETYPE = {
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".svg": "image/svg+xml",
}


def base64_encode_image(
    image_path: Union[str, Image.Image], mimetype: Optional[str] = None
) -> str:
    """Converts an image to base64 encoded string to be logged and rendered on Weave dashboard.

    Args:
        image_path (Union[str, Image.Image]): Path to the image or PIL Image object.
        mimetype (Optional[str], optional): Mimetype of the image. Defaults to None.

    Returns:
        str: Base64 encoded image string.
    """
    image = Image.open(image_path) if isinstance(image_path, str) else image_path
    mimetype = (
        EXT_TO_MIMETYPE[Path(image_path).suffix]
        if isinstance(image_path, str)
        else "image/png"
    )
    byte_arr = io.BytesIO()
    image.save(byte_arr, format="PNG")
    encoded_string = base64.b64encode(byte_arr.getvalue()).decode("utf-8")
    encoded_string = f"data:{mimetype};base64,{encoded_string}"
    return str(encoded_string)


def base64_decode_image(image: str) -> Image.Image:
    """Decodes a base64 encoded image string encoded using the function `hemm.utils.base64_encode_image`.

    Args:
        image (str): Base64 encoded image string encoded using the function `hemm.utils.base64_encode_image`.

    Returns:
        Image.Image: PIL Image object.
    """
    return Image.open(io.BytesIO(base64.b64decode(image.split(";base64,")[-1])))


def save_weave_dataset_rows_to_artifacts(
    dataset_rows: List[Dict], dump_dir: str
) -> None:
    """Saves the dataset rows to W&B artifacts.

    Args:
        dataset_rows (List[Dict]): List of dataset rows.
        dump_dir (str): Directory to dump the results.
    """
    with jsonlines.open(os.path.join(dump_dir, "data.jsonl"), mode="w") as writer:
        writer.write(dataset_rows)
    artifact = wandb.Artifact(name="t2i_compbench_spatial_prompts", type="dataset")
    artifact.add_file(local_path=os.path.join(dump_dir, "data.jsonl"))
    wandb.log_artifact(artifact)


def publish_dataset_to_weave(
    dataset_path,
    dataset_name: Optional[str] = None,
    prompt_column: Optional[str] = None,
    ground_truth_image_column: Optional[str] = None,
    split: Optional[str] = None,
    data_limit: Optional[int] = None,
    get_weave_dataset_reference: bool = True,
    dataset_transforms: Optional[List[Callable]] = None,
    column_transforms: Optional[Dict[str, Callable]] = None,
    dump_dir: Optional[str] = "./dump",
    *args,
    **kwargs,
) -> Union[ObjectRef, None]:
    """Publishes a HuggingFace dataset dictionary dataset as a Weave dataset.

    ??? example "Publish a subset of MSCOCO from Huggingface as a Weave Dataset"
        ```python
        import weave
        from hemm.utils import publish_dataset_to_weave

        if __name__ == "__main__":
            weave.init(project_name="t2i_eval")

            dataset_reference = publish_dataset_to_weave(
                dataset_path="HuggingFaceM4/COCO",
                prompt_column="sentences",
                ground_truth_image_column="image",
                split="validation",
                dataset_transforms=[
                    lambda item: {**item, "sentences": item["sentences"]["raw"]}
                ],
                data_limit=5,
            )
        ```

    Args:
        dataset_path ([type]): Path to the HuggingFace dataset.
        dataset_name (Optional[str], optional): Name of the Weave dataset.
        prompt_column (Optional[str], optional): Column name for prompt.
        ground_truth_image_column (Optional[str], optional): Column name for ground truth image.
        split (Optional[str], optional): Split to be used.
        data_limit (Optional[int], optional): Limit the number of data items.
        get_weave_dataset_reference (bool, optional): Whether to return the Weave dataset reference.
        dataset_transforms (Optional[List[Callable]], optional): List of dataset transforms.
        column_transforms (Optional[Dict[str, Callable]], optional): Column specific transforms.
        dump_dir (Optional[str], optional): Directory to dump the results.

    Returns:
        Union[ObjectRef, None]: Weave dataset reference if get_weave_dataset_reference is True.
    """
    os.makedirs(dump_dir, exist_ok=True)
    dataset_name = dataset_name or Path(dataset_path).stem
    dataset_dict = load_dataset(dataset_path, *args, **kwargs)
    dataset_dict = dataset_dict[split] if split else dataset_dict["train"]
    dataset_dict = (
        dataset_dict.select(range(data_limit))
        if data_limit is not None and data_limit < len(dataset_dict)
        else dataset_dict
    )
    if dataset_transforms:
        for transform in dataset_transforms:
            dataset_dict = dataset_dict.map(transform)
    dataset_dict = (
        dataset_dict.rename_column(prompt_column, "prompt")
        if prompt_column
        else dataset_dict
    )
    dataset_dict = (
        dataset_dict.rename_column(ground_truth_image_column, "ground_truth_image")
        if ground_truth_image_column
        else dataset_dict
    )
    column_transforms = (
        {**column_transforms, **{"ground_truth_image": base64_encode_image}}
        if column_transforms
        else {"ground_truth_image": base64_encode_image}
    )
    weave_dataset_rows = []
    for data_item in tqdm(dataset_dict):
        for key in data_item.keys():
            if column_transforms and key in column_transforms:
                data_item[key] = column_transforms[key](data_item[key])
        weave_dataset_rows.append(data_item)

    if wandb.run:
        save_weave_dataset_rows_to_artifacts(weave_dataset_rows, dump_dir)

    weave_dataset = weave.Dataset(name=dataset_name, rows=weave_dataset_rows)
    weave.publish(weave_dataset)
    return weave.ref(dataset_name).get() if get_weave_dataset_reference else None
