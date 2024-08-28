# import io
# import weave
# import PIL
# from PIL import Image
# from datasets import load_dataset


# def preprocess_images(example):
#     generic_image = Image.new(example["image"].mode, example["image"].size)
#     generic_image.putdata(list(example["image"].getdata()))
#     example["image"] = generic_image
#     return example

# dataset_dict = load_dataset("HuggingFaceM4/COCO")["train"].take(5)
# # dataset_dict = dataset_dict.map(preprocess_images)
# # print(dataset_dict[0])


# weave.init(project_name="test")


# dataset = weave.Dataset(rows=[{"image": dataset_dict[0]["image"]}], name="test")
# weave.publish(dataset)

import wandb
import weave
from hemm.utils import publish_dataset_to_weave


wandb.init(project="test", entity="hemm-eval")
weave.init(project_name="hemm-eval/test")


dataset_reference = publish_dataset_to_weave(
    dataset_path="HuggingFaceM4/COCO",
    prompt_column="sentences",
    ground_truth_image_column="image",
    split="validation",
    data_limit=10,
)
