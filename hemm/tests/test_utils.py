import weave

from hemm.utils import publish_dataset_to_weave


def test_parti_prompts():
    weave.init(project_name="hemm-eval/unit-tests")
    dataset_reference = publish_dataset_to_weave(
        dataset_path="nateraw/parti-prompts",
        prompt_column="Prompt",
        split="train",
        data_limit=10,
    )
    assert dataset_reference is not None


def test_coco():
    def preprocess_sentences_column(example):
        example["sentences"] = example["sentences"]["raw"]
        return example

    weave.init(project_name="hemm-eval/unit-tests")
    dataset_reference = publish_dataset_to_weave(
        dataset_path="HuggingFaceM4/COCO",
        prompt_column="sentences",
        ground_truth_image_column="image",
        split="validation",
        dataset_transforms=[preprocess_sentences_column],
        data_limit=10,
    )
    assert dataset_reference is not None
