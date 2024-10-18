import unittest

import weave

from hemm.utils import publish_dataset_to_weave


class TestUtils(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        weave.init(project_name="hemm-eval/unit-tests")

    def test_parti_prompts(self):
        dataset_reference = publish_dataset_to_weave(
            dataset_path="nateraw/parti-prompts",
            prompt_column="Prompt",
            split="train",
            data_limit=10,
        )
        self.assertIsNotNone(dataset_reference)

    def test_coco(self):
        def preprocess_sentences_column(example):
            example["sentences"] = example["sentences"]["raw"]
            return example

        dataset_reference = publish_dataset_to_weave(
            dataset_path="HuggingFaceM4/COCO",
            prompt_column="sentences",
            ground_truth_image_column="image",
            split="validation",
            dataset_transforms=[preprocess_sentences_column],
            data_limit=10,
        )
        self.assertIsNotNone(dataset_reference)
