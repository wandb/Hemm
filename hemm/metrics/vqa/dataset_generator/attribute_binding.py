import asyncio
import os
from typing import Dict, List, Optional, Union

import jsonlines
import wandb
import weave
from openai import OpenAI
from pydantic import BaseModel

from ....eval_pipelines.hemm_evaluation import AsyncHemmEvaluation
from ....utils import autogenerate_seed, str_to_json


class AttributeBindingEvaluation(AsyncHemmEvaluation):
    pass


class AttributeBindingEvaluationResponse(BaseModel):
    is_correct_json: bool = False
    num_correct_predictions: int = 0
    total_predictions: int = 0
    accuracy: float = 0.0


class AttributeBindingModel(weave.Model):
    """Weave Model to generate prompts for evaluation of attribute binding capability of
    image-generation models using an OpenAI model.

    Args:
        openai_model (Optional[str]): The OpenAI model to use for generating prompts.
        num_prompts (Optional[int]): Number of prompts to generate.
    """

    openai_model: Optional[str] = "gpt-3.5-turbo"
    num_prompts: Optional[int] = 20
    _openai_client: Optional[OpenAI] = None
    _system_prompt: Optional[str] = None
    _user_prompt: Optional[str] = None

    def _initialize(self):
        if self._system_prompt is None:
            self._system_prompt = """
            You are a helpful assistant designed to generate some sentences and additional metadata in JSON format.
            """
        if self._user_prompt is None:
            self._user_prompt = (
                """
            Please generate prompts in the format of “a {adj_1} {noun_1} and a {adj_2} {noun_2}”
            by using the shape adj.: long, tall, short, big, small, cubic, cylindrical,
            pyramidal, round, circular, oval, oblong, spherical, triangular, square, rectangular,
            conical, pentagonal, teardrop, crescent, and diamond.
            
            The output should be a list of """
                + str(self.num_prompts)
                + """ JSONs like the following:
            
            \{
                "data": \[
                    \{
                        "sentence": "a long balloon and a short giraffe",
                        "metadata": \{
                            "adj_1": "long",
                            "noun_1": "balloon",
                            "adj_2": "short",
                            "noun_2": "giraffe"
                        \}
                    \},
                    \{
                        "sentence": "a tall suitcase and a small frog",
                        "metadata": \{
                            "adj_1": "tall",
                            "noun_1": "suitcase",
                            "adj_2": "small",
                            "noun_2": "frog"
                        \}
                    \},
                    \{
                        "sentence": "a big horse and a small man",
                        "metadata": \{
                            "adj_1": "big",
                            "noun_1": "horse",
                            "adj_2": "small",
                            "noun_2": "man",
                        \}
                    \}
                \],
            \}
            """
            )
        self._openai_client = OpenAI()

    @weave.op()
    def predict(self, seed: int) -> Dict[str, str]:
        """Generate prompts and corresponding metadata for evaluation of attribute binding
        capability of image-generation models.

        Args:
            seed (int): OpenAI seed to use for generating prompts.
        """
        return {
            "response": self._openai_client.chat.completions.create(
                model=self.openai_model,
                response_format={"type": "json_object"},
                seed=seed,
                messages=[
                    {
                        "role": "system",
                        "content": self._system_prompt,
                    },
                    {
                        "role": "user",
                        "content": self._user_prompt,
                    },
                ],
            )
            .choices[0]
            .message.content
        }


class AttributeBindingDatasetGenerator:
    """Dataset generator for evaluation of attribute binding capability of image-generation models.
    This class enables us to generate the dataset consisting of prompts in the format
    `“a {adj_1} {noun_1} and a {adj_2} {noun_2}”` and the corresponding metadata using an LLM capable
    of generating json objects like GPT4-O. The dataset is then published both as a
    [W&B dataset artifact](https://docs.wandb.ai/guides/artifacts) and as a
    [weave dataset](https://wandb.github.io/weave/guides/core-types/datasets).

    ??? example "Sample usage"
        ```python
        from hemm.metrics.vqa import AttributeBindingDatasetGenerator

        dataset_generator = AttributeBindingDatasetGenerator(
            openai_model="gpt-4o",
            openai_seed=42,
            num_prompts_in_single_call=20,
            num_api_calls=50,
            project_name="disentangled_vqa",
        )

        dataset_generator(dump_dir="./dump")
        ```

    Args:
        openai_model (Optional[str]): The OpenAI model to use for generating prompts.
        openai_seed (Optional[Union[int, List[int]]]): Seed to use for generating prompts.
            If not provided, seeds will be auto-generated.
        num_prompts_in_single_call (Optional[int]): Number of prompts to generate in a single API call.
        num_api_calls (Optional[int]): Number of API calls to make.
        project_name (Optional[str]): Name of the Weave project to use for logging the dataset.
    """

    def __init__(
        self,
        openai_model: Optional[str] = "gpt-3.5-turbo",
        openai_seed: Optional[Union[int, List[int]]] = None,
        num_prompts_in_single_call: Optional[int] = 20,
        num_api_calls: Optional[int] = 50,
        project_name: Optional[str] = "diffusion_leaderboard",
    ) -> None:
        if not openai_seed:
            self.openai_seeds = [autogenerate_seed() for _ in range(num_api_calls)]
        elif isinstance(openai_seed, int):
            self.openai_seeds = [openai_seed] * num_api_calls
        elif isinstance(openai_seed, list) and len(openai_seed) != num_api_calls:
            raise ValueError(
                "Length of `openai_seed` should be equal to `num_api_calls`"
            )
        self.attribute_binding_model = AttributeBindingModel(
            openai_model=openai_model,
            num_prompts=num_prompts_in_single_call,
        )
        self.model_configs = self.attribute_binding_model.model_dump()
        self.attribute_binding_model._initialize()
        self.num_prompts_in_single_call = num_prompts_in_single_call
        self.num_api_calls = num_api_calls
        self.project_name = project_name
        self.dataset_rows: List[Dict] = []
        self.wandb_table = wandb.Table(
            columns=["prompt", "adj_1", "noun_1", "adj_2", "noun_2"]
        )

    def publish_dataset(self, dump_dir: str):
        os.makedirs(dump_dir, exist_ok=True)
        weave_dataset = weave.Dataset(
            name="attribute_binding_dataset", rows=self.dataset_rows
        )
        weave.publish(weave_dataset)
        with jsonlines.open(
            os.path.join(dump_dir, "dataset.jsonl"), mode="w"
        ) as writer:
            writer.write(self.dataset_rows)
        artifact = wandb.Artifact(name="attribute_binding_dataset", type="dataset")
        artifact.add_file(local_path=os.path.join(dump_dir, "dataset.jsonl"))
        wandb.log_artifact(artifact)
        wandb.log({"dataset/attribute_binding": self.wandb_table})

    @weave.op()
    async def evaluate_generated_response(
        self, prompt: str, model_output: Dict
    ) -> Dict:
        eval_response = AttributeBindingEvaluationResponse()
        model_output = str_to_json(model_output["response"])
        if model_output:
            if "data" not in model_output:
                return eval_response.model_dump()
            model_output = model_output["data"]
            eval_response.is_correct_json = True
            for idx in model_output:
                prompt = idx["sentence"]
                adj_1 = idx["metadata"]["adj_1"]
                adj_2 = idx["metadata"]["adj_2"]
                noun_1 = idx["metadata"]["noun_1"]
                noun_2 = idx["metadata"]["noun_2"]
                self.wandb_table.add_data(prompt, adj_1, noun_1, adj_2, noun_2)
                self.dataset_rows.append(
                    {
                        "prompt": prompt,
                        "adj_1": adj_1,
                        "noun_1": noun_1,
                        "adj_2": adj_2,
                        "noun_2": noun_2,
                    }
                )
                if f"a {adj_1} {noun_1} and a {adj_2} {noun_2}" == prompt:
                    eval_response.num_correct_predictions += 1
            eval_response.total_predictions = len(model_output)
            eval_response.accuracy = eval_response.num_correct_predictions / len(
                model_output
            )
        return eval_response.model_dump()

    def __call__(self, dump_dir: Optional[str] = "./dump") -> None:
        """Generate the dataset and publish it to Weave.

        Args:
            dump_dir (Optional[str]): Directory to dump the dataset.
        """
        wandb.init(
            project=self.project_name,
            job_type="attribute_binding_dataset",
            config=self.model_configs,
        )
        weave.init(project_name=self.project_name)
        evaluation = AttributeBindingEvaluation(
            dataset=[{"prompt": "", "seed": seed} for seed in self.openai_seeds],
            scorers=[self.evaluate_generated_response],
        )
        with weave.attributes(self.model_configs):
            asyncio.run(evaluation.evaluate(self.attribute_binding_model.predict))
        self.publish_dataset(dump_dir)
