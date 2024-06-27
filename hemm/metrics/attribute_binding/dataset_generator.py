import asyncio
import os
from typing import Dict, List, Optional

import jsonlines
import weave
from pydantic import BaseModel

import wandb

from ...eval_pipelines.hemm_evaluation import AsyncHemmEvaluation
from ...utils import str_to_json
from .attribute_binding_model import AttributeBindingModel


class AttributeBindingEvaluation(AsyncHemmEvaluation):
    pass


class AttributeBindingEvaluationResponse(BaseModel):
    is_correct_json: bool = False
    num_correct_predictions: int = 0
    total_predictions: int = 0
    accuracy: float = 0.0


class AttributeBindingDatasetGenerator:

    def __init__(
        self,
        openai_model: Optional[str] = "gpt-3.5-turbo",
        openai_seed: Optional[int] = None,
        num_prompts_in_single_call: Optional[int] = 20,
        num_api_calls: Optional[int] = 50,
        project_name: Optional[str] = "diffusion_leaderboard",
    ) -> None:
        self.attribute_binding_model = AttributeBindingModel(
            openai_model=openai_model,
            openai_seed=openai_seed,
            num_prompts=num_prompts_in_single_call,
        )
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
        wandb.init(project=self.project_name, job_type="attribute_binding_dataset")
        weave.init(project_name=self.project_name)
        evaluation = AttributeBindingEvaluation(
            dataset=[{"prompt": ""}] * self.num_api_calls,
            scorers=[self.evaluate_generated_response],
        )
        asyncio.run(evaluation.evaluate(self.attribute_binding_model.predict))
        self.publish_dataset(dump_dir)
