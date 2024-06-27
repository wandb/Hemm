import json
import os
from typing import Dict, List, Optional

import jsonlines
import weave
from openai import OpenAI

import wandb


class AttributeBindingEvaluation(weave.Evaluation):
    pass


class AttributeBindingModel(weave.Model):
    openai_model: Optional[str] = "gpt-3.5-turbo"
    openai_seed: Optional[int] = 42
    num_prompts: Optional[int] = 20

    @weave.op()
    def predict(self):
        openai_client = OpenAI()
        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            response_format={"type": "json_object"},
            seed=self.openai_seed,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to generate some sentences and additional metadata in JSON format.",
                },
                {
                    "role": "user",
                    "content": """
                    Please generate prompts in the format of “a {adj_1} {noun_1} and a {adj_2} {noun_2}”
                    by using the shape adj.: long, tall, short, big, small, cubic, cylindrical,
                    pyramidal, round, circular, oval, oblong, spherical, triangular, square, rectangular,
                    conical, pentagonal, teardrop, crescent, and diamond.
                    
                    The output should be a list of """
                    + self.num_prompts
                    + """ JSONs like the following:
                    
                    \{
                        "0": \{
                            "sentence": "a long balloon and a short giraffe",
                            "metadata": \{
                                "adj_1": "long",
                                "noun_1": "balloon",
                                "adj_2": "short",
                                "noun_2": "giraffe"
                            \}
                        \},
                        "1": \{
                            "sentence": "a tall suitcase and a small frog",
                            "metadata": \{
                                "adj_1": "tall",
                                "noun_1": "suitcase",
                                "adj_2": "small",
                                "noun_2": "frog"
                            \}
                        \},
                        "2": \{
                            "sentence": "a big horse and a small man",
                            "metadata": \{
                                "adj_1": "big",
                                "noun_1": "horse",
                                "adj_2": "small",
                                "noun_2": "man",
                            \}
                        \}
                    \}
                    """,
                },
            ],
        )
        openai_client.close()
        return {"response": json.loads(response.choices[0].message.content)}


class AttributeBindingDatasetGenerator:

    def __init__(
        self,
        openai_model: Optional[str] = "gpt-3.5-turbo",
        openai_seed: Optional[int] = 42,
        num_prompts_in_single_call: Optional[int] = 20,
        num_api_calls: Optional[int] = 50,
        project_name: str = "diffusion_leaderboard",
        dump_dir: str = "./dump",
    ) -> None:
        self.attribute_binding_model = AttributeBindingModel(
            openai_model=openai_model,
            openai_seed=openai_seed,
            num_prompts=num_prompts_in_single_call,
        )
        self.num_prompts_in_single_call = num_prompts_in_single_call
        self.num_api_calls = num_api_calls
        self.project_name = project_name
        self.dump_dir = dump_dir
        self.dataset_rows: List[Dict] = []
        self.wandb_table = wandb.Table(
            columns=["prompt", "adj_1", "noun_1", "adj_2", "noun_2"]
        )

    def publish_dataset(self):
        weave.publish(
            weave.Dataset(name="attribute_binding_dataset", rows=self.dataset_rows)
        )
        with jsonlines.open(
            os.path.join(self.dump_dir, "dataset.jsonl"), mode="w"
        ) as writer:
            writer.write(self.dataset_rows)
        artifact = wandb.Artifact(name="attribute_binding_dataset", type="dataset")
        artifact.add_file(local_path=os.path.join(self.dump_dir, "dataset.jsonl"))
        wandb.log_artifact(artifact)
        wandb.log({"dataset/attribute_binding": self.wandb_table})

    @weave.op()
    async def evaluate_generated_response(
        self, prompt: str, model_output: Dict
    ) -> Dict:
        num_correct_predictions = 0
        for idx in model_output.keys():
            prompt = model_output[str(idx)]["sentence"]
            adj_1 = model_output[str(idx)]["metadata"]["adj_1"]
            adj_2 = model_output[str(idx)]["metadata"]["adj_2"]
            noun_1 = model_output[str(idx)]["metadata"]["noun_1"]
            noun_2 = model_output[str(idx)]["metadata"]["noun_2"]
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
            self.dataset_rows.append()
            if f"a {adj_1} {noun_1} and a {adj_2} {noun_2}" == prompt:
                num_correct_predictions += 1
        return {
            "num_correct_predictions": num_correct_predictions,
            "total_predictions": len(model_output.keys()),
            "accuracy": num_correct_predictions / len(model_output.keys()),
        }

    def __call__(self) -> None:
        os.makedirs(self.dump_dir, exist_ok=True)
        evaluation = AttributeBindingEvaluation(
            dataset=[{"prompt": ""}] * self.num_api_calls,
            scorers=[self.evaluate_generated_response],
        )
        evaluation.evaluate(self.attribute_binding_model.predict)
