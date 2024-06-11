import asyncio
import json
import os
from typing import Dict, List, Optional

import jsonlines
import weave
from openai import OpenAI
from weave import Dataset, Evaluation, Model

import wandb


def chunk_all_prompts(prompt_list: List[str], chunk_size: int) -> List[str]:
    """Chunk all prompts into smaller chunks of size `chunk_size`.

    Args:
        prompt_list (List[str]): List of prompts to chunk.
        chunk_size (int): Size of each chunk.

    Returns:
        List[str]: List of chunked prompts.
    """
    prompts = []
    for idx in range(0, len(prompt_list), chunk_size):
        chunk = prompt_list[idx : idx + chunk_size]
        prompt = ""
        for idx, p in enumerate(chunk):
            prompt += f"{idx}: {p}\n"
        prompts.append({"prompt_chunk": prompt.strip()})
    return prompts


class SpatialPromptModel(Model):
    """Model to predict structured output from spatial prompts.

    Args:
        openai_model (str): OpenAI model to use for prediction.
        openai_seed (Optional[int], optional): Seed for OpenAI model. Defaults to None.
    """

    openai_model: str
    openai_seed: Optional[int] = None

    @weave.op()
    def predict(self, prompt_chunk: str) -> str:
        """Predict structured output from spatial prompts.

        Args:
            prompt_chunk (str): Chunk of spatial prompts.

        Returns:
            str: Predicted structured output.
        """
        client = OpenAI()
        response = client.chat.completions.create(
            model=self.openai_model,
            response_format={"type": "json_object"},
            seed=self.openai_seed,
            messages=[
                {
                    "role": "system",
                    "content": """
                You are a helpful assistant designed extract entities and the relationship between
                them from sentences in JSON format. Given a list of sentences like the following:
                
                    0: a balloon on the top of a giraffe
                    1: a suitcase on the top of a frog
                    2: a horse on the right of a man
                    3: a rabbit next to a balloon
                    4: a rabbit on the top of a bicycle
                
                The output should be a list of JSONs like the following:
                
                \{
                    "0": \{
                    "entities": \[
                        \{"name": "balloon", "numeracy": 1\},
                        \{"name": "giraffe", "numeracy": 1\}
                    \]
                    "relation": "on the top of"
                    \},
                    "1": \{
                    "entities": \[
                        \{"name": "suitcase", "numeracy": 1\},
                        \{"name": "frog", "numeracy": 1\}
                    \]
                    "relation": "on the top of"
                    \},
                    "2": \{
                    "entities": \[
                        \{"name": "horse", "numeracy": 1\},
                        \{"name": "man", "numeracy": 1\}
                    \]
                    "relation": "on the right of"
                    \},
                    "3": \{
                    "entities": \[
                        \{"name": "rabbit", "numeracy": 1\},
                        \{"name": "balloon", "numeracy": 1\}
                    \]
                    "relation": "next to"
                    \},
                    "4": \{
                    "entities": \[
                        \{"name": "rabbit", "numeracy": 1\},
                        \{"name": "bicycle", "numeracy": 1\}
                    \]
                    "relation": "on the top of"
                    \}
                \}

                Make sure you run through the entire list and not miss anything.
                I will be providing a total of 100 examples.
                Return me the json output with 100 elements.
                """,
                },
                {
                    "role": "user",
                    "content": prompt_chunk,
                },
            ],
        )
        return {"response": response.choices[0].message.content}


class SpatialPromptAnalyzer:
    """Analyze spatial prompts from T2I Compbench using a model using an OpenAI text-generation model.

    ??? example "Analyzing spatial prompts using OpenAI model"
        ```python
        import wandb
        import weave
        from hemm.metrics.spatial_relationship import SpatialPromptAnalyzer

        wandb.init(project=project_name, job_type="analyze_spatial_prompts")
        weave.init(project_name=project_name)
        analyzer = SpatialPromptAnalyzer(
            openai_model=openai_model, openai_seed=openai_seed, project_name=project_name
        )
        analyzer()
        ```

    Args:
        openai_model (str, optional): OpenAI model to use for prediction.
        openai_seed (Optional[int], optional): Seed for OpenAI model.
        project_name (str, optional): Name of the project.
        dump_dir (str, optional): Directory to dump the results.
    """

    def __init__(
        self,
        openai_model: str = "gpt-3.5-turbo-0125",
        openai_seed: Optional[int] = None,
        project_name: str = "diffusion_leaderboard",
        dump_dir: str = "./dump",
    ):
        self.model = SpatialPromptModel(
            openai_model=openai_model, openai_seed=openai_seed
        )
        self.dump_dir = dump_dir
        self.project_name = project_name
        self.spatial_prompts = self._fetch_spatial_prompts()
        self.spatial_weave_dataset = []
        self.spatial_prompts_weave_dataset = []
        self.wandb_table = wandb.Table(
            columns=["analyzer_model", "prompt", "predicted_response"]
        )

    def _fetch_spatial_prompts(self) -> None:
        artifact = wandb.use_artifact(
            "geekyrakshit/diffusion_leaderboard/t2i-compbench:v0", type="dataset"
        )
        artifact_dir = artifact.download()
        dataset_path = os.path.join(
            artifact_dir, "T2I-CompBench_dataset", "spatial.txt"
        )
        with open(dataset_path, "r") as f:
            spatial_prompts = f.read().strip().split("\n")
        return spatial_prompts

    def _save_prompt_analysis_result(self) -> None:
        # Publish prompt analysis results as a weave dataset
        weave.publish(
            weave.Dataset(
                name="t2i_compbench_spatial_prompts",
                rows=self.spatial_prompts_weave_dataset,
            )
        )
        # Publish prompt analysis results as a wandb dataset artifact
        with jsonlines.open(
            os.path.join(self.dump_dir, "spatial.jsonl"), mode="w"
        ) as writer:
            writer.write(self.spatial_prompts_weave_dataset)
        artifact = wandb.Artifact(name="t2i_compbench_spatial_prompts", type="dataset")
        artifact.add_file(local_path=os.path.join(self.dump_dir, "spatial.jsonl"))
        wandb.log_artifact(artifact)
        # Publish prompt analysis results as a wandb table
        wandb.log({"analysis/spatial_prompts": self.wandb_table})

    @weave.op()
    async def evaluate_structured_prompt_chunk(
        self, prompt_chunk: str, model_output: Dict
    ) -> List[Dict]:
        generated_response = model_output["response"]
        structured_respopse = json.loads(generated_response)
        evaluation_responses = {}
        for chunk_idx, prompt in enumerate(prompt_chunk.split("\n")):
            self.wandb_table.add_data(
                self.model.openai_model,
                prompt.split(":")[-1].strip(),
                structured_respopse[str(chunk_idx)],
            )
            self.spatial_prompts_weave_dataset.append(
                {
                    "prompt": prompt.split(":")[-1].strip(),
                    "response": structured_respopse[str(chunk_idx)],
                }
            )
            evaluation_responses[str(chunk_idx)] = {
                "prompt": prompt.split(":")[-1].strip(),
                "response": structured_respopse[str(chunk_idx)],
                "entity_1_correct": structured_respopse[str(chunk_idx)]["entities"][0][
                    "name"
                ]
                in prompt,
                "entity_2_correct": structured_respopse[str(chunk_idx)]["entities"][1][
                    "name"
                ]
                in prompt,
                "relation_correct": structured_respopse[str(chunk_idx)]["relation"]
                in prompt,
            }
        return evaluation_responses

    def __call__(self) -> None:
        os.makedirs(self.dump_dir, exist_ok=True)
        chunked_promts = chunk_all_prompts(self.spatial_prompts, chunk_size=50)
        chunked_prompt_dataset = Dataset(
            name="t2i_compbench_spatial_prompt_chunks", rows=chunked_promts
        )
        chunked_prompt_dataset_reference = weave.publish(chunked_prompt_dataset)
        evaluation = Evaluation(
            dataset=chunked_prompt_dataset_reference,
            scorers=[self.evaluate_structured_prompt_chunk],
        )
        eval_trace_configs = {
            "openai_model": self.model.openai_model,
            "openai_seed": self.model.openai_seed,
        }
        with weave.attributes(eval_trace_configs):
            asyncio.run(evaluation.evaluate(self.model.predict))
        self._save_prompt_analysis_result()
