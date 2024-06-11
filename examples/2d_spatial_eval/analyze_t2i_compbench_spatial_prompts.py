import asyncio
import json
import os
from functools import partial
from typing import Dict, List

import fire
import weave
from openai import OpenAI
from weave import Evaluation

import wandb

artifact = wandb.use_artifact(
    "geekyrakshit/diffusion_leaderboard/t2i-compbench:v0", type="dataset"
)
artifact_dir = artifact.download()
with open(os.path.join(artifact_dir, "T2I-CompBench_dataset", "spatial.txt"), "r") as f:
    spatial_prompts = f.read().split("\n")


def chunk_all_prompts(prompt_list: List[str], chunk_size: int) -> List[str]:
    prompts = []
    for idx in range(0, len(prompt_list), chunk_size):
        chunk = prompt_list[idx : idx + chunk_size]
        prompt = ""
        for idx, p in enumerate(chunk):
            prompt += f"{idx}: {p}\n"
        prompts.append({"prompt_chunk": prompt.strip()})
    return prompts


@weave.op()
def generate_structured_spatial_prompt_chunks(
    prompt_chunk: str, openai_model: str
) -> str:
    client = OpenAI()

    response = client.chat.completions.create(
        model=openai_model,
        response_format={"type": "json_object"},
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


@weave.op()
async def evaluate_structured_prompt_chunk(
    prompt_chunk: str, model_output: Dict
) -> List[Dict]:
    generated_response = model_output["response"]
    structured_respopse = json.loads(generated_response)
    evaluation_responses = {}
    for chunk_idx, prompt in enumerate(prompt_chunk.split("\n")):
        evaluation_responses[str(chunk_idx)] = {
            "prompt": prompt,
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


def main(
    openai_model: str = "gpt-3.5-turbo-0125",
    project_name: str = "diffusion_leaderboard",
):
    weave.init(project_name=project_name)
    wandb.init(project=project_name)
    chunked_promts = chunk_all_prompts(spatial_prompts[:-1], chunk_size=50)
    evaluation = Evaluation(
        dataset=chunked_promts, scorers=[evaluate_structured_prompt_chunk]
    )
    asyncio.run(
        evaluation.evaluate(
            partial(
                generate_structured_spatial_prompt_chunks, openai_model=openai_model
            )
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
