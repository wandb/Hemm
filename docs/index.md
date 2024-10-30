# Hemm: Holistic Evaluation of Multi-modal Generative Models

Hemm is a library for performing comprehensive benchmark of text-to-image diffusion models on image quality and prompt comprehension integrated with [Weights & Biases](https://wandb.ai/site) and [Weave](https://wandb.github.io/weave/). 

Hemm is highly inspired by the following projects:

- [Holistic Evaluation of Text-To-Image Models](https://crfm.stanford.edu/helm/heim/v1.0.0/)

- [T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation](https://karine-h.github.io/T2I-CompBench/)

- [T2I-CompBench++: An Enhanced and Comprehensive Benchmark for Compositional Text-to-image Generation](https://karine-h.github.io/T2I-CompBench-new/)

- [GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment](https://arxiv.org/abs/2310.11513)

| ![](./assets/evals.gif) | 
|:--:| 
| The evaluation pipeline will take each example, pass it through your application and score the output on multiple custom scoring functions using [Weave Evaluation](https://wandb.github.io/weave/guides/core-types/evaluations). By doing this, you'll have a view of the performance of your model, and a rich UI to drill into individual ouputs and scores. |

## Leaderboards

| Leaderboard | Weave Evals |
|---|---|
| [Rendering prompts with Complex Actions](https://wandb.ai/hemm-eval/mllm-eval-action/reports/Leaderboard-Rendering-prompts-with-Complex-Actions--Vmlldzo5Mjg2Nzky) | [Weave Evals](https://wandb.ai/hemm-eval/mllm-eval-action/weave/evaluations) |

## Installation

First, we recommend you install the PyTorch by visiting [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

```shell
git clone https://github.com/wandb/Hemm
cd Hemm
pip install -e ".[core]"
```

## Quickstart

First, you need to publish your evaluation dataset to Weave. Check out [this tutorial](https://weave-docs.wandb.ai/guides/core-types/datasets) that shows you how to publish a dataset on your project.

Once you have a dataset on your Weave project, you can evaluate a text-to-image generation model on the metrics.

```python
import asyncio
import weave
from hemm.metrics.vqa import MultiModalLLMEvaluationMetric
from hemm.metrics.vqa.judges.mmllm_judges import OpenAIJudge
from hemm.models import DiffusersModel

# Initialize Weave
weave.init(project_name="image-quality-leaderboard")

# The `DiffusersModel` is a `weave.Model` that uses a
# `diffusers.DiffusionPipeline` under the hood.
# You can write your own model `weave.Model` if your
# model is not diffusers compatible.
model = DiffusersModel(
    diffusion_model_name_or_path="stabilityai/stable-diffusion-2-1",
    image_height=1024,
    image_width=1024,
)

# Define the metric
metric = MultiModalLLMEvaluationMetric(judge=OpenAIJudge())

# Get the Weave dataset reference
dataset=weave.ref("Dataset:v2").get()

# Evaluate!
evaluation = weave.Evaluation(dataset=dataset, scorers=[metric])
summary = asyncio.run(evaluation.evaluate(model))
```
