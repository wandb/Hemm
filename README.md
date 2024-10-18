# Hemm: Holistic Evaluation of Multi-modal Generative Models

[![](https://img.shields.io/badge/Hemm-docs-blue)](https://wandb.github.io/Hemm/)

Hemm is a library for performing comprehensive benchmark of text-to-image diffusion models on image quality and prompt comprehension integrated with [Weights & Biases](https://wandb.ai/site) and [Weave](https://wandb.github.io/weave/).

Hemm is highly inspired by the following projects:
- [Holistic Evaluation of Text-To-Image Models](https://crfm.stanford.edu/helm/heim/v1.0.0/)
- [T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation](https://karine-h.github.io/T2I-CompBench/)
- [T2I-CompBench++: An Enhanced and Comprehensive Benchmark for Compositional Text-to-image Generation](https://karine-h.github.io/T2I-CompBench-new/)
- [GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment](https://arxiv.org/abs/2310.11513)

| ![](./docs/assets/evals.gif) | 
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
import wandb
import weave


from hemm.eval_pipelines import EvaluationPipeline
from hemm.metrics.prompt_alignment import CLIPImageQualityScoreMetric, CLIPScoreMetric
from hemm.models import BaseDiffusionModel


# Initialize Weave and WandB
wandb.init(project="image-quality-leaderboard", job_type="evaluation")
weave.init(project_name="image-quality-leaderboard")


# Initialize the diffusion model to be evaluated as a `weave.Model` using `BaseWeaveModel`
# The `BaseDiffusionModel` class uses a `diffusers.DiffusionPipeline` under the hood.
# You can write your own model `weave.Model` if your model is not diffusers compatible.
model = BaseDiffusionModel(diffusion_model_name_or_path="CompVis/stable-diffusion-v1-4")


# Add the model to the evaluation pipeline
evaluation_pipeline = EvaluationPipeline(model=model)


# Add PSNR Metric to the evaluation pipeline
psnr_metric = PSNRMetric(image_size=evaluation_pipeline.image_size)
evaluation_pipeline.add_metric(psnr_metric)


# Add SSIM Metric to the evaluation pipeline
ssim_metric = SSIMMetric(image_size=evaluation_pipeline.image_size)
evaluation_pipeline.add_metric(ssim_metric)


# Add LPIPS Metric to the evaluation pipeline
lpips_metric = LPIPSMetric(image_size=evaluation_pipeline.image_size)
evaluation_pipeline.add_metric(lpips_metric)


# Get the Weave dataset reference
dataset = weave.ref("COCO:v0").get()


# Evaluate!
evaluation_pipeline(dataset=dataset)
```
