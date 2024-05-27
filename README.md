# Hemm: Holistic Evaluation of Multi-modal Generative Models

Hemm is a library for performing comprehensive benchmark of text-to-image diffusion models on image quality and prompt comprehension integrated with [Weights & Biases](https://wandb.ai/site) and [Weave](https://wandb.github.io/weave/). Hemm is inspired by [Holistic Evaluation of Text-To-Image Models](https://crfm.stanford.edu/helm/heim/v1.0.0/) and [T2I-CompBench](https://karine-h.github.io/T2I-CompBench/).

## Installation

```shell
git clone https://github.com/soumik12345/Hemm
cd Hemm
pip install -e ".[core]"
```

## Quickstart

Evaluating Stable Diffusion on prompt alignment metrics is showcased in the following code snippet:

```python
from hemm.eval_pipelines import StableDiffusionEvaluationPipeline
from hemm.metrics.prompt_alignment import (
    BLIPScorer,
    CLIPImageQualityScorer,
    CLIPScorer,
)


diffuion_evaluation_pipeline = StableDiffusionEvaluationPipeline(
    "CompVis/stable-diffusion-v1-4"
)

# Add CLIP Scorer metric
clip_scorer = CLIPScorer()
diffuion_evaluation_pipeline.add_metric(clip_scorer)

# Add CLIP IQA Metric
clip_iqa_scorer = CLIPImageQualityScorer()
diffuion_evaluation_pipeline.add_metric(clip_iqa_scorer)

# Add BLIP Scorer Metric
blip_scorer = BLIPScorer()
diffuion_evaluation_pipeline.add_metric(blip_scorer)

diffuion_evaluation_pipeline(
    dataset="parti-prompts:v1",
    init_params=dict(project="t2i_eval", entity="geekyrakshit"),
)
```

Evaluating Stable Diffusion on image quality metrics is showcased in the following code snippet:

```python
from hemm.eval_pipelines import StableDiffusionEvaluationPipeline
from hemm.metrics.image_quality import LPIPSMetric, PSNRMetric, SSIMMetric


if __name__ == "__main__":
    diffuion_evaluation_pipeline = StableDiffusionEvaluationPipeline(
        "CompVis/stable-diffusion-v1-4"
    )

    # Add PSNR Metric
    psnr_metric = PSNRMetric(image_size=diffuion_evaluation_pipeline.image_size)
    diffuion_evaluation_pipeline.add_metric(psnr_metric)

    # Add SSIM Metric
    ssim_metric = SSIMMetric(image_size=diffuion_evaluation_pipeline.image_size)
    diffuion_evaluation_pipeline.add_metric(ssim_metric)
    
    # Add LPIPS Metric
    lpips_metric = LPIPSMetric(image_size=diffuion_evaluation_pipeline.image_size)
    diffuion_evaluation_pipeline.add_metric(lpips_metric)

    diffuion_evaluation_pipeline(
        dataset="COCO:v1",
        init_params=dict(project="t2i_eval", entity="geekyrakshit"),
    )
```