# Holistic Evaluation of Multi-modal Generative Models

`hemm` is a library for performing comprehensive benchmark of text-to-image diffusion models on image quality and prompt comprehension integrated with [Weights & Biases](https://wandb.ai/site) and [Weave](https://wandb.github.io/weave/). `hemm` is inspired by [Holistic Evaluation of Text-To-Image Models](https://crfm.stanford.edu/helm/heim/v1.0.0/).

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