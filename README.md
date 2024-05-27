# Hemm: Holistic Evaluation of Multi-modal Generative Models

Hemm is a library for performing comprehensive benchmark of text-to-image diffusion models on image quality and prompt comprehension integrated with [Weights & Biases](https://wandb.ai/site) and [Weave](https://wandb.github.io/weave/). Hemm is inspired by [Holistic Evaluation of Text-To-Image Models](https://crfm.stanford.edu/helm/heim/v1.0.0/).

## Installation

```shell
git clone https://github.com/soumik12345/Hemm
cd Hemm
pip install -e ".[core]"
```

## Quickstart

First let's publish a small subset of the MSCOCO validation set as a [Weave Dataset](https://wandb.github.io/weave/guides/core-types/datasets/).

```python
import weave
from hemm.utils import publish_dataset_to_weave


if __name__ == "__main__":
    weave.init(project_name="t2i_eval")

    dataset_reference = publish_dataset_to_weave(
        dataset_path="HuggingFaceM4/COCO",
        prompt_column="sentences",
        ground_truth_image_column="image",
        split="validation",
        dataset_transforms=[
            lambda item: {**item, "sentences": item["sentences"]["raw"]}
        ],
        data_limit=5,
    )
```

| ![](./docs/assets/weave_dataset.gif) | 
|:--:| 
|  |

Next, you can evaluate Stable Diffusion 1.4 on image quality metrics as shown in the following code snippet:

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

| ![](./docs/assets/weave_leaderboard.gif) | 
|:--:| 
| The evaluation pipeline will take each example, pass it through your application and score the output on multiple custom scoring functions using [Weave Evaluation](https://wandb.github.io/weave/guides/core-types/evaluations). By doing this, you'll have a view of the performance of your model, and a rich UI to drill into individual ouputs and scores. |