# Text-to-Image Diffusion Benchmark

`t2i_diffusion_benchmark` is a library for performing comprehensive benchmark of text-to-image diffusion models on image quality and prompt comprehension integrated with [Weights & Biases](https://wandb.ai/site) and [Weave](https://wandb.github.io/weave/).

```python
from t2i_diffusion_benchmark import CLIPScorer, StableDiffusionEvaluationPipeline


if __name__ == "__main__":
    dataset = [
        {"prompt": "a photo of an astronaut riding a horse on mars"},
        {"prompt": "A high tech solarpunk utopia in the Amazon rainforest"},
        {"prompt": "A pikachu fine dining with a view to the Eiffel Tower"},
        {"prompt": "A mecha robot in a favela in expressionist style"},
        {"prompt": "an insect robot preparing a delicious meal"},
        {
            "prompt": "A small cabin on top of a snowy mountain in the style of Disney, artstation"
        },
    ]

    diffuion_evaluation_pipeline = StableDiffusionEvaluationPipeline(
        "CompVis/stable-diffusion-v1-4"
    )

    clip_scorer = CLIPScorer()
    diffuion_evaluation_pipeline.add_metric(clip_scorer)

    diffuion_evaluation_pipeline(
        dataset=dataset, init_params=dict(project="t2i_eval", entity="geekyrakshit")
    )

```