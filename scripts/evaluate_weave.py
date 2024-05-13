import asyncio
from t2i_diffusion_benchmark import StableDiffusionEvaluationPipeline

import weave
from weave import Evaluation


if __name__ == "__main__":
    weave.init(project_name="t2i_eval")

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
    evaluation = Evaluation(
        dataset=dataset, scorers=[diffuion_evaluation_pipeline.calculate_clip_score]
    )
    asyncio.run(evaluation.evaluate(diffuion_evaluation_pipeline.infer))
    diffuion_evaluation_pipeline.log_summary()
