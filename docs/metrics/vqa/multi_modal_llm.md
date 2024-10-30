# Multi-modal LLM Based Evaluation

This module aims to implement the Multi-modal LLM based metric inspired by

- Section IV.D of the paper [T2I-CompBench++: An Enhanced and Comprehensive Benchmark for Compositional Text-to-image Generation](https://karine-h.github.io/T2I-CompBench-new/) and
- Section 4.4 of the paper [T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation](https://arxiv.org/abs/2307.06350).

| ![](../../assets/weave_mllm.png) | 
|:--:| 
| Using Multi-modal LLM based metric for evaluation a diffusion model. The Weave UI gives us a holistic view of the evaluations to drill into individual ouputs and scores. |


!!! example

    First, download the Spacy English langugage pipeline
    ```shell
    python -m spacy download en_core_web_sm
    ```
    Next, you need to set your OpenAI API key:
    ```
    export OPENAI_API_KEY="<INSERT-YOUR-OPENAI-API-KEY>"
    ```
    Finallly, you can run the following snippet to evaluate your model:
    ```python  
    import asyncio

    import weave

    from hemm.metrics.vqa import MultiModalLLMEvaluationMetric
    from hemm.metrics.vqa.judges.mmllm_judges import OpenAIJudge
    from hemm.models import DiffusersModel

    weave.init(project_name="hemm-eval/mllm-eval")

    model = DiffusersModel(
        diffusion_model_name_or_path="stabilityai/stable-diffusion-2-1",
        image_height=1024,
        image_width=1024,
    )
    metric = MultiModalLLMEvaluationMetric(judge=OpenAIJudge())

    evaluation = weave.Evaluation(dataset=weave.ref("Dataset:v2").get(), scorers=[metric])
    asyncio.run(evaluation.evaluate(model))
    ```

## Metrics

:::hemm.metrics.vqa.multi_modal_llm_eval

## Judges

:::hemm.metrics.vqa.judges.mmllm_judges.openai_judge