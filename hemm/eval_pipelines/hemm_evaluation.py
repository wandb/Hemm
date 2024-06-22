import inspect
import textwrap
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Union, cast

import numpy as np
import rich
import wandb
import weave
from tqdm.auto import tqdm
from weave.flow.dataset import Dataset
from weave.flow.model import Model, get_infer_method
from weave.flow.scorer import Scorer, auto_summarize, get_scorer_attributes
from weave.flow.util import async_foreach
from weave.trace.errors import OpCallError
from weave.trace.env import get_weave_parallelism
from weave.trace.op import Op, BoundOp


def replace_backslash_dot(d):
    if isinstance(d, dict):
        new_dict = {}
        for k, v in d.items():
            new_key = k.replace("\\.", ".")
            new_dict[new_key] = replace_backslash_dot(v)
        return new_dict
    elif isinstance(d, list):
        return [replace_backslash_dot(i) for i in d]
    else:
        return d


class AsyncHemmEvaluation(weave.Evaluation):
    dataset: Union[Dataset, list]
    scorers: Optional[list[Union[Callable, Op, Scorer]]] = None
    preprocess_model_input: Optional[Callable] = None
    trials: int = 1

    @weave.op()
    async def evaluate(self, model: Union[Callable, Model]) -> dict:
        eval_rows = []

        async def eval_example(example: dict) -> dict:
            try:
                eval_row = await self.predict_and_score(model, example)
            except OpCallError as e:
                raise e
            except Exception as e:
                rich.print("Predict and score failed")
                traceback.print_exc()
                return {"model_output": None, "scores": {}}
            return eval_row

        n_complete = 0
        dataset = cast(Dataset, self.dataset)
        _rows = dataset.rows
        trial_rows = list(_rows) * self.trials
        async for _, eval_row in async_foreach(
            trial_rows, eval_example, get_weave_parallelism()
        ):
            n_complete += 1
            rich.print(f"Evaluated {n_complete} of {len(trial_rows)} examples")
            if eval_row == None:
                eval_row = {"model_output": None, "scores": {}}
            if eval_row["scores"] == None:
                eval_row["scores"] = {}
            for scorer in self.scorers or []:
                scorer_name, _, _ = get_scorer_attributes(scorer)
                if scorer_name not in eval_row["scores"]:
                    eval_row["scores"][scorer_name] = {}
            eval_rows.append(eval_row)

        summary = await self.summarize(eval_rows)
        wandb.log(replace_backslash_dot(summary))
        rich.print("Evaluation summary", summary)
        return summary


class SyncHemmEvaluation(weave.Evaluation):
    dataset: Union[Dataset, List[Dict]]
    scorers: Optional[list[Union[Callable, Op, Scorer]]] = None
    preprocess_model_input: Optional[Callable] = None
    trials: int = 1

    @weave.op()
    def predict_and_score(
        self, model: Union[Callable, Model], example: dict
    ) -> Dict[str, Any]:
        if self.preprocess_model_input == None:
            model_input = example
        else:
            model_input = self.preprocess_model_input(example)  # type: ignore

        if callable(model):
            model_predict = model
        else:
            model_predict = get_infer_method(model)

        model_predict_fn_name = (
            model_predict.name
            if isinstance(model_predict, Op)
            else model_predict.__name__
        )

        if isinstance(model_predict, Op):
            predict_signature = model_predict.signature
        else:
            predict_signature = inspect.signature(model_predict)
        model_predict_arg_names = list(predict_signature.parameters.keys())
        # If the op is a `BoundOp`, then the first arg is automatically added at
        # call time and we should exclude it from the args required from the
        # user.
        if isinstance(model_predict, BoundOp):
            model_predict_arg_names = model_predict_arg_names[1:]

        if isinstance(model_input, dict):
            model_predict_args = {
                k: v for k, v in model_input.items() if k in model_predict_arg_names
            }
        else:
            if len(model_predict_arg_names) == 1:
                model_predict_args = {model_predict_arg_names[0]: model_input}
            else:
                raise ValueError(
                    f"{model_predict} expects arguments: {model_predict_arg_names}, provide a preprocess_model_input function that returns a dict with those keys."
                )
        try:
            model_start_time = time.time()
            model_output = model_predict(**model_predict_args)
        except OpCallError as e:
            dataset_column_names = list(example.keys())
            dataset_column_names_str = ", ".join(dataset_column_names[:3])
            if len(dataset_column_names) > 3:
                dataset_column_names_str += ", ..."
            required_arg_names = [
                param.name
                for param in predict_signature.parameters.values()
                if param.default == inspect.Parameter.empty
            ]
            if isinstance(model_predict, BoundOp):
                required_arg_names = required_arg_names[1:]

            message = textwrap.dedent(
                f"""
                Call error: {e}

                Options for resolving:
                a. change {model_predict_fn_name} argument names to match a subset of dataset column names: {dataset_column_names_str}
                b. change dataset column names to match expected {model_predict_fn_name} argument names: {required_arg_names}
                c. construct Evaluation with a preprocess_model_input function that accepts a dataset example and returns a dict with keys expected by {model_predict_fn_name}
                """
            )
            raise OpCallError(message)
        except Exception as e:
            rich.print("model_output failed")
            traceback.print_exc()
            model_output = None
        model_latency = time.time() - model_start_time

        scores = {}
        scorers = cast(list[Union[Op, Scorer]], self.scorers or [])
        for scorer in scorers:
            scorer_name, score_fn, _ = get_scorer_attributes(scorer)
            if isinstance(score_fn, Op):
                score_signature = score_fn.signature
            else:
                score_signature = inspect.signature(score_fn)
            score_arg_names = list(score_signature.parameters.keys())

            # If the op is a `BoundOp`, then the first arg is automatically added at
            # call time and we should exclude it from the args required from the
            # user.
            if isinstance(score_arg_names, BoundOp):
                score_arg_names = score_arg_names[1:]

            if "model_output" not in score_arg_names:
                raise OpCallError(
                    f"Scorer {scorer_name} must have a 'model_output' argument, to receive the output of the model function."
                )

            if isinstance(example, dict):
                score_args = {k: v for k, v in example.items() if k in score_arg_names}
            else:
                if len(score_arg_names) == 2:
                    score_args = {score_arg_names[0]: example}
                else:
                    raise ValueError(
                        f"{score_fn} expects arguments: {score_arg_names}, provide a preprocess_model_input function that returns a dict with those keys."
                    )
            score_args["model_output"] = model_output

            try:
                result = score_fn(**score_args)
            except OpCallError as e:
                dataset_column_names = list(example.keys())
                dataset_column_names_str = ", ".join(dataset_column_names[:3])
                if len(dataset_column_names) > 3:
                    dataset_column_names_str += ", ..."
                required_arg_names = [
                    param.name
                    for param in score_signature.parameters.values()
                    if param.default == inspect.Parameter.empty
                ]
                if isinstance(score_fn, BoundOp):
                    required_arg_names = required_arg_names[1:]
                required_arg_names.remove("model_output")

                message = textwrap.dedent(
                    f"""
                    Call error: {e}

                    Options for resolving:
                    a. change {scorer_name} argument names to match a subset of dataset column names ({dataset_column_names_str})
                    b. change dataset column names to match expected {scorer_name} argument names: {required_arg_names}
                    """
                )
                raise OpCallError(message)
            scores[scorer_name] = result

        return {
            "model_output": model_output,
            "scores": scores,
            "model_latency": model_latency,
        }

    @weave.op()
    def summarize(self, eval_table: Union[weave.WeaveList, list]) -> dict:
        summary = {}
        if not isinstance(eval_table, weave.WeaveList):
            eval_table = weave.WeaveList(eval_table)
        model_output_summary = auto_summarize(eval_table.column("model_output"))
        if model_output_summary:
            summary["model_output"] = model_output_summary
        scorers = self.scorers or []
        for scorer in scorers:
            scorer_name, _, summarize_fn = get_scorer_attributes(scorer)
            scorer_scores = eval_table.column("scores").column(scorer_name)
            summary[scorer_name] = summarize_fn(scorer_scores)  # type: ignore
        summary["model_latency"] = {
            "mean": float(np.mean(eval_table.column("model_latency"))),
        }
        return summary

    @weave.op()
    def evaluate(self, model: Union[Callable, Model]) -> dict:
        eval_rows = []
        n_complete = 0
        dataset = cast(Dataset, self.dataset)
        _rows = dataset.rows
        trial_rows = list(_rows) * self.trials

        def eval_example(example: Dict[str, Any]) -> Dict[str, Any]:
            try:
                eval_row = self.predict_and_score(model, example)
            except OpCallError as e:
                raise e
            except Exception as e:
                rich.print("Predict and score failed")
                traceback.print_exc()
                return {"model_output": None, "scores": {}}
            return eval_row

        for example in tqdm(trial_rows, desc="Evaluating", leave=False):
            eval_row = eval_example(example)
            n_complete += 1
            if eval_row == None:
                eval_row = {"model_output": None, "scores": {}}
            if eval_row["scores"] == None:
                eval_row["scores"] = {}
            for scorer in self.scorers or []:
                scorer_name, _, _ = get_scorer_attributes(scorer)
                if scorer_name not in eval_row["scores"]:
                    eval_row["scores"][scorer_name] = {}
            eval_rows.append(eval_row)

        summary = self.summarize(eval_rows)
        wandb.log(replace_backslash_dot(summary))
        rich.print("Evaluation summary", summary)
        return summary
