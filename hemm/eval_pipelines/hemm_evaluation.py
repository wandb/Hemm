import time
import traceback
from typing import Callable, Optional, Union, cast

import rich
import wandb
import weave
from weave.flow.dataset import Dataset
from weave.flow.model import Model
from weave.flow.util import async_foreach
from weave.flow.scorer import Scorer, get_scorer_attributes
from weave.trace.errors import OpCallError
from weave.trace.env import get_weave_parallelism
from weave.trace.op import Op


class HemmEvaluation(weave.Evaluation):
    dataset: Union[Dataset, list]
    scorers: Optional[list[Union[Callable, Op, Scorer]]] = None
    preprocess_model_input: Optional[Callable] = None
    trials: int = 1
    wandb_summary_table_name: str = None
    wandb_summary_table: wandb.Table = wandb.Table(columns=["summary"])

    @weave.op()
    async def evaluate(self, model: Union[Callable, Model]) -> dict:
        eval_rows = []

        start_time = time.time()

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
        async for example, eval_row in async_foreach(
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
        rich.print("Evaluation summary", summary)
        return summary
