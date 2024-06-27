from typing import Dict, Optional

import weave
from openai import OpenAI


class AttributeBindingModel(weave.Model):
    openai_model: Optional[str] = "gpt-3.5-turbo"
    num_prompts: Optional[int] = 20
    _openai_client: Optional[OpenAI] = None
    _system_prompt: Optional[str] = None
    _user_prompt: Optional[str] = None

    def _initialize(self):
        if self._system_prompt is None:
            self._system_prompt = """
            You are a helpful assistant designed to generate some sentences and additional metadata in JSON format.
            """
        if self._user_prompt is None:
            self._user_prompt = (
                """
            Please generate prompts in the format of “a {adj_1} {noun_1} and a {adj_2} {noun_2}”
            by using the shape adj.: long, tall, short, big, small, cubic, cylindrical,
            pyramidal, round, circular, oval, oblong, spherical, triangular, square, rectangular,
            conical, pentagonal, teardrop, crescent, and diamond.
            
            The output should be a list of """
                + str(self.num_prompts)
                + """ JSONs like the following:
            
            \{
                "data": \[
                    \{
                        "sentence": "a long balloon and a short giraffe",
                        "metadata": \{
                            "adj_1": "long",
                            "noun_1": "balloon",
                            "adj_2": "short",
                            "noun_2": "giraffe"
                        \}
                    \},
                    \{
                        "sentence": "a tall suitcase and a small frog",
                        "metadata": \{
                            "adj_1": "tall",
                            "noun_1": "suitcase",
                            "adj_2": "small",
                            "noun_2": "frog"
                        \}
                    \},
                    \{
                        "sentence": "a big horse and a small man",
                        "metadata": \{
                            "adj_1": "big",
                            "noun_1": "horse",
                            "adj_2": "small",
                            "noun_2": "man",
                        \}
                    \}
                \],
            \}
            """
            )
        self._openai_client = OpenAI()

    @weave.op()
    def predict(self, seed: int) -> Dict[str, str]:
        return {
            "response": self._openai_client.chat.completions.create(
                model=self.openai_model,
                response_format={"type": "json_object"},
                seed=seed,
                messages=[
                    {
                        "role": "system",
                        "content": self._system_prompt,
                    },
                    {
                        "role": "user",
                        "content": self._user_prompt,
                    },
                ],
            )
            .choices[0]
            .message.content
        }
