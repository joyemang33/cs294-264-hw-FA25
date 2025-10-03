import os
from openai import OpenAI
from abc import ABC, abstractmethod


class LLM(ABC):
    """Abstract base class for Large Language Models."""

    @abstractmethod
    def query(self, prompt: str) -> str:
        """
        Generate a response from the LLM given a prompt.
        Must include any required stop-token logic at the caller level.
        """
        raise NotImplementedError


class OpenAIModel(LLM):
    """
    Example LLM implementation using OpenAI's Responses API.

    TODO(student): Implement this class to call your chosen backend (e.g., OpenAI GPT-5 mini)
    and return the model's text output. You should ensure the model produces the response
    format required by ResponseParser and include the stop token in the output string.
    """

    def __init__(self, stop_token: str, model_name: str = "gpt-5-mini"):
        # TODO(student): Initialize your OpenAI client or chosen LLM provider here.
        self.stop_token = stop_token
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        print(f"Using OpenAI API key: {self.api_key}")
        self.client = OpenAI(api_key=self.api_key)


    def query(self, prompt: str) -> str:
        # TODO(student): Call the model, obtain text, and ensure the stop token is present.
        # Return the raw text including the terminal stop token required by the parser.
        response = self.client.responses.create(
            model=self.model_name,
            tools=[{ "type": "web_search_preview" }],
            input=prompt
        )
        print(response.output_text)
        return response.output_text