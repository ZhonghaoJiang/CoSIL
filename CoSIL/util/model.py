from abc import ABC, abstractmethod
from typing import Any, List

from litellm import completion, get_model_info


class DecoderBase(ABC):
    def __init__(
        self,
        name: str,
        logger,
        batch_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> None:
        self.name = name
        self.logger = logger
        self.batch_size = batch_size
        self.temperature = temperature
        try:
            model_info = get_model_info(name)
        except Exception:
            model_info = {}
        self.max_context_tokens = model_info.get("max_input_tokens") or model_info.get("max_tokens") or 100000
        self.max_new_tokens = max_new_tokens
        if self.max_new_tokens is None:
            self.max_new_tokens = model_info.get("max_output_tokens") or model_info.get("max_tokens") or 4096

    @abstractmethod
    def codegen(
            self, message: str | list, num_samples: int = 1, prompt_cache: bool = False, **kwargs
    ) -> List[dict]:
        pass

    @abstractmethod
    def is_direct_completion(self) -> bool:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class LiteLLMChatDecoder(DecoderBase):
    def __init__(
        self,
        name: str,
        logger,
        litellm_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(name, logger, **kwargs)
        self.litellm_kwargs = litellm_kwargs or {}

    def _messages(self, message: str | list) -> list[dict]:
        if isinstance(message, list):
            return message
        return [{"role": "user", "content": message}]

    def _serialize_message(self, message) -> dict:
        if isinstance(message, dict):
            return {k: v for k, v in message.items() if v is not None}
        if hasattr(message, "model_dump"):
            return message.model_dump(exclude_none=True)
        return message.to_dict()

    def _get_usage_value(self, usage, key: str) -> int:
        if usage is None:
            return 0
        if isinstance(usage, dict):
            return usage.get(key, 0) or 0
        return getattr(usage, key, 0) or 0

    def codegen(
            self,
            message: str | list,
            num_samples: int = 1,
            prompt_cache: bool = False,
            tools: list | None = None,
            tool_choice: str | dict | None = None,
            return_message: bool = False,
            **kwargs,
    ) -> List[dict]:
        if self.temperature == 0:
            assert num_samples == 1

        batch_size = min(self.batch_size, num_samples)
        config = {
            "model": self.name,
            "messages": self._messages(message),
            "temperature": self.temperature,
            "n": batch_size,
            **self.litellm_kwargs,
            **kwargs,
        }
        if self.max_new_tokens is not None:
            config["max_tokens"] = self.max_new_tokens
        if tools is not None:
            config["tools"] = tools
        if tool_choice is not None:
            config["tool_choice"] = tool_choice

        ret = completion(**config)
        choices = ret["choices"] if isinstance(ret, dict) else ret.choices
        messages = []
        for choice in choices:
            choice_message = choice["message"] if isinstance(choice, dict) else choice.message
            messages.append(self._serialize_message(choice_message))

        usage = ret.get("usage", {}) if isinstance(ret, dict) else getattr(ret, "usage", {})
        completion_tokens = self._get_usage_value(usage, "completion_tokens")
        prompt_tokens = self._get_usage_value(usage, "prompt_tokens")
        responses = [msg.get("content") or "" for msg in messages]

        traj = {
            "response": responses[0] if responses else "",
            "usage": {
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
            },
        }
        if return_message:
            traj["message"] = messages[0] if messages else {"role": "assistant", "content": ""}
        trajs = [traj]

        for idx, response in enumerate(responses[1:], start=1):
            traj = {
                "response": response,
                "usage": {
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                },
            }
            if return_message:
                traj["message"] = messages[idx] if idx < len(messages) else {"role": "assistant", "content": response}
            trajs.append(traj)
        return trajs

    def is_direct_completion(self) -> bool:
        return False


def make_model(
    model: str,
    logger,
    batch_size: int = 1,
    max_tokens: int | None = None,
    temperature: float = 0.0,
    litellm_kwargs: dict[str, Any] | None = None,
):
    return LiteLLMChatDecoder(
        name=model,
        logger=logger,
        batch_size=batch_size,
        max_new_tokens=max_tokens,
        temperature=temperature,
        litellm_kwargs=litellm_kwargs,
    )
