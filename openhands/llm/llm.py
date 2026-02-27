import copy
import os
from typing import Any, Callable, cast, override

from model_library.base import LLM as ValsLLM
from model_library.base import (
    InputItem,
    QueryResult,
    ToolDefinition,
)
from tokenizers import Tokenizer

from openhands.core.config.llm_config import LLMConfig
from openhands.core.logger import openhands_logger as logger
from openhands.llm.metrics import Metrics
from openhands.llm.utils import (
    PersistentEventLoopRunner,
    fetch_registry_model,
)

_persistent_loop: PersistentEventLoopRunner | None = None

MESSAGE_SEPARATOR = "\n\n----------\n\n"


class DebugMixin:
    def _truncate(self, string: str, max_length: int = 1000) -> str:
        if len(string) > max_length:
            return string[: max_length // 2] + "..." + string[-max_length // 2 :]

        return string

    def pretty_print(self, obj: Any) -> None:
        debug_message = ""
        if isinstance(obj, list):
            for item in obj:
                debug_message += f"{self._truncate(str(item))}\n"
        elif isinstance(obj, dict):
            for key, value in obj.items():
                debug_message += f"{key}: {self._truncate(str(value))}\n"
        elif isinstance(obj, str):
            debug_message += f"{self._truncate(obj)}\n"
        else:
            debug_message += f"{self._truncate(str(obj))}\n"

        if debug_message:
            logger.debug(debug_message)

    def vision_is_active(self) -> bool:
        raise NotImplementedError


class LLM(DebugMixin):
    def __init__(
        self,
        config: LLMConfig,
        service_id: str,
        metrics: Metrics | None = None,
        retry_listener: Callable[[int, int], None] | None = None,
    ) -> None:
        self.config: LLMConfig = copy.deepcopy(config)
        self.service_id: str = service_id
        self.metrics: Metrics = (
            metrics if metrics is not None else Metrics(model_name=config.model)
        )
        self._function_calling_active: bool = False
        if self.config.log_completions:
            if not self.config.log_completions_folder:
                raise RuntimeError(
                    "log_completions_folder is required when log_completions is enabled"
                )

            os.makedirs(self.config.log_completions_folder, exist_ok=True)

        self.tokenizer: Tokenizer | None = None

        model = fetch_registry_model(self.config)

        self.pretty_print(f"[Model] {model}")

        self.kwargs: dict[str, Any] = {}
        self.model: ValsLLM = model

        def wrapper(*args: Any, **kwargs: Any) -> QueryResult:
            input = cast(list[InputItem], kwargs.pop("input", []))
            history = cast(list[InputItem], kwargs.pop("history", []))
            tools = cast(list[ToolDefinition], kwargs.pop("tools", []))

            global _persistent_loop
            if _persistent_loop is None:
                _persistent_loop = PersistentEventLoopRunner()

            async def _query_llm() -> QueryResult:
                query_result: QueryResult = await self.model.query(
                    input=input,
                    history=history,
                    tools=tools,
                    **self.kwargs,
                )

                return query_result

            query_result = _persistent_loop.run(_query_llm())

            metadata = query_result.metadata

            self.metrics.add_response_latency(metadata.duration_seconds or 0, "")

            cost = metadata.cost.total if metadata.cost else 0

            self.pretty_print(
                "[Turn Cost] ${:.6f}\n[Turn Usage] {:}".format(cost, str(metadata))
            )
            self.metrics.add_cost(cost)

            self.metrics.add_token_usage(
                prompt_tokens=metadata.in_tokens,
                completion_tokens=metadata.out_tokens,
                cache_read_tokens=metadata.cache_read_tokens or 0,
                cache_write_tokens=metadata.cache_write_tokens or 0,
                context_window=0,
                response_id="",
            )

            self.pretty_print(
                "[Total Cost] ${:.6f}\n[Total Usage] {}".format(
                    self.metrics.accumulated_cost,
                    self.metrics.accumulated_token_usage,
                )
            )

            return query_result

        self._completion: Callable[..., QueryResult] = wrapper

    @property
    def completion(self) -> Callable[..., QueryResult]:
        return self._completion

    @override
    def vision_is_active(self) -> bool:
        return not self.config.disable_vision and self._supports_vision()

    def _supports_vision(self) -> bool:
        return self.model.supports_images


    def is_function_calling_active(self) -> bool:
        return self.model.supports_tools

    def get_token_count(self, history: list[InputItem]) -> int:
        raise NotImplementedError("Not implemented")

    @override
    def __str__(self) -> str:
        return f"LLM(model={self.config.model} \
        provider={self.model.provider} \
        max_tokens={self.model.max_tokens} \
        reasoning={self.config.reasoning_effort} \
        supports_vision={self.vision_is_active()} \
        supports_function_calling={self.is_function_calling_active()} \
        supports_files={self.model.supports_files} \
        supports_images={self.model.supports_images} \
        supports_batch={self.model.supports_batch} \
        supports_temperature={self.model.supports_temperature} \
        supports_tools={self.model.supports_tools} \
        )"

    @override
    def __repr__(self) -> str:
        return str(self)
