from asyncio import (
    AbstractEventLoop,
    new_event_loop,
    run_coroutine_threadsafe,
    set_event_loop,
)
from collections.abc import Coroutine
from concurrent.futures import Future
from threading import Lock, Thread
from typing import Any

from model_library.base import LLM, QueryResult
from model_library.base import LLMConfig as ValsLLMConfig
from model_library.registry_utils import get_registry_model

from openhands.core.config.llm_config import LLMConfig

MILLION_TOKENS = 1000000

def fetch_registry_model(llm_config: LLMConfig) -> LLM:
    try:
        config_kwargs: dict[str, Any] = {"supports_batch": False}

        if llm_config.max_output_tokens is not None:
            config_kwargs["max_tokens"] = llm_config.max_output_tokens

        if llm_config.temperature is not None:
            config_kwargs["temperature"] = llm_config.temperature

        if llm_config.top_p is not None:
            config_kwargs["top_p"] = llm_config.top_p

        if llm_config.reasoning_effort is not None:
            config_kwargs["reasoning_effort"] = llm_config.reasoning_effort

        if "xai" in llm_config.model.strip().lower():
            config_kwargs["sync_client"] = True

        vals_llm_config = ValsLLMConfig(**config_kwargs)

        registry_model = get_registry_model(
            llm_config.model, override_config=vals_llm_config
        )


    except ValueError as e:
        raise ValueError(
            f'Model name must be in the format "provider/model_name". Passed in name was: {llm_config.model}'
        ) from e

    return registry_model


class CostMetadata:
    input_cost: float
    output_cost: float

    def __init__(self, input_cost: float, output_cost: float) -> None:
        self.input_cost = input_cost
        self.output_cost = output_cost



class PersistentEventLoopRunner:
    """Runs a single asyncio event loop in a background thread.

    Thread-safe and reusable across calls. Coroutines are scheduled with
    asyncio.run_coroutine_threadsafe and awaited synchronously.
    """

    def __init__(self) -> None:
        self._loop: AbstractEventLoop | None = None
        self._thread: Thread | None = None
        self._start_lock: Lock = Lock()

    def _ensure_started(self) -> None:
        if self._loop and self._thread and self._thread.is_alive():
            return
        with self._start_lock:
            if self._loop and self._thread and self._thread.is_alive():
                return
            loop = new_event_loop()

            def _run_loop() -> None:
                set_event_loop(loop)
                loop.run_forever()

            thread = Thread(target=_run_loop, name='vals-event-loop', daemon=True)
            thread.start()
            self._loop = loop
            self._thread = thread

    def run(self, coro: Coroutine[Any, Any, QueryResult]) -> QueryResult:
        self._ensure_started()
        assert self._loop is not None
        fut: Future[QueryResult] = run_coroutine_threadsafe(coro, self._loop)
        return fut.result()
