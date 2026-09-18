from model_library.exceptions import (
    GatewayProviderError,
    MaxContextWindowExceededError,
)

from openhands.llm.llm import _is_model_library_max_context_window_error


def test_gateway_wrapped_max_context_error_is_recognized() -> None:
    message = 'request exceeded model token limit'
    gateway_error = GatewayProviderError(
        error_type='ProviderError',
        code=None,
        message=message,
        provider='kimi',
        raw_error={},
        exception_type='MaxContextWindowExceededError',
        status_code=400,
    )

    assert _is_model_library_max_context_window_error(
        MaxContextWindowExceededError(message)
    )
    assert _is_model_library_max_context_window_error(gateway_error)

    gateway_error.exception_type = 'AuthenticationError'
    assert not _is_model_library_max_context_window_error(gateway_error)
