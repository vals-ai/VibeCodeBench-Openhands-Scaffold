from unittest.mock import patch

import pytest

from openhands.llm import LLM


@pytest.fixture(autouse=True)
def isolate_model_initialization():
    with patch.object(LLM, '__init__', return_value=None):
        yield
