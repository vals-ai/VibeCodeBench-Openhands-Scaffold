"""
Shared fixtures for all tests.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from integrations.solvability.models.classifier import SolvabilityClassifier
from integrations.solvability.models.featurizer import (
    Feature,
    FeatureEmbedding,
    Featurizer,
)
from model_library.base import QueryResult, ToolCall
from model_library.base.output.result import QueryResultMetadata
from sklearn.ensemble import RandomForestClassifier

from openhands.core.config import LLMConfig


@pytest.fixture
def features() -> list[Feature]:
    """Create a list of features for testing."""
    return [
        Feature(identifier='feature1', description='Test feature 1'),
        Feature(identifier='feature2', description='Test feature 2'),
        Feature(identifier='feature3', description='Test feature 3'),
    ]


@pytest.fixture
def feature_embedding() -> FeatureEmbedding:
    """Create a feature embedding for testing."""
    return FeatureEmbedding(
        samples=[
            {'feature1': True, 'feature2': False, 'feature3': True},
            {'feature1': False, 'feature2': True, 'feature3': True},
        ],
        prompt_tokens=10,
        completion_tokens=5,
        response_latency=0.1,
    )


@pytest.fixture
def featurizer(mock_llm, features, monkeypatch) -> Featurizer:
    """
    Create a featurizer for testing.

    Mocks out any calls to LLM.completion
    """
    monkeypatch.setattr(
        'integrations.solvability.models.featurizer.LLM',
        lambda *args, **kwargs: mock_llm,
    )

    featurizer = Featurizer(
        system_prompt='Test system prompt',
        message_prefix='Test message prefix: ',
        features=features,
    )

    return featurizer


@pytest.fixture
def mock_completion_response() -> QueryResult:
    return QueryResult(
        tool_calls=[
            ToolCall(
                id='feature-call',
                name='call_featurizer',
                args={
                    'feature1': True,
                    'feature2': False,
                    'feature3': True,
                },
            )
        ],
        metadata=QueryResultMetadata(in_tokens=10, out_tokens=5),
    )


@pytest.fixture
def mock_llm(mock_completion_response):
    """Create a mock LLM instance."""
    mock_llm_instance = MagicMock()
    mock_llm_instance.completion.return_value = mock_completion_response
    return mock_llm_instance


@pytest.fixture
def mock_llm_config():
    """Create a mock LLM config for testing."""
    return LLMConfig(model='test-model')


@pytest.fixture
def mock_classifier():
    """Create a mock classifier for testing."""
    clf = RandomForestClassifier(random_state=42)
    # Initialize with some dummy data to avoid errors
    X = np.array([[0, 0, 0], [1, 1, 1]])  # noqa: N806
    y = np.array([0, 1])
    clf.fit(X, y)
    return clf


@pytest.fixture
def solvability_classifier(featurizer, mock_classifier):
    """Create a SolvabilityClassifier instance for testing."""
    return SolvabilityClassifier(
        identifier='test-classifier',
        featurizer=featurizer,
        classifier=mock_classifier,
        random_state=42,
    )
