from enum import Enum
from typing import Literal

from model_library.base import ToolCall
from pydantic import BaseModel, Field, model_serializer


class ContentType(Enum):
    TEXT = 'text'
    IMAGE_URL = 'image_url'


class Content(BaseModel):
    type: str
    cache_prompt: bool = False

    @model_serializer(mode='plain')
    def serialize_model(
        self,
    ) -> dict[str, str | dict[str, str]] | list[dict[str, str | dict[str, str]]]:
        raise NotImplementedError('Subclasses should implement this method.')


class TextContent(Content):
    type: str = ContentType.TEXT.value
    text: str

    @model_serializer(mode='plain')
    def serialize_model(self) -> dict[str, str | dict[str, str]]:
        data: dict[str, str | dict[str, str]] = {
            'type': self.type,
            'text': self.text,
        }
        if self.cache_prompt:
            data['cache_control'] = {'type': 'ephemeral'}
        return data


class ImageContent(Content):
    type: str = ContentType.IMAGE_URL.value
    image_urls: list[str]

    @model_serializer(mode='plain')
    def serialize_model(self) -> list[dict[str, str | dict[str, str]]]:
        images: list[dict[str, str | dict[str, str]]] = []
        for url in self.image_urls:
            images.append({'type': self.type, 'image_url': {'url': url}})
        if self.cache_prompt and images:
            images[-1]['cache_control'] = {'type': 'ephemeral'}
        return images


class Message(BaseModel):
    # NOTE: this is not the same as EventSource
    # These are the roles in the LLM's APIs
    role: Literal['user', 'system', 'assistant', 'tool']
    content: list[TextContent | ImageContent] = Field(default_factory=list)
    vision_enabled: bool = False
    # - tool calls (from LLM)
    tool_calls: list[ToolCall] = []
    # - tool execution result (to LLM)
    id: str | None = None
    # force string serializer
    force_string_serializer: bool = False

    @property
    def contains_image(self) -> bool:
        return any(isinstance(content, ImageContent) for content in self.content)
