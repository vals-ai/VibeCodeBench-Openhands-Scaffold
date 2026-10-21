import pytest
from model_library.base import ToolCall

from openhands.core.message import ImageContent, Message, TextContent


@pytest.mark.parametrize('vision_enabled', [True, False])
@pytest.mark.parametrize('has_image', [True, False])
def test_message_detects_images_independently_of_vision_configuration(
    vision_enabled, has_image
):
    content = [TextContent(text='Inspect this file')]
    if has_image:
        content.append(ImageContent(image_urls=['https://example.com/image.png']))
    message = Message(role='user', content=content, vision_enabled=vision_enabled)
    assert message.contains_image is has_image


def test_message_tool_call_serialization_preserves_model_library_arguments():
    message = Message(
        role='assistant',
        tool_calls=[
            ToolCall(id='call_123', name='test_function', args={'path': 'example.py'})
        ],
    )
    restored = Message.model_validate_json(message.model_dump_json())
    assert restored.tool_calls[0].id == 'call_123'
    assert restored.tool_calls[0].name == 'test_function'
    assert restored.tool_calls[0].parsed_args == {'path': 'example.py'}


def test_message_tool_response_serialization_preserves_call_id():
    message = Message(
        role='tool', id='call_123', content=[TextContent(text='Function result')]
    )
    restored = Message.model_validate_json(message.model_dump_json())
    assert restored.id == 'call_123'
    assert restored.content[0].text == 'Function result'
