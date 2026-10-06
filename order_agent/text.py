from langchain_core.messages import BaseMessage


def text_of(message: BaseMessage) -> str:
    """A message's text, whether the provider returned a string or content blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))
