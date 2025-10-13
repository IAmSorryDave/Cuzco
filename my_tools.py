from smolagents import tool

@tool
def my_custom_tool() -> str:
    
    """A tool that announces it has been called."""

    return "'my_custom_tool' was called."

