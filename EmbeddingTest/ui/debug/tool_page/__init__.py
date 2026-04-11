"""Debug tool page package."""

from .page import ToolPage
from .bindings import bind_tool_page

bind_tool_page(ToolPage)

__all__ = ["ToolPage"]
