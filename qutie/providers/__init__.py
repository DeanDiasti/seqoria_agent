"""
Maestra.core.llm: Package for managing LLM provider integrations.

This package provides modules for interacting with different
Large Language Model (LLM) providers, such as Groq, Bedrock,
and others. It defines a standardized interface for querying
these providers and handles common tasks like authentication,
error handling, and response formatting.
"""

# Import specific classes and functions to make them directly
# accessible when someone imports the llm package. This simplifies
# usage by avoiding the need to specify submodules.

from .base_provider import BaseProvider
from .errors import (LLMError,
                     EnvironmentError,
                     ConnectionError,
                     ModelNotFoundError,
                     ContextLengthExceededError,
                     ContentFilterError,
                     MaxRetriesExceededError,
                     InvalidRequestError,
                     QuotaExceededError,
                     ServerError)

# Import and expose the provider classes themselves.
# The provider classes are defined in separate files (e.g., groq.py, bedrock.py)
# but are made available directly under the llm package for convenience.

try:
    from .bedrock_provider import BedrockProvider
except ImportError:
    # Handle the case where the boto3 library is not installed, allowing
    # the rest of the system to function without it
    BedrockProvider = None  # Or a placeholder class/object

try:
    from .openai_provider import OpenAIProvider
except ImportError:
    # Handle the case where the OpenAI library is not installed, allowing
    # the rest of the system to function without it
    OpenAIProvider = None  # Or a placeholder class/object

try:
    from .anthropic_provider import AnthropicProvider
except ImportError:
    # Handle the case where the Anthropic library is not installed, allowing
    # the rest of the system to function without it
    AnthropicProvider = None  # Or a placeholder class/object

try:
    from .sambanova_provider import SambaNovaProvider
except ImportError:
    # Handle the case where required libraries are not installed
    SambaNovaProvider = None  # Or a placeholder class/object

# Define a list of all exported names from this package. This is used when
# someone does "from llm import *" to control which names get imported.
__all__ = ["BaseProvider",
           "LLMError",
           "EnvironmentError",
           "ConnectionError",
           "ModelNotFoundError",
           "ContextLengthExceededError",
           "ContentFilterError",
           "MaxRetriesExceededError",
           "InvalidRequestError",
           "QuotaExceededError",
           "ServerError",
           "GroqProvider",
           "BedrockProvider",
           "OpenAIProvider",
           "AnthropicProvider",
           "SambaNovaProvider",
           "GeminiProvider"]