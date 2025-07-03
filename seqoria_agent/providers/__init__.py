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

from seqoria_agent.providers.base_provider import BaseProvider
from seqoria_agent.providers.errors import (ConnectionError,
                                            ContentFilterError,
                                            ContextLengthExceededError,
                                            EnvironmentError,
                                            InvalidRequestError,
                                            LLMError,
                                            MaxRetriesExceededError,
                                            ModelNotFoundError,
                                            QuotaExceededError,
                                            ServerError)

# Import and expose the provider classes themselves.
# The provider classes are defined in separate files (e.g., groq.py, bedrock.py)
# but are made available directly under the llm package for convenience.
from seqoria_agent.providers.openai_provider import OpenAIProvider
from seqoria_agent.providers.anthropic_provider import AnthropicProvider


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
           "OpenAIProvider",
           "AnthropicProvider"]