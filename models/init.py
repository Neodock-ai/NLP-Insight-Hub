# models/__init__.py

from .llama import LlamaModel
from .falcon import FalconModel
from .mistral import MistralModel
from .deepseek import DeepSeekModel

# NEW: import our new GPT model
from .openai_gpt import OpenAIGPTModel

__all__ = [
    'LlamaModel',
    'FalconModel',
    'MistralModel',
    'DeepSeekModel',
    'OpenAIGPTModel'
]
