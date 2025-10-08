"""
Model wrappers for different API providers
Handles OpenAI, Anthropic, vLLM, and other model APIs
"""

import os
import re
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import logging
import requests

logger = logging.getLogger(__name__)

# Check for API libraries
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not installed")

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic library not installed")


class ModelWrapper(ABC):
    """
    Abstract base class for model wrappers
    """

    @abstractmethod
    def generate(self, prompt: str, scaling_level: int = 1, temperature: float = 0.7, max_tokens: int = 32768) -> Dict:
        """
        Generate response from model

        Args:
            prompt: Input prompt
            scaling_level: Scaling iteration level
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with 'text', 'total_tokens', 'completion_tokens'
        """
        pass

    @abstractmethod
    def extract_answer(self, response_text: str) -> str:
        """
        Extract numerical answer from response text

        Args:
            response_text: Model response text

        Returns:
            Extracted answer string
        """
        pass


class OpenAIWrapper(ModelWrapper):
    """
    Wrapper for OpenAI API models
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize OpenAI wrapper

        Args:
            model_name: OpenAI model name (e.g., 'gpt-4', 'o1')
            api_key: Optional API key (uses env var if not provided)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Run: pip install openai")

        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

        # Configure based on model type
        self.is_reasoning_model = model_name.startswith("o")

    def generate(self, prompt: str, scaling_level: int = 1, temperature: float = 0.7, max_tokens: int = 32768) -> Dict:
        """Generate response using OpenAI API"""

        # Adjust prompt based on scaling level
        if scaling_level > 1:
            system_prompt = f"Think step by step. Use {scaling_level}x more detailed reasoning."
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]

        # Special handling for o-series models
        if self.is_reasoning_model:
            # o-series models use different parameters
            response = self.client.chat.completions.create(model=self.model_name, messages=messages, max_completion_tokens=max_tokens * scaling_level)
        else:
            response = self.client.chat.completions.create(model=self.model_name, messages=messages, temperature=temperature, max_tokens=max_tokens * scaling_level)

        return {"text": response.choices[0].message.content, "total_tokens": response.usage.total_tokens, "completion_tokens": response.usage.completion_tokens}

    def extract_answer(self, response_text: str) -> str:
        """Extract answer from response"""
        # Look for boxed answer
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", response_text)
        if boxed_match:
            return boxed_match.group(1)

        # Look for "Answer: X" pattern
        answer_match = re.search(r"[Aa]nswer:?\s*([0-9\-.,]+)", response_text)
        if answer_match:
            return answer_match.group(1)

        # Look for last number in text
        numbers = re.findall(r"-?\d+(?:\.\d+)?", response_text)
        if numbers:
            return numbers[-1]

        return ""


class AnthropicWrapper(ModelWrapper):
    """
    Wrapper for Anthropic API models
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize Anthropic wrapper

        Args:
            model_name: Anthropic model name (e.g., 'claude-opus-4-1')
            api_key: Optional API key
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not installed. Run: pip install anthropic")

        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def generate(self, prompt: str, scaling_level: int = 1, temperature: float = 0.7, max_tokens: int = 32768) -> Dict:
        """Generate response using Anthropic API"""

        # Adjust system prompt based on scaling
        if scaling_level > 1:
            system = f"Think step by step with {scaling_level}x more detailed reasoning."
        else:
            system = "You are a helpful assistant."

        response = self.client.messages.create(model=self.model_name, max_tokens=max_tokens * scaling_level, temperature=temperature, system=system, messages=[{"role": "user", "content": prompt}])

        # Calculate token usage (approximate if not provided)
        text = response.content[0].text
        completion_tokens = response.usage.output_tokens if hasattr(response.usage, "output_tokens") else len(text.split()) * 1.3
        input_tokens = response.usage.input_tokens if hasattr(response.usage, "input_tokens") else len(prompt.split()) * 1.3

        return {"text": text, "total_tokens": int(input_tokens + completion_tokens), "completion_tokens": int(completion_tokens)}

    def extract_answer(self, response_text: str) -> str:
        """Extract answer from Claude's response"""
        # Similar extraction logic as OpenAI
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", response_text)
        if boxed_match:
            return boxed_match.group(1)

        answer_match = re.search(r"[Aa]nswer:?\s*([0-9\-.,]+)", response_text)
        if answer_match:
            return answer_match.group(1)

        numbers = re.findall(r"-?\d+(?:\.\d+)?", response_text)
        if numbers:
            return numbers[-1]

        return ""


class VLLMWrapper(ModelWrapper):
    """
    Wrapper for vLLM server (OpenAI-compatible endpoint)
    """

    def __init__(self, model_name: str, base_url: str = "http://localhost:8000"):
        """
        Initialize vLLM wrapper

        Args:
            model_name: Model name on vLLM server
            base_url: Base URL for vLLM server
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

        # Check if server is running
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code != 200:
                logger.warning(f"vLLM server at {base_url} may not be healthy")
        except:
            logger.warning(f"Cannot connect to vLLM server at {base_url}")

    def generate(self, prompt: str, scaling_level: int = 1, temperature: float = 0.7, max_tokens: int = 32768) -> Dict:
        """Generate response using vLLM server"""

        # Prepare request based on scaling level
        if scaling_level > 1:
            system_message = f"Think step by step with {scaling_level}x more detailed reasoning."
            messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]

        # Make request to vLLM server
        response = requests.post(f"{self.base_url}/v1/chat/completions", json={"model": self.model_name, "messages": messages, "temperature": temperature, "max_tokens": max_tokens * scaling_level})

        if response.status_code != 200:
            raise Exception(f"vLLM request failed: {response.text}")

        result = response.json()

        return {"text": result["choices"][0]["message"]["content"], "total_tokens": result["usage"]["total_tokens"], "completion_tokens": result["usage"]["completion_tokens"]}

    def extract_answer(self, response_text: str) -> str:
        """Extract answer from vLLM response"""
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", response_text)
        if boxed_match:
            return boxed_match.group(1)

        answer_match = re.search(r"[Aa]nswer:?\s*([0-9\-.,]+)", response_text)
        if answer_match:
            return answer_match.group(1)

        numbers = re.findall(r"-?\d+(?:\.\d+)?", response_text)
        if numbers:
            return numbers[-1]

        return ""


class DeepSeekWrapper(ModelWrapper):
    """
    Wrapper for DeepSeek API
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize DeepSeek wrapper

        Args:
            model_name: DeepSeek model name
            api_key: Optional API key
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com/v1"

    def generate(self, prompt: str, scaling_level: int = 1, temperature: float = 0.7, max_tokens: int = 32768) -> Dict:
        """Generate response using DeepSeek API"""

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        # Adjust for reasoning mode if using DeepSeek-R1
        if "r1" in self.model_name.lower() or "reasoner" in self.model_name.lower():
            # Enable reasoning mode
            data = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}], "temperature": temperature, "max_tokens": max_tokens * scaling_level, "reasoning_mode": True, "reasoning_depth": scaling_level}
        else:
            data = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}], "temperature": temperature, "max_tokens": max_tokens * scaling_level}

        response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data)

        if response.status_code != 200:
            raise Exception(f"DeepSeek API request failed: {response.text}")

        result = response.json()

        return {"text": result["choices"][0]["message"]["content"], "total_tokens": result["usage"]["total_tokens"], "completion_tokens": result["usage"]["completion_tokens"]}

    def extract_answer(self, response_text: str) -> str:
        """Extract answer from DeepSeek response"""
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", response_text)
        if boxed_match:
            return boxed_match.group(1)

        answer_match = re.search(r"[Aa]nswer:?\s*([0-9\-.,]+)", response_text)
        if answer_match:
            return answer_match.group(1)

        numbers = re.findall(r"-?\d+(?:\.\d+)?", response_text)
        if numbers:
            return numbers[-1]

        return ""


def get_model_wrapper(model_name: str, **kwargs) -> ModelWrapper:
    """
    Factory function to get appropriate model wrapper

    Args:
        model_name: Name of model
        **kwargs: Additional arguments for wrapper

    Returns:
        Appropriate ModelWrapper instance
    """
    model_lower = model_name.lower()

    # OpenAI models
    if any(prefix in model_lower for prefix in ["gpt", "o1", "o3"]):
        return OpenAIWrapper(model_name, **kwargs)

    # Anthropic models
    elif any(prefix in model_lower for prefix in ["claude", "opus", "sonnet"]):
        return AnthropicWrapper(model_name, **kwargs)

    # DeepSeek models
    elif "deepseek" in model_lower:
        return DeepSeekWrapper(model_name, **kwargs)

    # vLLM models (marked with vllm: prefix)
    elif model_lower.startswith("vllm:"):
        actual_model = model_name.split(":", 1)[1]
        base_url = kwargs.pop("base_url", "http://localhost:8000")
        return VLLMWrapper(actual_model, base_url)

    # Default to vLLM for open-source models
    else:
        logger.info(f"Assuming {model_name} is hosted on vLLM server")
        base_url = kwargs.pop("base_url", "http://localhost:8000")
        return VLLMWrapper(model_name, base_url)
