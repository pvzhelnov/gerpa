```python
"""
LLM Provider SDK - Unified interface for multiple LLM providers
"""

import os
import json
import yaml
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Type, Union, List
from abc import ABC, abstractmethod, ABCMeta
from pydantic import BaseModel

import requests
from google import genai
import ollama
from openai import OpenAI

from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

class NotReadyMeta(ABCMeta):
    def __new__(mcs, name, bases, dct):
        for attr_name, attr_value in dct.items():
            mask = ((not attr_name.startswith('__')) or (attr_name == '__init__'))
            if callable(attr_value) and mask:
                dct[attr_name] = NotReadyMeta.raise_not_implemented_wrapper(attr_value)
        return super().__new__(mcs, name, bases, dct)

    @staticmethod
    def raise_not_implemented_wrapper(func):
        def wrapper(*args, **kwargs):
            raise NotImplementedError(f"Class '{args[0].__class__.__name__}' is not ready for use yet. Method '{func.__name__}' is not implemented.")
        return wrapper

class LLMResponse(BaseModel):
    """Standard response format for all LLM providers"""
    content: BaseModel = None
    model: str
    provider: str
    timestamp: datetime
    token_usage: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = {}


class BaseLLMProvider(ABC):
    """Base class for all LLM providers"""
    
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.provider_name = self.__class__.__name__.lower().replace('provider', '')
        
    @abstractmethod
    def generate(self, prompt: Union[str, List[str]], response_schema: Optional[Type[BaseModel]] = None) -> LLMResponse:
        """Generate response from the LLM"""
        pass


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider"""
    
    def __init__(self, model: str = "gemini-2.0-flash", **kwargs):
        super().__init__(model, **kwargs)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        self.client = genai.Client(api_key=api_key)
        
    def generate(self, prompt: Union[str, List[str]], response_schema: Optional[Type[BaseModel]] = None) -> LLMResponse:
        try:
            # Handle both string and list inputs
            if isinstance(prompt, str):
                prompt_parts = [prompt]
            else:
                prompt_parts = prompt
                
            contents = []

            # Process each item in the prompt
            for item in prompt_parts:
                if self._is_valid_path_or_url(item):
                    uploaded_file = self._upload_file(item)
                    contents.append(uploaded_file)
                else:
                    contents.append(item)
            
            # Add schema instruction as separate item if response_schema is specified
            if response_schema:
                schema_instruction = f"Respond with valid JSON matching this schema: {response_schema.model_json_schema()}"
                contents.append(schema_instruction)
                
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=GenerateContentConfig(
                    temperature=0.8,
                    response_mime_type='application/json',
                    response_schema=response_schema
                )
            )
            
            return LLMResponse(
                content=response_schema.model_validate_json(response.text),
                model=self.model,
                provider="gemini",
                timestamp=datetime.now(),
                token_usage={
                    "prompt_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
                    "total_tokens": response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
                } if hasattr(response, 'usage_metadata') else None,
                metadata={"finish_reason": "completed"}
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")
        
    def _is_valid_path_or_url(self, prompt_part: str) -> bool:
        """Check if prompt is a valid local path or accessible URL"""
        # Check if it's a local file path
        if Path(prompt_part.strip()).exists():
            return True
        
        # Check if it's a URL
        if prompt_part.strip().startswith(('http://', 'https://')):
            try:
                response = requests.head(prompt_part.strip(), timeout=5)
                return response.status_code == 200
            except:
                return False
        
        return False

    def _upload_file(self, path_or_url: str):
        """Upload file to Gemini"""
        if path_or_url.startswith(('http://', 'https://')):
            # For URLs, download first then upload
            response = requests.get(path_or_url)
            temp_file = f"/tmp/{Path(path_or_url).name}"
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            uploaded_file = self.client.files.upload(file=temp_file)
            os.remove(temp_file)
            return uploaded_file
        else:
            # For local files
            return self.client.files.upload(file=path_or_url.strip())


class OpenRouterProvider(BaseLLMProvider, metaclass=NotReadyMeta):
    """OpenRouter provider"""
    
    def __init__(self, model: str = "anthropic/claude-3.5-sonnet", **kwargs):
        super().__init__(model, **kwargs)
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
    def generate(self, prompt: str, response_schema: Optional[Type[BaseModel]] = None) -> LLMResponse:
        try:
            messages = [{"role": "user", "content": prompt}]
            
            if response_schema:
                # Add JSON schema instruction
                schema_instruction = f"\nRespond with valid JSON matching this schema: {response_schema.model_json_schema()}"
                messages[0]["content"] = prompt + schema_instruction
                
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model,
                provider="openrouter",
                timestamp=datetime.now(),
                token_usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else None,
                metadata={"finish_reason": response.choices[0].finish_reason}
            )
        except Exception as e:
            raise RuntimeError(f"OpenRouter API error: {str(e)}")


class OllamaProvider(BaseLLMProvider, metaclass=NotReadyMeta):
    """Ollama local provider"""
    
    def __init__(self, model: str = "llama3.1", host: str = "http://localhost:11434", **kwargs):
        super().__init__(model, **kwargs)
        self.host = host
        
    def generate(self, prompt: str, response_schema: Optional[Type[BaseModel]] = None) -> LLMResponse:
        try:
            if response_schema:
                # Add JSON schema instruction
                schema_instruction = f"\nRespond with valid JSON matching this schema: {response_schema.model_json_schema()}"
                prompt = prompt + schema_instruction
                
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                host=self.host
            )
            
            return LLMResponse(
                content=response['response'],
                model=self.model,
                provider="ollama",
                timestamp=datetime.now(),
                token_usage={
                    "prompt_tokens": response.get('prompt_eval_count', 0),
                    "completion_tokens": response.get('eval_count', 0),
                    "total_tokens": response.get('prompt_eval_count', 0) + response.get('eval_count', 0)
                },
                metadata={
                    "total_duration": response.get('total_duration'),
                    "load_duration": response.get('load_duration'),
                    "prompt_eval_duration": response.get('prompt_eval_duration'),
                    "eval_duration": response.get('eval_duration')
                }
            )
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {str(e)}")


class LLMAgent:
    """Main agent class for LLM interactions"""
    
    def __init__(self, provider: str, model: str = None, response_schema: Optional[Type[BaseModel]] = None, **kwargs):
        self.response_schema = response_schema
        self.logger = self._setup_logger()
        
        # Provider mapping
        providers = {
            "gemini": GeminiProvider,
            "openrouter": OpenRouterProvider,
            "ollama": OllamaProvider
        }
        
        if provider not in providers:
            raise ValueError(f"Unsupported provider: {provider}. Supported: {list(providers.keys())}")
            
        provider_class = providers[provider]
        if model:
            self.provider = provider_class(model=model, **kwargs)
        else:
            self.provider = provider_class(**kwargs)
            
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with date-based file structure"""
        now = datetime.now()
        log_dir = Path("logs") / str(now.year) / f"{now.month:02d}" / f"{now.day:02d}"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the calling script name
        import inspect
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back.f_back
            script_name = Path(caller_frame.f_globals.get('__file__', 'unknown')).stem
        finally:
            del frame
            
        log_file = log_dir / f"{script_name}.log"
        
        logger = logging.getLogger(f"llm_agent_{script_name}")
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    def __call__(self, prompt: Union[str, List[str]], save_response: bool = True) -> LLMResponse:
        """Generate response from LLM"""
        try:
            # Version the prompt
            prompt_hash = self._version_prompt(prompt)
            
            # Generate response
            response = self.provider.generate(prompt, self.response_schema)

            json_response = response.content.model_dump_json()
            
            # Log the interaction
            log_data = {
                "prompt_hash": prompt_hash,
                "response_content": json_response[:200] + "..." if len(json_response) > 200 else json_response,
                "model": response.model,
                "provider": response.provider,
                "token_usage": response.token_usage,
                "timestamp": response.timestamp.isoformat()
            }
            self.logger.info(f"LLM Response: {json.dumps(log_data, indent=2, ensure_ascii=False)}")
            
            # Save response if requested
            if save_response:
                self._save_response(response, prompt_hash)
                
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise
            
    def _version_prompt(self, prompt: Optional[Union[str, List[str]]]) -> str:
        """Version a prompt and save it"""

        if prompt is None:
            return

        # Stringify if List[str] before hashing
        stringified_prompt = str(prompt)

        prompt_hash = hashlib.sha256(stringified_prompt.encode()).hexdigest()[:8]
        
        prompts_dir = Path("prompts")
        prompts_dir.mkdir(exist_ok=True)
        
        prompt_file = prompts_dir / f"{prompt_hash}.txt"
        if not prompt_file.exists():
            prompt_file.write_text(stringified_prompt)
            
        return prompt_hash
        
    def _save_response(self, response: LLMResponse, prompt_hash: str):
        """Save response as YAML file"""
        responses_dir = Path("responses")
        responses_dir.mkdir(exist_ok=True)
        
        timestamp_str = response.timestamp.strftime("%Y%m%d_%H%M%S")
        response_file = responses_dir / f"{timestamp_str}_{prompt_hash}_{response.provider}.yml"
        
        response_data = {
            "content": response.content.model_dump(),  # to dict
            "model": response.model,
            "provider": response.provider,
            "timestamp": response.timestamp.isoformat(),
            "token_usage": response.token_usage,
            "metadata": response.metadata,
            "prompt_hash": prompt_hash,
            "prompt_file": f"prompts/{prompt_hash}.txt",
            "response_schema": self.response_schema.model_json_schema() if self.response_schema else None,
            "evals": {},
            "ground_truth": None,
            "name": None
        }
        
        with open(response_file, 'w') as f:
            yaml.dump(response_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def agent(provider: str, response_schema: Optional[Type[BaseModel]] = None, **kwargs) -> LLMAgent:
    """Factory function to create LLM agent"""
    return LLMAgent(provider=provider, response_schema=response_schema, **kwargs)
```
