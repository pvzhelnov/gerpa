```python
"""
LLM Provider SDK - Unified interface for multiple LLM providers
"""

import os
import json
import re  # for fixing json in unconstrained output
import yaml
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import pytesseract  # for OCR
from PIL import Image  # for OCR

from IPython.display import display, Markdown  # for pretty print

if not load_dotenv():  # Try one level above modules dir
    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=dotenv_path)

def setup_logger(console: bool | None = None, file: bool | None = None) -> logging.Logger:
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

    # Get log level from environment
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
    # File handler
    LOG_FILE = os.getenv("LOG_FILE", "true").lower() == "true"
    need_log_file = file if file is not None else LOG_FILE  # arg takes precedence
    if need_log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    LOG_CONSOLE = os.getenv("LOG_CONSOLE", "true").lower() == "true"
    need_log_console = console if console is not None else LOG_CONSOLE  # arg takes precedence
    if need_log_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

from typing import Dict, Any, Optional, Type, Union, List, Literal
from pydantic import BaseModel, Field
from enum import Enum

from abc import ABC, abstractmethod, ABCMeta

import requests
from google import genai
import ollama
from openai import OpenAI

import base64  # for encoding images
import mimetypes  # for encoding images

from google.genai import types

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
    
class BasePrompt(BaseModel):
    prompt: Optional[Union[str, List[str]]] = None
    hash: Optional[str] = None
    reused: Optional[bool | None] = None

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and not kwargs:
            kwargs = {'prompt': args[0]}
        elif len(args) > 1:
            raise TypeError("Only one positional argument allowed.")
        super().__init__(**kwargs)
        self.hash, self.reused = self._version_prompt()

    def _version_prompt(self) -> tuple[str | None, bool | None]:
        """Version a prompt and save it"""

        if self.prompt is None:
            return None, None

        # Stringify if List[str] before hashing
        stringified_prompt = str(self.prompt)

        prompt_hash = hashlib.sha256(stringified_prompt.encode()).hexdigest()[:8]
        
        prompts_dir = Path("prompts")
        prompts_dir.mkdir(exist_ok=True)
        
        prompt_file = prompts_dir / f"{prompt_hash}.txt"
        if not prompt_file.exists():
            reused = False
            prompt_file.write_text(stringified_prompt)
        else:
            reused = True
            
        return prompt_hash, reused
    
    def __str__(self):
        return str(self.prompt or "")
    
    def parts(self):
        if isinstance(self.prompt, list):
            prompt_parts = self.prompt
        else:
            prompt_parts = [str(self.prompt)] if self.prompt else []
        return prompt_parts
    
class BaseResponseSchema(BaseModel):
    pass

class LLMResponse(BaseModel):
    """Standard response format for all LLM providers"""
    raw_content: str | None = None
    content: BaseModel = BaseResponseSchema()
    token_usage: Optional[Dict[str, Union[int, None]]] = None
    metadata: Dict[str, Any] = {}

    def _dump_json(self, pydantic_model: BaseModel) -> str:
        return json.dumps(
            pydantic_model.model_dump(),
            indent=2,
            ensure_ascii=False,
            default=str
        )
    
    def _dump_yaml(self, pydantic_model: BaseModel) -> str:
        # Custom representer for multiline strings
        def literal_presenter(dumper, data):
            if '\n' in data:
                return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
            return dumper.represent_scalar('tag:yaml.org,2002:str', data)
        # Add the custom representer
        yaml.add_representer(str, literal_presenter)
        return yaml.dump(
            pydantic_model.model_dump(mode='json'),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            indent=2,
        )
    
    def content_dump_json(self):
        """Returns validated content as a JSON string."""
        return self._dump_json(self.content)
    
    def raw_content_dump_json(self):
        """Returns raw content as a JSON string."""
        return self.raw_content

    def content_dump_yaml(self):
        """Returns validated content as a multiline YAML string."""
        return self._dump_yaml(self.content)

    def display_markdown_content(self):
        """Display validated content only, as Markdown."""
        markdown_content_parts = []
        if self.content.model_dump():
            markdown_content_parts.append(
                "Response received. Validated content:"
            )
            yaml_content = self.content_dump_yaml()
            markdown_content_parts.append(f'```yaml\n{yaml_content}\n```')
            markdown_content_parts.append(f'Detailed metadata available from full response variable or file.')
        elif self.raw_content:
            markdown_content_parts.append(
                "Response received but failed to validate its content. Raw content:"
            )
            raw_json_content = self.raw_content_dump_json()
            markdown_content_parts.append(f'```json\n{raw_json_content}\n```')
            markdown_content_parts.append(f'Detailed metadata available from full response variable or file.')
        else:
            markdown_content_parts.extend([
                "No response received or the content is empty.",
                "Please review the logs or the full response file."
            ])
            return
        
        display(Markdown("\n\n".join(markdown_content_parts)))

class BaseLLM(BaseModel):
    """Base class for all LLMs"""
    model_name: Optional[str] = None
    model_metadata: Optional[dict] = {}
    response_schema: Type[BaseModel] = BaseResponseSchema
    system_instruction: BasePrompt = BasePrompt()
    temperature: Optional[float] = 0.0
    top_k: Optional[int] = 40
    top_p: Optional[float] = 0.95  # Ollama and AI studio default is 0.95
    seed: Optional[int] = 42  # AI studio doesn't expose this, so using Ollama default
    max_tokens: Optional[int] = 8192  # AI studio default
    safety_settings: Optional[Any] = None

class GeminiLLM(BaseLLM):
    # https://aistudio.google.com/app/u/prompts/new_chat?model=gemma-3-27b-it
    model_name: Optional[str] = "gemma-3-27b-it"
    top_k: Optional[int] = 64  # AI studio doesn't expose this, so using Ollama's default for gemma3:27b-it-qat (29eb0b9aeda3)
    # also from: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/content-generation-parameters
    #'stop_sequences'=["STOP!"],  # perhaps default for API
    #'presence_penalty'=0.0,  # perhaps default for API
    #'frequency_penalty'=0.0,  # perhaps default for API

class OllamaLLM(BaseLLM):
    model_name: Optional[str] = "gemma3:27b-it-qat"
    # https://github.com/ollama/ollama/blob/4261a3b0b264430489921a1b4a16a6267711d595/docs/modelfile.md#valid-parameters-and-values
    num_ctx: Optional[int] = 4096  # Ollama default; the size of the context window used to generate the next token
    top_k: Optional[int] = 64  # Ollama's default for gemma3:27b-it-qat (29eb0b9aeda3), see here: https://ollama.com/library/gemma3:27b-it-qat
    min_p: Optional[float] = 0.05  # Ollama default
    #max_tokens: Optional[int] (Ollama default: -1, infinite generation)
    # also supported:
    # repeat_penalty (Default: 1.1)
    # repeat_last_n (Default: 64, 0 = disabled, -1 = num_ctx). Sets how far back for the model to look back to prevent repetition.

class OpenRouterLLM(BaseLLM):  # set to mirror OllamaLLM
    # https://openrouter.ai/google/gemma-3-27b-it:free
    model_name: Optional[str] = "google/gemma-3-27b-it:free"
    top_k: Optional[int] = 64
    min_p: Optional[float] = 0.05  # unsupported with Google AI Studio provider
    # also supported with Chutes provider: Stop, Frequency Penalty, Presence Penalty, Repetition Penalty, Logprobs, Logit Bias, Top Logprobs

class BaseLLMProvider(ABC):
    """Base class for all LLM providers"""
    
    def __init__(self, **kwargs):
        self.provider_name = self.__class__.__name__.lower().replace('provider', '')
        self.provider_metadata = {}
        llm_type = kwargs.pop('model', BaseLLM)
        safety_settings = kwargs.pop('safety_settings', self.unsafe_settings())
        system_instruction = BasePrompt(kwargs.pop('system_instruction', None))
        self.model: BaseLLM = llm_type(
            safety_settings=safety_settings,
            system_instruction=system_instruction,
            **kwargs
        )

    @abstractmethod
    def unsafe_settings(self):
        """Return provider-specific full uncensored safety settings"""
        pass
        
    @abstractmethod
    def generate(self, prompt: BasePrompt, response_schema: Optional[Type[BaseModel]] = None) -> LLMResponse:
        """Generate response from the LLM"""
        pass

    def _is_valid_path_or_url(self, prompt_part: str) -> bool:
        """Check if prompt is a valid local path or accessible URL"""
        # Check if it's a local file path
        try:
            if Path(prompt_part.strip()).exists():
                return True
        except:
            pass
        
        # Check if it's a URL
        if prompt_part.strip().startswith(('http://', 'https://')):
            try:
                response = requests.head(prompt_part.strip(), timeout=5)
                return response.status_code == 200
            except:
                return False
        
        return False

    def _clean_json_content(self, content_str: str) -> str:
        """Clean JSON content from markdown formatting"""
        stripped_content_str = content_str.strip()

        try:  # obvious formatting fix
            content_str = stripped_content_str.lstrip('```json').rstrip('```').strip()
            json.loads(content_str)  # just for testing - actual loads later
        except json.JSONDecodeError:
            def extract_json_block(text):  # for weird cases like starting with ':\n```json'
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    return json_str  # no json loads - will later
                else:
                    return text  # just stripped choices[0].message.content
            content_str = extract_json_block(stripped_content_str)
        
        return content_str
    
    def _ocr_simple(self, image_path):
        """
        Simple OCR function with caching support.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Extracted text from the image
        """
        logger = setup_logger()

        logger.info(f"Starting text extraction for image '{image_path}'")

        try:
            if not os.path.exists(image_path):
                logger.error(f"Error: File '{image_path}' not found")
                return
            
            # Create cache directory if it doesn't exist
            cache_dir = Path('.ocr_cache')
            cache_dir.mkdir(exist_ok=True)
            
            # Generate cache key based on file path and modification time
            file_stat = os.stat(image_path)
            cache_key = f"{image_path}_{file_stat.st_mtime}_{file_stat.st_size}"
            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
            cache_file = cache_dir / f"{cache_hash}.json"
            
            # Try to load from cache
            if cache_file.exists():
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                        logger.info(f"Loaded OCR cache from '{cache_file}'")
                        return cached_data.get('text', '')
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"Cache file '{cache_file}' corrupted, continue with OCR. Error text: {e}")
            
            # Perform OCR
            text = pytesseract.image_to_string(Image.open(image_path))
            result = text.strip() if text.strip() else "No text found in image"
            logger.info(f"Performed OCR for '{image_path}'")
            
            # Save to cache
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump({'text': result}, f, ensure_ascii=False, indent=2, default=str)
                logger.info(f"OCR cache saved to '{cache_file}'")
            except Exception as e:
                logger.error(f"Cache save failed, but OCR succeeded for '{image_path}'. Error text: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider"""
    
    def __init__(self, **kwargs):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        self.client = genai.Client(api_key=api_key)
        kwargs.setdefault('model', GeminiLLM)
        super().__init__(**kwargs)

    def unsafe_settings(self):
        # https://ai.google.dev/api/generate-content#v1beta.HarmCategory
        return [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            )
        ]
        
    def generate(self, prompt: BasePrompt) -> LLMResponse:
        logger = setup_logger()
        try:                
            contents = []

            # Process each item in the prompt
            for item in prompt.parts():
                if self._is_valid_path_or_url(item):
                    uploaded_file = self._upload_file(item)
                    mime_type, _ = mimetypes.guess_type(item)
                    if mime_type and mime_type.startswith('image/'):
                        contents.append(f"File name: '{os.path.basename(item)}'. OCR'd text: ```{self._ocr_simple(item)}```. Image:")
                    # elif it might be a pdf going straight to provider
                    contents.append(uploaded_file)
                else:
                    contents.append(item)
            
            # Add schema instruction as separate item if response_schema is specified
            response_schema = self.model.response_schema
            
            # Basic Gemini API compliant config
            # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/content-generation-parameters
            config_dict = {
                'temperature': self.model.temperature,
                'top_k': self.model.top_k,
                'top_p': self.model.top_p,
                'seed': self.model.seed,
                'max_output_tokens': self.model.max_tokens,
                'safety_settings': self.model.safety_settings,
                'response_mime_type': 'application/json',
                'response_schema': response_schema
                #'response_json_schema': response_schema.model_json_schema()  # not implemented
            }

            if response_schema == BaseResponseSchema:  # not set
                config_dict.pop("response_schema", None)

            if len(self.model.system_instruction.parts()) > 0:  # add only if set - in case unsupported
                config_dict['system_instruction'] = str(self.model.system_instruction)

            if 'gemma' in self.model.model_name:  # unsupported
                config_dict.pop("response_schema", None)
                config_dict.pop("response_mime_type", None)

            request_dict = {
                "contents": contents,
                "model": self.model.model_name,
                "config": types.GenerateContentConfig(**config_dict)
            }

            request_timestamp = datetime.now()
            logger = setup_logger()
            logger.info(f"Contacting '{self.provider_name}' LLM provider API...")
            response: types.GenerateContentResponse = self.client.models.generate_content(**request_dict)
            raw_content = response.text

            # Clean up potential markdown formatting
            cleaned_content = self._clean_json_content(raw_content)
            
            try:
                content = response_schema.model_validate_json(cleaned_content)
            except Exception as e:  # gracefully replace
                content = response_schema.model_construct()
            return LLMResponse(
                raw_content=raw_content,
                content=content,
                token_usage={
                    "prompt_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
                    "total_tokens": response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
                } if hasattr(response, 'usage_metadata') else None,
                metadata={
                    "request_timestamp": request_timestamp.isoformat(),
                    "response_timestamp": datetime.now().isoformat(),
                    "response_raw": response.model_dump_json(),  # str so that there is no python code here when dumping to yaml later; ascii is not ensured by default although the arg is not implemented afaik - so we're good and this should save unicode
                    "request_raw": json.dumps(request_dict, ensure_ascii=False, default=str)  # may want to redact for length; using json str to make safe from any python objects
                }
            )
        except Exception as e:
            logger.error(f"Error occurred when calling Gemini API: {str(e)}")
            return LLMResponse.model_construct()

    def _upload_file(self, path_or_url: str):
        """Upload file to Gemini"""
        logger = setup_logger()
        if path_or_url.startswith(('http://', 'https://')):
            # For URLs, download first then upload
            response = requests.get(path_or_url)
            temp_file = f"/tmp/{Path(path_or_url).name}"
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            uploaded_file = self.client.files.upload(file=temp_file)
            os.remove(temp_file)
        else:
            # For local files
            uploaded_file = self.client.files.upload(file=path_or_url.strip())
        logger.info(f"Uploaded '{os.path.basename(path_or_url)}' to Gemini")
        return uploaded_file


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter provider, mimics OllamaProvider structure but uses OpenRouter API"""

    def __init__(self, **kwargs):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required for OpenRouterProvider.")
        
        self.api_url = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")

        self.provider_metadata = {
            "url": self.api_url
        }

        # Set default model type for this provider if not specified by user
        kwargs.setdefault('model', OpenRouterLLM)
        super().__init__(**kwargs)

    def unsafe_settings(self):
        """OpenRouter is a router; safety settings are model-specific and not set at provider level."""
        return None

    def generate(self, prompt: BasePrompt) -> LLMResponse:
        logger = setup_logger()
        try:
            headers_openrouter = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                # Per OpenRouter docs, these are recommended:
                # "HTTP-Referer": "YOUR_SITE_URL",
                # "X-Title": "YOUR_APP_NAME",
            }

            if len(self.model.system_instruction.parts()) > 0:  # if not empty string
                messages_openai = [{
                    "role": "system",
                    "content": str(self.model.system_instruction)
                }]
            else:
                messages_openai = []

            # Process each item in the prompt
            for item in prompt.parts():
                if self._is_valid_path_or_url(item):
                    image_url = self._to_image_url(item)
                    if image_url:
                        messages_openai.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"File name: '{os.path.basename(item)}'. OCR'd text: ```{self._ocr_simple(item)}```. Image:"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url
                                    }
                                }
                            ]
                        })
                    else:
                        messages_openai.append({
                            "role": "user",
                            "content": {
                                "type": "text",
                                "text": f"File '{os.path.basename(item)}' failed to upload."
                            }
                        })
                else:
                    messages_openai.append({
                        "role": "user",
                        "content": item
                    })

            options_openai = {  # https://openrouter.ai/docs/api-reference/chat-completion
                "temperature": self.model.temperature,
                "top_p": self.model.top_p,
                "top_k": self.model.top_k,
                "min_p": self.model.min_p,
                "max_tokens": self.model.max_tokens,
                "seed": self.model.seed
            }
            
            # Add schema instruction as separate item if response_schema is specified
            response_schema = self.model.response_schema
            
            payload_openrouter = {
                'provider': {
                    'require_parameters': False  # Only use providers that support all parameters in your request, see: https://openrouter.ai/docs/features/provider-routing
                },
                "model": self.model.model_name,
                "options": options_openai,
                "response_format": {
                    "type": "json_schema", # Corrected from "json_object" if schema is provided
                    "json_schema": response_schema.model_json_schema()
                },
                "messages": messages_openai
            }

            if ((response_schema == BaseResponseSchema) or  # not set
                (f'class {response_schema.__name__}(BaseModel):' in str(messages_openai))):  # unsupported
                payload_openrouter.pop("response_format", None)

            request_openrouter = {
                "url": self.api_url,
                "headers": headers_openrouter,
                "json": payload_openrouter
            }
            ### debug ###
            #with open('untracked/tmp_request_openrouter.json', 'w') as f:
            #    f.write(json.dumps(request_openrouter, indent=2, ensure_ascii=False, default=str))
            ### debug ###
            request_timestamp = datetime.now()
            logger.info(f"Contacting '{self.provider_name}' LLM provider API...")
            response_openrouter = requests.post(**request_openrouter)
            response_openrouter.raise_for_status()

            response = response_openrouter.json()  # dict with raw json response

            raw_content = response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            # Clean up potential markdown formatting
            cleaned_content = self._clean_json_content(raw_content)

            # Remove image data URLs from request for saving
            def redact_request_raw(obj):
                if isinstance(obj, dict):
                    if 'headers' in obj:
                        obj['headers'] = 'REDACTED for privacy'
                    # If this dict contains an image_url, replace its "url"
                    if "image_url" in obj and isinstance(obj["image_url"], dict):
                        obj["image_url"]["url"] = r"data:text/plain,REDACTED%20for%20length"
                    elif "text" in obj and isinstance(obj["text"], str) and len(str(obj["text"])) > 100:
                        obj["text"] = "REDACTED for length"
                    else:
                        # Otherwise, recursively check all values
                        for k, v in obj.items():
                            obj[k] = redact_request_raw(v)
                elif isinstance(obj, list):
                    return [redact_request_raw(item) for item in obj]
                else:
                    pass
                return obj

            redacted_request_openrouter = request_openrouter.copy()
            try:
                redacted_request_openrouter = redact_request_raw(redacted_request_openrouter)
            except:
                pass
            
            try:
                content = response_schema.model_validate_json(cleaned_content)
            except Exception as e:  # gracefully replace
                content = response_schema.model_construct()
            return LLMResponse(
                raw_content=raw_content,
                content=content,
                token_usage={
                    "prompt_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": response.get("usage", {}).get("completion_tokens", 0),
                    "total_tokens": response.get("usage", {}).get("total_tokens", 0),
                },
                metadata={
                    "request_timestamp": request_timestamp.isoformat(),
                    "response_timestamp": datetime.now().isoformat(),
                    "response_raw": json.dumps(response, ensure_ascii=False, default=str),  # str for consistency; ascii is not ensured by default although the arg is not implemented afaik - so we're good and this should save unicode
                    "request_raw": json.dumps(redacted_request_openrouter, ensure_ascii=False, default=str)  # using json str to make safe from any python objects
                }
            )
        except Exception as e:
            logger.error(f"Error occurred when calling OpenRouter API: {str(e)}")
            return LLMResponse.model_construct()

    def _to_image_url(self, path_or_url: str) -> str | None:
        """Obtain a valid image URL based on given path or URL."""
        prompt_part = None

        # For URLs, download first
        if path_or_url.startswith(('http://', 'https://')):
            response = requests.get(path_or_url)
            path = f"/tmp/{Path(path_or_url).name}"
            with open(path, 'wb') as f:
                f.write(response.content)
        else:
            path = path_or_url
        
        # For local files
        with open(path, 'rb') as f:
            file_data = f.read()
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type and mime_type.startswith('image/'):
            base64_data = base64.b64encode(file_data).decode('utf-8')
            data_url = f"data:{mime_type};base64,{base64_data}"
            # Replace the original prompt_part with the data URL
            prompt_part = data_url
        
        return prompt_part

class OllamaProvider(BaseLLMProvider):
    """Ollama local provider"""
    
    def __init__(self, **kwargs):
        host = os.getenv("OLLAMA_HOST")
        if not host:
            #raise ValueError("OLLAMA_HOST environment variable is required")
            host = 'http://localhost:11434'
        self.host = host
        
        self.provider_metadata = {
            "release": {
                "tag": "v0.9.2",
                "url": "https://github.com/ollama/ollama/releases/tag/v0.9.2",
                "assets": [
                    {
                        "filename": "ollama-darwin.tgz",
                        "sha256_hash": "834cbf10e21ee42ed553784bffd770c7db50a567fb95d58059adfc8ba0225919"
                    }
                ]
            }
        }

        self.model.model_metadata = {
            "modelfile": {
                "url": "https://ollama.com/library/gemma3:27b-it-qat",
                "sha256_hash": "29eb0b9aeda35295ed728124d341b27e0c6771ea5c586fcabfb157884224fa93"
            }
        }

        kwargs.setdefault('model', OllamaLLM)
        super().__init__(**kwargs)

    def unsafe_settings(self):  # not offered for Ollama
        pass
        
    def generate(self, prompt: BasePrompt) -> LLMResponse:
        logger = setup_logger()
        try:    # https://ollama.com/library/gemma3:27b-it-qat
                # Updated Apr 18, 2025 2:08 AM UTC
                # gemma3:27b-it-qat
                # 18GB
                # 128K
                # Text, Image
                # 29eb0b9aeda3

            if len(self.model.system_instruction.parts()) > 0:  # if not empty string
                messages_openai = [{
                    "role": "system",
                    "content": str(self.model.system_instruction)
                }]
            else:
                messages_openai = []

            # Process each item in the prompt
            for item in prompt.parts():
                if self._is_valid_path_or_url(item):
                    messages_openai.append({
                        "role": "user",
                        "content": "",
                        "images": [item]
                    })
                else:
                    messages_openai.append({
                        "role": "user",
                        "content": item
                    })

            options_openai = {  # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
                "num_ctx": self.model.num_ctx,
                "temperature": self.model.temperature,
                "top_p": self.model.top_p,
                "top_k": self.model.top_k,
                "min_p": self.model.min_p,
                "num_predict": self.model.max_tokens,
                "seed": self.model.seed
            }
            
            # Add schema instruction as separate item if response_schema is specified
            response_schema = self.model.response_schema
            
            payload_ollama = {
                "model": self.model.model_name,
                "options": options_openai,
                "messages": messages_openai,
                "format": response_schema.model_json_schema()  # https://ollama.com/blog/structured-outputs
            }

            if response_schema == BaseResponseSchema:  # not set
                payload_ollama.pop("format", None)
            
            # ollama -v
            # ollama version is 0.9.2
            # 834cbf10e21e  ./ollama-darwin.tgz

            request_timestamp = datetime.now()
            response: ollama.ChatResponse = ollama.chat(**payload_ollama)

            def postprocess(dump: dict) -> dict:
                def nanoseconds_to_human_readable(nanoseconds):
                    return round(nanoseconds / 1_000_000_000.0 / 60, 4)
                for key in list(dump.keys()):
                    if key.endswith('_duration'):
                        dump[f"{key}_minutes"] = nanoseconds_to_human_readable(dump[key])
                return dump
            
            #response = postprocess(response_raw)  # won't work well because postprocess returns dict but response is is ollama.ChatResponse
            
            prompt_eval_count = response.prompt_eval_count if hasattr(response, 'prompt_eval_count') else 0
            eval_count = response.eval_count if hasattr(response, 'eval_count') else 0
            try:
                content = response_schema.model_validate_json(response.message.content)
            except Exception as e:  # gracefully replace
                content = response_schema()
            return LLMResponse(
                raw_content=response.message.content,
                content=content,
                token_usage={
                    "prompt_tokens": prompt_eval_count,
                    "completion_tokens": eval_count,
                    "total_tokens": prompt_eval_count + eval_count,
                },
                metadata={
                    "request_timestamp": request_timestamp.isoformat(),
                    "response_timestamp": datetime.now().isoformat(),
                    "response_raw": response.model_dump_json(),    # str for consistency; ascii is not ensured by default although the arg is not implemented afaik - so we're good and this should save unicode
                    "request_raw": json.dumps(payload_ollama, ensure_ascii=False, default=str)  # using json str to make safe from any python objects
                }
            )
        except Exception as e:
            logger.error(f"Error occurred when calling Ollama API: {str(e)}")
            return LLMResponse.model_construct()

    def _is_valid_path_or_url(self, prompt_part: str) -> bool:
        """Check if prompt is a valid local path or accessible URL"""
        # Check if it's a local file path
        try:
            if Path(prompt_part.strip()).exists():
                return True
        except:
            pass
        
        # Check if it's a URL
        if prompt_part.strip().startswith(('http://', 'https://')):
            try:
                response = requests.head(prompt_part.strip(), timeout=5)
                return response.status_code == 200
            except:
                return False
        
        return False


class LLMAgent:
    """Main agent class for LLM interactions"""
    
    def __init__(self, provider_name: str, **kwargs):
        self.logger = setup_logger()
        
        # Provider mapping
        providers: dict[str, BaseLLMProvider] = {
            "gemini": GeminiProvider,
            "openrouter": OpenRouterProvider,
            "ollama": OllamaProvider
        }
        
        if provider_name not in providers:
            raise ValueError(f"Unsupported provider: {provider_name}. Supported: {list(providers.keys())}")
            
        provider_class = providers[provider_name]
        self.provider: BaseLLMProvider = provider_class(**kwargs)
        
    def __call__(self, prompt: Union[str, List[str]], save_response: bool = True) -> LLMResponse:
        """Generate response from LLM"""
        try:
            hashed_prompt = BasePrompt(prompt)

            # Generate response
            response = self.provider.generate(hashed_prompt)

            json_response = response.content.model_dump_json()
            
            # Log the interaction
            log_data = {
                "provider": {
                    "provider_name": self.provider.provider_name,
                    "safety_settings": str(self.provider.model.safety_settings),
                    "provider_metadata": self.provider.provider_metadata
                },
                "model": {
                    "model_name": self.provider.model.model_name,
                    "config": {
                        "temperature": self.provider.model.temperature,
                        "top_k": self.provider.model.top_k,
                        "top_p": self.provider.model.top_p,
                        "seed": self.provider.model.seed,
                    },
                    "model_metadata": self.provider.model.model_metadata,
                },
                "request": {
                    "prompt_hash": hashed_prompt.hash,
                    "system_instruction_hash": self.provider.model.system_instruction.hash,
                    "response_schema": self.provider.model.response_schema.__name__,  # perhaps to be replaced with a hash later on
                },
                "response": {
                    "raw_content": response.raw_content[:200] + "..." if response.raw_content else None,
                    "content": json_response[:200] + "..." if len(json_response) > 200 else json_response,
                    "token_usage": response.token_usage,
                    "metadata": response.metadata,
                }
            }
            self.logger.info(f"LLM Response: {json.dumps(log_data, indent=2, ensure_ascii=False, default=str)}")
            
            # Save response if requested
            if save_response:
                self._save_response(response, hashed_prompt)
                
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return LLMResponse.model_construct()
        
    def _save_response(self, response: LLMResponse, prompt: BasePrompt):
        """Save response as YAML file"""
        responses_dir = Path("responses")
        responses_dir.mkdir(exist_ok=True)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        response_file = responses_dir / f"{timestamp_str}_{prompt.hash}_{self.provider.provider_name}.yml"
        
        response_data = {
            "provider": {
                "provider_name": self.provider.provider_name,
                "safety_settings": str(self.provider.model.safety_settings),
                "provider_metadata": self.provider.provider_metadata
            },
            "model": {
                "model_name": self.provider.model.model_name,
                "config": {
                    "temperature": self.provider.model.temperature,
                    "top_k": self.provider.model.top_k,
                    "top_p": self.provider.model.top_p,
                    "seed": self.provider.model.seed,
                },
                "model_metadata": self.provider.model.model_metadata,
            },
            "request": {
                "prompt_hash": prompt.hash,
                "prompt_file": f"prompts/{prompt.hash}.txt",
                "system_instruction_hash": self.provider.model.system_instruction.hash,
                "system_instruction_file": f"prompts/{hash}.txt" if (hash := self.provider.model.system_instruction.hash) else None,
                "response_schema": self.provider.model.response_schema.model_json_schema()
            },
            "response": {
                "raw_content": response.raw_content,
                "content": response.content.model_dump(mode='json'),  # to json - dict is hard to load for evals!
                "token_usage": response.token_usage,
                "metadata": response.metadata,
            },
            "evals": {},
            "manual_evals": {},
            "ground_truth": None,
            "name": None
        }
        
        with open(response_file, 'w') as f:
            # Custom representer for multiline strings
            def literal_presenter(dumper, data):
                if '\n' in data:
                    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
                return dumper.represent_scalar('tag:yaml.org,2002:str', data)
            # Add the custom representer
            yaml.add_representer(str, literal_presenter)
            yaml.dump(response_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True, indent=2)


def agent(prompt: Union[str, List[str]], response_schema: Type[BaseModel] = BaseResponseSchema, provider_name: str = "gemini", **kwargs) -> LLMResponse:
    """Factory function to call LLM agent"""
    llm_agent = LLMAgent(provider_name=provider_name, response_schema=response_schema, **kwargs)
    return llm_agent(prompt, save_response = True)

__all__ = ["agent", "BaseModel", "Field", "Dict", "Any", "Optional", "Type", "Union", "Literal", "List", "Optional", "Enum", "setup_logger"]
```
