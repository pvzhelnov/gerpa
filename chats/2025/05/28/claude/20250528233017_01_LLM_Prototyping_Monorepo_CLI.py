#!/usr/bin/env python3
"""
LLM Prototyping Monorepo CLI
A minimal CLI for fast LLM prototyping with versioned prompts, responses, and evaluations.
"""

import os
import sys
import json
import yaml
import click
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import hashlib

# Template files content
GITIGNORE_TEMPLATE = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# poetry
poetry.lock

# pdm
.pdm.toml

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
.idea/

# macOS
.DS_Store
.AppleDouble
.LSOverride
Icon
._*
.DocumentRevisions-V100
.fseventsd
.Spotlight-V100
.TemporaryItems
.Trashes
.VolumeIcon.icns
.com.apple.timemachine.donotpresent
.AppleDB
.AppleDesktop
Network Trash Folder
Temporary Items
.apdisk

# Project specific
logs/
responses/
*.env
"""

CONDA_ENV_TEMPLATE = """name: llm-proto
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11.9
  - pip=24.0
  - pip:
    - click==8.1.7
    - pydantic==2.7.1
    - python-dotenv==1.0.1
    - pyyaml==6.0.1
    - requests==2.31.0
    - openai==1.30.1
    - google-generativeai==0.5.4
    - ollama==0.2.1
    - pandas==2.2.2
    - numpy==1.26.4
    - matplotlib==3.8.4
    - seaborn==0.13.2
    - scikit-learn==1.4.2
    - jupyter==1.0.0
    - ipykernel==6.29.4
"""

LLMPROVIDER_CODE = '''"""
LLM Provider SDK - Unified interface for multiple LLM providers
"""

import os
import json
import yaml
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Type, Union
from abc import ABC, abstractmethod
from pydantic import BaseModel

import requests
import google.generativeai as genai
import ollama
from openai import OpenAI


class LLMResponse(BaseModel):
    """Standard response format for all LLM providers"""
    content: str
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
    def generate(self, prompt: str, response_schema: Optional[Type[BaseModel]] = None) -> LLMResponse:
        """Generate response from the LLM"""
        pass


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider"""
    
    def __init__(self, model: str = "gemini-1.5-flash", **kwargs):
        super().__init__(model, **kwargs)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)
        
    def generate(self, prompt: str, response_schema: Optional[Type[BaseModel]] = None) -> LLMResponse:
        try:
            if response_schema:
                # Add JSON schema instruction to prompt
                schema_instruction = f"\\nRespond with valid JSON matching this schema: {response_schema.model_json_schema()}"
                prompt = prompt + schema_instruction
                
            response = self.client.generate_content(prompt)
            
            return LLMResponse(
                content=response.text,
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


class OpenRouterProvider(BaseLLMProvider):
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
                schema_instruction = f"\\nRespond with valid JSON matching this schema: {response_schema.model_json_schema()}"
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


class OllamaProvider(BaseLLMProvider):
    """Ollama local provider"""
    
    def __init__(self, model: str = "llama3.1", host: str = "http://localhost:11434", **kwargs):
        super().__init__(model, **kwargs)
        self.host = host
        
    def generate(self, prompt: str, response_schema: Optional[Type[BaseModel]] = None) -> LLMResponse:
        try:
            if response_schema:
                # Add JSON schema instruction
                schema_instruction = f"\\nRespond with valid JSON matching this schema: {response_schema.model_json_schema()}"
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
        
    def __call__(self, prompt: str, save_response: bool = True) -> LLMResponse:
        """Generate response from LLM"""
        try:
            # Version the prompt
            prompt_hash = self._version_prompt(prompt)
            
            # Generate response
            response = self.provider.generate(prompt, self.response_schema)
            
            # Log the interaction
            log_data = {
                "prompt_hash": prompt_hash,
                "response_content": response.content[:200] + "..." if len(response.content) > 200 else response.content,
                "model": response.model,
                "provider": response.provider,
                "token_usage": response.token_usage,
                "timestamp": response.timestamp.isoformat()
            }
            self.logger.info(f"LLM Response: {json.dumps(log_data, indent=2)}")
            
            # Save response if requested
            if save_response:
                self._save_response(response, prompt_hash)
                
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise
            
    def _version_prompt(self, prompt: str) -> str:
        """Version a prompt and save it"""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        
        prompts_dir = Path("prompts")
        prompts_dir.mkdir(exist_ok=True)
        
        prompt_file = prompts_dir / f"{prompt_hash}.txt"
        if not prompt_file.exists():
            prompt_file.write_text(prompt)
            
        return prompt_hash
        
    def _save_response(self, response: LLMResponse, prompt_hash: str):
        """Save response as YAML file"""
        responses_dir = Path("responses")
        responses_dir.mkdir(exist_ok=True)
        
        timestamp_str = response.timestamp.strftime("%Y%m%d_%H%M%S")
        response_file = responses_dir / f"{timestamp_str}_{prompt_hash}_{response.provider}.yaml"
        
        response_data = {
            "content": response.content,
            "model": response.model,
            "provider": response.provider,
            "timestamp": response.timestamp.isoformat(),
            "token_usage": response.token_usage,
            "metadata": response.metadata,
            "prompt_hash": prompt_hash,
            "prompt_file": f"prompts/{prompt_hash}.txt",
            "response_schema": self.response_schema.__name__ if self.response_schema else None,
            "evals": {},
            "ground_truth": None,
            "name": None
        }
        
        with open(response_file, 'w') as f:
            yaml.dump(response_data, f, default_flow_style=False, sort_keys=False)


def agent(provider: str, response_schema: Optional[Type[BaseModel]] = None, **kwargs) -> LLMAgent:
    """Factory function to create LLM agent"""
    return LLMAgent(provider=provider, response_schema=response_schema, **kwargs)
'''

NOTEBOOK_CODE = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# LLM Prototyping Notebook\\n",
    "\\n",
    "Quick experiments with LLM providers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load environment and imports\\n",
    "from dotenv import load_dotenv\\n",
    "load_dotenv()\\n",
    "\\n",
    "import sys\\n",
    "sys.path.append('.')\\n",
    "\\n",
    "from llm_provider import agent\\n",
    "from pydantic import BaseModel\\n",
    "import logging\\n",
    "\\n",
    "# Setup logging\\n",
    "logging.basicConfig(level=logging.INFO)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define response schema (optional)\\n",
    "class SomeResponseSchema(BaseModel):\\n",
    "    summary: str\\n",
    "    key_points: list[str]\\n",
    "    confidence: float"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create agent\\n",
    "LLMPROVIDER = \\"gemini\\"  # or \\"openrouter\\", \\"ollama\\"\\n",
    "llm_agent = agent(LLMPROVIDER, SomeResponseSchema)\\n",
    "\\n",
    "# Test prompt\\n",
    "prompt = \\"Explain quantum computing in simple terms\\"\\n",
    "response = llm_agent(prompt)\\n",
    "\\n",
    "print(f\\"Content: {response.content}\\")\\n",
    "print(f\\"Model: {response.model}\\")\\n",
    "print(f\\"Provider: {response.provider}\\")\\n",
    "print(f\\"Token usage: {response.token_usage}\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Experiment with different providers\\n",
    "providers = [\\"gemini\\", \\"openrouter\\", \\"ollama\\"]\\n",
    "\\n",
    "for provider in providers:\\n",
    "    try:\\n",
    "        print(f\\"\\\\n--- Testing {provider} ---\\")\\n",
    "        test_agent = agent(provider)\\n",
    "        response = test_agent(\\"What is the capital of France?\\")\\n",
    "        print(f\\"Response: {response.content[:100]}...\\")\\n",
    "    except Exception as e:\\n",
    "        print(f\\"Error with {provider}: {e}\\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''

ENV_TEMPLATE = """# LLM API Keys
GOOGLE_API_KEY=your_google_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434

# Other settings
LOG_LEVEL=INFO
"""

EVALUATOR_CODE = '''"""
Evaluation system for LLM responses
"""

import os
import yaml
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import logging
from sklearn.metrics import confusion_matrix, classification_report
import hashlib


class EvalResult:
    """Container for evaluation results"""
    def __init__(self, test_name: str, result: str, expected: Any = None, actual: Any = None, 
                 score: float = None, metadata: Dict = None):
        self.test_name = test_name
        self.result = result  # "pass", "fail", "skip"
        self.expected = expected
        self.actual = actual
        self.score = score
        self.metadata = metadata or {}
        self.timestamp = datetime.now()


class BaseEvaluator:
    """Base class for all evaluators"""
    
    def __init__(self, name: str):
        self.name = name
        
    def evaluate(self, response_content: str, ground_truth: Any = None, **kwargs) -> EvalResult:
        """Evaluate a response"""
        raise NotImplementedError


class ContainsEvaluator(BaseEvaluator):
    """Check if response contains specific text"""
    
    def __init__(self, text: str, case_sensitive: bool = False):
        super().__init__(f"CONTAINS_{text.upper()}")
        self.text = text
        self.case_sensitive = case_sensitive
        
    def evaluate(self, response_content: str, ground_truth: Any = None, **kwargs) -> EvalResult:
        content_check = response_content if self.case_sensitive else response_content.lower()
        text_check = self.text if self.case_sensitive else self.text.lower()
        
        contains = text_check in content_check
        return EvalResult(
            test_name=self.name,
            result="pass" if contains else "fail",
            expected=f"Contains '{self.text}'",
            actual=f"Contains: {contains}",
            score=1.0 if contains else 0.0
        )


class NotContainsEvaluator(BaseEvaluator):
    """Check if response does NOT contain specific text"""
    
    def __init__(self, text: str, case_sensitive: bool = False):
        super().__init__(f"NOT_CONTAINS_{text.upper()}")
        self.text = text
        self.case_sensitive = case_sensitive
        
    def evaluate(self, response_content: str, ground_truth: Any = None, **kwargs) -> EvalResult:
        content_check = response_content if self.case_sensitive else response_content.lower()
        text_check = self.text if self.case_sensitive else self.text.lower()
        
        not_contains = text_check not in content_check
        return EvalResult(
            test_name=self.name,
            result="pass" if not_contains else "fail",
            expected=f"Does not contain '{self.text}'",
            actual=f"Contains: {not not_contains}",
            score=1.0 if not_contains else 0.0
        )


class LengthEvaluator(BaseEvaluator):
    """Check response length constraints"""
    
    def __init__(self, min_length: int = None, max_length: int = None):
        super().__init__(f"LENGTH_{min_length or 0}_TO_{max_length or 'INF'}")
        self.min_length = min_length
        self.max_length = max_length
        
    def evaluate(self, response_content: str, ground_truth: Any = None, **kwargs) -> EvalResult:
        length = len(response_content)
        
        passes = True
        reasons = []
        
        if self.min_length and length < self.min_length:
            passes = False
            reasons.append(f"Too short: {length} < {self.min_length}")
            
        if self.max_length and length > self.max_length:
            passes = False
            reasons.append(f"Too long: {length} > {self.max_length}")
            
        return EvalResult(
            test_name=self.name,
            result="pass" if passes else "fail",
            expected=f"Length between {self.min_length or 0} and {self.max_length or 'inf'}",
            actual=f"Length: {length}",
            score=1.0 if passes else 0.0,
            metadata={"reasons": reasons}
        )


class GroundTruthEvaluator(BaseEvaluator):
    """Compare response against ground truth"""
    
    def __init__(self, comparison_type: str = "exact"):
        super().__init__(f"GROUND_TRUTH_{comparison_type.upper()}")
        self.comparison_type = comparison_type
        
    def evaluate(self, response_content: str, ground_truth: Any = None, **kwargs) -> EvalResult:
        if ground_truth is None:
            return EvalResult(
                test_name=self.name,
                result="skip",
                expected="Ground truth available",
                actual="No ground truth provided",
                score=None
            )
            
        if self.comparison_type == "exact":
            matches = str(response_content).strip() == str(ground_truth).strip()
            score = 1.0 if matches else 0.0
        elif self.comparison_type == "contains":
            matches = str(ground_truth).lower() in str(response_content).lower()
            score = 1.0 if matches else 0.0
        else:
            # Semantic similarity (placeholder - could implement with embeddings)
            matches = False
            score = 0.0
            
        return EvalResult(
            test_name=self.name,
            result="pass" if matches else "fail",
            expected=str(ground_truth),
            actual=str(response_content)[:200] + "..." if len(str(response_content)) > 200 else str(response_content),
            score=score
        )


class EvaluationRunner:
    """Main evaluation runner"""
    
    def __init__(self):
        self.evaluators = {}
        self.logger = logging.getLogger(__name__)
        self._register_default_evaluators()
        
    def _register_default_evaluators(self):
        """Register commonly used evaluators"""
        # Common negative checks
        self.register_evaluator("HAS_NO_SHIT_IN_RESPONSE", NotContainsEvaluator("shit"))
        self.register_evaluator("HAS_NO_FUCK_IN_RESPONSE", NotContainsEvaluator("fuck"))
        self.register_evaluator("HAS_NO_DAMN_IN_RESPONSE", NotContainsEvaluator("damn"))
        
        # Length checks
        self.register_evaluator("REASONABLE_LENGTH", LengthEvaluator(min_length=10, max_length=5000))
        self.register_evaluator("SHORT_RESPONSE", LengthEvaluator(max_length=500))
        self.register_evaluator("LONG_RESPONSE", LengthEvaluator(min_length=1000))
        
        # Ground truth
        self.register_evaluator("GROUND_TRUTH_EXACT", GroundTruthEvaluator("exact"))
        self.register_evaluator("GROUND_TRUTH_CONTAINS", GroundTruthEvaluator("contains"))
        
    def register_evaluator(self, name: str, evaluator: BaseEvaluator):
        """Register a custom evaluator"""
        self.evaluators[name] = evaluator
        
    def run_evals(self, responses_dir: str = "responses", output_dir: str = "eval_results") -> Dict[str, Any]:
        """Run evaluations on all response files"""
        responses_path = Path(responses_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if not responses_path.exists():
            self.logger.warning(f"Responses directory {responses_path} does not exist")
            return {}
            
        # Load all response files
        response_files = list(responses_path.glob("*.yaml"))
        if not response_files:
            self.logger.warning(f"No YAML files found in {responses_path}")
            return {}
            
        all_results = []
        
        for response_file in response_files:
            try:
                with open(response_file, 'r') as f:
                    response_data = yaml.safe_load(f)
                    
                # Run evaluations
                file_results = self._evaluate_response(response_data, response_file.name)
                all_results.extend(file_results)
                
            except Exception as e:
                self.logger.error(f"Error processing {response_file}: {e}")
                continue
                
        # Aggregate results
        aggregated = self._aggregate_results(all_results)
        
        # Save results
        self._save_results(aggregated, output_path)
        
        # Generate visualizations
        self._generate_visualizations(aggregated, output_path)
        
        return aggregated
        
    def _evaluate_response(self, response_data: Dict, filename: str) -> List[Dict]:
        """Evaluate a single response"""
        results = []
        
        content = response_data.get('content', '')
        evals = response_data.get('evals', {})
        ground_truth = response_data.get('ground_truth')
        name = response_data.get('name', filename)
        
        # Run specified evaluations
        for eval_name, expected_result in evals.items():
            if eval_name in self.evaluators:
                evaluator = self.evaluators[eval_name]
                result = evaluator.evaluate(content, ground_truth)
                
                results.append({
                    'response_file': filename,
                    'response_name': name,
                    'provider': response_data.get('provider'),
                    'model': response_data.get('model'),
                    'timestamp': response_data.get('timestamp'),
                    'eval_name': eval_name,
                    'expected': expected_result,
                    'actual_result': result.result,
                    'matches_expected': (result.result == expected_result),
                    'score': result.score,
                    'eval_expected': result.expected,
                    'eval_actual': result.actual,
                    'metadata': result.metadata
                })
            else:
                self.logger.warning(f"Unknown evaluator: {eval_name}")
                
        return results
        
    def _aggregate_results(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate evaluation results"""
        if not all_results:
            return {}
            
        df = pd.DataFrame(all_results)
        
        # Overall metrics
        total_evals = len(df)
        total_passed = len(df[df['matches_expected'] == True])
        total_failed = len(df[df['matches_expected'] == False])
        pass_rate = total_passed / total_evals if total_evals > 0 else 0
        
        # By evaluator
        by_evaluator = df.groupby
