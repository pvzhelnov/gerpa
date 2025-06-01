```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# LLM Prototyping Notebook\n",
    "\n",
    "Quick experiments with LLM providers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load environment and imports\n",
    "from dotenv import load_dotenv\n",
    "load_dotenv()\n",
    "\n",
    "import sys\n",
    "sys.path.append('.')\n",
    "\n",
    "from modules.llm_provider import agent\n",
    "from pydantic import BaseModel\n",
    "import logging\n",
    "\n",
    "# Setup logging\n",
    "logging.basicConfig(level=logging.INFO)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define response schema (optional)\n",
    "class SomeResponseSchema(BaseModel):\n",
    "    summary: str\n",
    "    key_points: list[str]\n",
    "    confidence: float"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create agent\n",
    "LLMPROVIDER = \"gemini\"  # or \"openrouter\", \"ollama\"\n",
    "llm_agent = agent(LLMPROVIDER, SomeResponseSchema)\n",
    "\n",
    "# Test prompt\n",
    "prompt = \"Explain quantum computing in simple terms\"\n",
    "response = llm_agent(prompt)\n",
    "\n",
    "print(f\"Content: {response.content}\")\n",
    "print(f\"Model: {response.model}\")\n",
    "print(f\"Provider: {response.provider}\")\n",
    "print(f\"Token usage: {response.token_usage}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Experiment with different providers\n",
    "providers = [\"gemini\", \"openrouter\", \"ollama\"]\n",
    "\n",
    "for provider in providers:\n",
    "    try:\n",
    "        print(f\"\\n--- Testing {provider} ---\")\n",
    "        test_agent = agent(provider)\n",
    "        response = test_agent(\"What is the capital of France?\")\n",
    "        print(f\"Response: {response.content[:100]}...\")\n",
    "    except Exception as e:\n",
    "        print(f\"Error with {provider}: {e}\")"
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
}
```
