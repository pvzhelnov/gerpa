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
    "# Environment variables are loaded from .env by llm_provider\n",
    "from modules.llm_provider import *  # imports: ['agent', 'BaseModel', 'Field', 'Any', 'Dict', 'Optional', 'Type', 'Union', 'Literal', 'List', 'Optional', 'Union', 'Enum']\n",
    "# Define response schema. Uncommment lines if model doesn't support response format\n",
    "response_schema = [\"Respond with valid JSON object matching this Python class:\\n```python\", \"\"\"\n",
    "                   \n",
    "class GenericResponseSchema(BaseModel):\n",
    "    generic_response: str = Field(..., description=\"Generic response.\")\n",
    "\n",
    "\"\"\", \"```\"]; exec(response_schema[1])  # careful - this executes this code!\n",
    "# Test prompt\n",
    "system_instruction = \"You are a helpful assistant.\"  # optional\n",
    "prompt = [\n",
    "    system_instruction,  # if unsupported as argument\n",
    "    \"Explain quantum computing in simple terms\",\n",
    "    \"\".join(response_schema)  # and pass it directly in prompt - if not supported\n",
    "]\n",
    "# Run agent\n",
    "response = agent(\n",
    "    prompt,\n",
    "    GenericResponseSchema,\n",
    "    provider_name = \"gemini\",  # or \"openrouter\", \"ollama\"\n",
    "    model_name=\"gemma-3-27b-it\",  # doesn't support response schema nor system instruction in this API\n",
    "    #system_instruction=system_instruction,\n",
    "    temperature=0.0\n",
    ")\n",
    "# This will print the full request/response log."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Print only the response schema model filled in\n",
    "import json; from IPython.display import Markdown\n",
    "Markdown('```json\\n' + json.dumps(response.content.model_dump(), indent=2, ensure_ascii=False) + '\\n```')"
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
