# Handling Local Files and URLs in Gemini API

*Exported on: 5/29/2025, 10:21:34 AM*

## User

modify the below to be able to recognize that if the prompt is a path that exists locally (or an accessible url over the web), then use google genai functionality for working with files, else process like text as before. ONLY PRINT the lines that need to be edited/added for this to work, dont rewrite everything!class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider"""    def init(self, model: str = "gemini-1.5-flash", kwargs):
        super().init(model, kwargs)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        self.client = genai.Client(api_key=api_key)    def generate(self, prompt: str, response_schema: Optional[Type[BaseModel]] = None) -> LLMResponse:
        try:
            if response_schema:
                # Add JSON schema instruction to prompt
                schema_instruction = f"\nRespond with valid JSON matching this schema: {response_schema.model_json_schema()}"
                prompt = prompt + schema_instruction            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
                config=GenerateContentConfig(
                    temperature=0.8,
                    response_mime_type='application/json',
                    response_schema=response_schema
                )
            )            return LLMResponse(
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

## Claude

Here are the lines that need to be added/edited:

```python
import os
from pathlib import Path
import requests

def _is_valid_path_or_url(self, prompt: str) -> bool:
    """Check if prompt is a valid local path or accessible URL"""
    # Check if it's a local file path
    if Path(prompt.strip()).exists():
        return True
    
    # Check if it's a URL
    if prompt.strip().startswith(('http://', 'https://')):
        try:
            response = requests.head(prompt.strip(), timeout=5)
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
        uploaded_file = genai.upload_file(temp_file)
        os.remove(temp_file)
        return uploaded_file
    else:
        # For local files
        return genai.upload_file(path_or_url.strip())

def generate(self, prompt: str, response_schema: Optional[Type[BaseModel]] = None) -> LLMResponse:
    try:
        # Check if prompt is a file path or URL
        if self._is_valid_path_or_url(prompt):
            uploaded_file = self._upload_file(prompt)
            contents = [uploaded_file]
        else:
            contents = [prompt]
            
        if response_schema:
            # Add JSON schema instruction to prompt for text-only requests
            if not self._is_valid_path_or_url(prompt):
                schema_instruction = f"\nRespond with valid JSON matching this schema: {response_schema.model_json_schema()}"
                contents = [prompt + schema_instruction]
```

## User

now implement handling of cases when prompt can also be a list of strings, and in that case accordingly upload all files and add them to prompt or pass as text. note that user is allowed to pass a mix of plain text and links so all items on list must be reviewed independently. ONLY PRINT lines that are changed/added, don't rewrite anything

## Claude

Here are the lines that need to be added/changed:

```python
def generate(self, prompt: Union[str, List[str]], response_schema: Optional[Type[BaseModel]] = None) -> LLMResponse:
    try:
        # Handle both string and list inputs
        if isinstance(prompt, str):
            prompt_items = [prompt]
        else:
            prompt_items = prompt
            
        contents = []
        text_parts = []
        
        # Process each item in the prompt
        for item in prompt_items:
            if self._is_valid_path_or_url(item):
                uploaded_file = self._upload_file(item)
                contents.append(uploaded_file)
            else:
                text_parts.append(item)
        
        # Add combined text parts if any
        if text_parts:
            combined_text = "\n".join(text_parts)
            if response_schema:
                schema_instruction = f"\nRespond with valid JSON matching this schema: {response_schema.model_json_schema()}"
                combined_text += schema_instruction
            contents.append(combined_text)
```

And add this import at the top:

```python
from typing import Union, List
```

## User

nope, each text part must go as its own part, egagent.generate(prompt: ['here is file for you:', 'http://example.com/file.pdf', 'please process it')ONLY print lines that changed after this edit, dont rewrite anything

## Claude

Here are the lines that need to be changed:

```python
        # Process each item in the prompt
        for item in prompt_items:
            if self._is_valid_path_or_url(item):
                uploaded_file = self._upload_file(item)
                contents.append(uploaded_file)
            else:
                contents.append(item)
        
        # Add schema instruction to last text item if response_schema is specified
        if response_schema and contents and isinstance(contents[-1], str):
            schema_instruction = f"\nRespond with valid JSON matching this schema: {response_schema.model_json_schema()}"
            contents[-1] += schema_instruction
        elif response_schema and contents and not isinstance(contents[-1], str):
            # If last item is a file, add schema as separate text part
            schema_instruction = f"Respond with valid JSON matching this schema: {response_schema.model_json_schema()}"
            contents.append(schema_instruction)
```

## User

no, schema instruction must not "# Add schema instruction to last text item if response_schema is specified" but instead it should be added as the latest part item on the list. ONLY PRINT the lines that changed

## Claude

```python
       # Add schema instruction as separate item if response_schema is specified
       if response_schema:
           schema_instruction = f"Respond with valid JSON matching this schema: {response_schema.model_json_schema()}"
           contents.append(schema_instruction)
```

## User

here is updated docs for using files with new google-genai sdk. review it and ONLY PRINT the lines that i must change.

**Pasted:** `pasted.md`

## Claude

Looking at the updated documentation, here are the lines that need to be changed:

```python
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
```

The key change is replacing genai.upload_file() with self.client.files.upload(file=...) to match the new SDK structure.

`genai.upload_file()``self.client.files.upload(file=...)`

