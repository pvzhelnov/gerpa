from llm_provider import agent
llm_agent = agent("gemini", ResponseSchema)
response = llm_agent("your prompt")