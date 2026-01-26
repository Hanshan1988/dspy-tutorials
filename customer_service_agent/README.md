# dspy Customer Service Agent

Basic implemenatation of customer service agent with airline mcp server. 
* MCP server with `fastmcp`
* AI agent with `dspy` -> agent loop
    * Remote HF Inference Endpoint through LiteLLM as the Language Model 
    * Asynchronous LM calls
* Observability with `langfuse`

The current query is "Please help me book a flight from SFO to JFK on 09/01/2025, my name is Adam".
Run example with `python dspy_mcp_agent.py`.