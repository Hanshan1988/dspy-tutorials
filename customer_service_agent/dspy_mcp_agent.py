from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import dspy
from langfuse import get_client
from openinference.instrumentation.dspy import DSPyInstrumentor

import os
import base64
from dotenv import load_dotenv
load_dotenv()  # Load environment variables

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Warning: OPENAI_API_KEY not found in environment variables. Agent may fail to authenticate or hit rate limits.")
api_base = os.getenv("OPENAI_API_BASE", "https://api.huggingface.co")
# model_id = "openai/gpt-4o-mini"
model_id = "openai/openai/gpt-oss-120b:novita" # litellm model needs to have extra openai/ prefix

# --- Langfuse + OTLP exporter config ---

# os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-...")
# os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-...")
# os.environ["LANGFUSE_BASE_URL"] = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")

LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST") 
PK = os.environ["LANGFUSE_PUBLIC_KEY"]
SK = os.environ["LANGFUSE_SECRET_KEY"]

# # OTLP endpoint + auth header for Langfuse's OTel API
# os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{LANGFUSE_HOST}/api/public/otel"
# auth = base64.b64encode(f"{PK}:{SK}".encode()).decode()
# os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth}"

langfuse = get_client()
 
# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")
# Enable tracing for DSPy
DSPyInstrumentor().instrument()

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",  # Executable
    args=["airline_mcp_server.py"],  # Optional command line arguments
    env=None,  # Optional environment variables
)


class DSPyAirlineCustomerService(dspy.Signature):
    """You are an airline customer service agent. You are given a list of tools to handle user requests.
    You should decide the right tool to use in order to fulfill users' requests."""

    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
            "Message that summarizes the process result, and the information users need, "
            "e.g., the confirmation_number if it's a flight booking request."
        )
    )

# Configure DSPy with the desired LM
lm = dspy.LM(model_id, api_key=api_key, api_base=api_base)
dspy.configure(
    lm=lm,
    track_usage=True,
)

async def run(user_request):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            # List available tools
            tools = await session.list_tools()

            # Convert MCP tools to DSPy tools
            dspy_tools = []
            for tool in tools.tools:
                dspy_tools.append(dspy.Tool.from_mcp_tool(session, tool))

            # Create the agent
            react = dspy.ReAct(DSPyAirlineCustomerService, tools=dspy_tools)

            result = await react.acall(user_request=user_request)
            print(result)


if __name__ == "__main__":
    import asyncio
    asyncio.run(run("Please help me book a flight from SFO to JFK on 09/01/2025, my name is Adam"))
    get_client().flush()
