import dspy

from langfuse import get_client
from openinference.instrumentation.dspy import DSPyInstrumentor

import os
import base64
import pprint
from typing import Literal
from dotenv import load_dotenv

load_dotenv('/Users/hanshan/Downloads/.env')  # Load environment variables

api_key = os.getenv("HF_TOKEN")
if not api_key:
    print("Warning: HF_TOKEN not found in environment variables. Agent may fail to authenticate or hit rate limits.")
api_base = os.getenv("HF_API_BASE_URL", "https://api.huggingface.co")
# model_id = "openai/gpt-4o-mini"
model_id = "openai/openai/gpt-oss-120b:novita"

# # --- Langfuse + OTLP exporter config ---

os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_BASE_URL"] = os.getenv("LANGFUSE_HOST")

# # LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST") 
# # PK = os.environ["LANGFUSE_PUBLIC_KEY"]
# # SK = os.environ["LANGFUSE_SECRET_KEY"]

# # # OTLP endpoint + auth header for Langfuse's OTel API
# # os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{LANGFUSE_HOST}/api/public/otel"
# # auth = base64.b64encode(f"{PK}:{SK}".encode()).decode()
# # os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth}"

langfuse = get_client()
 
# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")
# Enable tracing for DSPy
DSPyInstrumentor().instrument()

# Configure DSPy with the desired LM
lm = dspy.LM(model_id, api_key=api_key, api_base=api_base)
dspy.configure(lm=lm, track_usage=True)

# Section 1: Basic DSPy usage

question = "Who is best between Google, Anthropic, and OpenAI? Provide a one-word answer."

class Answer(dspy.Signature):
    question: str = dspy.InputField()
    answer: Literal["Google", "Anthropic", "OpenAI"] = dspy.OutputField()

with dspy.context(lm=lm):
    answer = dspy.Predict(Answer)
    # Use a fixed-width format for aligned columns
    print(f"{model_id} ---> {answer(question=question).answer}")

# Section 2: Custom DSPy Signature with Instructions

SentimentClassifier = dspy.Signature("text: str -> sentiment: float").with_instructions(
    "Classify the sentiment of the text between 0 and 1 where 0 means very negative, 1 means very positive."
)

predict = dspy.Predict(SentimentClassifier)
output = predict(text="The movie started great but ended up terrible.")
print(f"Sentiment score: {output.sentiment}")

# Pretty print LM usage stats
pprint.pprint(output.get_lm_usage())