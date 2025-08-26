from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import os

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Define schema
class WeatherInfo(BaseModel):
    location: str
    temperature_c: float
    condition: str

# Correct model 
model = OpenAIModel(
    'gpt-4o-mini',
    provider=OpenAIProvider(api_key=api_key)
)

# Creating agent with output_type as WeatherInfo
agent = Agent(
    model,
    output_type=WeatherInfo,
)

# Run the agent

qurey = "What's the weather in vizag today?"
result = agent.run_sync(qurey)
print(result.output)