import json
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
from pydantic import BaseModel, Field


APP_NAME = "Basic LLM Agent"
USER_ID = "test_user_001"
SESSION_ID = "test_session_001"
model_name = "gemini-2.0-flash"


# define the agents

root_agent = LlmAgent(
    model=model_name,
    name = "root_agent",
    description="Ansers users questions and provides precise and accurate answers.",
    instruction=""" You are an agent that answers any question asked by the user.
    When user asks a question:
    1. You use "google_search" tool to search for the answer.
    2. You present the anser in a clear and concise manner.
    3. You answer the question as accurately as possible.
    4. You provide the source of the information you used to answer the question.
    5. if you don't know the answer, you you say it to the user politely
    """,
    tools=[google_search],
)

session_service = InMemorySessionService()
# Create separate sessions for clarity, though not strictly necessary if context is managed
session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

runner = Runner(
    agent = root_agent,
    app_name= APP_NAME,
    session_service = session_service
    )

async def run_agent(input_text: str):
    """Sends query to the agent and returns the response."""
    print(f"User query: {input_text}")
    content = types.Content(role = "user", parts=[types.Part(text=input_text)])
    final_response_content = "No final response received."

    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
        if event.content and event.content.parts:
            print(f"Intermediate response: {event.content.parts[0].text}")

        if event.is_final_response() and event.content and event.content.parts:
            # For output_schema, the content is the JSON string itself
            final_response_content = event.content.parts[0].text
            break

    print(f"<<< Agent '{root_agent.name}', Response: {final_response_content}")
    return final_response_content


async def main():
    input_text = input("Enter your question: ")
    print(run_agent(input_text))

if __name__ == "__main__":
    main()