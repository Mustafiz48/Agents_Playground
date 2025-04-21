from google.adk.agents import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from google.adk.tools import google_search



APP_NAME = "Basic LLM Agent"
USER_ID = "test_user_002"
SESSION_ID = "test_session_002"
model_name = "gemini-2.0-flash"
session = InMemorySessionService()

code_generation_agent = LlmAgent(
    model=model_name, 
    name="code_generation_agent",
    description="Generates code based on user input.",
    instruction = """
    You are an agent that generates code based on user input.
    When user asks a question:
    1. You analyze the question or task
    2. If user mentions a specific programming language, you generate code in that language.
    3. If user doesn't specify programming language, you generate code in Python.
    4. you output *only* the code, without any explanation or comments. 
    """,
    output_key="generated_code",
)


code_review_agent = LlmAgent(
    model=model_name, 
    name="code_review_agent",
    description="Reviews a code for error correction and feedback.",
    instruction = """
    You are an agent that reviews code.
    1. You analyze the code in session state under the key named "generated_code" and check for errors or improvements.
    2. You provide feedback on the code, including any errors or improvements that can be made.
    3. You only provide the feedback, where to change, and how to change. But you don't change the code.
    """,
    output_key="code_review",
)

code_refactoring_agent = LlmAgent(
    model=model_name, 
    name="code_refactoring_agent",
    description="Refactors a code from feedback",
    instruction = """
    You are an agent that refactors code given to you with the provided feedback.
    1. You first analyze the code in session state under the key named "generated_code"
    2. Analyze the feedback provided by the code_review_agent in sesion state named under key "code_review".
    3. You refactor the code based on the feedback provided by the code_review_agent.
    3. You refactor the code to make it more efficient and error free and aptimized.
    """,
)



# Create a sequential agent with the above agents

root_agent = SequentialAgent(
    name="code_pipeline_agent",
    description="A sequential agent that generates, reviews, and refactors code.",
    sub_agents=[code_generation_agent, code_review_agent, code_refactoring_agent],
)

session_service = InMemorySessionService()
session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

runner = Runner(app_name=APP_NAME, agent=root_agent, session_service=session_service)

def call_agent(query):
    content = types.Content(role="user", parts=[types.Part(text = query)])
    
    final_response_content = "No final response received."

    for event in runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
        if event.content and event.content.parts:
            print(f"Intermediate response: {event.content.parts[0].text}")

        if event.is_final_response() and event.content and event.content.parts:
            # For output_schema, the content is the JSON string itself
            final_response_content = event.content.parts[0].text
            break

    print(f"<<< Agent '{root_agent.name}', Response: {final_response_content}")
    return final_response_content


if __name__ == "__main__":
    query = input("Enter your question: ")
    call_agent(query)