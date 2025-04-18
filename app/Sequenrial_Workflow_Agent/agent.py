from google.adk.agents import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

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
    3. You refactor the code to make it more efficient and readable.
    """,
)
