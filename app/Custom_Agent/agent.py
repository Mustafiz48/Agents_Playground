import logging
from typing import AsyncGenerator
from typing_extensions import override

from google.adk.agents import BaseAgent, LlmAgent, SequentialAgent, LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.events import Event
from pydantic import BaseModel, Field


APP_NAME = "Basic LLM Agent"
USER_ID = "test_user_003"
SESSION_ID = "test_session_003"
model_name = "gemini-2.0-flash"
session_service= InMemorySessionService()

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StoryFlowAgent(BaseAgent):
    """
    Custom agent for story generation and revise.

    This agent generates a story, critique it, refactors it, check grammer and tone, potentially regenrate it if tone is negative.
    """
    # Field declaration for Pydantic

    story_generator: LlmAgent
    critic: LlmAgent
    reviser: LlmAgent
    grammar_checker: LlmAgent
    tone_checker: LlmAgent

    loop_agent: LoopAgent
    sequential_agent: SequentialAgent
    
    # model_config allows setting Pydantic configurations if needed, e.g., arbitrary_types_allowed
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, 
        name: str,
        story_generator: LlmAgent,
        critic: LlmAgent,
        reviser: LlmAgent,
        grammar_checker: LlmAgent,
        tone_checker: LlmAgent,         
        ):
        """
        Initialize the StoryFlowAgent with the provided agents.
        Args:
            story_generator (LlmAgent): Agent for generating stories.
            critic (LlmAgent): Agent for critiquing stories.
            reviser (LlmAgent): Agent for revising stories.
            grammar_checker (LlmAgent): Agent for checking grammar.
            tone_checker (LlmAgent): Agent for checking tone.

        """
        # create internal agents before calling super().__init__()

        loop_agent = LoopAgent(
            name="Critique_and_revise_Loop",
            description="Loop agent for story generation and revise.",
            sub_agents=[critic, reviser,],
            max_iterations=2,
        )

        sequential_agent = SequentialAgent(
            name="Story_revise",
            description="Sequential agent for grammar and tone check.",
            sub_agents=[grammar_checker, tone_checker],
        )
        sub_agents = [story_generator, loop_agent, sequential_agent]
        super().__init__(
            name=name,
            story_generator=story_generator,
            critic=critic,
            reviser=reviser,
            grammar_checker=grammar_checker,
            tone_checker=tone_checker,
            loop_agent=loop_agent,
            sequential_agent=sequential_agent,
            sub_agents=sub_agents,
            )
        
    @override
    async def _run_async_impl(self, ctx:InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Implements the custom orchestration logic for the story workflow.
        Uses the instance attributes assigned by Pydantic (e.g., self.story_generator).
        """

        logger.info(f"[{self.name}] Starting story generation workflow.")
        
        # Step 1: Generate an initial story using the story_generator agent
        async for event in self.story_generator.run_async(ctx):
            yield event

        if "current_story" not in ctx.session.state or not ctx.session.state["current_story"]:
            logger.error("No story generated.")
            return # stop if initial story generation fails
        
        logger.info(f"[{self.name}] Generated story: {ctx.session.state['current_story']}")

        # Step 2: Critique the story using the critic agent
        async for event in self.loop_agent.run_async(ctx):
            yield event
        logger.info(f"[{self.name}] Critique and revise completed.")

        # step 3: Check grammar and tone using the sequential agent
        async for event in self.sequential_agent.run_async(ctx):
            yield event
        logger.info(f"[{self.name}] Grammar and tone check completed.")

        # Step 4: Check if the tone is negative and regenerate the story if needed
        tone = ctx.session.state.get("tone_check_result")

        if tone == "negative":
            logger.info(f"[{self.name}] Tone is negative. Regenerating story.")
            async for event in self.story_generator.run_async(ctx):
                yield event
            logger.info(f"[{self.name}] Regenerated story: {ctx.session.state['current_story']}")
        
        else:
            logger.info(f"[{self.name}] Tone is positive. No need to regenerate the story.")
            pass

        logger.info(f"[{self.name}] Story generation workflow completed.")


# --- Define individual LLM agents ---

# Story generation agent
story_generator = LlmAgent(
    model=model_name, 
    name="story_generator",
    description="Generates a story based on the given prompt.",
    instruction = """
    You are an agent that generates a story.
    1. You take a prompt from the user and generate a short story based on it.
    """,
    output_key="current_story",
)

critic = LlmAgent(
    model=model_name, 
    name="critic",
    description="Critiques the story for errors and improvements.",
    instruction = """
    You are an agent that critiques a story.
    1. You analyze the story in session state under the key named "current_story" and check for errors or improvements.
    2. Provide 1-2 sentences of constructive criticism on how to improve it. 
    3. You Focus on plot or character..
    """,
    output_key="critique",
)

reviser = LlmAgent(
    model=model_name, 
    name="reviser",
    description="Revises the story based on the critique.",
    instruction = """
    You are an agent that revises a story.
    1. You take the critique from the critic agent found undrer the key anmed "critique" and revise the story in session state under the key named "current_story".
    2. You make changes to the story based on the critique provided by the critic agent.
    """,
    output_key="current_story", # this will be the updated story
)

grammar_checker = LlmAgent(
    model=model_name, 
    name="grammar_checker",
    description="Checks the grammar of the story.",
    instruction = """
    You are an agent that checks the grammar of a story.
    1. You analyze the story in session state under the key named "current_story" and check for grammatical errors.
    2. You provide feedback on the grammatical errors found in the story.
    """,
    output_key="grammar_check_result",
)

tone_checker = LlmAgent(
    model=model_name, 
    name="tone_checker",
    description="Checks the tone of the story.",
    instruction = """
    You are an agent that checks the tone of a story.
    1. You analyze the story in session state under the key named "current_story" and check for the tone of the story.
    2. Output only one word: 'positive' if the tone is generally positive, 'negative' if the tone is generally negative, or 'neutral' otherwise.
    """,
    output_key="tone_check_result", # this output will be used to check if the tone is negative or not
)

# --- Create the main StoryFlowAgent ---
root_agent = StoryFlowAgent(
    name="story_flow_agent",
    story_generator=story_generator,
    critic=critic,
    reviser=reviser,
    grammar_checker=grammar_checker,
    tone_checker=tone_checker,
)

initial_state = {"topic": "a brave kitten exploring a haunted house"}
session = session_service.create_session(
    app_name=APP_NAME, 
    user_id=USER_ID, 
    session_id=SESSION_ID,
    state=initial_state,
    )
runner = Runner(
    app_name=APP_NAME, 
    agent=root_agent, 
    session_service=session,
)

def call_agent(story_topic:str):
    """
    Call the agent with a story topic.
    Args:
        story_topic (str): The topic for the story to be generated.
    """

    current_session = session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    if not current_session:
        logger.error("Session not found!")
        return

    current_session.state["topic"] = story_topic
    logger.info(f"Updated session state topic to: {story_topic}")

    content = types.Content(role="user", parts=[types.Part(text = f"Generate a story about: {story_topic}")])

    events = runner.run(
        user_id=USER_ID, 
        session_id=SESSION_ID, 
        new_message=content,
    )

    finally_response_content = "No final response received."

    for event in events:
        if event.content and event.content.parts:
            print(f"Intermediate response: {event.content.parts[0].text}")

        if event.is_final_response() and event.content and event.content.parts:
            # For output_schema, the content is the JSON string itself
            finally_response_content = event.content.parts[0].text
            break
    
    final_session = session_service.get_session(app_name=APP_NAME, 
                                                user_id=USER_ID, 
                                                session_id=SESSION_ID)
    print("Final Session State:")
    import json
    print(json.dumps(final_session.state, indent=2))
    print("-------------------------------\n")

if __name__ == "__main__":
    query = input("Enter your question: ")
    call_agent(query)