import logging
from typing import AsyncGenerator
from typing_extensions import override

from google.adk.agents import BaseAgent, LlmAgent, SequentialAgent, LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.events import Event
from google.adk.tools import google_search
from pydantic import BaseModel, Field


APP_NAME = "Custom Blog Generator Agent"
USER_ID = "test_user_004"
SESSION_ID = "test_session_004"
model_name = "gemini-2.0-flash"
session_service= InMemorySessionService()

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlogGeneratorAgent(BaseAgent):
    """"
    Custom agent for blog generation from a given topic.
    This agenst orchastrates a sequence of LLM agent to generate a title, blog structucture, detailed written blog post,
    optimize it for SEO, generate HTML code and reviews it for any errors or suggestions.
    Finally it generates a detailed written, long, SEO friendly, nicely html bofrmatted blog post.
    """

    title_generator: LlmAgent
    structure_generator: LlmAgent
    blog_generator: LlmAgent
    seo_optimizer: LlmAgent
    html_generator: LlmAgent
    review_agent: LlmAgent

    loop_agent: LoopAgent
    sequential_agent: SequentialAgent
    second_loop_agent: LoopAgent
    # model_config allows setting Pydantic configurations if needed, e.g., arbitrary_types_allowed
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self,
        name: str,
        title_generator: LlmAgent,
        structure_generator: LlmAgent,
        blog_generator: LlmAgent,
        seo_optimizer: LlmAgent,
        html_generator: LlmAgent,
        review_agent: LlmAgent,
        ):
        """
        Initialize the BlogGeneratorAgent with the provided agents.
        Args:
            name (str): Name of the agent.
            title_generator (LlmAgent): Agent for generating blog titles.
            structure_generator (LlmAgent): Agent for generating blog structures.
            blog_generator (LlmAgent): Agent for generating blog content.
            seo_optimizer (LlmAgent): Agent for optimizing SEO.
            html_generator (LlmAgent): Agent for generating HTML code.
            review_agent (LlmAgent): Agent for reviewing the final output.
        """

        sequential_agent = SequentialAgent(
            name="Blog_Title_and_Structure_Generation",
            description="A sequrntial agent to Generate a blog title and structure",
            sub_agents=[title_generator, structure_generator, ],
        )

        loop_agent = LoopAgent(
            name="Blog_Generation_and_SEO_Optimization",
            description="Loop agent for blog generation and SEO optimization.",
            sub_agents=[blog_generator,seo_optimizer,],
            max_iterations=3,
        )
        second_loop_agent = LoopAgent(
            name="Blog_HTML_Generation_and_Review",
            description="A sequrntial agent to Generate a blog HTML and review it",
            sub_agents=[html_generator, review_agent,],
            max_iterations=3,
        )
        sub_agents = [sequential_agent, loop_agent, second_loop_agent]

        super().__init__(
            name=name,
            title_generator=title_generator,
            structure_generator=structure_generator,
            blog_generator=blog_generator,
            seo_optimizer=seo_optimizer,
            html_generator=html_generator,
            review_agent=review_agent,
            sequential_agent=sequential_agent,
            loop_agent=loop_agent,
            second_loop_agent=second_loop_agent,
            sub_agents=sub_agents
        )

    @override
    async def _run_async_impl(self, ctx:InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Implements the custom orchestration logic for the blog generation workflow.
        Args:
            ctx (InvocationContext): The context for the invocation.
        Yields:
            Event: The generated events during the run.
        """

        logger.info("Starting blog generation process...")

        # Step 1: Generate a blog title and structure using the sequential_agent
        async for event in self.sequential_agent.run_async(ctx):
            logger.info(f"Event from sequential_agent: {event}")
            yield event

        if "blog_structure" not in ctx.session.state or not ctx.session.state["blog_structure"]:
            logger.error("No blog structure generated.")
            return

        logger.info(f"Generated blog structure: {ctx.session.state['blog_structure']}")



        # Step 2: Generate blog content and optimize for SEO using the loop_agent
        async for event in self.loop_agent.run_async(ctx):
            logger.info(f"Event from loop_agent: {event}")
            yield event

        logger.info("Blog content generation and SEO optimization completed.")

        if "blog_content" not in ctx.session.state or not ctx.session.state["blog_content"]:
            logger.error("No blog content generated.")
            return
        logger.info(f"Generated blog content: {ctx.session.state['blog_content']}")



        # Step 3: Generate HTML code and review it using the second_loop_agent
        async for event in self.second_loop_agent.run_async(ctx):
            logger.info(f"Event from second_loop_agent: {event}")
            yield event
        logger.info("HTML generation and review completed.")
        if "html_code" not in ctx.session.state or not ctx.session.state["html_code"]:
            logger.error("No HTML code generated.")
            return
        logger.info(f"Generated HTML code: {ctx.session.state['html_code']}")


# Let;s define  the individual agents for title generation, structure generation, blog content generation, SEO optimization, HTML generation, and review.

title_generator = LlmAgent(
    model=model_name,
    name="Title_Generator",
    description="Generates a catchy, SEO firendly title for the blog post.",
    instruction= """
    You are a title generator for a blog post.
    Given a topic, generate a catchy, SEO friendly title for the blog post.
    """,
    output_key="blog_title",
)

structure_generator = LlmAgent(
    model=model_name,
    name="Structure_Generator",
    description="Generates a detailed structure for the blog post.",
    instruction= """
    You are a structure generator for a blog post.
    1. You take the blog title from  the key "blog_title"
    2. You take the blog topic from the key "blog_topic"
    2. You use google_search tool to search for the topic and generate a detailed structure for the blog post.
    """,
    output_key="blog_structure",
    tools=[google_search],
)

blog_generator = LlmAgent(
    model=model_name,
    name="Blog_Generator",
    description="Generates a detailed blog post based on the given blog title and structure.",
    instruction= """
    You are a detailed blog generator.
    1. You take the blog topic from the key "blog_topic"
    2. You take the blog title found under the key "blog_title"
    3. You take the blog structure from the key "blog_structure"
    4. You search the web using search_tool for the topic, blog title anad analyze the information.
    5. From the information you find, generate a detailed blog post with the blog title following blog structure.
    """,
    output_key="blog_content",
    tools=[google_search],
)

seo_optimizer = LlmAgent(
    model=model_name,
    name="SEO_Optimizer",
    description="Optimizes the blog post for SEO.",
    instruction= """
    You are an SEO optimizer for a blog post.
    1. You take the blog content from the key "blog_content"
    2. You analyze the content and optimize it for SEO.
    """,
    output_key="optimized_blog_content",
)

html_generator = LlmAgent(
    model=model_name,
    name="HTML_Generator",
    description="Generates HTML code for the blog post.",
    instruction= """
    You are an HTML generator for a blog post.
    1. You take the optimized blog content from the key "optimized_blog_content"
    2. You generate well written, error free HTML code with proper css styling for the blog post.
    """,
    output_key="html_code",
)

review_agent = LlmAgent(
    model=model_name,
    name="Review_Agent",
    description="Reviews the HTML code for errors and suggestions.",
    instruction= """
    You are a review agent for a blog post.
    1. You take the HTML code from the key "html_code"
    2. You review the HTML code for errors and suggestions.
    3. based on the review, you provide a detailed feedback on how to improve the HTML code.
    4. Based on your review and feedback, you modify the HTML code to make it better.
    """,
    output_key="review_result",
)

root_agent = BlogGeneratorAgent(
    name="Blog_generation_agent",
    title_generator=title_generator,
    structure_generator=structure_generator,
    blog_generator=blog_generator,
    seo_optimizer=seo_optimizer,
    html_generator=html_generator,
    review_agent=review_agent,
)

# --- Run the agent ---

session = session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID,
)
runner = Runner(
    app_name=APP_NAME,
    agent=root_agent,
    session_service=session_service,
)

def call_agent(blog_topic: str):
    """
    Call the agent with the given blog topic.
    Args:
        blog_topic (str): The topic for the blog post.
    """
    current_session = session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )
    if not current_session:
        logger.error("Session not found.")
        return
    current_session.state["blog_topic"] = blog_topic

    logger.info(f"Blog topic set to: {blog_topic}")

    content = types.Content(role="user", parts=[types.Part(text=f"Generate a blog post about: {blog_topic}")])
    events = runner.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content,
    )

    for event in events:
        if event.content and event.content.parts:
            yield f"Intermediate response: {event.content.parts[0].text}"

        if event.is_final_response() and event.content and event.content.parts:
            # For output_schema, the content is the JSON string itself
            finally_response_content = event.content.parts[0].text
            yield f"Final response: {finally_response_content}"
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