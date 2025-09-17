import logging
from typing import AsyncGenerator
from google.adk.agents import LlmAgent, LoopAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai.types import Content, Part
from pydantic import BaseModel, Field
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.load_artifacts_tool import load_artifacts_tool
from .tools.post_creator_tool import generate_image, edit_image

# Configure logging
logger = logging.getLogger(__name__)

class LoopDecision(BaseModel):
    """Decision about whether to continue or end the deep think loop."""
    should_continue: bool = Field(description="Whether to continue the deep think iteration loop")
    reason: str = Field(description="Brief explanation for the decision")

class ContentReview(BaseModel):
    """Review feedback for generated marketing content."""
    adheres_to_request: bool = Field(description="Does the content match the user's original request?")
    visual_appeal: bool = Field(description="Is the visual composition appealing and well-designed?")
    obvious_issues: bool = Field(description="Are there any obvious problems or issues?")
    typos_in_text: bool = Field(description="Are there any unintentional typos in the text overlayed on the image?")
    feedback_addressed: bool = Field(description="Has previous feedback been properly addressed?")
    specific_issues: list[str] = Field(default=[], description="List of specific problems found")
    improvement_suggestions: list[str] = Field(default=[], description="Specific actionable improvements")

content_generation_agent = LlmAgent(
    name="ContentGenAgent",
    instruction="""
    You are a helpful and creative design assistant who helps to create images based on the user's requirements. 
    
    Carefully study the user's input and extract all the details.
    
    if deep_think_iteration: {deep_think_iteration} is 1, call the generate_image tool, else call the edit_image tool
    Use the below feedback (if any) given by the review agent when you draft your inputs for the edit_image tool to ensure that the content is corrected to meet the user's requirements.
    you may use the load_artifact_tool to load and study the image if needed before you call the tool.

    **Important**:
    1. when calling the generate_image or edit_image tools, be very clear and succinct with your instructions. make explicit suggestions on what needs to be changed in order to meet the user's requirements. include only the details needed and omit any unnecessary phrases like "urgent request" or "critical feature"
    2. avoid vague instructions. for example "improve contrast" and "reduce font size" are vague. instead be explicit "add a black gradient background to the top of the image behind the text to increase contrast" and "reduce font size of the subtitle by 2 points".
    3. use your creativity to figure out how the user requirements and improvement suggestions can be implemented in the design to address the suggestion.

    Feedback from previous iterations as follows:
    {content_review}
    """,
    tools=[generate_image, edit_image, load_artifacts_tool]
)

content_review_agent = LlmAgent(
    name="ContentReviewAgent",
            model="gemini-2.5-flash",
            instruction="""You are a marketing content reviewer. Your job is to evaluate generated marketing content and provide constructive feedback.

Load the generated image named {last_generated_image} using load_artifacts_tool and evaluate it against the original user request and provide feedback on:

1. **Adherence to Request**: Does the content match what the user originally asked for?
2. **Visual Appeal**: Is the composition, colors, and overall design appealing and professional?
3. **Obvious Issues**: Are there any clear problems like poor text readability, distorted elements, or technical issues?
4. **Previous Feedback**: If this is a revision, has the previous feedback been properly addressed?
5: **Typos**: Are there any misspelt words on the image?

Provide specific, actionable suggestions for improvement. Focus on practical issues that can be addressed in the next iteration.

Be constructive but honest in your assessment. The goal is to help create the best possible marketing content for the user.

Original user request: {original_prompt}
Current iteration: {iteration_count}
Previous feedback: {previous_feedback}""",
            output_schema=ContentReview,
            output_key="content_review",
            disallow_transfer_to_parent=True,
            disallow_transfer_to_peers=True,
            tools=[load_artifacts_tool]
)

loop_control_agent = LlmAgent(
    name="LoopControlAgent",
            model="gemini-2.5-flash",
            instruction="""You are responsible for determining whether the deep think content creation process should continue or conclude.

                Analyze the review feedback from the ContentReviewAgent and decide:

                **Continue Loop If:**
                - The content doesn't match the user's original request
                - There are significant visual appeal issues
                - Obvious problems or technical issues exist
                - Previous feedback hasn't been properly addressed
                - The content could be significantly improved

                **End Loop If:**
                - The content matches the user's request well
                - Visual appeal is good and professional
                - No obvious issues or problems
                - Previous feedback has been addressed
                - Only minor improvements could be made
                - Maximum iterations have been reached

                **Decision Criteria:**
                The content should be "good enough" - it doesn't need to be perfect, but it should meet the user's core requirements and be visually appealing.

                If continuing, briefly summarize the key areas that need improvement. If ending, confirm that the content is ready for finalization.

                Current iteration: {iteration_count}
                Max iterations: 4
                Review feedback: {content_review}""",
            output_schema=LoopDecision,
            output_key="loop_decision",
            disallow_transfer_to_parent=True,
            disallow_transfer_to_peers=True,
)

class LoopTerminationAgent(BaseAgent):
    """Checks loop control decision and manages termination."""
    
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        loop_decision = ctx.session.state.get("loop_decision")
        iteration_count = ctx.session.state.get("deep_think_iteration", 0)
        
        should_continue = True
        reason = "No decision found"
        
        if isinstance(loop_decision, LoopDecision):
            should_continue = loop_decision.should_continue
            reason = loop_decision.reason
        elif isinstance(loop_decision, dict):
            should_continue = loop_decision.get("should_continue", True)
            reason = loop_decision.get("reason", "No reason provided")
        
        # Force stop if we've reached max iterations
        if iteration_count >= 4:
            should_continue = False
            reason = "Maximum iterations reached"
        
        if should_continue:
            yield Event(
                author=self.name,
                content=Content(parts=[Part(text=f"Continuing deep think refinement: {reason}")]),
            )
        else:
            # End the loop and clean up deep think state
            final_image = ctx.session.state.get("last_generated_image")
            yield Event(
                author=self.name,
                content=Content(parts=[Part(text=f"Deep think mode complete: {reason}. Final marketing content: {final_image}")]),
                actions=EventActions(
                    state_delta={
                        "deep_think_mode": False,
                        "deep_think_iteration": 0,
                        "original_deep_think_prompt": None,
                        "content_review": None,
                        "loop_decision": None
                    },
                    escalate=True
                )
            )

prompt_capture_agent = LlmAgent(
    name="PromptCaptureAgent",
    instruction="""
    You are tasked to analyse the conversation history and extract the key requirements from the user's latest prompt and store in the state variable. output your response as a string. ensure that you have captured all the key details that the user mentioned.
    """,
    model="gemini-2.5-flash",
    output_key="original_prompt",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

class DeepThinkPreparationAgent(BaseAgent):
    """Prepares context for deep think content generation."""
    
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # Initialise state variables that would be used later
        if not ctx.session.state.get("deep_think_iteration"):
            ctx.session.state["deep_think_iteration"] = 0

        if not ctx.session.state.get("iteration_count"):
            ctx.session.state["iteration_count"] = 0

        if not ctx.session.state.get("previous_feedback"):
            ctx.session.state["previous_feedback"] = {}

        original_prompt = ctx.session.state.get("original_deep_think_prompt", "")
        iteration_count = ctx.session.state.get("deep_think_iteration", 0) + 1
        
        context_message = f"""
        Starting deep think content generation iteration {iteration_count}.

        Original user request: {original_prompt}
        """
        
        if iteration_count > 1:
            previous_review = ctx.session.state.get("content_review")
            if previous_review:
                context_message += f"\nPrevious review feedback: {previous_review}"
        else:
            ctx.session.state["content_review"] = "" 
        
        yield Event(
            author=self.name,
            content=Content(parts=[Part(text=context_message)]),
        )


# Create the deep think loop structure
deep_think_loop = LoopAgent(
    name="DeepThinkLoop",
    sub_agents=[
        DeepThinkPreparationAgent(name="DeepThinkPreparationAgent"),
        prompt_capture_agent,
        content_generation_agent,
        content_review_agent,
        loop_control_agent,
        LoopTerminationAgent(name="LoopTerminationAgent"),
    ],
    max_iterations=5,
)

# Create an agent tool wrapper for the deep think loop
deep_think_agent_tool = AgentTool(
    agent=deep_think_loop,
)