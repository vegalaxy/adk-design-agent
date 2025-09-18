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
from .prompt import (
    CONTENT_GENERATION_AGENT_INSTRUCTION,
    CONTENT_REVIEW_AGENT_INSTRUCTION,
    LOOP_CONTROL_AGENT_INSTRUCTION,
    PROMPT_CAPTURE_AGENT_INSTRUCTION
)

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
    instruction=CONTENT_GENERATION_AGENT_INSTRUCTION,
    tools=[generate_image, edit_image, load_artifacts_tool]
)

content_review_agent = LlmAgent(
    name="ContentReviewAgent",
    model="gemini-2.5-flash",
    instruction=CONTENT_REVIEW_AGENT_INSTRUCTION,
    output_schema=ContentReview,
    output_key="content_review",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    tools=[load_artifacts_tool]
)

loop_control_agent = LlmAgent(
    name="LoopControlAgent",
    model="gemini-2.5-flash",
    instruction=LOOP_CONTROL_AGENT_INSTRUCTION,
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
    instruction=PROMPT_CAPTURE_AGENT_INSTRUCTION,
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

# Create an agent tool wrapper for the deep think loop (used if we don't want the user to see the outputs of the deep think loop)
deep_think_agent_tool = AgentTool(
    agent=deep_think_loop,
)