import logging
import uuid
import time
import hashlib
from typing import Optional, List
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.tools.load_artifacts_tool import load_artifacts_tool
from google.genai.types import Content, Part
from google.genai import types
from .tools.post_creator_tool import generate_image, edit_image, list_asset_versions, list_reference_images, store_reference_image
from .deep_think_loop import deep_think_agent_tool, deep_think_loop
from .prompt import SOCIAL_MEDIA_AGENT_INSTRUCTION

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple callback to auto-store uploaded reference images
async def auto_store_reference_images(callback_context, llm_request):
    """Automatically store uploaded images as reference images"""
    if not llm_request.contents:
        return None
    
    for message in llm_request.contents:
        for part in message.parts:
            if hasattr(part, 'inline_data') and part.inline_data and part.inline_data.mime_type.startswith("image/"):
                try:
                    # Generate filename
                    image_hash = hashlib.md5(part.inline_data.data).hexdigest()[:8]
                    filename = f"reference_{image_hash}_{int(time.time())}.jpg"
                    
                    # Store using ADK's native artifact service
                    version = await callback_context.save_artifact(filename, part)
                    
                    # Simple state tracking
                    callback_context.state["latest_reference_image"] = filename
                    
                    logger.info(f"Auto-stored reference image: {filename}")
                    
                except Exception as e:
                    logger.error(f"Error auto-storing reference image: {e}")
    
    return None

# --- Define the Agent ---
root_agent = LlmAgent(
    name="wardrobe_design_agent",
    model="gemini-2.5-flash",
    instruction=SOCIAL_MEDIA_AGENT_INSTRUCTION,
    tools=[generate_image, edit_image, list_asset_versions, list_reference_images, load_artifacts_tool],
    sub_agents=[deep_think_loop],
    before_model_callback=auto_store_reference_images
)

# --- Configure and Expose the Runner ---
runner = Runner(
    agent=root_agent,
    app_name="wardrobe_design_app",
    session_service=None,  # Using default InMemorySessionService
    artifact_service=InMemoryArtifactService(),
)