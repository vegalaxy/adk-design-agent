import logging
import uuid
from typing import Optional
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.tools.load_artifacts_tool import load_artifacts_tool
from google.genai.types import Content, Part
from .tools.post_creator_tool import generate_image, edit_image, list_asset_versions, list_reference_images
from .deep_think_loop import deep_think_agent_tool, deep_think_loop

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def detect_deep_think_callback(callback_context: CallbackContext) -> Optional[Content]:
    """
    A before_agent_callback to detect deep think mode in user input.
    
    This function checks for "deep think" phrase in the user's input and sets up
    the session state accordingly. It also cleans the prompt for agent processing.
    """
    # Initialize deep think mode state if not present
    if "deep_think_mode" not in callback_context.state:
        callback_context.state["deep_think_mode"] = False
    
    # Get the user content from the callback context
    user_content = callback_context.user_content
    if not user_content or not user_content.parts:
        # Default to regular mode if no user content
        callback_context.state["deep_think_mode"] = False
        return None
    
    # Extract text from user's message
    user_text = ""
    for part in user_content.parts:
        if part.text:
            user_text += part.text + " "
    
    if not user_text.strip():
        # Default to regular mode if no text content
        callback_context.state["deep_think_mode"] = False
        return None
    
    # Check for "deep think" phrase (case insensitive)
    if "deep think" in user_text.lower():
        logger.info("Deep think mode detected in user prompt")
        callback_context.state["deep_think_mode"] = True
        
        # Store original prompt for the deep think loop
        callback_context.state["original_deep_think_prompt"] = user_text.strip()
        
        # Clean "deep think" from the prompt to avoid confusion
        cleaned_text = user_text.replace("deep think", "").replace("Deep Think", "").replace("DEEP THINK", "")
        cleaned_text = " ".join(cleaned_text.split())  # Clean up extra spaces
        
        # Update the user content with cleaned text
        # Create new content with cleaned text
        cleaned_parts = []
        for part in user_content.parts:
            if part.text:
                if not cleaned_parts:  # Only update the first text part
                    cleaned_parts.append(Part(text=cleaned_text))
                # Skip other text parts to avoid duplication
            else:
                cleaned_parts.append(part)  # Keep non-text parts (like images)
        
        # Update the user content parts
        user_content.parts = cleaned_parts
        
        logger.info(f"Cleaned prompt: '{cleaned_text}'")
    else:
        # Ensure deep think mode is off for regular requests
        callback_context.state["deep_think_mode"] = False
    
    return None

async def process_reference_images_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[Content]:
    """
    A before_model_callback to process uploaded reference images.
    
    This function intercepts the request before it goes to the LLM.
    If it finds an image upload, it saves it as a reference artifact.
    """
    if not llm_request.contents:
        return None
        
    latest_user_message = llm_request.contents[-1]
    image_part = None
    
    # Look for uploaded images in the latest user message
    for part in latest_user_message.parts:
        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
            logger.info(f"Found reference image to process: {part.inline_data.mime_type}")
            image_part = part
            break
    
    # Process reference image if found
    if image_part:
        # Generate versioned filename for reference image
        reference_images = callback_context.state.get("reference_images", {})
        ref_count = len(reference_images) + 1
        filename = f"reference_image_v{ref_count}.png"
        logger.info(f"Saving reference image as artifact: {filename}")
        
        try:
            version = await callback_context.save_artifact(
                filename=filename, artifact=image_part
            )
            
            # Store reference image info in session state
            if "reference_images" not in callback_context.state:
                callback_context.state["reference_images"] = {}
            
            callback_context.state["reference_images"][filename] = {
                "version": ref_count,
                "uploaded_version": version
            }
            callback_context.state["latest_reference_image"] = filename
            
            logger.info(f"Saved reference image as '{filename}' version {version}")
            
        except Exception as e:
            logger.error(f"Error saving reference image artifact: {e}")
    
    return None

# --- Define the Agent ---
root_agent = LlmAgent(
    name="social_media_agent",
    model="gemini-2.5-flash",
    instruction="""You are a social media post agent. Your goal is to help users create and iterate on social media posts.

First, ask the user what kind of post they would like to create, the desired aspect ratio, any text overlays, and any other relevant details needed to generate the image.

**Deep Think Mode**: If the user says they want you to "deep think" or use any instructions along those lines, then call the deep_think_loop to perform a deeper generation process.

If the user does not make any request to think deeply, proceed with the regular mode.

**Regular Mode**: For normal requests, use the `generate_image` tool to create the first version of the image.

After the image is generated, ask the user for feedback. If they want to make changes, use the `edit_image` tool to modify the image based on their feedback.

You can iterate on the image multiple times until the user is happy with the result.

If the user asks to see a previously generated image, use the `load_artifacts_tool` tool.

You can use `list_asset_versions` to show the user all marketing assets and their versions that have been created in this session.

Users can upload reference images to provide visual inspiration for their marketing content. When a user uploads an image, it will be automatically saved as a reference image. You can use `list_reference_images` to show available reference images.

When generating or editing images, users can specify a reference image filename (or use 'latest' for the most recent upload) to guide the visual style, composition, or elements.

Use the load_artifacts_tool to read the image and understand what the user is saying, especially when they are referencing elements of the image that you have not yet seen.

When creating new images, ask the user for a meaningful asset name (e.g., 'holiday_promo', 'product_launch', 'brand_awareness') instead of using generic names. This helps with organization and iteration.""",
    tools=[generate_image, edit_image, list_asset_versions, list_reference_images, load_artifacts_tool],
    sub_agents=[deep_think_loop],
    # before_agent_callback=detect_deep_think_callback,
    before_model_callback=process_reference_images_callback
)

# --- Configure and Expose the Runner ---
runner = Runner(
    agent=root_agent,
    app_name="social_media_agent_app",
    session_service=None,  # Using default InMemorySessionService
    artifact_service=InMemoryArtifactService(),
)