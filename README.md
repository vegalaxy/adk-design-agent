# Social Media Design Agent

An intelligent multi-agent system built with Google ADK for creating and iterating on social media content. This agent demonstrates advanced AI orchestration patterns with two distinct operational modes: fast interactive design assistance and autonomous quality assurance through "Deep Think Mode."

## Features

- **Dual-Mode Operation**: Choose between speed (Regular Mode) or quality (Deep Think Mode)
- **Intelligent Image Generation**: Powered by Gemini's image generation capabilities
- **Reference Image Support**: Upload images for style and composition guidance
- **Versioned Asset Management**: Automatic versioning and organization of created content
- **Autonomous Quality Loop**: Multi-agent review and refinement system
- **Context Awareness**: Maintains session state for iterative improvements

## Architecture

The agent consists of several specialized sub-agents:

- **Main Social Media Agent**: Handles user interactions and routing
- **Content Generation Agent**: Creates and edits images with detailed instructions
- **Content Review Agent**: Evaluates quality across multiple dimensions
- **Loop Control Agent**: Makes intelligent decisions about iteration continuation
- **Deep Think Loop**: Orchestrates autonomous creative iteration

## Prerequisites

- Python 3.13+
- Google ADK (`pip install google-adk`)
- Gemini API key with image generation access

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd social_media_agent
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Verify installation**:
   ```bash
   python -c "import google.adk; print('ADK installed successfully')"
   ```

## Running the Agent

### Development UI (Recommended for testing)

Start the ADK development server:

```bash
adk web
```

Then navigate to `http://127.0.0.1:8000/dev-ui` in your browser and interact with the agent through the web interface.

## Usage Examples

### Regular Mode (Fast)

Simple, direct interaction for quick iterations:

```
User: "Create a holiday promotion poster in 16:9 aspect ratio"
Agent: [Generates image quickly using generate_image tool]
User: "Make the text larger"
Agent: [Uses edit_image tool for quick refinement]
```

### Deep Think Mode (Quality)

Autonomous quality assurance with multiple iterations:

```
User: "Deep think create a professional product launch announcement"
Agent: [Enters autonomous loop]
       1. Generates initial content
       2. Reviews for quality and adherence
       3. Refines based on feedback
       4. Repeats until professional standard
       5. Presents final result
```

### With Reference Images

Upload inspiration images for style guidance:

```
User: [Uploads reference image] "Create a social media post in this style"
Agent: [Automatically saves reference as reference_image_v1.png]
       [Uses reference for style and composition guidance]
```

## Key Commands

- **Regular Generation**: `"Create a [description]"`
- **Deep Think Mode**: `"Deep think create a [description]"`
- **With Reference**: Upload image + `"Create something in this style"`
- **List Assets**: `"Show me all my assets"`
- **List References**: `"What reference images do I have?"`
- **Load Previous**: `"Show me [asset_name_v2.png]"`

## File Structure

```
social_media_agent/
├── agent.py                 # Main agent configuration
├── deep_think_loop.py       # Deep think mode implementation
├── tools/
│   └── post_creator_tool.py # Image generation and editing tools
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Configuration

The agent can be customized by modifying:

- **Max iterations**: Change `max_iterations=5` in `deep_think_loop.py`
- **Models**: Update model names in agent configurations
- **Instructions**: Modify agent instructions for different behaviors
- **Tools**: Add or remove tools from the agent's toolkit

## Troubleshooting

### Common Issues

1. **"GEMINI_API_KEY environment variable not set"**
   - Ensure your `.env` file contains the API key
   - Verify the key has image generation permissions

2. **"Artifact service is not initialized"**
   - The agent uses `InMemoryArtifactService` by default
   - For production, consider using `GcsArtifactService`

3. **"Deep think mode not activating"**
   - Ensure you include "deep think" in your prompt
   - Check that `sub_agents=[deep_think_loop]` is configured

### Debug Mode

Enable detailed logging:

```bash
export PYTHONPATH=.
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from social_media_agent.agent import runner
# Your agent interaction code here
"
```

## Advanced Usage

### Custom Asset Names

Provide meaningful names for better organization:

```python
# In your prompts, specify asset names
"Create a holiday_promotion poster"
# Results in: holiday_promotion_v1.png, holiday_promotion_v2.png, etc.
```

### Reference Image Management

```python
# List all reference images
"What reference images do I have?"

# Use specific reference
"Create a design using reference_image_v2.png as inspiration"

# Use latest uploaded reference
"Create something based on the latest reference image"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Support

For issues and questions:
- Check the troubleshooting section above
- Review ADK documentation: https://google.github.io/adk-docs/
- Open an issue in the repository
