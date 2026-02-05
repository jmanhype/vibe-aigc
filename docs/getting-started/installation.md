# Installation

## Requirements

- Python 3.12 or higher
- An OpenAI API key (or compatible LLM endpoint)

## Install from PyPI

```bash
pip install vibe-aigc
```

## Install from Source

```bash
git clone https://github.com/jmanhype/vibe-aigc.git
cd vibe-aigc
pip install -e ".[dev]"
```

## Configuration

### OpenAI API Key

Set your OpenAI API key as an environment variable:

=== "Linux/macOS"

    ```bash
    export OPENAI_API_KEY="sk-..."
    ```

=== "Windows (PowerShell)"

    ```powershell
    $env:OPENAI_API_KEY="sk-..."
    ```

=== "Windows (CMD)"

    ```cmd
    set OPENAI_API_KEY=sk-...
    ```

### Custom LLM Endpoint

You can use any OpenAI-compatible endpoint:

```python
from vibe_aigc import MetaPlanner
from vibe_aigc.llm import LLMConfig

config = LLMConfig(
    api_key="your-api-key",
    base_url="https://your-endpoint.com/v1",
    model="your-model-name"
)

planner = MetaPlanner(llm_config=config)
```

## Verify Installation

```python
from vibe_aigc import Vibe, MetaPlanner

print("Vibe AIGC installed successfully!")

# Create a test vibe
vibe = Vibe(description="Test vibe")
print(f"Created vibe: {vibe.description}")
```

## Development Setup

For contributing or development:

```bash
# Clone the repository
git clone https://github.com/jmanhype/vibe-aigc.git
cd vibe-aigc

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -v

# Run linting
ruff check .
black --check .
```
