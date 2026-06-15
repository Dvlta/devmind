# DevMind

DevMind is an AI codebase understanding tool designed to answer natural-language questions about unfamiliar repositories with source-grounded citations.

Current implementation status:

- Local repository indexing
- Python symbol-aware and fallback line-window chunking
- Hybrid semantic, keyword, path, symbol, and import retrieval
- File inspection with line numbers
- LangGraph planning, retrieval, inspection, synthesis, and grounding validation
- Configurable deterministic or OpenAI LLM synthesis
- CLI commands for indexing, searching, inspecting, and cited answers

## Quick Start

```bash
python -m devmind.cli index /path/to/repo
python -m devmind.cli search auth
python -m devmind.cli ask "Where is authentication handled?" --trace
```

By default, DevMind stores its local index at `.devmind/devmind.sqlite`.

## LLM Configuration

DevMind defaults to a deterministic provider, so indexing, retrieval, LangGraph execution, and tests work without an API key.

To enable OpenAI structured planning, synthesis, and grounding validation:

```bash
pip install -e ".[agent,llm,dev]"
devmind ask "How does authentication work?" --repo your-repo --trace
```

DevMind automatically loads the nearest `.env` file for local development. Copy `.env.example` to `.env` and set:

```env
DEVMIND_LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
DEVMIND_LLM_MODEL=gpt-5.5
DEVMIND_LLM_MODE=fast
```

Exported environment variables take precedence over values in `.env`. Production deployments can inject the same variables through their platform or secret manager.

You can also override the provider and model per command with `--provider` and `--model`.

`fast` mode is the default and makes one LLM call for concise synthesis. Use `DEVMIND_LLM_MODE=full` or `--llm-mode full` to also use the model for planning and grounding validation.

## Tests

```bash
python -m pytest
```
