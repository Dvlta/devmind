# DevMind

DevMind is an AI codebase understanding tool designed to answer natural-language questions about unfamiliar repositories with source-grounded citations.

Current implementation status:

- Local repository indexing
- Line-window code chunking
- SQLite FTS keyword retrieval
- Path search
- File inspection with line numbers
- CLI commands for indexing, searching, inspecting, and basic cited answers

## Quick Start

```bash
python3 -m devmind.cli index /path/to/repo
python3 -m devmind.cli search auth
python3 -m devmind.cli ask "Where is authentication handled?"
```

By default, DevMind stores its local index at `.devmind/devmind.sqlite`.

