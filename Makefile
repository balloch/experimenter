.PHONY: install install-dev test lint mcp-server mcp-dev clean

# ── Setup ──────────────────────────────────────────────────────────────────────
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,nlp]"

# copy .env template if missing
env:
	@if [ ! -f .env ]; then cp .env.example .env && echo "Created .env — fill in your credentials"; fi

# ── Quality ────────────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/

test:
	pytest tests/ -v

# ── Running ────────────────────────────────────────────────────────────────────
# Start the MCP server (used by Claude Desktop / Claude Code)
mcp-server:
	python -m experimenter.mcp_server

# Development: stream MCP server logs to stdout
mcp-dev:
	LOG_LEVEL=DEBUG python -m experimenter.mcp_server

# Run a quick end-to-end smoke test locally
smoke:
	experimenter run "Does income level predict life satisfaction?" \
		--compute local --time-budget 5

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf dist/ build/ .pytest_cache/
