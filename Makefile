.DEFAULT_GOAL := help
.PHONY: help install serve web mcp test test-fast lint format web-lint web-build sample testset eval-rules eval-real summary docker-build docker-up clean

FREE_MODEL ?= nvidia/nemotron-3-super-120b-a12b:free
REAL = RECUT_LLM_PROVIDER=openrouter RECUT_LLM_MODEL=$(FREE_MODEL)

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Python (with faster-whisper and the test-set voices) and web dependencies
	uv sync --extra asr --extra synth
	cd web && npm ci --no-audit --no-fund

serve:  ## API on http://localhost:8000 (imports the bundled sample on first start)
	uv run recut serve --port 8000

web:  ## Review UI on http://localhost:3000
	cd web && npm run dev

mcp:  ## MCP server over stdio for Claude Code / Claude Desktop (see README)
	uv run recut mcp

sample:  ## Process the bundled sample talk from the command line with the offline model
	uv run recut process data/samples/creator-margin-ep12.mp4 --license "CC0 1.0" --render-top 1

test:  ## All Python tests (no keys, no network; ffmpeg needed for the media tests)
	uv run pytest

test-fast:  ## Tests without real renders
	uv run pytest -m "not slow"

lint:  ## Ruff lint + format check + mypy (strict)
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy

format:  ## Auto-format and fix lint issues
	uv run ruff format src tests
	uv run ruff check --fix src tests

web-lint:  ## ESLint + TypeScript type check
	cd web && npm run lint && npm run typecheck

web-build:  ## Production build of the review UI
	cd web && npm run build

testset:  ## Fetch the NASA videos (checksummed) and render the synthetic talks (Kokoro voices, ~15 min CPU)
	uv run recut testset download
	uv run recut testset synth

eval-rules:  ## The whole evaluation with the offline rule-based model (the baseline; no API calls)
	RECUT_LLM_PROVIDER=fake uv run recut eval run --name rules --no-judge

eval-real:  ## The real run behind the README (needs OPENROUTER_API_KEY; every model :free; ~150 requests)
	$(REAL) uv run recut eval run --name main
	uv run recut eval summary

summary:  ## Fold results/<run>/items/*.json and the call ledger into results/summary.json
	uv run recut eval summary

docker-build:  ## Build the API and UI images
	docker compose build

docker-up:  ## Everything in Docker: UI :3000, API :8000
	docker compose up --build

clean:  ## Remove caches (keeps results/ and projects)
	rm -rf .pytest_cache .mypy_cache .ruff_cache web/.next
