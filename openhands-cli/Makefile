export UV_EXCLUDE_NEWER := 7 days

.PHONY: help install install-dev test format clean run

# Default target
help:
	@echo "OpenHands CLI - Available commands:"
	@echo "  install                  - Install the package"
	@echo "  install-dev              - Install with development dependencies"
	@echo "  test                     - Run tests"
	@echo "  format                   - Format code with ruff"
	@echo "  clean                    - Clean build artifacts"
	@echo "  run                      - Run the CLI"

# Install the package
install:
	uv sync --exclude-newer "7 days"

# Install with development dependencies
install-dev:
	uv sync --exclude-newer "7 days" --group dev

# Run tests
test:
	uv run pytest

# Format code
format:
	uv run ruff format openhands_cli/

# Clean build artifacts
clean:
	rm -rf .venv/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Run the CLI
run:
	uv run openhands

# Install UV if not present
install-uv:
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "Installing UV..."; \
		installer=$$(mktemp) || exit $$?; \
		trap 'rm -f "$$installer"' EXIT; \
		curl -LsSf https://astral.sh/uv/0.12.5/install.sh -o "$$installer" && \
		printf '%s  %s\n' '504511fbbbd811aeaba6738abc79408956b6c7da0ca35437b3dcc24a41efc111' "$$installer" | shasum -a 256 -c - && \
		sh "$$installer"; \
	else \
		echo "UV is already installed"; \
	fi
