.PHONY: help lint-check lint-fix format-check format-fix

help:
	@echo "Available targets:"
	@echo "  lint-check    Check for linting errors"
	@echo "  lint-fix      Fix auto-fixable linting errors"
	@echo "  format-check  Check formatting without modifying files"
	@echo "  format-fix    Format code"

lint-check:
	uvx ruff check .

lint-fix:
	uvx ruff check --fix .

format-check:
	uvx ruff format --check .

format-fix:
	uvx ruff format .
