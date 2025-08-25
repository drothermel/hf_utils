# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Development
- `us` runs `uv sync` - Install all dependencies
- `pt` runs `uv run pytest` - Run tests with pytest

### Code Quality (Manual Only)
**DO NOT RUN THESE COMMANDS** - The user runs these manually to avoid interfering with git commit setup:
- `lint` runs `uv run ruff check --fix .` - Lint code with ruff and apply autofixes where possible
- `ft` runs `uv run ruff format .` - Format code with ruff  
- `mp` runs `uv run mypy src/` - Type check with mypy

### Configuration Philosophy
- Compatible with datadec library integration
- HuggingFace-focused utilities and tools
- Clean separation from main ddpred library
- Uses assertions over exceptions for validation (performance requirement)

## Project Purpose

This repository contains HuggingFace-specific utilities extracted from the main ddpred library:
- DataDecide dataset processing
- HuggingFace model/checkpoint management
- Branch and weight utilities
- Integration scripts and notebooks

The goal is clean separation of concerns, allowing ddpred to focus on ML prediction while this library handles HF-specific functionality.