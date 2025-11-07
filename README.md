# Project Setup

A simple guide to get started with this project using UV.

## Clone repository

```
git clone https://github.com/rockerBOO/qwen3-vl-test
cd qwen3-vl-test
```

## Installation

Install UV using the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Windows, use PowerShell:

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Setup

Sync all project dependencies:

```bash
uv sync
```

This will read your `pyproject.toml` and install all required dependencies in a virtual environment.

## Usage

Run the main script:

```bash
uv run main.py
```

That's it! UV will automatically handle the virtual environment and execute your script.
