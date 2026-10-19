# justfile —— make 的替代入口（Windows 友好）
set shell := ["bash", "-c"]

py := ".venv/bin/python"

default:
    @just --list

install:
    uv pip install --python {{py}} -e ".[dev,serve,deploy]"

check: lint type test

lint:
    .venv/bin/ruff check src scripts tests

fmt:
    .venv/bin/ruff format src scripts tests
    .venv/bin/ruff check src scripts tests --fix

type:
    .venv/bin/mypy src

test:
    {{py}} -m pytest tests -q

demo:
    {{py}} scripts/demo_synthetic.py

validate manifest="data/manifests/icing_v1.jsonl":
    {{py}} scripts/validate_dataset.py --manifest {{manifest}}

export task="seg":
    {{py}} scripts/export_onnx.py --task {{task}}

serve:
    {{py}} -m grid_ice_vision.cli serve
