# justfile —— 常用命令入口（本机没有 make）
#
# 两处本机实测的坑，改这个文件前先看：
#   1) uv 生成的 .venv/Scripts/mypy.exe 是 trampoline，直接调用报
#      "uv trampoline failed to canonicalize script path"；`python -m mypy` 正常。
#   2) just 用 bash 作 shell（本机是 WSL 的 bash），Windows 的 venv 可执行文件
#      在 bash 里必须带 .exe 后缀，否则报 "No such file or directory"。
set shell := ["bash", "-c"]

b := if os_family() == "windows" { ".venv/Scripts" } else { ".venv/bin" }
ext := if os_family() == "windows" { ".exe" } else { "" }
py := b + "/python" + ext

default:
    @just --list

install:
    uv pip install --python {{py}} -e ".[dev,serve,deploy]"

check: lint type test

lint:
    {{py}} -m ruff check src scripts tests

fmt:
    {{py}} -m ruff format src scripts tests
    {{py}} -m ruff check src scripts tests --fix

type:
    {{py}} -m mypy src

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