# grid-ice-vision —— 常用命令入口
# Windows 下请用 `make PY=.venv/Scripts/python.exe` 或在 Git Bash 中运行

PY ?= .venv/bin/python
RUFF ?= .venv/bin/ruff
MYPY ?= .venv/bin/mypy

.PHONY: help install check lint fmt type test test-unit test-integration test-e2e \
        demo validate export serve clean

help:
	@echo "install            安装依赖（含 dev/serve/deploy）"
	@echo "check              ruff + mypy + 全部测试（提交前必跑）"
	@echo "lint / fmt         ruff 检查 / 格式化"
	@echo "type               mypy 类型检查"
	@echo "test               全部测试"
	@echo "demo               合成自检（验证物理量算链路）"
	@echo "validate           校验 manifest 是否泄漏（需 MANIFEST=...）"
	@echo "export             导出 ONNX（需 TASK=seg|precip|detector）"
	@echo "serve              启动服务"

install:
	uv pip install --python $(PY) -e ".[dev,serve,deploy]"

check: lint type test
	@echo "✓ 全部检查通过"

lint:
	$(RUFF) check src scripts tests

fmt:
	$(RUFF) format src scripts tests
	$(RUFF) check src scripts tests --fix

type:
	$(MYPY) src

test:
	$(PY) -m pytest tests -q

test-unit:
	$(PY) -m pytest tests/unit -q

test-integration:
	$(PY) -m pytest tests/integration -q

test-e2e:
	$(PY) -m pytest tests/e2e -q

demo:
	$(PY) scripts/demo_synthetic.py

MANIFEST ?= data/manifests/icing_v1.jsonl
validate:
	$(PY) scripts/validate_dataset.py --manifest $(MANIFEST)

TASK ?= seg
export:
	$(PY) scripts/export_onnx.py --task $(TASK)

serve:
	$(PY) -m grid_ice_vision.cli serve

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
