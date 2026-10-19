"""命令行入口（`giv`）。

命令只做装配：读配置 → 建数据/模型 → 调对应模块。不含算法逻辑。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from grid_ice_vision.common.config import config_hash, get, load_config
from grid_ice_vision.common.logging import get_logger

app = typer.Typer(no_args_is_help=True, help="Grid ice vision CLI")
console = Console()
log = get_logger("cli")


def _cfg(path: str) -> dict[str, Any]:
    cfg = load_config(path)
    log.info("配置 %s（hash=%s）", path, config_hash(cfg)[:12])
    return cfg


def _build_loaders(cfg: dict[str, Any], task: str) -> tuple[Any, Any, Any]:
    from torch.utils.data import DataLoader

    from grid_ice_vision.data.datasets import detection as det_ds
    from grid_ice_vision.data.datasets import segmentation as seg_ds
    from grid_ice_vision.data.datasets import weather as wx_ds
    from grid_ice_vision.data.manifest import load_manifest

    records = load_manifest(get(cfg, "data.manifest"))
    root = get(cfg, "data.root", "data/processed")
    size = int(get(cfg, "data.size", 384))
    bs = int(get(cfg, "train.batch", 4))
    nw = int(get(cfg, "data.num_workers", 0))
    tr = [r for r in records if r.split == "train"]
    va = [r for r in records if r.split == "val"]

    if task == "precip":
        return (
            DataLoader(
                wx_ds.WeatherDataset(tr, root, size, True),
                batch_size=bs,
                shuffle=True,
                num_workers=nw,
                collate_fn=wx_ds.collate,
            ),
            DataLoader(
                wx_ds.WeatherDataset(va, root, size, False),
                batch_size=bs,
                num_workers=nw,
                collate_fn=wx_ds.collate,
            ),
            wx_ds,
        )
    if task == "seg":
        return (
            DataLoader(
                seg_ds.SegmentationDataset(tr, root, size, True),
                batch_size=bs,
                shuffle=True,
                num_workers=nw,
                collate_fn=seg_ds.collate,
            ),
            DataLoader(
                seg_ds.SegmentationDataset(va, root, size, False),
                batch_size=bs,
                num_workers=nw,
                collate_fn=seg_ds.collate,
            ),
            seg_ds,
        )
    if task == "detector":
        coco = get(cfg, "data.coco")
        return (
            DataLoader(
                det_ds.DetectionDataset(tr, root, coco, size, True),
                batch_size=bs,
                shuffle=True,
                num_workers=nw,
                collate_fn=det_ds.collate,
            ),
            DataLoader(
                det_ds.DetectionDataset(va, root, coco, size, False),
                batch_size=bs,
                num_workers=nw,
                collate_fn=det_ds.collate,
            ),
            det_ds,
        )
    raise typer.BadParameter(f"未知任务: {task}")


@app.command()
def train(
    task: str = typer.Argument(..., help="precip | seg | detector"),
    config: str = typer.Option(..., "--config"),
) -> None:
    """训练单个任务。"""
    from grid_ice_vision.features.detection.model import DetectionNet
    from grid_ice_vision.features.ice.model import UNetLite
    from grid_ice_vision.features.precipitation.model import PrecipitationNet
    from grid_ice_vision.training.engine import TrainConfig, fit
    from grid_ice_vision.training.steps import det_step, precip_step, seg_step

    cfg = _cfg(config)
    train_loader, val_loader, _ = _build_loaders(cfg, task)
    backbone = str(get(cfg, "model.backbone", "mobilenetv4_conv_small"))
    pretrained = bool(get(cfg, "model.pretrained", False))
    if task == "precip":
        model: Any = PrecipitationNet(backbone, pretrained)
        step = precip_step
        monitor, mode = "macro_f1_micro", "max"
    elif task == "seg":
        model = UNetLite(backbone, pretrained)
        step = seg_step
        monitor, mode = "dice_ice", "max"
    else:
        model = DetectionNet(backbone, pretrained)
        step = det_step
        monitor, mode = "loss", "min"

    tcfg = TrainConfig(
        epochs=int(get(cfg, "train.epochs", 30)),
        lr=float(get(cfg, "train.lr", 1e-3)),
        weight_decay=float(get(cfg, "train.weight_decay", 1e-4)),
        amp=bool(get(cfg, "train.amp", True)),
        patience=int(get(cfg, "train.patience", 8)),
        out_dir=str(get(cfg, "train.out_dir", "artifacts")),
        task=task,
        version=str(get(cfg, "train.version", "0.1.0")),
    )
    best = fit(
        model,
        train_loader,
        val_loader,
        step,
        tcfg,
        monitor=monitor,
        mode=mode,
        device=str(get(cfg, "device", "cpu")),
    )
    console.print_json(json.dumps(best, ensure_ascii=False))


@app.command()
def evaluate(
    config: str = typer.Option(..., "--config"), split: str = typer.Option("test")
) -> None:
    """在指定 split 上评估并输出指标 JSON。"""
    cfg = _cfg(config)
    task = str(get(cfg, "task", "seg"))
    _, loader, _ = _build_loaders(cfg, task)
    from grid_ice_vision.training.engine import evaluate as run_eval
    from grid_ice_vision.training.steps import det_step, precip_step, seg_step

    ckpt = get(cfg, "model.weights")
    if task == "precip":
        from grid_ice_vision.features.precipitation.model import PrecipitationNet

        model: Any = PrecipitationNet(str(get(cfg, "model.backbone", "mobilenetv4_conv_small")))
        step = precip_step
    elif task == "seg":
        from grid_ice_vision.features.ice.model import UNetLite

        model = UNetLite(str(get(cfg, "model.backbone", "mobilenetv4_conv_small")))
        step = seg_step
    else:
        from grid_ice_vision.features.detection.model import DetectionNet

        model = DetectionNet(str(get(cfg, "model.backbone", "mobilenetv4_conv_small")))
        step = det_step

    if ckpt:
        from grid_ice_vision.training.checkpoint import load_into

        load_into(model, ckpt, strict=False)
    metrics = run_eval(model, loader, step, device=str(get(cfg, "device", "cpu")))
    out = {"split": split, "loss": metrics.loss, **metrics.metrics}
    console.print_json(json.dumps(out, ensure_ascii=False))


@app.command()
def infer(
    image: str = typer.Option(..., "--image"),
    config: str = typer.Option("configs/model/pipeline.yaml", "--config"),
    out: str = typer.Option("out.json", "--out"),
    site_id: str = typer.Option("S00"),
    camera_id: str = typer.Option("C00"),
    d0_mm: float = typer.Option(None, help="导线外径 mm"),
    w0_px: float = typer.Option(None, help="裸导线像素宽度"),
) -> None:
    """单图推理，输出契约 JSON + 证据图。"""
    import cv2

    from grid_ice_vision.application.pipeline import InferencePipeline
    from grid_ice_vision.common.schemas import CalibInput, SourceInfo

    cfg = _cfg(config)
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    if img is None:
        raise typer.BadParameter(f"无法读取图像: {image}")
    pipe = InferencePipeline.from_config(cfg, device=str(get(cfg, "device", "cpu")))
    src = SourceInfo(site_id=site_id, camera_id=camera_id, timestamp="2026-01-15T08:30:12+08:00")
    result = pipe.run(img, Path(image).stem, src, CalibInput(d0_mm=d0_mm, w0_px=w0_px))
    Path(out).write_text(json.dumps(result.to_json_dict(), ensure_ascii=False, indent=2), "utf-8")
    console.print(f"[green]已写出 {out}[/green]")


@app.command()
def export(
    task: str = typer.Option("seg", help="precip | seg | detector"),
    config: str = typer.Option("configs/deploy/onnx_cpu.yaml", "--config"),
    out: str = typer.Option(None, "--out"),
) -> None:
    """导出 ONNX 并做数值一致性校验。"""
    from grid_ice_vision.deployment.export import export_and_verify
    from grid_ice_vision.features.detection.model import DetectionNet
    from grid_ice_vision.features.ice.model import UNetLite
    from grid_ice_vision.features.precipitation.model import PrecipitationNet

    cfg = _cfg(config)
    backbone = str(get(cfg, "model.backbone", "mobilenetv4_conv_small"))
    size = int(get(cfg, "deploy.input_size", 512))
    if task == "precip":
        model: Any = PrecipitationNet(backbone)
        keys = ("weather_logits", "intensity_logits")
    elif task == "seg":
        model = UNetLite(backbone)
        keys = None
    else:
        model = DetectionNet(backbone)
        keys = None
    target = out or f"artifacts/onnx/{task}.onnx"
    res = export_and_verify(model, target, (1, 3, size, size), keys=keys)
    console.print_json(json.dumps(res, ensure_ascii=False))


@app.command()
def serve(
    config: str = typer.Option("configs/model/pipeline.yaml", "--config"),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
) -> None:
    """启动 FastAPI 服务。"""
    import uvicorn

    from grid_ice_vision.service.api import create_app

    uvicorn.run(create_app(config=_cfg(config)), host=host, port=port)


@app.command()
def validate(manifest: str = typer.Option(..., "--manifest")) -> None:
    """校验 manifest 的分组划分是否泄漏。"""
    from grid_ice_vision.data.manifest import assert_no_leakage, load_manifest, summarize

    records = load_manifest(manifest)
    try:
        assert_no_leakage(records)
    except AssertionError as exc:
        console.print(f"[red]✗ {exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print_json(json.dumps(summarize(records), ensure_ascii=False))
    console.print("[green]✓ 无分组泄漏[/green]")


@app.command()
def quality(image: str = typer.Option(..., "--image")) -> None:
    """只跑质量门控，用于现场排查摄像头。"""
    import cv2

    from grid_ice_vision.features.quality.gate import assess_quality

    img = cv2.imread(image, cv2.IMREAD_COLOR)
    if img is None:
        raise typer.BadParameter(f"无法读取图像: {image}")
    console.print_json(json.dumps(assess_quality(img).model_dump(mode="json"), ensure_ascii=False))


if __name__ == "__main__":
    app()
