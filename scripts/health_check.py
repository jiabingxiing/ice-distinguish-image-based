"""工作区与跨盘迁移健康巡检（计划 §5.1 T03 / §7.5 必跑第四步）。

检查项：
1. 位置无关环境：editable `.pth` 与 `grid_ice_vision` 导入路径是否指向当前 `PROJ`；
2. 外部数据源：`SCUT_ROOT` 可达性、CSV 与年份子目录完整可读；
3. 数据契约与清单：`data/splits/*.jsonl` 与 `data/manifests/scut_ice_type_v1.jsonl`
   存在性、无泄漏校验（`assert_no_leakage`）、可选 `--check-sha` 核验；
4. 产物与缓存目录：`artifacts/` 可读写状态、`hf-cache` / `torch-cache` / 模型产物计数；
5. 异盘备份与哈希核验（可选 `--backup-dir` / `--backup-sha`）：核对备份与工作区是否位于
   不同物理磁盘、备份时效（`--max-backup-age-days`）及 SHA-256 清单一致性。

退出码：0 全部通过；1 存在阻断性失败。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from grid_ice_vision.common import paths  # noqa: E402
from grid_ice_vision.data.manifest import (  # noqa: E402
    assert_no_leakage,
    load_manifest,
    sha256_file,
)


def _check_preflight() -> dict[str, Any]:
    """复用 scripts/preflight.py 的核心位置无关自检。"""
    import preflight

    stale = preflight.find_stale_pth(PROJ / ".venv")
    imp_ok, imp_detail = preflight.check_import()
    return {
        "ok": (not stale) and imp_ok,
        "stale_editable_count": len(stale),
        "import_ok": imp_ok,
        "import_detail": imp_detail,
    }


def _check_scut_root() -> dict[str, Any]:
    try:
        root = paths.scut_root(must_exist=True)
        csv_path = paths.scut_csv(must_exist=True)
        year_dirs = [str(y) for y in range(2014, 2019) if (root / str(y)).is_dir()]
        sample_img = next((root / year_dirs[0]).glob("*.jpg"), None) if year_dirs else None
        readable = False
        if sample_img is not None:
            with sample_img.open("rb") as f:
                readable = len(f.read(16)) > 0
        return {
            "ok": len(year_dirs) == 5 and readable,
            "root": str(root),
            "csv": csv_path.name,
            "year_dirs": year_dirs,
            "sample_readable": readable,
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _check_manifests(check_sha: bool = False) -> dict[str, Any]:
    targets = [
        PROJ / "data" / "splits" / "scut_monitor_v2.jsonl",
        PROJ / "data" / "splits" / "scut_temporal_v2.jsonl",
        PROJ / "data" / "manifests" / "scut_ice_type_v1.jsonl",
    ]
    scut_dir: Path | None = None
    with contextlib.suppress(Exception):
        scut_dir = paths.scut_root(must_exist=True)

    items: list[dict[str, Any]] = []
    all_ok = True
    for p in targets:
        rel = p.relative_to(PROJ).as_posix()
        if not p.exists():
            # scut_ice_type_v1.jsonl 在跑完 build_scut_manifest.py 前可暂缺，splits 必须存在
            required = "splits" in p.parts
            if required:
                all_ok = False
            items.append({"path": rel, "exists": False, "required": required, "ok": not required})
            continue
        try:
            recs = load_manifest(p)
            assert_no_leakage(recs)
            sha_populated = sum(1 for r in recs if r.sha256)
            sha_bad = 0
            if check_sha and scut_dir is not None and sha_populated > 0:
                for r in recs:
                    if r.sha256:
                        fp = scut_dir / r.path
                        if not fp.exists() or sha256_file(fp) != r.sha256:
                            sha_bad += 1
            item_ok = len(recs) > 0 and sha_bad == 0
            all_ok = all_ok and item_ok
            items.append(
                {
                    "path": rel,
                    "exists": True,
                    "count": len(recs),
                    "sha256_populated": sha_populated,
                    "sha256_mismatches": sha_bad,
                    "leakage_ok": True,
                    "ok": item_ok,
                }
            )
        except Exception as exc:
            all_ok = False
            items.append({"path": rel, "exists": True, "ok": False, "error": str(exc)})
    return {"ok": all_ok, "files": items}


def _check_artifacts() -> dict[str, Any]:
    art = PROJ / "artifacts"
    writable = False
    try:
        art.mkdir(parents=True, exist_ok=True)
        probe = art / ".health_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        writable = True
    except OSError:
        writable = False

    feat_dir = art / "features"
    npz_files = sorted(p.name for p in feat_dir.glob("*.npz")) if feat_dir.is_dir() else []
    ice_dir = art / "ice_type"
    ice_versions = (
        sorted(p.name for p in ice_dir.iterdir() if p.is_dir()) if ice_dir.is_dir() else []
    )
    onnx_dir = art / "onnx"
    onnx_files = sorted(p.name for p in onnx_dir.glob("*.onnx")) if onnx_dir.is_dir() else []
    return {
        "ok": writable,
        "writable": writable,
        "feature_caches": npz_files,
        "ice_type_versions": ice_versions,
        "onnx_models": onnx_files,
    }


def _same_physical_drive(a: Path, b: Path) -> bool:
    """尽力判断是否在同一磁盘分区/挂载点（Windows 看盘符，POSIX 看 st_dev）。"""
    ra = a.resolve()
    rb = b.resolve()
    if os.name == "nt":
        return ra.drive.upper() == rb.drive.upper() and bool(ra.drive)
    return ra.stat().st_dev == rb.stat().st_dev


def _check_backup(
    backup_dir: Path | None,
    sha_file: Path | None,
    max_age_days: float,
) -> dict[str, Any]:
    if backup_dir is None and sha_file is None:
        return {
            "ok": True,
            "configured": False,
            "note": "未指定 --backup-dir / --backup-sha，跳过异盘备份核验",
        }

    issues: list[str] = []
    info: dict[str, Any] = {"configured": True}
    if backup_dir is not None:
        if not backup_dir.exists():
            issues.append(f"备份目录不存在: {backup_dir}")
        else:
            same_drive = _same_physical_drive(PROJ, backup_dir)
            info["backup_dir"] = str(backup_dir.resolve())
            info["different_drive"] = not same_drive
            if same_drive:
                issues.append(f"备份目录与工作区位于同一磁盘/分区（{backup_dir}）")
            mtimes = [p.stat().st_mtime for p in backup_dir.rglob("*") if p.is_file()]
            if not mtimes:
                issues.append("备份目录为空，无备份文件")
            else:
                age_days = (time.time() - max(mtimes)) / 86400.0
                info["latest_age_days"] = round(age_days, 2)
                if age_days > max_age_days:
                    issues.append(f"最近备份距今 {age_days:.1f} 天，超过阈值 {max_age_days:.1f} 天")

    if sha_file is not None:
        if not sha_file.exists():
            issues.append(f"SHA-256 校验清单不存在: {sha_file}")
        else:
            base = backup_dir if backup_dir is not None else sha_file.parent
            verified = 0
            mismatched: list[str] = []
            for raw in sha_file.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                expected_sha, rel = parts[0].lower(), parts[1].lstrip("*").strip()
                target = base / rel
                if not target.exists():
                    mismatched.append(f"missing:{rel}")
                elif sha256_file(target) != expected_sha:
                    mismatched.append(f"sha_mismatch:{rel}")
                else:
                    verified += 1
            info["sha_verified"] = verified
            info["sha_mismatches"] = mismatched
            if mismatched:
                issues.append(f"SHA-256 校验失败 {len(mismatched)} 项")

    info["issues"] = issues
    info["ok"] = not issues
    return info


def run_health_check(
    *,
    check_sha: bool = False,
    backup_dir: Path | None = None,
    backup_sha: Path | None = None,
    max_backup_age_days: float = 7.0,
) -> dict[str, Any]:
    preflight_res = _check_preflight()
    scut_res = _check_scut_root()
    manifest_res = _check_manifests(check_sha=check_sha)
    artifacts_res = _check_artifacts()
    backup_res = _check_backup(backup_dir, backup_sha, max_backup_age_days)

    overall_ok = (
        preflight_res["ok"]
        and scut_res["ok"]
        and manifest_res["ok"]
        and artifacts_res["ok"]
        and backup_res["ok"]
    )
    return {
        "ok": overall_ok,
        "project_root": str(PROJ),
        "preflight": preflight_res,
        "scut": scut_res,
        "manifests": manifest_res,
        "artifacts": artifacts_res,
        "backup": backup_res,
    }


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(errors="backslashreplace")  # type: ignore[union-attr]

    ap = argparse.ArgumentParser(description="工作区磁盘、环境、清单与异盘备份健康巡检")
    ap.add_argument("--check-sha", action="store_true", help="逐文件核对 manifest 中的 sha256")
    ap.add_argument("--backup-dir", default=None, help="异盘备份目录（可选）")
    ap.add_argument("--backup-sha", default=None, help="异盘备份 SHA-256 清单文件（可选）")
    ap.add_argument("--max-backup-age-days", type=float, default=7.0)
    ap.add_argument("--json", action="store_true", help="以 JSON 格式输出完整报告")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(PROJ / "scripts"))
    report = run_health_check(
        check_sha=args.check_sha,
        backup_dir=Path(args.backup_dir) if args.backup_dir else None,
        backup_sha=Path(args.backup_sha) if args.backup_sha else None,
        max_backup_age_days=args.max_backup_age_days,
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"[health] project_root : {report['project_root']}")
        print(f"[health] preflight    : {'OK' if report['preflight']['ok'] else 'FAIL'}")
        print(f"[health] scut         : {'OK' if report['scut']['ok'] else 'FAIL'}")
        print(f"[health] manifests    : {'OK' if report['manifests']['ok'] else 'FAIL'}")
        print(
            f"[health] artifacts    : {'OK' if report['artifacts']['ok'] else 'FAIL'} "
            f"(npz={len(report['artifacts']['feature_caches'])}, "
            f"ice_type={report['artifacts']['ice_type_versions']}, "
            f"onnx={report['artifacts']['onnx_models']})"
        )
        bkp = report["backup"]
        if bkp["configured"]:
            issues_msg = bkp.get("issues", [])
            print(f"[health] backup       : {'OK' if bkp['ok'] else 'FAIL'} ({issues_msg})")
        else:
            print(f"[health] backup       : SKIP ({bkp['note']})")
        print(f"[health] overall      : {'OK' if report['ok'] else 'FAIL'}")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
