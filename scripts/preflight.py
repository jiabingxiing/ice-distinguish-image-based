"""换盘/换机自检（计划 §7.6 第 1 步）。

检出三类「位置无关」违规：
1. **陈旧 editable 安装** —— `.venv` 内 `_editable_impl_*.pth` 指向旧盘符/旧父目录，
   这是本仓实测发生过的 `0xfffffffe` 启动失败根因；
2. **导入来源错误** —— 实际 imported `grid_ice_vision` 不在当前项目内；
3. **外部数据缺失** —— `paths.external()` 无法解析 SCUT 等外部目录。

退出码：0 全部通过；1 存在任一问题。供 CI 与迁移演练直接调用。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from grid_ice_vision.common import paths  # noqa: E402


def find_stale_pth(venv: Path) -> list[tuple[Path, str, bool]]:
    """返回 [(pth 文件, 内容行, 是否指向当前项目)]，仅含可疑项。"""
    out: list[tuple[Path, str, bool]] = []
    site_dirs = list(venv.glob("Lib/site-packages")) + list(
        venv.glob("lib/python*/site-packages")
    )
    for site in site_dirs:
        for pth in site.glob("_editable_impl_*.pth"):
            for line in pth.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith(("import ", "#")):
                    continue
                target = Path(line)
                resolved = target.resolve() if target.exists() else None
                ok = resolved is not None and (
                    paths.PROJ in resolved.parents or resolved == paths.PROJ / "src"
                )
                if not ok:
                    out.append((pth, line, False))
    return out


def check_import() -> tuple[bool, str]:
    code = (
        "import grid_ice_vision, sys; "
        "print(grid_ice_vision.__file__)"
    )
    # 必须显式指定 utf-8：Windows 子进程默认按 ANSI(GBK) 解码，
    # 项目路径含中文时 stdout 读取会抛 UnicodeDecodeError。
    # 同时强制子进程用 utf-8 输出，否则子进程按 cp936 写、父进程按 utf-8 读，
    # 中文路径会被替换成 U+FFFD，导致误判“导入自项目外副本”。
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        r = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            env=child_env,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return False, f"导入检查无法执行: {exc}"
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()[-1:] or ["(无输出)"]
        return False, f"import grid_ice_vision 失败: {tail}"
    loc = Path((r.stdout or "").strip())
    try:
        loc.resolve().relative_to(paths.PROJ.resolve())
    except ValueError:
        return False, f"导入自项目外副本: {loc}（应为 {paths.PROJ / 'src'} 之下）"
    return True, str(loc)


def check_external() -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for key in ("SCUT_ROOT",):
        try:
            p = paths.external(key)
            results.append({"key": key, "ok": True, "path": str(p)})
        except paths.ExternalDataMissingError as exc:
            results.append({"key": key, "ok": False, "error": str(exc)})
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="位置无关自检：陈旧 editable 安装 / 导入来源 / 外部数据"
    )
    ap.add_argument("--venv", default=str(PROJ / ".venv"), help="虚拟环境目录")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出")
    ap.add_argument("--skip-import", action="store_true", help="跳过子进程导入检查")
    args = ap.parse_args(argv)

    # Windows 默认控制台是 CP936(GBK)：路径里出现无法编码的字符（如 U+FFFD）时
    # print 会抛 UnicodeEncodeError，让本来只是“报告一个问题”的自检自己崩掉。
    # 这里把 stdout 的重定向错误策略改成 backslashreplace，坏字符不再中断自检。
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(errors="backslashreplace")  # type: ignore[union-attr]

    problems: list[str] = []
    report: dict[str, object] = {"project_root": str(paths.PROJ)}

    print(f"[preflight] 项目根  : {paths.PROJ}")

    stale = find_stale_pth(Path(args.venv))
    report["stale_editable"] = [{"pth": str(a), "target": b} for a, b, _ in stale]
    if stale:
        for pth, line, _ in stale:
            problems.append(
                f"陈旧 editable 安装：{pth.name} → {line}\n"
                f"    修复：uv sync --reinstall-package grid-ice-vision"
            )
        print(f"[preflight] editable: FAIL（{len(stale)} 条指向旧位置）")
    else:
        print("[preflight] editable: OK")

    if not args.skip_import:
        ok, msg = check_import()
        report["import"] = {"ok": ok, "detail": msg}
        if ok:
            print(f"[preflight] import  : OK → {msg}")
        else:
            problems.append(f"导入来源异常：{msg}")
            print(f"[preflight] import  : FAIL → {msg}")

    ext = check_external()
    report["external"] = ext
    for e in ext:
        if e["ok"]:
            print(f"[preflight] external: OK  {e['key']} → {e['path']}")
        else:
            problems.append(f"外部数据缺失 {e['key']}：{e['error']}")
            print(f"[preflight] external: FAIL {e['key']}")

    report["ok"] = not problems
    report["problems"] = problems

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif problems:
        print(f"\n发现 {len(problems)} 个问题：")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")

    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())