"""公开资源台账与可用性检查（§11.1、附录 A）。

**不自动下载**：多数资源许可未确认，自动下载会造成「默认合规」的错觉。
本脚本只做两件事：打印台账、可选地做 HEAD 可达性检查。

许可状态取值：confirmed / pending / restricted / rejected
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Resource:
    name: str
    url: str
    content: str
    use: str
    license: str
    status: str
    note: str = ""


LEDGER: tuple[Resource, ...] = (
    Resource(
        "TTPLA",
        "https://github.com/R3ab/ttpla_dataset",
        "航拍杆塔/导线 COCO 多边形",
        "ROI 预训练、W0_px 来源",
        "仓库未见 LICENSE",
        "pending",
        "域为航拍俯视，与杆塔侧视差异大，只能预训练",
    ),
    Resource(
        "CPLID",
        "https://github.com/InsulatorData/InsulatorDataSet",
        "绝缘子正常/缺陷",
        "绝缘子检测预训练",
        "学术用途",
        "restricted",
        "仅学术；商用需另行确认",
    ),
    Resource(
        "InsPLAD",
        "https://github.com/andreluizbvs/InsPLAD",
        "无人机电力资产",
        "ROI 预训练",
        "待核实",
        "pending",
    ),
    Resource(
        "DTL-IceNet",
        "https://zenodo.org/records/15718305",
        "覆冰样例 + 分割代码",
        "架构参考、样例验证",
        "页面未标注许可",
        "pending",
        "不得作为唯一主训练集；商用前须书面确认",
    ),
    Resource(
        "PLAIR-AW",
        "https://github.com/ntuhubin/PLAIR-AW",
        "Haze/Rain/Snow × 3 数据集",
        "退化图像 + L/M/H 标签",
        "待核实",
        "pending",
        "仓库只有数据集与基线，**无合成脚本**",
    ),
    Resource(
        "Kaggle Weather Dataset",
        "https://www.kaggle.com/datasets/jehanbhathena/weather-dataset",
        "6862 张 11 类（含 glaze/rime/frost/hail）",
        "天气分类预训练",
        "随 Kaggle 数据集条款",
        "pending",
        "原始代码仓库为 HOG+SVM，非深度模型",
    ),
    Resource(
        "CDD-11 / Snow100K / LOL",
        "https://github.com/JudicialHaze/CDD-11",
        "通用退化",
        "复原模块（P4 才用）",
        "各自不同",
        "pending",
    ),
    Resource(
        "firc-dataset",
        "https://gitee.com/FIRC/power-industry-dataset",
        "电力行业数据集（交易平台）",
        "不作主训练集",
        "不明",
        "rejected",
        "明确声明不对模型精度作保证",
    ),
    Resource(
        "NWPUYRCC2",
        "https://github.com/Junjue-Wang/NWPUYRCC2",
        "遥感河冰分割",
        "仅纹理先验",
        "待核实",
        "restricted",
        "域差过大，不混训",
    ),
    Resource(
        "CoInfra",
        "https://github.com/CoInfra/CoInfra",
        "路侧协同感知",
        "域适应消融",
        "CC-BY-NC-4.0",
        "restricted",
        "NC 条款禁止商用",
    ),
)


def check(url: str, timeout: float = 8.0) -> str:
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "giv-ledger/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return f"HTTP {resp.status}"
    except Exception as exc:
        return f"unreachable: {type(exc).__name__}"


def main() -> int:
    ap = argparse.ArgumentParser(description="资源台账")
    ap.add_argument("--check", action="store_true", help="做可达性检查（HEAD）")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = []
    for r in LEDGER:
        row = asdict(r)
        if args.check:
            row["reachable"] = check(r.url)
        rows.append(row)

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0

    print(f"{'资源':<24}{'许可':<18}{'状态':<12}{'用途'}")
    print("-" * 100)
    for r in rows:
        print(f"{r.name:<24}{r.license:<18}{r.status:<12}{r.use}")
    n_confirmed = sum(1 for r in rows if r.status == "confirmed")
    print(f"\n已确认许可：{n_confirmed} / {len(rows)}")
    print("⚠️ 许可未确认的资源只能进 data/external/（只读、不参与训练）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
