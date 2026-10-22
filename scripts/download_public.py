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
        "航拍杆塔/导线 COCO 多边形，1234 张 3840x2160（split 905/109/220）",
        "ROI 预训练、W0_px 来源",
        "Apache-2.0",
        "confirmed",
        "2026-10-20 实读 LICENSE；图像本体只在 Google Drive（本机不可达），走 hf 镜像",
    ),
    Resource(
        "TTPLA（hf 镜像）",
        "https://hf-mirror.com/datasets/hjjiao/TTPLA",
        "data_original_size.tar.gz，4,301.6 MB",
        "TTPLA 图像的可用通路",
        "镜像未标（上游 Apache-2.0）",
        "confirmed",
        "2026-10-20 实测可达；解压后 1242 对 jpg/json（labelme v4.2.7）",
    ),
    Resource(
        "宝鸡输电线路覆冰图像",
        "https://www.scidb.cn/detail?dataSetId=983d28a746c046f0909817f4baa4ccd5",
        "17,114,951 B / 1 文件（DOI 10.57760/sciencedb.35688）",
        "现场域定性检查、示例图",
        "CC BY 4.0",
        "confirmed",
        "2026-01-22 发布；样本量小，只作定性检查，不作训练集/指标来源",
    ),
    Resource(
        "浙江 220kV 覆冰预测气象集",
        "https://doi.org/10.57760/sciencedb.nyjs.00009",
        "多源气象观测（无图像）",
        "气象侧对照、先验合理性",
        "CC BY-NC-SA 4.0",
        "restricted",
        "NC 条款禁止商用",
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
        "无人机电力资产（10607 张 / 17 类）",
        "绝缘子/防振锤/间隔棒检测",
        "CC BY-NC 3.0",
        "restricted",
        "禁商用；可做研究性评测，不得用于商用交付权重",
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
        "6862 张 11 类（原始出处为 WEAPD，DOI 10.7910/DVN/M8JQCR）",
        "天气分类 P1",
        "CC0-1.0",
        "confirmed",
        "2026-10-20 核实；frost/glaze/rime 多为植物特写，域外",
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
