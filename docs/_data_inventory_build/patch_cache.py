# -*- coding: utf-8 -*-
"""为公式单元格注入缓存值（<v>），使其在 pandas / 预览器中可直接读取。

做法：按明细表真实数据重算汇总表应有的数值，再把这些值作为 <v> 写进
缺口汇总 sheet 的 XML。公式本身保留，Excel/WPS 打开后仍可重算。
"""
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import openpyxl

OUT = Path("D:/NWKYYSJZX/2026年工作/1019 基于图像的电网覆冰、降水类型及程度识别算法/数据需求清单_v1.0.xlsx")

# ---------- 1. 从明细表重算真实值
wb = openpyxl.load_workbook(OUT)
det = wb["数据需求清单"]
FIRST, LAST = 4, 54

groups = []
counts = defaultdict(lambda: defaultdict(int))
for r in range(FIRST, LAST + 1):
    grp = det.cell(row=r, column=2).value
    status = det.cell(row=r, column=9).value
    prio = det.cell(row=r, column=10).value
    if grp not in counts:
        groups.append(grp)
    counts[grp]["total"] += 1
    counts[grp][status] += 1
    counts[grp][prio] += 1

assert len(groups) == 8, groups
assert sum(counts[g]["total"] for g in groups) == 51

# ---------- 2. 生成 cell -> value 映射
values = {}
S_FIRST = 3
for k, g in enumerate(groups):
    r = S_FIRST + k
    c = counts[g]
    row_vals = [c["total"], c["已到位"], c["部分"], c["空缺"], c["P0"], c["P1"], c["P2"]]
    for i, v in enumerate(row_vals, start=2):
        col = openpyxl.utils.get_column_letter(i)
        values[f"{col}{r}"] = v

TOTAL_ROW = S_FIRST + len(groups)
totals = [sum(counts[g][k] for g in groups) for k in ("total", "已到位", "部分", "空缺", "P0", "P1", "P2")]
for i, v in enumerate(totals, start=2):
    col = openpyxl.utils.get_column_letter(i)
    values[f"{col}{TOTAL_ROW}"] = v

print("groups:", {g: dict(counts[g]) for g in groups})
print("totals:", totals)

# ---------- 3. 定位缺口汇总 sheet 的 XML
tmp = OUT.with_suffix(".tmp.xlsx")
patched = 0
with zipfile.ZipFile(OUT, "r") as zin:
    names = zin.namelist()
    target = None
    for n in names:
        if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"):
            if b"COUNTIF" in zin.read(n):
                target = n
                break
    assert target, "未找到含公式的工作表"
    print("target sheet:", target)

    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == target:
                xml = data.decode("utf-8")
                for ref, val in values.items():
                    # openpyxl 已写出 <f>...</f><v />，把空 <v /> 替换为 <v>数值</v>
                    pattern = re.compile(
                        r'(<c r="%s"(?![0-9])[^>]*>)(.*?)(</c>)' % re.escape(ref),
                        re.S,
                    )

                    def repl(m):
                        global patched
                        head, body, close = m.group(1), m.group(2), m.group(3)
                        if f"<v>{val}</v>" in body:
                            return m.group(0)
                        if "<v />" in body:
                            body = body.replace("<v />", f"<v>{val}</v>", 1)
                        elif "<v/>" in body:
                            body = body.replace("<v/>", f"<v>{val}</v>", 1)
                        else:
                            body = body + f"<v>{val}</v>"
                        patched += 1
                        return f"{head}{body}{close}"

                    xml = pattern.sub(repl, xml)
                data = xml.encode("utf-8")
            zout.writestr(item, data)

shutil.move(str(tmp), str(OUT))
print(f"patched cells: {patched} / expected {len(values)}")
assert patched == len(values), "缓存值注入不完整"
