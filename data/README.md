# data/

四级数据流（§11.2），目录骨架入库，内容不入库：

| 目录 | 内容 | 规则 |
|---|---|---|
| `raw/` | 原始下载/采集 | **只读，永不修改** |
| `external/` | 第三方数据集（许可待确认） | 只读，**不参与训练** |
| `interim/` | 去重、裁切、格式转换中间产物 | 可重建 |
| `processed/` | 训练就绪（图像 + 掩码 + COCO） | 由脚本生成 |
| `manifests/` | 版本化清单（JSONL） | 每次训练绑定一个 |

生成方式：

```bash
python scripts/build_manifests.py --root data/processed/icing --out data/manifests/icing_v1.jsonl
python scripts/split_manifest.py --manifest data/manifests/icing_v1.jsonl
python scripts/validate_dataset.py --manifest data/manifests/icing_v1.jsonl --check-sha
```
