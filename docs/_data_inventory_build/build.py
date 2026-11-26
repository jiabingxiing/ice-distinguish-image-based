# -*- coding: utf-8 -*-
"""生成《数据需求清单_v1.0.xlsx》：3 个 sheet（明细台账 / 缺口汇总 / 许可台账摘要）。"""

try:
    import openpyxl
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "openpyxl>=3.1.0"])
    import openpyxl

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.formatting.rule import FormulaRule
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.properties import PageSetupProperties
from openpyxl.utils import get_column_letter


def xl_color(css_hex: str) -> str:
    value = css_hex.removeprefix("#").upper()
    if len(value) != 6:
        raise ValueError(f"Expected #RRGGBB, got: {css_hex}")
    return "FF" + value


XL_PRIMARY = xl_color("#4472C4")
XL_PRIMARY_DARK = xl_color("#2F5597")
XL_LIGHT = xl_color("#D9E2F3")
XL_WHITE = xl_color("#FFFFFF")
XL_BORDER = xl_color("#BFBFBF")
XL_TEXT = xl_color("#1A1A1A")
XL_RED_BG = xl_color("#FFC7CE")
XL_RED_FG = xl_color("#9C0006")
XL_YEL_BG = xl_color("#FFEB9C")
XL_YEL_FG = xl_color("#9C6500")
XL_GRN_BG = xl_color("#C6EFCE")
XL_GRN_FG = xl_color("#006100")
XL_GRAY_BG = xl_color("#F2F2F2")

thin_side = Side(style="thin", color=XL_BORDER)
BORDER_ALL = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)

OUT = "D:/NWKYYSJZX/2026年工作/1019 基于图像的电网覆冰、降水类型及程度识别算法/数据需求清单_v1.0.xlsx"
TITLE = "数据可行性需求调研 · 数据需求清单 v1.0"

# ---------------------------------------------------------------- 数据定义
HEADERS = [
    "编号", "数据组", "数据项", "服务的任务", "关键字段/格式",
    "采集精度期望", "标注精度期望", "数量期望",
    "当前状态", "优先级", "获取途径", "责任方", "阻塞影响", "备注",
]
WIDTHS = [7, 19, 27, 22, 30, 30, 20, 24, 9, 8, 22, 15, 34, 30]
WRAP_COLS = {"C", "D", "E", "F", "G", "H", "K", "M", "N"}

A = "A 图像本体"
B = "B 设备台账与元数据"
C = "C 标定数据"
D = "D 标注数据"
E = "E 气象与环境观测"
F = "F 时序与实测真值"
G = "G 业务规则数据"
H = "H 合成与增强素材"

ROWS = [
    # ---------------- A 图像本体
    ("A1", A, "输电线路固定监控可见光图像", "T2/T3/T4/T5 主输入", "JPG；image_id、source.ptz",
     "≥1920×1080；导线在图中≥8px；JPEG质量≥90", "—", "P5现场精标≥300张；≥3站点×≥2机位",
     "空缺", "P0", "运检部门调取监控历史录像", "运检部门",
     "无同源域数据，公开集指标不可外推", "选点须先按导线像素宽≥8px筛机位"),
    ("A2", A, "无人机巡检4K图像", "T3 精细掩码；标定补充", "JPG；EXIF焦距；asset_id",
     "短边≥2000px；含侧视+俯视两种视角", "—", "≥500张",
     "空缺", "P2", "无人机班组作业影像", "设备管理",
     "固定机位视野死角无法覆盖", "非必需，可后补"),
    ("A3", A, "覆冰过程连续帧序列", "T5 EKF；告警去抖", "同机位+同预置位+同变倍",
     "间隔≤30min；连续≥6帧；时间戳≤1s", "—", "≥20个完整过程×≥24帧",
     "空缺", "P1", "监控平台历史录像导出", "运检部门",
     "alarm_level 无时序去抖，告警闪烁", "连续性比数量更重要"),
    ("A4", A, "晴天无冰基准图（同预置位同变倍）", "L3标定W0_px；质量阈值按机位标定", "JPG；同预置位",
     "同预置位同变倍；覆盖不同光照时段", "—", "≥5张/机位",
     "空缺", "P1", "历史录像按预置位抽帧", "运检部门",
     "覆盖率无法输出，只能停在 visual_only", "最便宜的一项，不必专门出差"),
    ("A5", A, "夜间/红外/低照度图像", "夜间可用性评估", "可见光+红外配对",
     "同机位同时刻配对", "—", "可选",
     "空缺", "P2", "现有夜视设备", "设备管理",
     "夜间直接判 unusable", "边界声明明确不承诺夜间性能"),
    ("A6", A, "镜头污染·结冰·遮挡负样本", "T0 质量门控；LensIcingTracker", "连续帧+遮挡区域标注",
     "同机位连续≥10帧", "遮挡区域 mask/bbox", "≥30组",
     "空缺", "P1", "监控平台异常帧筛查", "运检部门",
     "lens_icing_suspected 误报率高", "需连续帧才有效"),
    ("A7", A, "困难场景样本（逆光/雨雾雪/抖动/飞掠雪花）", "Set-C 鲁棒性；场景分层采样", "JPG + scene_group",
     "按 normal/adverse/device_bg 分层，每组≥100张", "—", "≥300张",
     "部分", "P2", "算子A合成+真实抽帧", "算法团队",
     "鲁棒性无法评估", "合成数据不得进 test"),
    ("A8", A, "多站点多机位覆盖", "分组防泄漏；跨站点泛化", "site_id、camera_id",
     "≥3站点×≥2机位；地理气候差异明显", "—", "≥6机位",
     "空缺", "P1", "运检部门协调多站", "运检部门",
     "分组键为构造值，测试集可能泄漏", "至少留1个站点完全不参与训练"),

    # ---------------- B 设备台账
    ("B1", B, "导线/地线外径 D0", "T5 α=D0/W0_px", "d0_mm（CalibInput）",
     "±0.5mm", "—", "覆盖全部被测线路型号",
     "空缺", "P0", "PMS导出/设计图纸", "运检部门",
     "d_radial、d_eq、M 全部 null", "全项目性价比最高的一项"),
    ("B2", B, "分裂导线根数·间距·子导线径", "测量口径；conductor 定义", "根数、间距mm、子导线径",
     "间距±5mm", "—", "覆盖被测型号",
     "空缺", "P2", "设计图纸", "设计院",
     "分裂导线按单根还是整体测口径不清", "影响覆盖率计算口径"),
    ("B3", B, "设计冰厚 d_design", "T5 γ=d_eq/d_design", "design_ice_mm",
     "按规程取值（常见10/15/20/30mm）", "—", "逐线路逐段",
     "空缺", "P0", "设计文件", "运检/设计部门",
     "gamma 恒 null", "预警矩阵少一条判据"),
    ("B4", B, "刚性参考金具物理尺寸", "L1标定；reference_fitting", "d_ref_mm（耐张线夹/防振锤/盘径/钢帽）",
     "±1mm；须明确测量部位", "—", "覆盖视野内金具型号",
     "空缺", "P0", "设备台账+现场量测", "设备管理",
     "L1标定不可用", "必须写清量的是哪个部位"),
    ("B5", B, "绝缘子型号·片数·伞裙间距", "η_bridge 分母", "型号、片数、间距mm",
     "片数准确；间距±2mm", "—", "覆盖被测型号",
     "空缺", "P1", "设备台账", "运检部门",
     "bridge_ratio 不可算", "—"),
    ("B6", B, "杆塔型号·呼高·档距", "几何上下文；弧垂判断", "型号、呼高m、档距m",
     "呼高±0.5m；档距±5m", "—", "逐基",
     "空缺", "P2", "PMS导出", "运检部门",
     "误检抑制变弱", "非阻塞项"),
    ("B7", B, "机位台账（预置位/安装高/俯仰/朝向）", "分组键；L3标定前提", "site_id、camera_id、预置位、安装高、俯仰、朝向",
     "预置位唯一；安装高±0.5m；俯仰±1°", "—", "逐机位",
     "空缺", "P1", "设备管理台账", "设备管理",
     "跨帧对齐失败；分组键为构造值", "一切采集工作的前提"),
    ("B8", B, "PTZ逐帧读数（zoom/pan/tilt）", "L2标定 α_t=α0·(f0/ft)", "zoom、pan、tilt",
     "zoom±2%；pan/tilt±0.5°", "—", "逐帧随图",
     "空缺", "P1", "监控平台PTZ日志", "设备管理",
     "球机变焦后厚度偏差2-3倍", "pipeline 当前硬编码 None"),
    ("B9", B, "相机内参与畸变系数", "亚像素测量误差校正；双目视差", "fx、fy、cx、cy、k1、k2",
     "焦距±1%；畸变残差<0.5px", "—", "逐机位",
     "空缺", "P2", "相机标定", "设备管理",
     "图像边缘区域系统偏差", "可选"),
    ("B10", B, "时间戳（带时区+NTP同步）", "EKF Δt；气象配对；事件归并", "timestamp ISO8601+时区",
     "同机位帧间≤1s；跨源≤60s", "—", "逐帧",
     "部分", "P1", "平台日志/EXIF", "设备管理",
     "时区错8h会把隔夜增长算成瞬时跳变", "TTPLA 当前为 unknown"),
    ("B11", B, "采集参数（曝光/增益/白平衡/焦距）", "跨机位一致性；质量阈值标定", "EXIF 或 sidecar",
     "逐帧记录", "—", "逐帧",
     "空缺", "P2", "EXIF", "设备管理",
     "质量门控阈值无法按机位标定", "block_highfreq_energy 随分辨率变化"),

    # ---------------- C 标定数据
    ("C1", C, "L1刚性金具标定对", "fuse_scale_references 最高优先级源", "d_ref_mm + w_ref_px",
     "w_ref_px量测≤0.5px；d_ref_mm±1mm", "量测误差≤0.5px", "≥100对/机位，覆盖不同变倍",
     "空缺", "P0", "现场标定", "设备管理/现场班组",
     "measured 状态不可达，厚度永远出不来", "需覆盖不同变倍档"),
    ("C2", C, "L2 PTZ变倍-尺度查找表/双目视差标定", "变倍后 α 修正", "zoom档位→α",
     "插值误差≤3%", "—", "≥8个常用zoom档",
     "空缺", "P1", "PTZ标定", "设备管理",
     "球机不可用", "解决2-3倍系统性偏差"),
    ("C3", C, "L3基线档案 W0_px", "baseline_archive 标定源", "W0_px（per 预置位）",
     "W0_px±0.5px", "—", "每机位每预置位1组",
     "空缺", "P1", "晴天图像建档", "算法团队+运检",
     "只能输出 visual_only", "依赖 A4"),
    ("C4", C, "标定一致性校验样本", "三源一致性检验（max/min>1.20 即降级）", "同时具备L1/L2/L3 三源α",
     "三源差异≤20%", "—", "≥20组",
     "空缺", "P2", "标定过程留存", "算法团队",
     "无法验证一致性检验是否有效", "差异>20%降级为 visual_only"),

    # ---------------- D 标注数据
    ("D1", D, "ROI检测标注（4类bbox）", "T2；assets[]", "bbox+类别；reference_fitting附d_ref_mm/w_ref_px",
     "—", "IoU≥0.7；8-20px细长目标单独规则", "insulator_string≥500实例；reference_fitting≥300实例",
     "部分", "P0", "内部标注+prelabel预标注", "标注团队",
     "桥接率不可算；L1标定零正样本", "conductor 10102/tower 968 已有"),
    ("D2", D, "覆冰语义分割掩码（4类像素）", "T3；覆盖率/厚度/桥接率", "4类mask PNG；必须含exposed_asset",
     "—", "边界像素误差≤2px", "真实精标≥300-500；合成2000-8000",
     "空缺", "P0", "标注团队+合成算子B", "标注团队",
     "覆盖率分母为0 → 等级恒 G0", "现有仅8张合成"),
    ("D3", D, "覆冰类型5类", "T4；决定密度ρ", "ice_type",
     "—", "κ≥0.75", "≥300精标",
     "空缺", "P0", "专家标注", "标注团队+专家",
     "密度只能取900，d_eq折算失真", "单类<50并入mixed"),
    ("D4", D, "视觉等级 G0-G4", "T4 CORAL；visual_grade", "G0-G4",
     "—", "QWK≥0.70", "≥300",
     "空缺", "P1", "专家按 rubric 定级", "专家",
     "等级恒 G0", "覆盖率阈值 3/15/30/50%"),
    ("D5", D, "天气8类多标签", "T1；precipitation.labels", "weather_visual[]",
     "—", "κ≥0.75", "已有4869条",
     "已到位", "已具备", "WEAPD（CC0-1.0）", "—",
     "无需再采集", "实测 Macro-F1 0.891"),
    ("D6", D, "降水强度 L/M/H", "T1 CORAL强度头", "intensity",
     "—", "QWK≥0.55", "≥1000张带强度标注",
     "空缺", "P0", "在现有WEAPD上补标", "标注团队",
     "CORAL头无监督，恒输出 unknown", "最便宜的高价值标注"),
    ("D7", D, "冰凌桥接标注", "η_bridge；III/IV级判据", "icicle_bridge mask + 伞裙间隙",
     "—", "边界≤2px", "≥200张绝缘子串",
     "空缺", "P1", "标注团队", "标注团队",
     "η_bridge 不可算", "依赖 D1 的 insulator_string"),
    ("D8", D, "困难负样本库（反光/角钢/积雪/云）", "抑制误检 FP", "负样本图+无覆冰依据",
     "—", "需明确无覆冰判定依据", "≥500",
     "部分", "P2", "TTPLA裁切+现场抽帧", "算法团队",
     "晴天场景 FP 偏高", "现有为航拍域，需补现场"),

    # ---------------- E 气象观测
    ("E1", E, "逐时气温 T_amb", "weather_sensor；prior.py；FiLM门控", "t_amb_c",
     "±0.5℃；时间对齐≤10min", "—", "与图像同期覆盖",
     "空缺", "P1", "气象API/线路观测站", "气象/运检",
     "门控关闭，退化为纯视觉", "—"),
    ("E2", E, "相对湿度 RH", "同 E1；Makkonen虚拟气象配对", "rh（0-100）",
     "±5%", "—", "同 E1",
     "空缺", "P1", "同 E1", "气象/运检",
     "同 E1", "—"),
    ("E3", E, "风速风向", "同 E1；rime 迎风面判据", "v_wind_ms + 风向",
     "±1m/s；风向±15°", "—", "同 E1",
     "空缺", "P1", "同 E1", "气象/运检",
     "rime 判据失效", "—"),
    ("E4", E, "气压 P_atm", "p_atm_hpa；空气密度修正", "p_atm_hpa",
     "±1hPa", "—", "同 E1",
     "空缺", "P1", "同 E1", "气象/运检",
     "影响<1%，可忽略", "优先级可下调"),
    ("E5", E, "降水相态观测 precip_confirmed", "唯一能确认冻雨的物理真值", "precip_confirmed",
     "人工观测记录；对齐≤10min", "—", "与覆冰过程同期",
     "空缺", "P2", "人工观测/气象站", "气象部门",
     "冻雨无法确认", "双体系原则的“确认”侧"),
    ("E6", E, "降水量/降水强度", "强度标签外部校验", "降水量 mm/h",
     "±0.1mm", "—", "同 E5",
     "空缺", "P2", "气象站", "气象部门",
     "强度标签无外部校验", "不要求与图像强度一致"),
    ("E7", E, "覆冰观测站/拉力/称重数据", "三源互校；差异>30%降级", "等值冰厚或载荷时序",
     "±2mm 或载荷±5%", "—", "≥1个站点覆盖被测线路",
     "空缺", "P2", "覆冰观测站", "运检",
     "视觉厚度无人校验", "连续3帧同向偏差>0.20冻结机位"),
    ("E8", E, "微地形/海拔/迎风面", "站点分层；先验合理性", "海拔、坡向",
     "海拔±10m；坡向±10°", "—", "逐机位",
     "空缺", "P2", "GIS数据", "运检",
     "rime 分布无法解释", "—"),
    ("E9", E, "历史冰灾记录/故障工单", "真实 weather_event_id；事件级标签", "事件起止、厚度记录",
     "事件起止±1h；厚度±2mm", "—", "近5年",
     "空缺", "P2", "工单系统", "业务方",
     "分组键无真实过程号", "—"),

    # ---------------- F 时序与真值
    ("F1", F, "逐帧 d_eq 历史序列", "EKF冷启动与收敛；增长率统计", "时间序列 d_eq",
     "同机位同时刻；≥6帧/过程", "—", "≥20个过程",
     "空缺", "P2", "系统输出积累", "算法团队",
     "EKF 冷启动慢", "依赖 A3"),
    ("F2", F, "人工实测冰厚（游标卡尺/冰尺）", "唯一真值；MAPE验收；标定校核", "d_measured_mm",
     "±1mm；须记录测量位置与时刻", "±1mm", "≥50组（P5.3 验收门槛）",
     "空缺", "P1", "除冰/特巡作业时顺采", "运检班组",
     "厚度 MAPE 不可验证", "决定能否宣称生产可用"),
    ("F3", F, "除冰/融冰事件记录", "验证厚度归零与状态切换", "事件时刻",
     "事件时刻±10min", "—", "覆盖被测线路",
     "空缺", "P2", "工单系统", "运检",
     "状态切换无法验证", "—"),
    ("F4", F, "力学传感器时序（倾角/应力/称重）", "三源互校；系统性漂移检测", "倾角/应力/称重时序",
     "采样≤10min；载荷±5%", "—", "≥1站点",
     "空缺", "P2", "在线监测装置", "设备管理",
     "系统性标定漂移无法发现", "连续3帧同向偏差>0.20冻结机位"),

    # ---------------- G 业务规则
    ("G1", G, "本地区覆冰预警规程 I-IV 阈值", "alarm.yaml 校准", "d_eq / γ / η_bridge 阈值",
     "按本地区现行规程，逐电压等级", "—", "覆盖被测电压等级",
     "空缺", "P1", "现行规程文件", "业务方/安监",
     "只能用占位值，不敢真正下发指令", "当前 5/10/20mm 为工程占位值"),
    ("G2", G, "设备缺陷/隐患台账", "风险加权；ROI检测优先级", "缺陷类型、等级",
     "台账完整性", "—", "被测线路",
     "空缺", "P2", "PMS", "运检",
     "风险加权缺失", "—"),
    ("G3", G, "运维处置与人工复核记录", "误报推翻率；阈值调优", "处置结论",
     "逐条记录", "—", "试运行期全量",
     "空缺", "P2", "工单系统", "业务方",
     "误报率无法评估", "推翻率>30%触发信任危机"),

    # ---------------- H 合成素材
    ("H1", H, "真实 glaze/rime 纹理块", "算子B纹理贴合（Poisson融合）", "≥256×256 PNG",
     "—", "—", "glaze/rime 各≥100块",
     "空缺", "P2", "从真实覆冰图裁切", "算法团队",
     "合成冰外观不真实", "data/raw/textures 缺失"),
    ("H2", H, "裸导线 mask 素材", "算子B沿法向外扩的基底", "mask PNG",
     "—", "边界≤2px", "≥200",
     "部分", "P2", "TTPLA 提取", "算法团队",
     "合成基底不足", "可从现有数据提取"),
    ("H3", H, "大气散射参数标定素材", "算子A的β/A参数校准；L/M/H标签", "(clean, hazy) 配对",
     "—", "—", "≥50组",
     "空缺", "P2", "自采/合成", "算法团队",
     "强度标签不准", "—"),
    ("H4", H, "合成数据隔离与零误差真值校验样本", "is_synthetic隔离；几何回归测试", "jpg+mask+truth.jsonl",
     "真值误差为0（几何构造）", "—", "≥100",
     "部分", "P2", "算子B", "算法团队",
     "合成泄漏进 test 的风险", "现有8张；单张生成9.4s，需先优化"),

    # ---------------- 许可台账
]

LICENSE_ROWS = [
    ("TTPLA", "航拍杆塔/导线 COCO 多边形，1,234张 3840×2160，11,070实例", "ROI 预训练；W0_px 来源", "Apache-2.0", "confirmed", "域为航拍俯视，只能预训练；官方 split 为图像级随机划分"),
    ("WEAPD", "6,862张 11类（本机落地6,877张）", "天气分类 P1", "CC0-1.0", "confirmed", "权威原始出处（Harvard Dataverse）"),
    ("Kaggle Weather Dataset", "6,862张 11类（jehanbhathena 镜像）", "同 WEAPD", "CC0: Public Domain", "confirmed", "与本机 WEAPD 重复，不要二次下载"),
    ("宝鸡输电线路覆冰图像", "覆冰观测精灵图像，17,114,951 B", "现场域定性检查、示例图", "CC BY 4.0", "confirmed", "样本量小，只作定性检查，不作训练集/指标来源"),
    ("TTPLA 派生镜像", "1536短边 parquet，975.9 MB", "备选通路", "other（上游 Apache-2.0）", "pending", "仅在原始 tarball 缺标注时考虑"),
    ("CPLID", "绝缘子正常/缺陷", "绝缘子检测预训练", "学术用途", "restricted", "商用需另行确认"),
    ("InsPLAD", "10,607张 UAV 图像 / 17类资产 / 6类缺陷", "绝缘子/防振锤/间隔棒检测", "CC BY-NC 3.0", "restricted", "禁商用；不得用于商用交付权重"),
    ("DTL-IceNet", "覆冰样例+分割代码（130.9MB+146.1kB）", "架构参考、样例验证", "页面未标注", "pending", "需向通讯作者书面确认商用授权"),
    ("PLAIR-AW", "Haze/Rain/Snow × 3数据集", "退化图像 + L/M/H 标签", "待核实", "pending", "仓库只有数据集与基线，无合成脚本"),
    ("浙江220kV覆冰预测气象集", "多源气象观测（不含图像）", "气象侧对照、先验合理性", "CC BY-NC-SA 4.0", "restricted", "NC 条款禁止商用"),
    ("CDD-11 / Snow100K / LOL", "通用退化数据集", "复原模块（P4 才用）", "各自不同", "pending", "—"),
    ("NWPUYRCC2 / Sentinel-LakeIce", "遥感河冰/湖冰分割", "仅纹理先验", "待核实", "restricted", "域差过大，不混训"),
    ("CoInfra", "路侧协同感知（冻雨）", "域适应消融", "CC-BY-NC-4.0", "restricted", "NC 条款禁止商用"),
    ("firc-dataset", "Gitee/FIRC 电力行业数据交易平台", "不作主训练集", "不明", "rejected", "明确声明不对模型精度作保证"),
    ("自采 / 现场", "电网监控图像", "域适应、验收（Set-D）", "内部授权", "pending", "暂无"),
]

GROUPS = [A, B, C, D, E, F, G, H]

wb = Workbook()

# ================================================== Sheet1 数据需求清单
ws = wb.active
ws.title = "数据需求清单"
ws.sheet_properties.tabColor = XL_PRIMARY
ws.sheet_view.showGridLines = False

HDR_ROW = 3
FIRST = 4
LAST = FIRST + len(ROWS) - 1
LASTCOL = get_column_letter(len(HEADERS))

# 标题区
ws.merge_cells(f"A1:{LASTCOL}1")
c = ws["A1"]
c.value = TITLE
c.font = Font(name="微软雅黑", size=15, bold=True, color=XL_WHITE)
c.fill = PatternFill("solid", fgColor=XL_PRIMARY_DARK)
c.alignment = Alignment(horizontal="center", vertical="center")
ws.row_dimensions[1].height = 30

ws.merge_cells(f"A2:{LASTCOL}2")
c = ws["A2"]
c.value = ("共 51 项 ｜ 优先级：P0 阻塞（缺失则核心字段恒 null）· P1 影响验收 · P2 增强 · 已具备 ｜ "
           "状态：已到位 / 部分 / 空缺 ｜ 数据组 A-H 对应报告第 3 章")
c.font = Font(name="微软雅黑", size=9, color=XL_TEXT)
c.fill = PatternFill("solid", fgColor=XL_LIGHT)
c.alignment = Alignment(horizontal="left", vertical="center", indent=1)
ws.row_dimensions[2].height = 22

# 表头
for i, name in enumerate(HEADERS, start=1):
    cell = ws.cell(row=HDR_ROW, column=i, value=name)
    cell.font = Font(name="微软雅黑", size=10, bold=True, color=XL_WHITE)
    cell.fill = PatternFill("solid", fgColor=XL_PRIMARY)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell.border = BORDER_ALL
ws.row_dimensions[HDR_ROW].height = 34

# 数据
for r, row in enumerate(ROWS, start=FIRST):
    for i, val in enumerate(row, start=1):
        cell = ws.cell(row=r, column=i, value=val)
        col = get_column_letter(i)
        cell.font = Font(name="微软雅黑", size=9, color=XL_TEXT)
        cell.border = BORDER_ALL
        if col in WRAP_COLS:
            cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
        else:
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[r].height = 44

# 列宽
for i, w in enumerate(WIDTHS, start=1):
    ws.column_dimensions[get_column_letter(i)].width = w

# 冻结 + 筛选
ws.freeze_panes = f"D{FIRST}"
ws.auto_filter.ref = f"A{HDR_ROW}:{LASTCOL}{LAST}"

# 数据验证
dv_status = DataValidation(type="list", formula1='"已到位,部分,空缺"', allow_blank=False)
dv_prio = DataValidation(type="list", formula1='"P0,P1,P2,已具备"', allow_blank=False)
ws.add_data_validation(dv_status)
ws.add_data_validation(dv_prio)
dv_status.add(f"I{FIRST}:I{LAST}")
dv_prio.add(f"J{FIRST}:J{LAST}")

# 条件格式：状态列
rng = f"I{FIRST}:I{LAST}"
ws.conditional_formatting.add(rng, FormulaRule(
    formula=[f'$I{FIRST}="空缺"'], stopIfTrue=False,
    fill=PatternFill("solid", fgColor=XL_RED_BG), font=Font(color=XL_RED_FG, bold=True)))
ws.conditional_formatting.add(rng, FormulaRule(
    formula=[f'$I{FIRST}="部分"'], stopIfTrue=False,
    fill=PatternFill("solid", fgColor=XL_YEL_BG), font=Font(color=XL_YEL_FG, bold=True)))
ws.conditional_formatting.add(rng, FormulaRule(
    formula=[f'$I{FIRST}="已到位"'], stopIfTrue=False,
    fill=PatternFill("solid", fgColor=XL_GRN_BG), font=Font(color=XL_GRN_FG, bold=True)))

# 条件格式：优先级列
rng = f"J{FIRST}:J{LAST}"
ws.conditional_formatting.add(rng, FormulaRule(
    formula=[f'$J{FIRST}="P0"'], stopIfTrue=False,
    fill=PatternFill("solid", fgColor=XL_RED_BG), font=Font(color=XL_RED_FG, bold=True)))
ws.conditional_formatting.add(rng, FormulaRule(
    formula=[f'$J{FIRST}="P1"'], stopIfTrue=False,
    fill=PatternFill("solid", fgColor=XL_YEL_BG), font=Font(color=XL_YEL_FG, bold=True)))
ws.conditional_formatting.add(rng, FormulaRule(
    formula=[f'$J{FIRST}="已具备"'], stopIfTrue=False,
    fill=PatternFill("solid", fgColor=XL_GRN_BG), font=Font(color=XL_GRN_FG, bold=True)))

# ================================================== Sheet2 缺口汇总
ws2 = wb.create_sheet("缺口汇总")
ws2.sheet_properties.tabColor = XL_PRIMARY_DARK
ws2.sheet_view.showGridLines = False

ws2.merge_cells("A1:H1")
c = ws2["A1"]
c.value = "缺口汇总（按数据组统计，引用“数据需求清单”）"
c.font = Font(name="微软雅黑", size=14, bold=True, color=XL_WHITE)
c.fill = PatternFill("solid", fgColor=XL_PRIMARY_DARK)
c.alignment = Alignment(horizontal="center", vertical="center")
ws2.row_dimensions[1].height = 28

SUM_HDR = ["数据组", "项数", "已到位", "部分", "空缺", "P0 阻塞", "P1 影响验收", "P2 增强"]
for i, name in enumerate(SUM_HDR, start=1):
    cell = ws2.cell(row=2, column=i, value=name)
    cell.font = Font(name="微软雅黑", size=10, bold=True, color=XL_WHITE)
    cell.fill = PatternFill("solid", fgColor=XL_PRIMARY)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell.border = BORDER_ALL
ws2.row_dimensions[2].height = 30

S_FIRST = 3
S_LAST = S_FIRST + len(GROUPS) - 1
TOTAL_ROW = S_LAST + 1
DET = "数据需求清单"

for k, grp in enumerate(GROUPS):
    r = S_FIRST + k
    ws2.cell(row=r, column=1, value=grp)
    ws2.cell(row=r, column=2, value=f"=COUNTIF({DET}!$B:$B,$A{r})")
    ws2.cell(row=r, column=3, value=f'=COUNTIFS({DET}!$B:$B,$A{r},{DET}!$I:$I,"已到位")')
    ws2.cell(row=r, column=4, value=f'=COUNTIFS({DET}!$B:$B,$A{r},{DET}!$I:$I,"部分")')
    ws2.cell(row=r, column=5, value=f'=COUNTIFS({DET}!$B:$B,$A{r},{DET}!$I:$I,"空缺")')
    ws2.cell(row=r, column=6, value=f'=COUNTIFS({DET}!$B:$B,$A{r},{DET}!$J:$J,"P0")')
    ws2.cell(row=r, column=7, value=f'=COUNTIFS({DET}!$B:$B,$A{r},{DET}!$J:$J,"P1")')
    ws2.cell(row=r, column=8, value=f'=COUNTIFS({DET}!$B:$B,$A{r},{DET}!$J:$J,"P2")')

for r in range(S_FIRST, S_LAST + 1):
    for cidx in range(1, 9):
        cell = ws2.cell(row=r, column=cidx)
        cell.border = BORDER_ALL
        if cidx == 1:
            cell.font = Font(name="微软雅黑", size=10, color=XL_TEXT)
            cell.alignment = Alignment(horizontal="left", vertical="center", indent=1)
        else:
            cell.font = Font(name="微软雅黑", size=10, color=XL_TEXT)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.number_format = "0"
    ws2.row_dimensions[r].height = 22

# 合计行
ws2.cell(row=TOTAL_ROW, column=1, value="合计")
for cidx in range(2, 9):
    col = get_column_letter(cidx)
    ws2.cell(row=TOTAL_ROW, column=cidx, value=f"=SUM({col}{S_FIRST}:{col}{S_LAST})")
for cidx in range(1, 9):
    cell = ws2.cell(row=TOTAL_ROW, column=cidx)
    cell.font = Font(name="微软雅黑", size=10, bold=True, color=XL_WHITE)
    cell.fill = PatternFill("solid", fgColor=XL_PRIMARY_DARK)
    cell.border = BORDER_ALL
    cell.alignment = Alignment(
        horizontal="left" if cidx == 1 else "center", vertical="center", indent=1 if cidx == 1 else 0)
    if cidx > 1:
        cell.number_format = "0"
ws2.row_dimensions[TOTAL_ROW].height = 24

# 条件格式：P0 列数值 >0 标红
ws2.conditional_formatting.add(
    f"F{S_FIRST}:F{S_LAST}",
    FormulaRule(formula=[f"$F{S_FIRST}>0"], fill=PatternFill("solid", fgColor=XL_RED_BG),
                font=Font(color=XL_RED_FG, bold=True)))
ws2.conditional_formatting.add(
    f"G{S_FIRST}:G{S_LAST}",
    FormulaRule(formula=[f"$G{S_FIRST}>0"], fill=PatternFill("solid", fgColor=XL_YEL_BG),
                font=Font(color=XL_YEL_FG, bold=True)))

for i, w in enumerate([30, 8, 9, 8, 8, 10, 12, 10], start=1):
    ws2.column_dimensions[get_column_letter(i)].width = w

# 说明区
note_row = TOTAL_ROW + 2
ws2.merge_cells(f"A{note_row}:H{note_row}")
c = ws2.cell(row=note_row, column=1)
c.value = ("读法：P0 项缺失会让核心输出字段恒为 null（不是“精度差一点”）；P1 项影响验收指标达标；"
           "P2 项为增强项，不影响主链路。当前 51 项中仅 1 项（D5 天气8类）已完全具备。")
c.font = Font(name="微软雅黑", size=9, color=XL_TEXT)
c.fill = PatternFill("solid", fgColor=XL_GRAY_BG)
c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True, indent=1)
ws2.row_dimensions[note_row].height = 34

# ================================================== Sheet3 许可台账摘要
ws3 = wb.create_sheet("许可台账摘要")
ws3.sheet_properties.tabColor = xl_color("#70AD47")
ws3.sheet_view.showGridLines = False

ws3.merge_cells("A1:F1")
c = ws3["A1"]
c.value = "数据与资源许可台账摘要（摘自 grid-ice-vision/docs/data_license_ledger.md）"
c.font = Font(name="微软雅黑", size=14, bold=True, color=XL_WHITE)
c.fill = PatternFill("solid", fgColor=XL_PRIMARY_DARK)
c.alignment = Alignment(horizontal="center", vertical="center")
ws3.row_dimensions[1].height = 28

LIC_HDR = ["资源", "内容", "用途", "许可", "状态", "备注 / 待闭环"]
for i, name in enumerate(LIC_HDR, start=1):
    cell = ws3.cell(row=2, column=i, value=name)
    cell.font = Font(name="微软雅黑", size=10, bold=True, color=XL_WHITE)
    cell.fill = PatternFill("solid", fgColor=XL_PRIMARY)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell.border = BORDER_ALL
ws3.row_dimensions[2].height = 28

L_FIRST = 3
for k, row in enumerate(LICENSE_ROWS):
    r = L_FIRST + k
    for i, val in enumerate(row, start=1):
        cell = ws3.cell(row=r, column=i, value=val)
        cell.font = Font(name="微软雅黑", size=9, color=XL_TEXT)
        cell.border = BORDER_ALL
        cell.alignment = Alignment(horizontal="left" if i in (1, 2, 3, 6) else "center",
                                   vertical="top", wrap_text=True)
    ws3.row_dimensions[r].height = 40

L_LAST = L_FIRST + len(LICENSE_ROWS) - 1
for i, w in enumerate([26, 44, 28, 20, 12, 46], start=1):
    ws3.column_dimensions[get_column_letter(i)].width = w

ws3.freeze_panes = "A3"
ws3.auto_filter.ref = f"A2:F{L_LAST}"

# 状态着色
rng3 = f"E{L_FIRST}:E{L_LAST}"
ws3.conditional_formatting.add(rng3, FormulaRule(
    formula=[f'$E{L_FIRST}="confirmed"'], fill=PatternFill("solid", fgColor=XL_GRN_BG),
    font=Font(color=XL_GRN_FG, bold=True)))
ws3.conditional_formatting.add(rng3, FormulaRule(
    formula=[f'$E{L_FIRST}="pending"'], fill=PatternFill("solid", fgColor=XL_YEL_BG),
    font=Font(color=XL_YEL_FG, bold=True)))
ws3.conditional_formatting.add(rng3, FormulaRule(
    formula=[f'$E{L_FIRST}="restricted"'], fill=PatternFill("solid", fgColor=xl_color("#FFE699")),
    font=Font(color=xl_color("#7F6000"), bold=True)))
ws3.conditional_formatting.add(rng3, FormulaRule(
    formula=[f'$E{L_FIRST}="rejected"'], fill=PatternFill("solid", fgColor=XL_RED_BG),
    font=Font(color=XL_RED_FG, bold=True)))

note3 = L_LAST + 2
ws3.merge_cells(f"A{note3}:F{note3}")
c = ws3.cell(row=note3, column=1)
c.value = ("硬门槛：许可未确认的资源只能进 data/external/（只读、不参与训练）；"
           "进入 data/processed/ 的必须是 confirmed 或 restricted 且已确认使用场景合规。")
c.font = Font(name="微软雅黑", size=9, color=XL_TEXT)
c.fill = PatternFill("solid", fgColor=XL_GRAY_BG)
c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True, indent=1)
ws3.row_dimensions[note3].height = 30

# 打印设置
for sheet in (ws, ws2, ws3):
    sheet.page_setup.orientation = "landscape"
    sheet.page_setup.fitToWidth = 1
    sheet.page_setup.fitToHeight = 0
    sheet.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)

wb.properties.title = TITLE
wb.calculation.fullCalcOnLoad = True
wb.save(OUT)
print(f"OK rows={len(ROWS)} first={FIRST} last={LAST} sheets={wb.sheetnames}")
