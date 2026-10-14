# -*- coding: utf-8 -*-
"""
================================================================================
共享数据模块：农业物联网 / 作物生长数据集（合成数据）
================================================================================

后续 4 个教学脚本（01~04）全部 import 本模块，使用同一份数据，便于把
「超参数调优 → 特征工程 → 模型优化 → 不平衡处理」串成一条完整的学习主线。

业务背景
--------
某农业物联网试验田部署了大量传感器，记录每块地块一个生长旬（10 天）的
环境指标，需要根据这些指标预测该地块当旬是否发生作物病害（二分类）：

    目标 y = disease：1 = 发生病害（少数类，约占 10%~15%）
                       0 = 健康   （多数类）

为了让特征工程、不平衡处理等章节“有东西可做”，这份数据被故意做“脏”：
    1. 含缺失值（传感器掉线）；
    2. 含异常值（传感器故障导致超量程读数）；
    3. 特征量纲差异巨大（℃、%、小时、mm、pH、mg/kg）；
    4. 含偏态分布特征（降雨量、含氮量）；
    5. 含冗余特征（摄氏温度与华氏温度完全线性相关）；
    6. 含低方差无信息特征（传感器电量几乎恒定）；
    7. 含纯噪声特征；
    8. 类别不平衡（病害样本稀少）。

所有脚本均使用固定随机种子，保证结果可复现。
================================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

# 统一随机种子，保证任何人、任何时候运行结果都一致
RANDOM_STATE = 42

# 特征列名分组（各脚本会反复用到）
TARGET = "disease"
NUMERIC_COLS = [
    "temperature",   # 平均气温，单位 ℃
    "humidity",      # 平均空气相对湿度，单位 %
    "sunshine",      # 日均日照时长，单位 小时
    "rainfall",      # 旬累计降雨量，单位 mm（右偏分布）
    "soil_ph",       # 土壤 pH 值
    "nitrogen",      # 土壤速效氮含量，单位 mg/kg（右偏分布）
]
CATEGORICAL_COLS = [
    "soil_type",     # 土壤质地：砂土 / 壤土 / 黏土（名义变量，无序）
    "irrigation",    # 灌溉方式：滴灌 / 喷灌 / 漫灌（名义变量，无序）
]
# 仅供特征工程章节“找问题”用的特殊列
REDUNDANT_COL = "temperature_f"   # 华氏温度 = 摄氏温度的线性变换（冗余）
LOW_VAR_COL = "battery_level"     # 传感器电量，几乎恒定（低方差）
NOISE_COL = "white_noise"         # 纯高斯噪声（无信息）


def load_agri_data(n_samples: int = 1200, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    生成一份“带脏数据”的农业传感器数据集。

    参数
    ----
    n_samples : int
        样本数量（地块条数）。
    seed : int
        随机种子。

    返回
    ----
    pd.DataFrame，每行一块地块一个生长旬的观测，列含义见模块顶部注释。
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # 1. 生成与病害有真实因果关系的“干净”特征
    # ------------------------------------------------------------------ #
    temperature = rng.normal(loc=25.0, scale=4.0, size=n_samples)      # 气温
    humidity = rng.normal(loc=70.0, scale=12.0, size=n_samples)        # 湿度
    sunshine = rng.normal(loc=6.5, scale=2.0, size=n_samples)          # 日照
    rainfall = rng.exponential(scale=3.0, size=n_samples) * 20.0       # 降雨量（右偏）
    soil_ph = rng.normal(loc=6.5, scale=0.6, size=n_samples)           # pH
    nitrogen = rng.lognormal(mean=3.0, sigma=0.4, size=n_samples)      # 含氮量（右偏）

    soil_type = rng.choice(
        ["砂土", "壤土", "黏土"], size=n_samples, p=[0.30, 0.45, 0.25]
    )
    irrigation = rng.choice(
        ["滴灌", "喷灌", "漫灌"], size=n_samples, p=[0.40, 0.35, 0.25]
    )

    # ------------------------------------------------------------------ #
    # 2. 按农业常识构造病害发生概率（逻辑回归式的“真实世界”）
    #    高湿、多雨、寡照、黏土排水差、漫灌湿度大 → 更容易发病
    #    logit = b0 + Σ bi*xi，p = sigmoid(logit) = 1/(1+e^-logit)
    # ------------------------------------------------------------------ #
    soil_effect = np.where(
        soil_type == "黏土", 0.90, np.where(soil_type == "砂土", -0.30, 0.0)
    )
    irr_effect = np.where(
        irrigation == "漫灌", 0.70, np.where(irrigation == "喷灌", 0.20, -0.20)
    )

    logit = (
        -2.60                         # 截距：控制整体发病率（不平衡）
        + 0.09 * (humidity - 70.0)    # 湿度越高越容易发病
        + 0.012 * (rainfall - 60.0)   # 降雨越多越容易发病
        - 0.28 * (sunshine - 6.5)     # 日照越少越容易发病
        + 0.004 * (nitrogen - 22.0)   # 氮肥偏多、植株徒长也易感病
        + soil_effect
        + irr_effect
        + rng.normal(0.0, 0.5, size=n_samples)  # 不可观测的随机扰动
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    disease = (rng.random(n_samples) < prob).astype(int)

    df = pd.DataFrame(
        {
            "temperature": temperature,
            "humidity": humidity,
            "sunshine": sunshine,
            "rainfall": rainfall,
            "soil_ph": soil_ph,
            "nitrogen": nitrogen,
            "soil_type": soil_type,
            "irrigation": irrigation,
            TARGET: disease,
        }
    )

    # ------------------------------------------------------------------ #
    # 3. 故意加入“教学用脏东西”
    # ------------------------------------------------------------------ #
    # (a) 冗余特征：华氏温度 F = C * 9/5 + 32，与 temperature 完全线性相关
    df[REDUNDANT_COL] = df["temperature"] * 9.0 / 5.0 + 32.0

    # (b) 低方差特征：传感器电量几乎总是 100%（方差约 0.05，供方差过滤演示）
    df[LOW_VAR_COL] = rng.choice(
        [99, 100], size=n_samples, p=[0.05, 0.95]
    )

    # (c) 纯噪声特征：与目标毫无关系
    df[NOISE_COL] = rng.normal(loc=0.0, scale=1.0, size=n_samples)

    # (d) 缺失值：模拟传感器掉线 / 数据未回传
    def _inject_missing(col: str, ratio: float) -> None:
        idx = rng.choice(n_samples, size=int(n_samples * ratio), replace=False)
        df.loc[idx, col] = np.nan

    _inject_missing("temperature", 0.02)
    _inject_missing("humidity", 0.015)
    _inject_missing("nitrogen", 0.04)

    # (e) 异常值：传感器故障导致的超量程 / 不可能读数
    idx = rng.choice(n_samples, size=15, replace=False)
    df.loc[idx, "humidity"] = rng.uniform(120, 180, size=15)   # 湿度不可能 >100%
    idx = rng.choice(n_samples, size=10, replace=False)
    df.loc[idx, "rainfall"] = rng.uniform(300, 500, size=10)   # 极端暴雨（离群）
    idx = rng.choice(n_samples, size=6, replace=False)
    df.loc[idx, "soil_ph"] = rng.choice([0.0, 14.0], size=6)   # pH 越界

    # 调整列顺序，把目标列放到最后
    feature_cols = (
        NUMERIC_COLS
        + [REDUNDANT_COL, LOW_VAR_COL, NOISE_COL]
        + CATEGORICAL_COLS
        + [TARGET]
    )
    return df[feature_cols]


def get_output_dir() -> Path:
    """返回并（如需要则）创建 output 目录，所有图表都保存在这里。"""
    out = Path(__file__).resolve().parent / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_fig(fig, filename: str) -> Path:
    """把 matplotlib 图统一保存到 output/ 目录并打印路径。"""
    path = get_output_dir() / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"[图表已保存] {path}")
    return path


def section(title: str) -> None:
    """打印醒目的分节标题，方便阅读控制台输出。"""
    bar = "=" * 78
    print("\n" + bar)
    print(f"  {title}")
    print(bar)


if __name__ == "__main__":
    # 直接运行本文件时，快速查看数据集长什么样
    data = load_agri_data()
    section("农业传感器数据集速览")
    print(f"样本数: {data.shape[0]}，列数: {data.shape[1]}")
    print("\n前 5 行：")
    print(data.head().to_string())
    print("\n各列数据类型与缺失情况：")
    info = pd.DataFrame(
        {
            "dtype": data.dtypes.astype(str),
            "missing": data.isna().sum(),
            "missing_ratio": (data.isna().mean() * 100).round(2).astype(str) + "%",
        }
    )
    print(info.to_string())
    print("\n标签分布（类别不平衡）：")
    counts = data[TARGET].value_counts().sort_index()
    for label, c in counts.items():
        name = "健康(0)" if label == 0 else "病害(1)"
        print(f"  {name}: {c:4d} 条，占比 {c / len(data):.2%}")
    print("\n数值特征统计（注意异常值导致的 max 越界）：")
    print(data[NUMERIC_COLS].describe().round(2).to_string())
