# -*- coding: utf-8 -*-
"""
================================================================================
第 02 章 · 进阶能力（2/4）：特征工程 (Feature Engineering)
================================================================================

业界常说“数据和特征决定了机器学习的上限，模型只是在逼近这个上限”。
特征工程就是利用领域知识和数据变换，从原始数据中制造出更容易被模型学习的
输入。本章覆盖表格数据最常用的全套手法：

  1. 缺失值处理   ：均值/中位数/众数填充、KNN 填充、缺失指示变量；
  2. 异常值处理   ：Z-score 法、IQR 箱线图法、缩尾截断 (winsorize)；
  3. 类别特征编码 ：Label / Ordinal / One-Hot 编码及适用模型；
  4. 数值变换     ：标准化、归一化、Robust 缩放、对数变换（治偏态）；
  5. 特征构造     ：业务交互特征、多项式特征、分箱 (binning)；
  6. 特征选择     ：过滤式 Filter / 包裹式 Wrapper / 嵌入式 Embedded；
  7. 降维         ：PCA 主成分分析（原理 + 解释方差 + 二维可视化）；
  8. 流水线整合   ：ColumnTransformer + Pipeline，可复现、防泄漏。

★ 贯穿全章的第一原则——防数据泄漏 (data leakage)：
  填充中位数、缩放均值方差、PCA 方向等所有“从数据中统计出来的量”，
  都【只能在训练集上 fit】，再去 transform 验证集/测试集。否则等于提前
  偷看了测试集分布，交叉验证分数虚高，上线必然掉点。

运行方式：python 02_特征工程.py
================================================================================
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)
from sklearn.decomposition import PCA

from agri_data import (
    CATEGORICAL_COLS,
    LOW_VAR_COL,
    NOISE_COL,
    NUMERIC_COLS,
    RANDOM_STATE,
    REDUNDANT_COL,
    TARGET,
    load_agri_data,
    save_fig,
    section,
)

# 全部数值列（含故意加入的冗余列、低方差列、噪声列）
ALL_NUMERIC = NUMERIC_COLS + [REDUNDANT_COL, LOW_VAR_COL, NOISE_COL]


# ============================================================================ #
# 自定义转换器：缩尾截断 Winsorizer
# ---------------------------------------------------------------------------- #
# fit 时在训练集上学习每列的下/上分位数界限，transform 时把超出界限的值
# “压”到界限上（不是删除样本）。写成 fit/transform 形式才能放进 Pipeline，
# 保证交叉验证每折只用该折训练集的分位数，避免泄漏。
# ============================================================================ #
class Winsorizer(BaseEstimator, TransformerMixin):#继承两个 sklearn 的基类
    #BaseEstimator 是什么？
    #它是 sklearn 所有"估计器"（estimator）的基类。继承它后，你的类自动获得：
    #get_params()：返回所有超参数。
    #set_params()：设置超参数。
    #能配合 GridSearchCV、Pipeline 等使用

    #TransformerMixin 是什么？
    #它是 sklearn "转换器"（transformer）的混入类（mixin）。继承它后，你的类自动获得：
    #fit_transform(X)：等价于 fit(X).transform(X)
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile#lower_quantile=0.01：下界分位数，默认 1%
        self.upper_quantile = upper_quantile#upper_quantile=0.99：上界分位数，默认 99%
        #self.xxx = xxx：把参数存进对象属性

    def fit(self, X, y=None):# y成默认None
        X = np.asarray(X, dtype=float)#np.asarray：转成数组，dtype=float：强制为浮点型
        # nanquantile：即使有 NaN 也能算（本流水线中放在填充之后，其实无 NaN）
        self.lower_bounds_ = np.nanquantile(X, self.lower_quantile, axis=0)
        #np.nanquantile：算分位数，能忽略 NaN;X：数据;self.lower_quantile：要算的分位数
        #axis=0：沿着行方向(方向：向下)算，即每列分别算分位数
        self.upper_bounds_ = np.nanquantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        #np.clip(数组, 下界, 上界)：小于下界置为下界，大于上界置为上界
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


def main() -> None:
    df = load_agri_data()
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"训练集 {X_train.shape}，测试集 {X_test.shape}")

    # ================================================================== #
    # 1. 缺失值处理
    # ================================================================== #
    section("1. 缺失值处理")
    print("各列缺失数量：")
    print(X_train.isna().sum()[X_train.isna().sum() > 0].to_string())
    #X_train.isna()：对每个格子判断"是不是 NaN"，是 NaN → True，不是 NaN → False
    #在 True/False 表格上调用 .sum()： 默认按列求和
    #[X_train.isna().sum() > 0]：布尔索引；只保留 True 对应的行
    print(
        "\n常用策略：\n"
        " * 数值列：均值(mean，受异常值影响大) / 中位数(median，偏态/有异常值首选)；\n"
        " * 类别列：众数(most_frequent) 或 单独作为一个 'missing' 类别；\n"
        " * 缺失本身可能携带信息（传感器掉线往往与故障/极端天气相关），\n"
        "   可用 add_indicator=True 额外增加“是否缺失”的 0/1 指示列；\n"
        " * KNNImputer：用欧氏距离最近的 k 个样本均值填充，更精细但更慢。"
        #举例：样本 A 的 humidity 缺失。找和 A 在其他特征上最像的 5 个样本（k=5），用它们的 humidity 平均值填 A。
    )

    # 1.1 中位数填充 + 缺失指示
    num_imputer = SimpleImputer(strategy="median", add_indicator=True)#add_indicator=True：除了填缺失，额外加 0/1 列表示"原来是不是缺失"
    train_imputed = num_imputer.fit_transform(X_train[NUMERIC_COLS])
    test_imputed = num_imputer.transform(X_test[NUMERIC_COLS])
    print(
        f"\n中位数填充后训练矩阵形状 {train_imputed.shape}（多出的列是缺失指示列），"
        f"剩余缺失 {np.isnan(train_imputed).sum()} 个"
    )

    # 1.2 KNN 填充演示（在含缺失的 3 列上）
    miss_cols = ["temperature", "humidity", "nitrogen"]
    knn = KNNImputer(n_neighbors=5)
    knn.fit(X_train[miss_cols])
    knn_filled = knn.transform(X_test[miss_cols])
    print(f"KNNImputer(k=5) 填充测试集后剩余缺失 {np.isnan(knn_filled).sum()} 个")

    # ================================================================== #
    # 2. 异常值检测与处理
    # ================================================================== #
    section("2. 异常值检测：Z-score 法 与 IQR 法")
    # 先用中位数填充，避免 NaN 干扰统计
    filled = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X_train[NUMERIC_COLS]),
        columns=NUMERIC_COLS,
    )

    # ---- 方法 A：Z-score（标准化分数）法 ---- #
    # 公式：z = (x - μ) / σ；经验上 |z| > 3 视为异常值（落在 3 倍标准差之外）
    z = (filled - filled.mean()) / filled.std()#filled.mean()：每列的均值
    outlier_z = (z.abs() > 3).sum()

    # ---- 方法 B：IQR（四分位距）法，箱线图“触须”的画法 ---- #
    # IQR = Q3 - Q1；正常区间 [Q1 - 1.5*IQR, Q3 + 1.5*IQR]，区间外即异常
    q1, q3 = filled.quantile(0.25), filled.quantile(0.75)#filled.quantile(0.25)：每列的 25% 分位点（Q1），返回 Series
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_iqr = ((filled < low) | (filled > high)).sum()

    outlier_report = pd.DataFrame(
        {"Zscore法_异常数": outlier_z, "IQR法_异常数": outlier_iqr}
    )
    print("两种方法检出的异常样本数：")
    print(outlier_report.to_string())

    # ---- 处理：缩尾截断 winsorize（保留样本、只压缩极端值）---- #
    winsor = Winsorizer(lower_quantile=0.01, upper_quantile=0.99)
    filled_winsor = winsor.fit_transform(filled)
    filled_winsor = pd.DataFrame(filled_winsor, columns=NUMERIC_COLS)
    print("\n缩尾截断前后 humidity 的最大值：",
          f"{filled['humidity'].max():.1f} → {filled_winsor['humidity'].max():.1f}")
    print("缩尾截断前后 rainfall 的最大值：",
          f"{filled['rainfall'].max():.1f} → {filled_winsor['rainfall'].max():.1f}")

    # 图：humidity / rainfall 处理前后箱线图
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for row, col in enumerate(["humidity", "rainfall"]):
        axes[row, 0].boxplot(filled[col])
        axes[row, 0].set_title(f"{col} BEFORE winsorize")
        axes[row, 1].boxplot(filled_winsor[col])
        axes[row, 1].set_title(f"{col} AFTER winsorize")
        for ax in axes[row]:
            ax.set_ylabel(col)
    fig.suptitle("Outlier handling: before vs after 1%/99% winsorization")
    fig.tight_layout()
    save_fig(fig, "02a_outliers_boxplot.png")
    plt.close(fig)

    # ================================================================== #
    # 3. 类别特征编码
    # ================================================================== #
    section("3. 类别特征编码")
    # 3.1 One-Hot 编码：土壤类型、灌溉方式是【无序名义变量】，不能比大小
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)#设成 "ignore" → 把未知类别全部编成 0
    onehot.fit(X_train[CATEGORICAL_COLS])
    cat_oh = onehot.transform(X_train[CATEGORICAL_COLS])
    print("One-Hot 编码后的列名：", list(onehot.get_feature_names_out()))#get_feature_names_out():返回编码后的列名列表,命名规则：原列名_类别值
    print(f"原始 2 个类别列 → {cat_oh.shape[1]} 个 0/1 列，前 3 行：")
    print(pd.DataFrame(cat_oh, columns=onehot.get_feature_names_out()).head(3).to_string())

    # 3.2 Ordinal 编码：只适用于【有天然大小顺序】的有序类别
    # 这里把土壤 pH 分箱成“强酸/弱酸/中性/弱碱/强碱”作为有序变量演示
    #把连续数值切分成几段，每段给一个标签
    ph_levels = pd.cut(
        X_train["soil_ph"],#要切的数据
        bins=[-np.inf, 5.0, 6.0, 7.0, 8.0, np.inf],#切分边界(左开右闭),-np.inf：负无穷,np.inf：正无穷
        labels=["强酸", "弱酸", "中性", "弱碱", "强碱"],#每段的名字
    ).to_frame()#pd.cut 返回的是 Series（一列）。.to_frame() 转成 DataFrame
    ordinal = OrdinalEncoder(
        categories=[["强酸", "弱酸", "中性", "弱碱", "强碱"]]
    )
    ph_encoded = ordinal.fit_transform(ph_levels)
    print("\nOrdinal 编码（顺序信息被保留，0<1<2<3<4）示例：")
    print(pd.DataFrame({"pH等级": ph_levels["soil_ph"], "编码": ph_encoded.ravel()})#.ravel()把它压成 (960,) 一维数组
          .drop_duplicates().sort_values("编码").to_string(index=False))#drop_duplicates():去重

    print(
        "\n编码选型经验：\n"
        " * 线性回归 / 逻辑回归 / SVM / KNN / 神经网络：基于距离或内积，\n"
        "   无序类别必须 One-Hot，否则模型会误以为 黏土(2)>砂土(0)；\n"
        " * 决策树 / 随机森林 / GBDT / XGBoost / LightGBM：对单调编码不敏感，\n"
        "   One-Hot 或序数编码均可，LightGBM/XGBoost 还有原生类别处理；\n"
        " * LabelEncoder 按字母序编号，【只用于编码目标 y】，不要用于无序特征。"
    )

    # ================================================================== #
    # 4. 数值缩放（量纲统一）与对数变换
    # ================================================================== #
    section("4. 数值缩放与分布变换")
    sample_cols = ["temperature", "rainfall"]
    demo = filled[sample_cols].copy()

    # 4.1 标准化 Z-score（StandardScaler）
    #   公式：x' = (x - μ) / σ；变换后均值≈0、标准差≈1，不限制范围，
    #   是逻辑回归/SVM/PCA/KNN 的默认选择。
    std_scaler = StandardScaler().fit(demo)
    demo_std = std_scaler.transform(demo)

    # 4.2 归一化 Min-Max（MinMaxScaler）
    #   公式：x' = (x - min) / (max - min)；结果落在 [0,1]，对异常值敏感。
    mm_scaler = MinMaxScaler().fit(demo)
    demo_mm = mm_scaler.transform(demo)

    # 4.3 Robust 缩放：用中位数和四分位距，异常值不会把范围拉爆
    #   x' = (x - median) / IQR
    rb_scaler = RobustScaler().fit(demo)
    demo_rb = rb_scaler.transform(demo)

    cmp_df = pd.DataFrame(
        {
            "原始_mean": demo.mean().round(2),
            "原始_std": demo.std().round(2),
            "标准化_mean": np.mean(demo_std, axis=0).round(2),
            "标准化_std": np.std(demo_std, axis=0).round(2),
            "归一化_min": np.min(demo_mm, axis=0).round(3),
            "归一化_max": np.max(demo_mm, axis=0).round(3),
        },
        index=sample_cols,
    )
    print("三种缩放效果对比：")
    print(cmp_df.to_string())

    # 4.4 对数变换治右偏：rainfall、nitrogen 都是非负右偏分布
    #   x' = log(1 + x)（log1p），把大值压缩、长尾变短，利于线性模型
    skew_before = X_train["rainfall"].skew()#返回偏度值
    log_rainfall = np.log1p(X_train["rainfall"])
    skew_after = log_rainfall.skew()
    print(f"\nrainfall 偏度：{skew_before:.2f}（右偏）→ log1p 后 {skew_after:.2f}（接近对称）")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(X_train["rainfall"], bins=40, color="#c0392b", alpha=0.8)
    axes[0].set_title(f"rainfall raw (skew={skew_before:.2f})")
    axes[1].hist(log_rainfall, bins=40, color="#27ae60", alpha=0.8)
    axes[1].set_title(f"log1p(rainfall) (skew={skew_after:.2f})")
    for ax in axes:
        ax.set_xlabel("value")
        ax.set_ylabel("count")
    fig.suptitle("Log transform fixes right-skewed distribution")
    fig.tight_layout()
    save_fig(fig, "02b_log_transform.png")
    plt.close(fig)

    # ================================================================== #
    # 5. 特征构造（创造新特征）
    # ================================================================== #
    section("5. 特征构造")
    feat = filled.copy()#.copy() 创建独立副本
    #想改 DataFrame 又保留原版，必须 .copy()
    # 5.1 基于农业领域知识的业务特征
    feat["temp_humid"] = feat["temperature"] * feat["humidity"]      # 高温高湿协同促病，两者相乘：逐元素相乘（pandas 自动对齐索引）
    feat["wet_index"] = feat["humidity"] + feat["rainfall"] / 10.0   # 综合田间湿润度
    feat["sun_gap"] = feat["sunshine"] - 6.5                          # 相对正常日照的盈亏
    print("业务构造特征示例（前 3 行）：")
    print(feat[["temp_humid", "wet_index", "sun_gap"]].head(3).round(2).to_string())

    # 5.2 多项式特征：为线性模型制造非线性/交互能力
    #   degree=2 会生成 [1, a, b, a^2, a*b, b^2]；interaction_only=True 只留交互项
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    #degree=2：最高次数为 2，interaction_only=True：只保留交互项，不要平方项，include_bias=False：不要常数项 1
    poly_mat = poly.fit_transform(filled[["temperature", "humidity"]])
    print("\nPolynomialFeatures(degree=2, 仅交互项) 生成的列：",
          poly.get_feature_names_out(["temperature", "humidity"]))#传入原始列名，返回生成后的列名

    # 5.3 特征分箱（离散化）：把连续值切成区间，增强稳健性、捕捉非线性
    #   strategy: uniform 等宽 / quantile 等频 / kmeans 按聚类中心
    binner = KBinsDiscretizer(n_bins=5, encode="onehot", strategy="quantile")
    #n_bins=5：分成 5 个箱子，encode="onehot"：编码方式
    #"uniform"	等宽：每个箱子宽度相同，"quantile"	等频：每个箱子样本数相同，"kmeans"	按 K-Means 聚类中心分

    ph_bin = binner.fit_transform(filled[["soil_ph"]])
    print(f"\nsoil_ph 等频分 5 箱 → One-Hot 矩阵形状 {ph_bin.shape}，"
          f"分箱边界：{np.round(binner.bin_edges_[0], 2)}")
    binner.bin_sdges-[0]
    # ================================================================== #
    # 6. 特征选择：从一堆特征里挑出真正有用的，降噪、防过拟合、提速
    # ================================================================== #
    section("6. 特征选择（Filter / Wrapper / Embedded 三大流派）")
    # 准备矩阵：全部 9 个数值列（含冗余华氏温度、低方差电量、纯噪声），中位数填充
    X_num = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X_train[ALL_NUMERIC]),
        columns=ALL_NUMERIC,
    )

    # ---- 6.1 过滤式 Filter：按统计指标独立给每个特征打分，与模型无关 ---- #
    print("【Filter-1】VarianceThreshold：删除方差过小、几乎不变的特征")
    print(f"  处理前 {X_num.shape[1]} 个特征；battery_level 方差 = "
          f"{X_num[LOW_VAR_COL].var():.4f}")
    vt = VarianceThreshold(threshold=0.05)#创建过滤器，阈值 0.05，方差 < 0.05 的特征被删除
    vt.fit(X_num)#计算每列方差
    sel_cols = np.array(ALL_NUMERIC)[vt.get_support()]#返回一个布尔数组，True 表示保留，False 表示删除
    dropped = np.array(ALL_NUMERIC)[~vt.get_support()]#~ 是取反
    # 后续 Filter/Wrapper/Embedded/PCA 都在“已剔除低方差特征”的 8 列上进行
    X_sel = pd.DataFrame(vt.transform(X_num), columns=sel_cols)
    print(f"  阈值 0.05 后保留 {X_sel.shape[1]} 个，被删除：{list(dropped)}")

    print("\n【Filter-2】相关性过滤：删除与其他特征高度线性相关的冗余列")
    corr = X_sel.corr().abs()#算相关系数矩阵，corr[i, j] = 第 i 列和第 j 列的相关系数
    high_corr_pairs = []
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    #.astype(bool)：把 0/1 矩阵转成 True/False
    #np.triu(M, k=1)：保留主对角线以上的元素，其他变0，k=1：不含对角线(主对角线也变0)
    #DataFrame.where(条件)：条件为 True 保留原值，False 变 NaN

    for col_ in upper.columns:
        hit = upper.index[upper[col_] > 0.9].tolist()#.tolist()：转成列表
        #upper.index[...]：用布尔 Series 筛选行索引，得到所有满足条件的行名
        for row_ in hit:
            high_corr_pairs.append((row_, col_, round(corr.loc[row_, col_], 3)))
    for a, b, r in high_corr_pairs:
        print(f"  高相关对：{a} ~ {b}，|r| = {r} → 可删除其中之一（temperature_f）")

    # 画相关性热力图
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(sel_cols)))
    ax.set_yticks(range(len(sel_cols)))
    ax.set_xticklabels(sel_cols, rotation=45, ha="right")
    ax.set_yticklabels(sel_cols)
    for i in range(len(sel_cols)):
        for j in range(len(sel_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    ax.set_title("Feature correlation matrix (|r| near 1 = redundant)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    save_fig(fig, "02c_correlation_heatmap.png")
    plt.close(fig)

    print("\n【Filter-3】SelectKBest：按单变量统计检验选 k 个最强特征")
    # f_classif：方差分析 F 检验（数值特征 vs 分类标签），衡量组间均值差异
    kbest_f = SelectKBest(score_func=f_classif, k=5).fit(X_sel, y_train)
    #f_classif：对于每个特征，不同类别（病害 0 / 1）下的均值差异大不大
    #如果某个特征在“健康”和“病害”两组里的均值差很多，说明这个特征对区分类别有用，F 值就大
    #SelectKBest(score_func=f_classif, k=5)用 F 检验给每个特征打分，选前 5 个
    print("  ANOVA F 检验得分前 5：", list(sel_cols[kbest_f.get_support()]))
    #kbest_f.get_support()：返回一个布尔数组，长度等于特征数，True 表示被选中
    # 互信息 MI：知道某个特征后，对标签的不确定性减少了多少，MI = 0 表示特征和标签完全独立，没有任何信息
    #MI 越大，说明特征越有用
    kbest_mi = SelectKBest(score_func=mutual_info_classif, k=5).fit(X_sel, y_train)
    mi_score = pd.Series(kbest_mi.scores_, index=sel_cols).sort_values(ascending=False)
    #kbest_mi.scores_：所有特征的互信息得分
    #pd.Series(..., index=sel_cols)：把得分和特征名对应起来，方便按特征名查看
    print("  互信息 Top5：", list(mi_score.head(5).index))
    print("  （white_noise 的 MI 应接近 0，符合预期）")
    # 卡方检验 chi2：检验非负特征与标签的独立性，【要求特征非负】，先 MinMax
    X_nonneg = MinMaxScaler().fit_transform(X_sel)
    kbest_chi2 = SelectKBest(score_func=chi2, k=5).fit(X_nonneg, y_train)
    #chi2：卡方检验，它检验的是：特征与标签之间是否独立，如果特征分布和标签分布有关联，卡方值就大，说明特征有用
    #卡方检验要求特征非负
    print("  卡方检验 Top5（需先归一化到非负）：", list(sel_cols[kbest_chi2.get_support()]))
    print("  注意两点：① temperature_f 可能也被选中——单变量 Filter 各自独立打分，\n"
          "  无法识别它与 temperature 的冗余，需要相关性过滤/PCA 兜底；\n"
          "  ② white_noise 偶尔会因随机巧合入选（假阳性），实战中要结合多种方法\n"
          "  和业务知识交叉验证，不要盲信单一统计检验。")

    # ---- 6.2 包裹式 Wrapper：反复训练模型，用模型成绩搜索特征子集 ---- #
    print("\n【Wrapper】RFE 递归特征消除：")
    print("  从全特征开始，每轮删除模型认为最不重要的 1 个，递归到剩下 k 个")
    scaler_for_selection = StandardScaler()
    X_sel_scaled = scaler_for_selection.fit_transform(X_sel)
    rfe = RFE(
        estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        n_features_to_select=5,
        step=1,
    ).fit(X_sel_scaled, y_train)
    #estimator=LogisticRegression(...)：指定用什么模型来评估特征重要性
    #max_iter=2000：增加迭代次数，确保模型收敛
    #n_features_to_select=5：最终要保留 5 个特征
    #step=1：每轮删除 1 个最不重要的特征
    rfe_result = pd.DataFrame(
        {"feature": sel_cols, "selected": rfe.support_, "ranking": rfe.ranking_}
    ).sort_values("ranking")#按'排名'列升序排列
    #rfe.ranking_：整数数组，表示特征被淘汰的顺序
    #1表示最终保留的特征。
    #2表示第一轮被删除的特征（最不重要）
    #3表示第二轮被删除的特征
    print(rfe_result.to_string(index=False))

    # ---- 6.3 嵌入式 Embedded：在模型训练过程中自动完成选择 ---- #
    print("\n【Embedded-1】L1 正则（Lasso 思想）：")
    print("  损失 = 交叉熵 + C0*Σ|w|，惩罚项会把【无用特征的系数压成恰好 0】，")
    print("  非零系数对应的特征即被选中（C 越小正则越强、越稀疏）")
    # 注：部分新版本 sklearn 会提示 penalty 参数弃用、建议改用 l1_ratio，警告可忽略
    import warnings
    with warnings.catch_warnings():
        # 不同 sklearn 版本对 penalty 参数可能有弃用/兼容提示，与本节原理无关
        warnings.simplefilter("ignore", (FutureWarning, UserWarning))

        l1 = LogisticRegression(
            penalty="l1", solver="liblinear", C=0.2, random_state=RANDOM_STATE
        ).fit(X_sel_scaled, y_train)
        #penalty="l1"：使用 L1 正则
        #solver="liblinear"：指定求解器
        #C=0.2：正则强度的倒数,C 越小 → 正则越强 → 越多系数被压成 0 → 越稀疏
    l1_coef = pd.Series(l1.coef_.ravel(), index=sel_cols)
    #l1.coef_：模型的系数数组，形状通常是 (1, n_features)，因为是二分类
    print("  L1 逻辑回归系数（0 即被淘汰）：")
    print(l1_coef.round(3).to_string())
    #系数为 0 的特征,被 L1 自动淘汰

    print("\n【Embedded-2】树模型特征重要性 + SelectFromModel：")
    '''
    特征重要性的计算逻辑：
    某个特征在所有树中被用来分裂的次数越多；
    每次分裂带来的纯度提升越大；
    这个特征就越重要
    '''
    rf_sel = RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    ).fit(X_sel, y_train)
    importances = pd.Series(rf_sel.feature_importances_, index=sel_cols)
    sfm = SelectFromModel(rf_sel, threshold="median").fit(X_sel, y_train)
    print("  随机森林重要性（按重要性高于中位数自动保留）：")
    for name, imp in importances.sort_values(ascending=False).items():
        mark = "保留" if name in sel_cols[sfm.get_support()] else "淘汰"
        print(f"    {name:15s} {imp:.4f}  [{mark}]")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    importances.sort_values().plot.barh(ax=ax, color="#2980b9")
    ax.set_title("Random Forest feature importance")
    ax.set_xlabel("importance")
    fig.tight_layout()
    save_fig(fig, "02d_feature_importance.png")
    plt.close(fig)

    # ================================================================== #
    # 7. PCA 主成分分析（无监督线性降维）
    # ================================================================== #
    section("7. PCA 主成分分析")
    print(
        "原理：PCA 对【标准化后】特征的协方差矩阵 C = (1/(n-1)) XᵀX 做特征分解，\n"
        "找出若干相互正交、使投影后方差依次最大的方向（主成分 PC）。\n"
        "  * 特征值 λᵢ = 该主成分解释的方差大小；\n"
        "  * 解释方差比 = λᵢ / Σλⱼ；累计解释方差达到 85%~95% 即可保留大部分信息；\n"
        "  * 作用：去相关、压缩维度、可视化、降噪；缺点：主成分失去原始业务含义。"
    )
    pca_full = PCA().fit(X_sel_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_for_90 = int(np.argmax(cumvar >= 0.90) + 1)
    print(f"\n各主成分解释方差比：{np.round(pca_full.explained_variance_ratio_, 3)}")
    print(f"累计解释方差：{np.round(cumvar, 3)}")
    print("→ 最后一个主成分方差几乎为 0，方向正是 temperature 与 temperature_f")
    print(f"  的完全共线方向（冗余信息）；保留 90% 信息需前 {n_for_90} 个主成分"
          f"（原 {X_sel_scaled.shape[1]} 维）")

    # 降到 2 维可视化（病害/健康两类是否可分）
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca2.fit_transform(X_sel_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(range(1, len(cumvar) + 1), cumvar, "o-", color="#8e44ad")
    axes[0].axhline(0.9, color="gray", ls="--", label="90% threshold")
    axes[0].set_xlabel("number of principal components")
    axes[0].set_ylabel("cumulative explained variance")
    axes[0].set_title("PCA scree plot")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for label, color, name in [(0, "#27ae60", "healthy"), (1, "#c0392b", "disease")]:
        m = y_train.to_numpy() == label
        axes[1].scatter(coords[m, 0], coords[m, 1], s=10, alpha=0.5, c=color, label=name)
    axes[1].set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%} variance)")
    axes[1].set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.1%} variance)")
    axes[1].set_title("2D PCA projection")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "02e_pca.png")
    plt.close(fig)

    # ================================================================== #
    # 8. 用 ColumnTransformer + Pipeline 把特征工程固化为一条流水线
    # ================================================================== #
    section("8. 流水线整合：一次 fit，自动防泄漏、可复现")
    # 数值分支：中位数填充 → 缩尾截断去异常 → 标准化
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("winsor", Winsorizer(0.01, 0.99)),
            ("scale", StandardScaler()),
        ]
    )
    # 类别分支：众数填充 → One-Hot
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_COLS),       # 只喂 6 个有效数值列
            ("cat", categorical_pipe, CATEGORICAL_COLS),
        ]
    )
    final_pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=4,
                random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 对照组：把全部“脏列”（含缺失、异常、冗余、噪声）直接喂给模型
    # 对照组需要自己的 ColumnTransformer（所有列都进，类别只做 onehot）
    raw_preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), ALL_NUMERIC),
            ("cat", categorical_pipe, CATEGORICAL_COLS),
        ]
    )
    raw_model = Pipeline([("preprocess", raw_preprocess),
                          ("clf", RandomForestClassifier(
                              n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1))])

    auc_engineered = cross_val_score(
        final_pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
    ).mean()
    auc_raw = cross_val_score(
        raw_model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
    ).mean()
    print(f"原始脏特征直接建模，5 折 CV ROC-AUC = {auc_raw:.4f}")
    print(f"特征工程流水线建模，  5 折 CV ROC-AUC = {auc_engineered:.4f}")

    # 在训练集 fit 后，对测试集只做 transform + predict，全程无泄漏
    final_pipe.fit(X_train, y_train)
    from sklearn.metrics import roc_auc_score
    test_auc = roc_auc_score(y_test, final_pipe.predict_proba(X_test)[:, 1])
    print(f"流水线在测试集上的 ROC-AUC = {test_auc:.4f}")
    print(
        "\n这条 Pipeline 可以直接塞进 GridSearchCV / RandomizedSearchCV 一起调参，\n"
        "也可以用 joblib.dump 整体保存上线，是工业界表格建模的标准形态。"
    )
    print("\n全部完成，图表见 output/02a~02e。")


if __name__ == "__main__":
    main()
