# -*- coding: utf-8 -*-
"""
================================================================================
第 02 章 · 进阶能力（4/4）：不平衡数据集处理
================================================================================

农业病害预警、设备故障检测、欺诈识别等场景的共同特点：正类（出事）很少。
若病害样本只占 15%，模型把所有地块都预测为“健康”就能拿到 85% 准确率——
这就是【准确率悖论】。本章系统演示工业界四类主流对策：

  1. 改变评价方式：抛弃 Accuracy，改用 Recall / F1 / ROC-AUC / PR-AUC；
  2. 类别加权 class_weight='balanced'（算法层，不改变样本）；
  3. 重采样（数据层）：
       随机过采样 RandomOverSampler、SMOTE 合成新样本、随机欠采样、
       过采样 + 欠采样组合；
  4. 阈值移动 threshold moving：把默认 0.5 的判定阈值调低，优先抓正类。

关键公式
--------------------------------------------------------------------------------
* balanced 类别权重：对第 j 类，
      w_j = n_samples / (n_classes × n_j)
  少数类 n_j 小 → 权重 w_j 大，模型错分一个少数类的惩罚更重。
* SMOTE（Synthetic Minority Over-sampling Technique）合成新少数类样本：
  对少数类样本 x_i，随机选它的 k 近邻（同为少数类）中的一个 x_zi，
      x_new = x_i + λ · (x_zi − x_i),   λ ~ Uniform(0, 1)
  即在两个少数类样本的连线上随机取点，凭空“造”出新样本（不是简单复制）。
* 阈值选择：
  - 最大化 F1：threshold* = argmax F1(t)；
  - Youden 指数：J = TPR − FPR = Recall + Specificity − 1，取使 J 最大的阈值。

★ 最重要的工程纪律：重采样必须放在交叉验证【每折内部】做
  错误做法：先对整个训练集 SMOTE，再划分折 → 合成样本的“原型”可能落在
  验证折里，验证集信息泄漏，CV 成绩虚高。
  正确做法：用 imblearn.pipeline.Pipeline 把 SMOTE 串进去，每折只对该折
  训练部分重采样。本章第 5 节会用实验对比两种做法。

运行方式：python 04_不平衡数据集处理.py
依赖：pip install imbalanced-learn
================================================================================
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

from agri_data import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    RANDOM_STATE,
    TARGET,
    load_agri_data,
    save_fig,
    section,
)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def build_preprocess(scale: bool = True) -> ColumnTransformer:#scale 为布尔类型（bool），默认为 True，作为一个条件开关，用于控制是否在预处理流程中加入特征缩放步骤
    """数值：中位数填充（SMOTE 基于距离，需标准化）；类别：众数 + OneHot。"""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))#StandardScaler = 标准化，把每列变成 均值 0、标准差 1
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), NUMERIC_COLS),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_COLS,
            ),
        ]
    )


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1,
    )


def evaluate_cv(pipeline, X, y, cv, name="") -> dict:
    """在 5 折 CV 下同时计算不平衡问题关心的 4 个指标。"""
    result = cross_validate(
        pipeline, X, y, cv=cv, n_jobs=-1,
        scoring=["f1", "recall", "roc_auc", "average_precision"],#PR_AUC（average_precision）
    )
    scores = {
        "方法": name,
        "F1": result["test_f1"].mean(),
        "Recall": result["test_recall"].mean(),
        "ROC_AUC": result["test_roc_auc"].mean(),
        "PR_AUC": result["test_average_precision"].mean(),
    }
    return scores


def main() -> None:
    df = load_agri_data()
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    preprocess = build_preprocess(scale=True)
    X_train_t = preprocess.fit_transform(X_train)
    X_test_t = preprocess.transform(X_test)
    feature_names = list(preprocess.get_feature_names_out())#get_feature_names_out() 返回处理后的每个列的名字

    # ================================================================== #
    # 1. 不平衡现状与“准确率悖论”
    # ================================================================== #
    section("1. 不平衡现状与准确率悖论")
    counts = y.value_counts().sort_index()#y.value_counts() 统计每个值出现多少次，默认按频次降序排
    ratio = counts[1] / counts[0]
    print(f"健康(0)：{counts[0]} 条；病害(1)：{counts[1]} 条；正负比 ≈ 1:{counts[0] / counts[1]:.1f}")
    dumb_acc = (y_test == 0).mean()
    print(f"如果模型“躺平”把所有地块都预测为健康，测试集准确率 = {dumb_acc:.4f}")
    print("→ Accuracy 完全失效：它同时掩盖了 100% 的病害漏报。")
    print("  不平衡任务应主看 Recall（漏报率）、F1、PR-AUC，ROC-AUC 作参考。")

    # ================================================================== #
    # 2. 基线：不做任何不平衡处理
    # ================================================================== #
    section("2. 基线模型（不做不平衡处理）")
    base = make_rf().fit(X_train_t, y_train)
    base_pred = base.predict(X_test_t)
    base_proba = base.predict_proba(X_test_t)[:, 1]
    cm = confusion_matrix(y_test, base_pred)
    print("测试集混淆矩阵：")
    print(cm)
    print(classification_report(y_test, base_pred,
                                target_names=["健康", "病害"], zero_division=0))
    print(f"Recall = {recall_score(y_test, base_pred):.4f}（病害漏报严重）")
    print(f"F1 = {f1_score(y_test, base_pred):.4f}，PR-AUC = "
          f"{average_precision_score(y_test, base_proba):.4f}")

    # ================================================================== #
    # 3. 方法一：类别加权 class_weight='balanced'
    # ================================================================== #
    section("3. 方法一：类别加权 class_weight='balanced'")
    n = len(y_train)
    n0, n1 = (y_train == 0).sum(), (y_train == 1).sum()
    w0 = n / (2 * n0)
    w1 = n / (2 * n1)
    print("按公式 w_j = n_samples / (n_classes × n_j) 手算：")
    print(f"  健康类权重 w0 = {n}/{2*n0} = {w0:.4f}")
    print(f"  病害类权重 w1 = {n}/{2*n1} = {w1:.4f}  ← 少数类权重高得多")
    weighted = make_rf()
    weighted.set_params(class_weight="balanced")#改参数
    weighted.fit(X_train_t, y_train)
    w_pred = weighted.predict(X_test_t)
    print(f"\n加权后测试集 Recall = {recall_score(y_test, w_pred):.4f}，"
          f"F1 = {f1_score(y_test, w_pred):.4f}")
    print("（与 sklearn 自动计算的 class_weight 完全一致；也可手动传字典 "
          "class_weight={0: w0, 1: w1}）")

    # ================================================================== #
    # 4. 方法二：重采样
    # ================================================================== #
    section("4. 方法二：重采样（过采样 / SMOTE / 欠采样）")
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    smote = SMOTE(k_neighbors=5, random_state=RANDOM_STATE)
    rus = RandomUnderSampler(random_state=RANDOM_STATE)

    X_ros, y_ros = ros.fit_resample(X_train_t, y_train)
    X_smote, y_smote = smote.fit_resample(X_train_t, y_train)
    X_rus, y_rus = rus.fit_resample(X_train_t, y_train)
    '''
    为什么方法叫 fit_resample？
    它把"学习采样策略 + 生成新数据"合成一步：
    fit：分析原数据（比如 SMOTE 要建 KD 树找近邻、ROS/RUS 要数类别数量）。
    resample：生成新的 X 和 y。
    返回两个数组：新的特征矩阵和新的标签。
    '''
    print(f"原始训练集：    {np.bincount(y_train)}")
    #统计每个非负整数出现的次数，返回数组，下标就是值
    print(f"随机过采样后：  {np.bincount(y_ros)}（复制少数类样本直到平衡）")
    print(f"SMOTE 后：      {np.bincount(y_smote)}（在少数类连线间合成新样本）")
    print(f"随机欠采样后：  {np.bincount(y_rus)}（丢弃多数类样本，损失信息）")

    # 图：各类方法的类别分布
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    datasets = [
        ("Original", np.bincount(y_train)),
        ("RandomOver", np.bincount(y_ros)),
        ("SMOTE", np.bincount(y_smote)),
        ("RandomUnder", np.bincount(y_rus)),
    ]
    for ax, (name_, cnt) in zip(axes, datasets):
        bars = ax.bar(["healthy", "disease"], cnt, color=["#27ae60", "#c0392b"])
        ax.set_title(name_)
        for b, c in zip(bars, cnt):
            ax.text(b.get_x() + b.get_width() / 2, c, str(c),
                    ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("samples")
    fig.suptitle("Resampling strategies for the imbalanced training set")
    fig.tight_layout()
    save_fig(fig, "04a_resampling_distribution.png")
    plt.close(fig)

    # SMOTE 二维直观演示：在 humidity / sunshine 两个标准化特征上看合成点
    fig, ax = plt.subplots(figsize=(7, 5.5))
    demo_cols = [feature_names.index("num__humidity"),
                 feature_names.index("num__sunshine")]
    Xd = X_train_t[:, demo_cols]
    Xd_sm, _ = SMOTE(k_neighbors=5, random_state=RANDOM_STATE).fit_resample(Xd, y_train)
    pos = Xd[y_train == 1]
    synth = Xd_sm[len(Xd):]  # SMOTE 输出中原样本在前、合成样本在后
    ax.scatter(Xd[y_train == 0, 0], Xd[y_train == 0, 1], s=12,
               c="#27ae60", alpha=0.4, label="majority (healthy)")
    ax.scatter(pos[:, 0], pos[:, 1], s=28, c="#2980b9", label="real minority")
    ax.scatter(synth[:, 0], synth[:, 1], s=14, marker="x",
               c="#c0392b", label="SMOTE synthetic")
    ax.set_xlabel("humidity (standardized)")
    ax.set_ylabel("sunshine (standardized)")
    ax.set_title("SMOTE synthesizes new minority points along neighbor lines")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "04b_smote_demo.png")
    plt.close(fig)
    print("\nSMOTE 注意事项：① 基于距离，数值特征必须先标准化、类别先编码；")
    print("  ② 若少数类存在噪声/离群点，SMOTE 会在错误区域造样本，可改用")
    print("     BorderlineSMOTE / SMOTEN(纯类别) / SMOTENC(混合) / ADASYN。")

    # 过采样 + 欠采样组合：先 SMOTE 到多数类的一半，再欠采样多数类到同样规模
    over = SMOTE(sampling_strategy=0.5, random_state=RANDOM_STATE)#sampling_strategy=0.5：少数类补到多数类的 50%
    under = RandomUnderSampler(sampling_strategy=1.0, random_state=RANDOM_STATE)#sampling_strategy=1.0：把多数类砍到和少数类一样多
    combo = ImbPipeline(steps=[("over", over), ("under", under)])
    #sklearn 的 Pipeline 只允许"中间步骤必须 transform、最后一步可以 fit"，
    # 而重采样器在 fit 之后还会改变样本数量。
    # sklearn 的 Pipeline 不支持这种"样本数改变"的中间步骤
    X_combo, y_combo = combo.fit_resample(X_train_t, y_train)
    print(f"SMOTE+欠采样组合后：{np.bincount(y_combo)}（折中方案，少造点、少删点）")

    # ================================================================== #
    # 5. 关键纪律：重采样必须在 CV 每折【内部】进行（防泄漏）
    # ================================================================== #
    section("5. 防泄漏：错误做法（先重采样再划分）vs 正确做法（折内重采样）")
    # 错误：SMOTE 作用于整个训练集后再交叉验证
    X_leak, y_leak = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_t, y_train)
    wrong_scores = cross_validate(
        make_rf(), X_leak, y_leak, cv=cv,
        scoring=["f1", "roc_auc"], n_jobs=-1,
    )
    # 正确：SMOTE 放进 imblearn 流水线，每折只在训练部分重采样
    right_pipe = ImbPipeline(steps=[("smote", SMOTE(random_state=RANDOM_STATE)),
                                    ("clf", make_rf())])
    right_scores = cross_validate(
        right_pipe, X_train_t, y_train, cv=cv,
        scoring=["f1", "roc_auc"], n_jobs=-1,
    )
    print(f"错误做法（先SMOTE后划分）：F1 = {wrong_scores['test_f1'].mean():.4f}，"
          f"ROC-AUC = {wrong_scores['test_roc_auc'].mean():.4f}  ← 虚高")
    print(f"正确做法（折内SMOTE）：  F1 = {right_scores['test_f1'].mean():.4f}，"
          f"ROC-AUC = {right_scores['test_roc_auc'].mean():.4f}  ← 可信")
    print("原因：先重采样时，验证折样本的 SMOTE 近邻可能就是训练折里的合成原型，")
    print("      模型相当于“提前见过验证集的近亲”，成绩不可信。")

    # ================================================================== #
    # 6. 方法三：阈值移动
    # ================================================================== #
    section("6. 方法三：阈值移动（不重训模型，只改判定门槛）")
    # 用独立验证集选阈值（不能用测试集选，否则又泄漏）
    # 这里故意用【不加权】的普通模型：它在默认 0.5 阈值下几乎不报警，
    # 正好用来演示“调低阈值”如何把漏报的病害找回来
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_t, y_train, test_size=0.3, random_state=RANDOM_STATE, stratify=y_train
    )
    thr_model = make_rf().fit(X_fit, y_fit)
    val_proba = thr_model.predict_proba(X_val)[:, 1]

    prec, rec, thresholds = precision_recall_curve(y_val, val_proba)
    # precision_recall_curve 返回的阈值比 P/R 少一个，对齐长度
    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    best_idx = int(np.argmax(f1s[:-1]))
    best_thr_f1 = thresholds[best_idx]

    fpr, tpr, roc_thr = roc_curve(y_val, val_proba)
    j_scores = tpr - fpr
    best_thr_j = roc_thr[int(np.argmax(j_scores))]
    print(f"默认阈值 0.5；按验证集 F1 最大选出的阈值 = {best_thr_f1:.3f}")
    print(f"按 Youden 指数 J=TPR-FPR 最大选出的阈值 = {best_thr_j:.3f}（两者通常都 <0.5）")

    # 图：P / R / F1 随阈值变化
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(thresholds, prec[:-1], label="Precision")
    ax.plot(thresholds, rec[:-1], label="Recall")
    ax.plot(thresholds, f1s[:-1], label="F1")
    ax.axvline(best_thr_f1, color="red", ls="--", label=f"best F1 @ {best_thr_f1:.2f}")
    ax.axvline(0.5, color="gray", ls=":", label="default 0.5")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("score on validation set")
    ax.set_title("Threshold moving: trade Precision for Recall")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "04c_threshold_curve.png")
    plt.close(fig)

    # 在测试集上验证阈值移动的效果
    test_proba = thr_model.predict_proba(X_test_t)[:, 1]
    pred_05 = (test_proba >= 0.5).astype(int)
    pred_best = (test_proba >= best_thr_f1).astype(int)
    print("\n同一模型在测试集上，仅改变阈值：")
    print(f"  阈值 0.5：Recall = {recall_score(y_test, pred_05):.3f}，"
          f"F1 = {f1_score(y_test, pred_05):.3f}")
    print(f"  阈值 {best_thr_f1:.2f}：Recall = {recall_score(y_test, pred_best):.3f}，"
          f"F1 = {f1_score(y_test, pred_best):.3f}")
    print("阈值越低漏报越少、误报越多——按业务代价（漏报减产 vs 复查成本）定阈值。")

    # ================================================================== #
    # 7. 全部方法横向对比（全部在折内处理，CV 公平比较）
    # ================================================================== #
    section("7. 全部方法横向对比（分层 5 折 CV）")
    methods = {
        "不处理(基线)": ImbPipeline(steps=[("clf", make_rf())]),
        "class_weight": ImbPipeline(
            steps=[("clf", make_rf().set_params(class_weight="balanced"))]),
        "随机过采样": ImbPipeline(steps=[
            ("ros", RandomOverSampler(random_state=RANDOM_STATE)),
            ("clf", make_rf())]),
        "SMOTE": ImbPipeline(steps=[
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", make_rf())]),
        "随机欠采样": ImbPipeline(steps=[
            ("rus", RandomUnderSampler(random_state=RANDOM_STATE)),
            ("clf", make_rf())]),
        "SMOTE+欠采样": ImbPipeline(steps=[
            ("over", SMOTE(sampling_strategy=0.5, random_state=RANDOM_STATE)),
            ("under", RandomUnderSampler(sampling_strategy=1.0, random_state=RANDOM_STATE)),
            ("clf", make_rf())]),
    }
    rows = []
    for name_, pipe in methods.items():
        rows.append(evaluate_cv(pipe, X_train_t, y_train, cv, name_))
    table = pd.DataFrame(rows).set_index("方法").round(4)
    print(table.to_string())

    # 分组柱状图
    fig, ax = plt.subplots(figsize=(10, 4.6))
    metric_names = ["F1", "Recall", "ROC_AUC", "PR_AUC"]
    x = np.arange(len(table.index))
    width = 0.2
    for i, m in enumerate(metric_names):
        ax.bar(x + (i - 1.5) * width, table[m], width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(table.index, rotation=15)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("CV score")
    ax.set_title("Imbalance handling methods comparison (5-fold CV)")
    ax.legend(ncol=4, loc="upper center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "04d_methods_comparison.png")
    plt.close(fig)

    # ================================================================== #
    # 8. 方法选型建议
    # ================================================================== #
    section("8. 选型建议与常见坑")
    print(
        "选型顺序（成本从低到高）：\n"
        " 1) 先换指标：Recall/F1/PR-AUC + 混淆矩阵，别再看 Accuracy；\n"
        " 2) class_weight='balanced'：零成本、不造数据，通常先试；\n"
        " 3) 阈值移动：不重训即可按业务代价平衡漏报/误报，上线必备；\n"
        " 4) 还不够再上重采样：\n"
        "    * 正类极少（<几十条）优先 SMOTE 家族，别用简单复制的过采样；\n"
        "    * 数据量很大可欠采样，或只在决策边界附近欠采样（Tomek Links / ENN）；\n"
        "    * 过采样+欠采样组合常比单一方法稳。\n"
        "常见坑：\n"
        " × 在【整个数据集】上重采样/拟合填充器/标准化 → 数据泄漏；\n"
        " × SMOTE 前不缩放数值特征 → 距离被大量纲特征主导；\n"
        " × 用测试集调阈值、挑模型 → 评估失真；\n"
        " × 极端不平衡（如 1:1000）考虑异常检测（One-Class SVM / Isolation\n"
        "   Forest），把“病害/故障”当异常点处理——即图片中第 3 部分内容。"
    )
    print("\n全部完成，图表见 output/04a~04d。")


if __name__ == "__main__":
    main()
