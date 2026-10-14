# -*- coding: utf-8 -*-
"""
================================================================================
第 02 章 · 进阶能力（3/4）：模型优化
================================================================================

调参只是模型优化的一部分。本章系统讲解“怎么科学地衡量模型、怎么诊断模型
的问题、怎么对症下药”，覆盖：

  1. 交叉验证 Cross-Validation：留出法、K 折、分层 K 折、留一法、多指标 CV；
  2. 分类评估指标体系（全部给出公式）：
       混淆矩阵、准确率 Accuracy、精确率 Precision、召回率 Recall、
       特异度 Specificity、F1、ROC 曲线与 AUC、PR 曲线与平均精度 AP；
  3. 偏差-方差诊断：学习曲线 (learning curve) 识别欠拟合 / 过拟合；
  4. 验证曲线 (validation curve)：定位单个超参数的合理区间；
  5. 正则化：L1/L2/弹性网公式，正则强度 C 对系数的影响；
  6. 早停 Early Stopping：GBDT 用 staged_predict 画训练/验证误差曲线；
  7. 综合优化闭环：Pipeline + 分层 K 折 + 网格搜索 + 测试集终评。

核心公式（先混个脸熟，代码里逐个演示）
--------------------------------------------------------------------------------
记混淆矩阵四个格子：TP 真阳（真病害报病害）、FN 假阴（漏报）、
                    FP 假阳（误报）、TN 真阴（健康报健康）。
  Accuracy    = (TP+TN) / (TP+TN+FP+FN)
  Precision   = TP / (TP+FP)        报了警的里面有多少是真的（查准）
  Recall/TPR  = TP / (TP+FN)        真正发病的有多少被抓到（查全 / 灵敏度）
  Specificity = TN / (TN+FP) = 1-FPR
  F1          = 2·P·R / (P+R)       P 与 R 的调和平均，不平衡数据首选
  ROC 横轴 FPR = FP/(FP+TN)，纵轴 TPR = TP/(TP+FN)，随分类阈值从高到低画出
  AUC = ROC 曲线下面积，0.5=瞎猜，1=完美；排序意义：随机一正一负，正样本得分
        排在负样本前面的概率。
  PR 曲线：横轴 Recall、纵轴 Precision，不平衡问题上比 ROC 更敏感、更难“刷分”。

运行方式：python 03_模型优化.py
================================================================================
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    learning_curve,
    train_test_split,
    validation_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agri_data import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    RANDOM_STATE,
    TARGET,
    load_agri_data,
    save_fig,
    section,
)


def prepare_data():
    """切分数据并用 02 章的标准做法完成预处理（中位数填充 + One-Hot）。"""
    df = load_agri_data()
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_COLS),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),#sparse_output=False：输出普通 NumPy 数组
                    ]
                ),
                CATEGORICAL_COLS,
            ),
        ]
    )
    X_train_t = preprocess.fit_transform(X_train)
    X_test_t = preprocess.transform(X_test)
    feature_names = preprocess.get_feature_names_out()#get_feature_names_out() 返回预处理后每一列的名字
    return X_train_t, X_test_t, y_train, y_test, feature_names
    #X_train_t：预处理后的训练特征
def main() -> None:
    X_train, X_test, y_train, y_test, feature_names = prepare_data()
    print(f"预处理后训练集 {X_train.shape}，测试集 {X_test.shape}")
    print("特征列表：", list(feature_names))

    # ================================================================== #
    # 1. 交叉验证
    # ================================================================== #
    section("1. 交叉验证：用“轮流当验证集”获得稳定、可信的模型评分")
    print(
        "为什么不能只做一次 train_test_split？\n"
        " 单次划分的成绩受“运气”影响大（某些病害样本恰好进了验证集）。\n"
        "K 折交叉验证：把训练集均分成 K 份，每次拿 1 份当验证集、其余训练，\n"
        "共训练 K 次，报告 K 个成绩的均值±标准差，方差更小、更接近真实水平。"
    )

    # 1.1 普通 KFold vs 分层 StratifiedKFold
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    print("\n普通 KFold 各折正类（病害）样本数：",
          [int(y_train.iloc[val].sum()) for _, val in kf.split(X_train, y_train)])
    #kf.split(X_train, y_train) 返回训练索引和验证索引
    print("分层 StratifiedKFold 各折正类样本数：",
          [int(y_train.iloc[val].sum()) for _, val in skf.split(X_train, y_train)])
    print("→ 不平衡分类必须用 StratifiedKFold，保证每折病害比例一致。")

    # 图：分层 5 折划分示意（每根横条是一折，金色=该折验证集）
    fig, ax = plt.subplots(figsize=(10, 3.2))
    order = np.argsort(y_train.to_numpy())#按标签从小到大排，各样本应处的位置”这一排列
    #order[0] 是排在最左边的那个样本的原始行号。因为标签只有 0/1，排序后前半段全是健康（0）、后半段全是病害（1）
    assignment = np.empty(len(y_train), dtype=int)
    #np.empty：预分配一个未初始化的数组（比逐个 append 快）
    for fold, (_, val_idx) in enumerate(
        skf.split(X_train, y_train.to_numpy())
        #skf.split 每次返回 (训练索引, 验证索引)；这里用 _ 丢弃训练索引，只留 val_idx。
        #循环 5 次（fold = 0~4），把“样本 i 在第几折当验证集”写进 assignment[i]。循环结束后每个样本恰好被赋值一次。
        ):
        assignment[val_idx] = fold
    grid = np.zeros((5, len(y_train)))
    for i in range(5):
        grid[i, assignment[order] == i] = 1
    ax.imshow(grid[:, :], aspect="auto", cmap="coolwarm", interpolation="nearest")
    ax.set_xlabel("samples (sorted by label: healthy left, disease right)")
    ax.set_ylabel("fold")
    ax.set_title("Stratified 5-Fold split: gold = validation fold")
    fig.tight_layout()
    save_fig(fig, "03a_cv_folds.png")
    plt.close(fig)

    # 1.2 cross_val_score（单指标）与 cross_validate（多指标 + 耗时）
    # class_weight='balanced' 让模型不会因病害样本少而“全部预测健康”，
    # 从而保证下面混淆矩阵四个格子都有样本、指标公式都能演示；
    # 它的原理与更多不平衡处理方法是第 04 章的主题。
    model = RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=4,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    auc_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc")
    print(f"\ncross_val_score(roc_auc) 各折：{np.round(auc_scores, 4)}")
    print(f"均值 {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")

    cv_multi = cross_validate(
        model, X_train, y_train, cv=skf,
        scoring=["accuracy", "f1", "roc_auc", "precision", "recall"],
        n_jobs=-1, return_train_score=True,
    )
    '''
    cross_validate可以一次计算多个指标:
    scoring 列表包括：
    accuracy：准确率，预测正确的比例。
    f1：精确率和召回率的调和平均。
    roc_auc：AUC。
    precision：精确率，预测为病害的样本中真正病害的比例。
    recall：召回率，真正病害中被预测出来的比例。
    return_train_score=True 还会返回训练集分数，方便比较训练和验证表现，判断是否过拟合。
    cv_multi 是一个字典，里面每个键对应一个数组
    '''
    cv_table = pd.DataFrame(cv_multi).agg(["mean", "std"]).T.round(4)
    #.agg(["mean", "std"]) 对每列求均值和标准差,.T 转置，让指标变成行
    print("\ncross_validate 多指标汇总（mean ± std）：")
    print(cv_table.to_string())
    print(
        "\n其他 CV 形式：\n"
        " * LeaveOneOut(LOO)：每次留 1 个样本验证，n 个样本训练 n 次，适合小数据；\n"
        " * RepeatedStratifiedKFold：多次重复 K 折再平均，评分更稳，代价是更慢；\n"
        " * TimeSeriesSplit：时序数据专用，验证折总在训练折之后（第 3 章场景适配）。"
    )

    # ================================================================== #
    # 2. 评估指标体系
    # ================================================================== #
    section("2. 评估指标：混淆矩阵 / P / R / F1 / ROC-AUC / PR-AP")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    #predict_proba 返回两列，第一列是健康概率，第二列是病害概率，所以 [:, 1](取所有行，第 1 列) 取病害概率

    # 2.1 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()#把 2×2 矩阵拉平成一维数组，顺序是 [tn, fp, fn, tp]
    print(f"测试集混淆矩阵：\n{cm}")
    print(f"  TP={tp}（病害被抓到）  FN={fn}（漏报，农业上代价最高）")
    print(f"  FP={fp}（误报）        TN={tn}（健康判健康）")

    # 2.2 按公式手算一遍，再与 sklearn 对照
    acc_manual = (tp + tn) / (tp + tn + fp + fn)
    # 分母为 0 时指标无定义（例如一个正样本都没预测出来），用 nan 表示
    precision_manual = tp / (tp + fp) if (tp + fp) else float("nan")
    recall_manual = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity_manual = tn / (tn + fp) if (tn + fp) else float("nan")#真实健康样本中，被正确判为健康的比例
    f1_manual = (
        2 * precision_manual * recall_manual / (precision_manual + recall_manual)
        if (precision_manual + recall_manual) > 0 else float("nan")
    )
    print("\n按公式手算 vs sklearn 计算（应完全一致）：")
    print(f"  Accuracy    = {acc_manual:.4f} | sklearn {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision   = {precision_manual:.4f} | "
          f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall      = {recall_manual:.4f} | {recall_score(y_test, y_pred):.4f}")
    print(f"  Specificity = {specificity_manual:.4f}")
    print(f"  F1          = {f1_manual:.4f} | {f1_score(y_test, y_pred):.4f}")
    print("\nclassification_report 一次性给出各类的 P/R/F1：")
    print(classification_report(y_test, y_pred, target_names=["健康", "病害"], zero_division=0))
    #target_names=["健康", "病害"] 把 0 和 1 换成中文标签
    print(
        "指标取舍：病害漏报（FN）会导致减产绝收，代价远大于多跑一趟田间复查（FP），\n"
        "因此农业病害预警更看重 Recall；若要兼顾误报成本，用 F1 或 PR-AUC。"
    )

    # 2.3 ROC 曲线、PR 曲线（阈值变化的全景图）
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)#计算不同阈值下的假正率 FPR 和真正率 TPR
    auc = roc_auc_score(y_test, y_proba)
    prec_curve, rec_curve, thr_pr = precision_recall_curve(y_test, y_proba)#计算不同阈值下的精确率和召回率
    ap = average_precision_score(y_test, y_proba)#计算 PR 曲线下的平均精度 AP。AP 越高越好

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["healthy", "disease"]).plot(
        ax=axes[0], cmap="Blues", colorbar=False
    )
    axes[0].set_title("Confusion matrix")
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc).plot(ax=axes[1])
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_title(f"ROC curve (AUC={auc:.3f})")
    PrecisionRecallDisplay(
        precision=prec_curve, recall=rec_curve, average_precision=ap
    ).plot(ax=axes[2])
    axes[2].set_title(f"PR curve (AP={ap:.3f}), baseline={y_test.mean():.3f}")
    fig.tight_layout()
    save_fig(fig, "03b_confusion_roc_pr.png")
    plt.close(fig)
    print(f"\n测试集 ROC-AUC = {auc:.4f}，PR-AP = {ap:.4f}")
    print("（PR 图中的水平虚线是“全部预测为正”的基线，等于正类比例，AP 必须显著高于它）")

    # ================================================================== #
    # 3. 学习曲线：诊断偏差（欠拟合）与方差（过拟合）
    # ================================================================== #
    section("3. 学习曲线 Learning Curve：诊断欠拟合 / 过拟合")
    train_sizes = np.linspace(0.1, 1.0, 8)#np.linspace(0.1, 1.0, 8) 生成 8 个均匀分布的数，从 0.1 到 1.0
    shallow = RandomForestClassifier(
        n_estimators=200, max_depth=2, random_state=RANDOM_STATE, n_jobs=-1
    )  # 容量不足 → 高偏差/欠拟合
    deep = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=1,
        random_state=RANDOM_STATE, n_jobs=-1,
    )  # 容量过剩 → 高方差/过拟合

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, mdl, title in [
        (axes[0], shallow, "Shallow tree (max_depth=2): underfitting"),
        (axes[1], deep, "Deep forest (max_depth=None): overfitting"),
    ]:
        sizes, train_sc, val_sc = learning_curve(
            mdl, X_train, y_train, train_sizes=train_sizes,
            cv=skf, scoring="roc_auc", n_jobs=-1,
        )
        '''
    对每个 train_size（比如 10%、22.86%... 100%），从训练集中取对应比例的数据。
    用这些数据做交叉验证（这里用 skf，即分层 5 折）。
    记录每次交叉验证中：
    模型在训练子集上的得分（train_sc）
    模型在验证折上的得分（val_sc）
    返回值：
    sizes：实际使用的训练样本数量（不是百分比，是具体数字）。因为 train_sizes 给的是比例，函数会自动换算成样本数。
    train_sc：二维数组，形状 (n_train_sizes, n_cv_folds)。每一行对应一个训练集大小，每一列对应一次交叉验证的训练分数。
    val_sc：同样形状，对应验证分数。
        '''
        ax.plot(sizes, train_sc.mean(axis=1), "o-", label="train score")
        ax.plot(sizes, val_sc.mean(axis=1), "s-", label="CV score")
        ax.fill_between(sizes, val_sc.mean(axis=1) - val_sc.std(axis=1),
                        val_sc.mean(axis=1) + val_sc.std(axis=1), alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("training samples")
        ax.set_ylabel("ROC-AUC")
        ax.set_ylim(0.5, 1.02)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "03c_learning_curve.png")
    plt.close(fig)
    print(
        "读图方法：\n"
        " * 训练分和验证分【都低且接近】→ 欠拟合（高偏差）：模型太简单，\n"
        "   应加特征、增大模型（更深的树、更小正则）；给更多数据帮助不大。\n"
        " * 训练分很高、验证分低，两条曲线【裂口很大】→ 过拟合（高方差）：\n"
        "   应加强正则、做特征选择、获取更多训练数据。\n"
        " * 两条曲线都高且收敛在一起 → 拟合良好。"
    )

    # ================================================================== #
    # 4. 验证曲线：定位单个超参数的最佳区间
    # ================================================================== #
    section("4. 验证曲线 Validation Curve：给单个超参数找拐点")
    depth_range = [2, 4, 6, 8, 12, None]
    tr_sc, val_sc = validation_curve(
        RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        X_train, y_train,
        param_name="max_depth",#要调整的超参数名称
        param_range=depth_range,#超参数的候选值列表
        cv=skf, scoring="roc_auc", n_jobs=-1,
    )
    #tr_sc：训练分数，二维数组，形状为 (len(param_range), n_splits)

    x_labels = ["2", "4", "6", "8", "12", "None"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(x_labels, tr_sc.mean(axis=1), "o-", label="train score")
    ax.plot(x_labels, val_sc.mean(axis=1), "s-", label="CV score")
    best_i = int(np.argmax(val_sc.mean(axis=1)))
    ax.axvline(x_labels[best_i], color="red", ls="--",
               label=f"best max_depth={x_labels[best_i]}")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Validation curve for max_depth")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "03d_validation_curve.png")
    plt.close(fig)
    print(f"各 max_depth 的 CV ROC-AUC：{np.round(val_sc.mean(axis=1), 4)}")
    print("CV 曲线先升后降：升段是欠拟合被解除，降段是开始过拟合；拐点即近似最优。")

    # ================================================================== #
    # 5. 正则化：L1 / L2 与正则强度 C
    # ================================================================== #
    section("5. 正则化 Regularization：给模型的“复杂度”加罚")
    print(
        "以逻辑回归为例，训练目标 = 经验损失 + 正则惩罚：\n"
        "  L2（岭/Ridge）：min  CrossEntropy + (1/C)·Σ wᵢ²   权重整体缩小、趋近 0；\n"
        "  L1（Lasso）   ：min  CrossEntropy + (1/C)·Σ |wᵢ|  产生稀疏解、部分权重=0；\n"
        "  ElasticNet    ：L1 与 L2 按比例混合。\n"
        "sklearn 中 C = 1/λ 是正则强度的【倒数】：C 越小，惩罚越重、模型越简单。"
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    c_grid = np.logspace(-3, 3, 13)#生成对数等距的数列，指数代入底数10计算
    coef_paths = []
    import warnings
    with warnings.catch_warnings():
        # 不同 sklearn 版本对 penalty 参数可能有弃用提示，与本节原理无关，屏蔽之
        warnings.simplefilter("ignore", FutureWarning)
        for c in c_grid:
            lr = LogisticRegression(C=c, penalty="l2", max_iter=3000,#最大迭代次数
                                    random_state=RANDOM_STATE)
            lr.fit(X_train_s, y_train)
            coef_paths.append(lr.coef_.ravel())#模型的系数，形状是(1, n_features)
    coef_paths = np.array(coef_paths)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, name in enumerate(feature_names):
        ax.plot(np.log10(c_grid), coef_paths[:, i], label=name.replace("num__", "").replace("cat__", ""))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("log10(C)   (left = stronger regularization)")
    ax.set_ylabel("coefficient value")
    ax.set_title("L2 regularization path: coefficients shrink as C decreases")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "03e_regularization_path.png")
    plt.close(fig)
    print("图中越往左（C 越小）所有系数越被压向 0，模型越简单、越不容易过拟合。")

    # ================================================================== #
    # 6. 早停 Early Stopping（以 GBDT 为例）
    # ================================================================== #
    section("6. 早停 Early Stopping：验证误差不再下降就停止迭代")
    # 再切出一块独立验证集用于监控（也可以用交叉验证的每折验证集）
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train
    )
    gbdt = GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=3,
        random_state=RANDOM_STATE,
    ).fit(X_tr2, y_tr2)

    # staged_predict_proba：返回第 1、2、…、500 棵树时的预测，用于画误差曲线
    train_loss, val_loss = [], []
    for proba in gbdt.staged_predict_proba(X_tr2):
        #staged_predict_proba：它返回一个生成器，每次迭代给出“只用前 k 棵树”的预测概率
        train_loss.append(log_loss(y_tr2, proba))
    for proba in gbdt.staged_predict_proba(X_val):
        val_loss.append(log_loss(y_val, proba))
    best_iter = int(np.argmin(val_loss)) + 1
    #np.argmin(val_loss)：找到 val_loss 中最小值的位置索引，从 0 开始
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(train_loss, label="train log-loss")
    ax.plot(val_loss, label="validation log-loss")
    ax.axvline(best_iter, color="red", ls="--", label=f"best iteration ≈ {best_iter}")
    ax.set_xlabel("number of boosting trees")
    ax.set_ylabel("log-loss")
    ax.set_title("Early stopping: stop when validation loss bottoms out")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "03f_early_stopping.png")
    plt.close(fig)
    print(f"验证集 log-loss 在第 {best_iter} 棵树左右最低，之后继续训练只会过拟合。")

    # sklearn 内置自动早停：连续 n_iter_no_change 轮验证分数没有改善就停
    gbdt_es = GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=3,
        validation_fraction=0.2, n_iter_no_change=20,
        #validation_fraction=0.2：从训练集中自动切出 20% 作为验证集，用于监控
        #n_iter_no_change=20：连续 20 轮验证分数没有改善，就停止训练
        random_state=RANDOM_STATE,
    ).fit(X_train, y_train)
    print(f"开启 n_iter_no_change=20 后，模型实际只训练了 "
          f"{gbdt_es.n_estimators_} 棵树就自动停止（XGBoost/LightGBM 的 early_stopping_rounds 同理）。")
          #gbdt_es.n_estimators_：训练结束后，实际使用的树的数量

    # ================================================================== #
    # 7. 综合优化闭环
    # ================================================================== #
    section("7. 综合闭环：Pipeline + 分层CV + 网格搜索 + 测试集终评")
    # 注意这里在【原始未变换】的 X/y 上重建流水线，让预处理也进入每折 CV
    df = load_agri_data()
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    from sklearn.model_selection import GridSearchCV
    final_pipe = Pipeline(
        [
            ("preprocess", ColumnTransformer(
                [
                    ("num", SimpleImputer(strategy="median"), NUMERIC_COLS),
                    ("cat", Pipeline([
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]), CATEGORICAL_COLS),
                ]
            )),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )
    param_grid = {
        "clf__n_estimators": [150, 300],
        "clf__max_depth": [3, 5, 8],
        "clf__min_samples_leaf": [1, 4],
    }
    search = GridSearchCV(
        final_pipe, param_grid, scoring="roc_auc",
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    ).fit(X_train_raw, y_train_raw)
    print(f"最优参数：{search.best_params_}")
    print(f"最优 CV ROC-AUC = {search.best_score_:.4f}")
    final_pred = search.predict(X_test_raw)
    final_proba = search.predict_proba(X_test_raw)[:, 1]
    print("测试集最终成绩：")
    print(classification_report(y_test_raw, final_pred,
                                target_names=["健康", "病害"], zero_division=0))
    print(f"测试集 ROC-AUC = {roc_auc_score(y_test_raw, final_proba):.4f}，"
          f"F1 = {f1_score(y_test_raw, final_pred):.4f}")
    print(
        "⚠ 注意这里出现的典型“不平衡陷阱”：ROC-AUC 并不低，但默认 0.5 阈值下\n"
        "  F1=0、一个病害都没报出来——这正是第 04 章要专门解决的问题\n"
        "  （类别权重 / 重采样 / 阈值移动）。"
    )
    print(
        "\n优化方法论总结（按顺序排查）：\n"
        " 1) 学习曲线判断是欠拟合还是过拟合；\n"
        " 2) 欠拟合 → 加特征/特征交叉、换更强模型、放松正则；\n"
        "    过拟合 → 加正则/剪枝/早停、特征选择、补数据；\n"
        " 3) 验证曲线/网格/随机搜索定位超参数；\n"
        " 4) 用与业务代价匹配的指标（病害预警看 Recall/F1/PR-AUC）定方案；\n"
        " 5) 测试集只在最后用一次，给出无偏的上线预期。"
    )
    print("\n全部完成，图表见 output/03a~03f。")


if __name__ == "__main__":
    main()
