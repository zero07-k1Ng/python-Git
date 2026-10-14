# -*- coding: utf-8 -*-
import time

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401 必须先导入
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from agri_data import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    RANDOM_STATE,
    TARGET,
    load_agri_data,
    save_fig,
    section,
)

def build_basic_pipeline(classifier) -> Pipeline:
    #数值特征的缺失填充
    numeric_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])
    #SimpleImputer：负责填补缺失
    #strategy="median"（中位数）：因为这份数据里 humidity 有 15 条 120~180 的异常值。平均值会被这些异常值拉偏
    #类别特征的缺失填充 + 独热编码
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),#类别特征不能算中位数，所以用"出现最多的类别"来填
            # handle_unknown='ignore'：验证/测试集出现训练集没见过的类别时不报错,设成 "ignore" 就会把未知类别全部编成 0，不崩溃
            ("onehot", OneHotEncoder(handle_unknown="ignore")),#独热编码的做法：给每个类别单独开一列，是它就填 1，不是就填 0。
        ]
    )
    #对不同列做不同处理
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_COLS),# 这几列走数值流程
            ("cat", categorical_pipe, CATEGORICAL_COLS),# 这几列走类别流程
        ]
    )
    return Pipeline(steps=[("preprocess", preprocess), ("clf", classifier)])
    #[("preprocess", preprocess)：预处理步骤，("clf", classifier)：分类器


def cv_auc(model, X, y, cv) -> float:
    """返回交叉验证 ROC-AUC 的均值。"""
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def main() -> None:
    # ------------------------------------------------------------------ #
    # 加载数据，按 8:2 分层切分训练集 / 测试集
    # stratify=y：保证训练集、测试集中病害比例一致（不平衡数据必须这样做）
    # 测试集在整个调参过程中被“锁进柜子”，只在最后评估时用一次
    # ------------------------------------------------------------------ #
    df = load_agri_data()
    X = df.drop(columns=TARGET)#去掉答案列，剩下的都是特征
    y = df[TARGET]#只留答案列
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,#保证训练集和测试集的病害比例一致,分层（Stratify）
    )

    # 分层 5 折交叉验证：每一折里正负样本比例都与整体一致
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    #n_splits=5 = 切 5 折

    section("0. 调参前的基线模型（全部使用默认超参数）")
    baseline = build_basic_pipeline(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    )
    base_auc = cv_auc(baseline, X_train, y_train, cv)
    print(f"默认参数随机森林，5 折 CV 平均 ROC-AUC = {base_auc:.4f}")

    # ================================================================== #
    # 1. 网格搜索 GridSearchCV
    # ================================================================== #
    section("1. 网格搜索 GridSearchCV：穷举所有超参数组合")
    # 注意：流水线里参数名要加“步骤名__”前缀（两根下划线）
    grid = {
        "clf__n_estimators": [100, 200],       # 森林里树的数量
        "clf__max_depth": [4, 8, None],        # 每棵树最大深度，None=不限制
        "clf__min_samples_leaf": [1, 4],       # 叶子节点最少样本数（正则）
        "clf__max_features": ["sqrt", 0.6],    # 每次分裂考虑的特征比例
        # "sqrt"：考虑 √(总特征数) 个。比如 12 个特征，√12 ≈ 3.5 → 考虑约 3 个
    }
    '''
    sklearn 里，带下划线的属性是 fit() 之后才有的（学出来的结果）。
    best_score_、best_params_、cv_results_ 都是 fit 之后才存在
    '''
    n_combinations = int(np.prod([len(v) for v in grid.values()]))
    #np.prod(...)连乘：2 × 3 × 2 × 2 = 24
    print(
        f"超参数候选空间：{grid}\n"
        f"组合总数 N = 2×3×2×2 = {n_combinations}，"
        f"按 5 折 CV 共需训练 {n_combinations * 5} 次模型。"#每组参数都要做 5 折交叉验证（每组参数训练 5 次）
    )

    grid_search = GridSearchCV(
        estimator=build_basic_pipeline(
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        ),#estimator：要调优的模型
        param_grid=grid,#要搜的网格
        scoring="roc_auc",   # 用 ROC-AUC 作为每组参数的评分（不平衡数据比准确率合适）
        cv=cv,#怎么交叉验证,用之前定义好的 StratifiedKFold(5)
        n_jobs=-1,
        refit=True,          # 找到最优的一组参数后，在全训练集上再训一次
        return_train_score=True,#把"训练集上的分数"也记录下来
    )
    t0 = time.time()
    grid_search.fit(X_train, y_train)
    grid_time = time.time() - t0
    print(f"\n网格搜索耗时 {grid_time:.1f} 秒")
    print(f"最优参数：{grid_search.best_params_}")
    print(f"最优 CV ROC-AUC = {grid_search.best_score_:.4f}")

    # cv_results_ 是一个大字典，转成 DataFrame 可查看每一组参数的详细成绩
    grid_res = pd.DataFrame(grid_search.cv_results_)
    print("\n网格搜索成绩前 5 名（按交叉验证得分排序）：")
    print(
        grid_res.sort_values("rank_test_score")[
            [
                "rank_test_score",
                "param_clf__n_estimators",
                "param_clf__max_depth",
                "param_clf__min_samples_leaf",
                "param_clf__max_features",
                "mean_test_score",
                "std_test_score",
            ]#只挑出这几列
        ].head(5).to_string(index=False)#打印时不显示行号
    )

    # ================================================================== #
    # 2. 随机搜索 RandomizedSearchCV
    # ================================================================== #
    section("2. 随机搜索 RandomizedSearchCV：从概率分布中随机抽样")
    # 离散超参数给候选列表；连续超参数直接给“分布”，搜索时按分布随机采样
    # randint(a, b)：在 [a, b) 上均匀抽整数；uniform(a, b)：在 [a, a+b) 上均匀抽浮点
    distributions = {
        "clf__n_estimators": randint(100, 400),#在 [100, 400) 上均匀抽整数。
        "clf__max_depth": [4, 6, 8, 12, None],#随机从里面挑
        "clf__min_samples_leaf": randint(1, 9),
        "clf__max_features": uniform(0.5, 0.4),   # 0.5 ~ 0.9 之间的连续值
        #uniform(loc, scale)在 [loc, loc + scale) 上均匀分布
        "clf__criterion": ["gini", "entropy"],#指定"不纯度衡量标准",候选值列表：gini 或 entropy，二选一
        #在随机搜索里：每次抽一组参数时，从 ["gini", "entropy"] 里随机挑一个。
    }
    n_iter = 24  #抽 24 组，训练次数 = 24×5 = 120，与上面的网格搜索预算相同
    #n_iter 越大，探索越充分，但越慢
    random_search = RandomizedSearchCV(
        estimator=build_basic_pipeline(
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        ),
        param_distributions=distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        random_state=RANDOM_STATE,
        return_train_score=True,
    )
    t0 = time.time()
    random_search.fit(X_train, y_train)
    random_time = time.time() - t0
    print(f"随机搜索（n_iter={n_iter}）耗时 {random_time:.1f} 秒")
    print(f"最优参数：{random_search.best_params_}")
    print(f"最优 CV ROC-AUC = {random_search.best_score_:.4f}")

    rand_res = pd.DataFrame(random_search.cv_results_)
    print("\n随机搜索成绩前 5 名：")
    cols = [
        "rank_test_score",
        "param_clf__n_estimators",
        "param_clf__max_depth",
        "param_clf__min_samples_leaf",
        "param_clf__max_features",
        "param_clf__criterion",
        "mean_test_score",
    ]
    print(
        rand_res.sort_values("rank_test_score")[cols]
        .head(5)
        .to_string(index=False)
    )

    # ================================================================== #
    # 3. 网格 vs 随机：同等预算下谁探索得更充分？
    # ================================================================== #
    section("3. 网格搜索 vs 随机搜索 对比")
    compare = pd.DataFrame(
        {
            "方法": ["不调参(默认)", "网格搜索", "随机搜索"],
            "CV_ROC_AUC": [
                base_auc,
                grid_search.best_score_,
                random_search.best_score_,
            ],
            "耗时(秒)": [np.nan, round(grid_time, 1), round(random_time, 1)],
            "训练组合数": [1, n_combinations, n_iter],
        }
    )
    print(compare.to_string(index=False))
    print(
        "\n结论：\n"
        " * 网格搜索【可复现、不遗漏】，但组合数随超参数个数指数增长，"
        "连续超参数（如 max_features=0.63）根本无法穷举；\n"
        " * 随机搜索的预算只由 n_iter 决定，与超参数维度无关，且每个重要超参数"
        "都能抽到多个不同取值，实践中常以更小/相同预算得到不输网格搜索的结果。"
    )

    # 图：把两种搜索的所有组合得分从高到低排序，直观对比“成绩分布”
    fig, ax = plt.subplots(figsize=(8, 4.5))

    grid_sorted = np.sort(grid_res["mean_test_score"])[::-1]
    rand_sorted = np.sort(rand_res["mean_test_score"])[::-1]
    ax.plot(range(1, len(grid_sorted) + 1), grid_sorted, "o-", label="Grid Search")
    ax.plot(range(1, len(rand_sorted) + 1), rand_sorted, "s-", label="Random Search")
    ax.set_xlabel("Hyperparameter combination (rank)")
    ax.set_ylabel("CV ROC-AUC")
    ax.set_title("Grid vs Random Search: CV score of all combinations")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.savefig("output/01_grid_vs_random_scores.png")
    plt.show()

    # ================================================================== #
    # 4. Successive Halving（逐次减半）：资源自适应的搜索
    # ================================================================== #
    section("4. 逐次减半搜索 HalvingXxxSearchCV（迭代减半算法）")
    print(
        "思想（两阶段）：\n"
        " ① 先让【所有】候选组合只用【很小一部分】训练数据训练，筛掉明显差的；\n"
        " ② 每轮把存活组合数【减半】(factor=3 即只剩 1/3)，"
        "同时给存活者分配更多数据，直到数据用完。\n"
        "好处：把算力集中在有希望的组合上，比普通网格/随机搜索更省时间。"
    )
    # min_resources：第一轮给每个候选的最少样本数。不能设得太小，否则某一折里
    # 可能一个病害样本都没有，ROC-AUC 无法定义（sklearn 会报 nan 警告）
    halving_grid = HalvingGridSearchCV(
        estimator=build_basic_pipeline(
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        ),
        param_grid=grid,
        factor=3,                 # 每轮保留比例的倒数
        min_resources=120,        #第 1 轮每个候选用的样本数
        # 首轮资源：120 → 360 → 1080
        # (截断为全量:但训练集一共才 800 个。所以第 3 轮直接用全部 800 个)
        #首轮资源120*factor=第2轮资源，第2轮资源360*factor=第3轮资源
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ).fit(X_train, y_train)
    print(f"HalvingGrid 最优参数：{halving_grid.best_params_}")
    print(f"HalvingGrid 最优 CV ROC-AUC = {halving_grid.best_score_:.4f}")

    halving_random = HalvingRandomSearchCV(
        estimator=build_basic_pipeline(
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        ),
        param_distributions=distributions,
        n_candidates=24,#从分布里随机抽 24 个候选，进入第 1 轮。
        factor=3,
        min_resources=120,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ).fit(X_train, y_train)
    print(f"HalvingRandom 最优参数：{halving_random.best_params_}")
    print(f"HalvingRandom 最优 CV ROC-AUC = {halving_random.best_score_:.4f}")
    print(
        "\n口径提示：Halving 的 best_score_ 是【最后一轮、所用数据子集上】的 CV 分数，\n"
        "与普通搜索在全量训练集上的 CV 分数不能直接横向比较；它的核心价值是\n"
        "“用同样时间能筛更多候选”。确定参数后仍应用 refit 出的模型在测试集评估。"
    )

    # ================================================================== #
    # 5. 用测试集做“一锤定音”的最终评估
    # ================================================================== #
    section("5. 在全程未参与调参的测试集上做最终评估")
    from sklearn.metrics import roc_auc_score

    for name, search in [
        ("默认参数", baseline),
        ("GridSearchCV 最优", grid_search),
        ("RandomizedSearchCV 最优", random_search),
    ]:
        if name == "默认参数":
            search.fit(X_train, y_train)
        test_auc = roc_auc_score(y_test, search.predict_proba(X_test)[:, 1])
        #: 表示"所有行"，1 表示"第 2 列"
        print(f"{name:24s} 模型的测试集 ROC-AUC = {test_auc:.4f}")
    print(
        "\n注意：绝对不能用测试集成绩反过来挑选超参数——否则测试集信息泄漏，"
        "成绩会虚高、上线后翻车。测试集只允许在方案完全确定后使用【一次】。"
    )

    # ================================================================== #
    # 6. 延伸：贝叶斯优化（Optuna）
    # ================================================================== #
    section("6. 延伸阅读：贝叶斯优化（以 Optuna 为例）")
    print(
        "网格/随机搜索都【不利用】已尝试组合的成绩，彼此独立。\n"
        "贝叶斯优化则用一个“代理模型”根据历史成绩建模“超参数→得分”的响应面，\n"
        "智能地决定下一组尝试什么（在 探索 exploration 与 利用 exploitation 间平衡），\n"
        "通常 20~50 次试验就能超过随机搜索上百次的效果。XGBoost/LightGBM 调参首选。\n"
        "安装：pip install optuna。标准写法见下方代码（取消注释即可运行）："
    )
    OPTUNA_DEMO = '''
    import optuna
    from sklearn.model_selection import cross_val_score

    def objective(trial: optuna.Trial) -> float:
        # 告诉 Optuna 每个超参数的搜索空间（支持 int/float/categorical/log 均匀）
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        }
        model = build_basic_pipeline(
            RandomForestClassifier(random_state=42, n_jobs=-1, **params)
        )
        # 目标函数返回交叉验证得分；Optuna 会自动最大化它
        return cross_val_score(
            model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        ).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40)          # 只做 40 次试验
    print(study.best_params, study.best_value)
    best_model = build_basic_pipeline(
        RandomForestClassifier(random_state=42, n_jobs=-1, **study.best_params)
    ).fit(X_train, y_train)
    '''
    print(OPTUNA_DEMO)
    print("全部完成。图表见 output/01_grid_vs_random_scores.png")


if __name__ == "__main__":
    main()
    #如果这个文件是直接被运行的，就执行 main() 函数；
    #如果这个文件是被别的文件 import 的，就不执行 main()。
