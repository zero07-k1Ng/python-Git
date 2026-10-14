import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)

# 二分类真实标签与预测
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 0])
y_prob = np.array([0.9, 0.1, 0.8, 0.4, 0.2, 0.7, 0.3, 0.6, 0.85, 0.05])#模型预测为1的概率

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)#混淆矩阵
print("混淆矩阵：")
print(cm)
print("行：真实类别，列：预测类别；[[TN, FP], [FN, TP]]")
'''
TN：True Negative，真实是 0，预测也是 0
FP：False Positive，真实是 0，预测成 1
FN：False Negative，真实是 1，预测成 0
TP：True Positive，真实是 1，预测也是 1
'''
# 各项指标
acc = accuracy_score(y_true, y_pred)        # 准确率:准确率在类别不平衡时可能骗人
precision = precision_score(y_true, y_pred) # 精确率: 预测为 1 的样本中，真正是 1 的比例
recall = recall_score(y_true, y_pred)       # 召回率:真实为 1 的样本中，被模型找出来的比例
f1 = f1_score(y_true, y_pred)               # F1分数:2× (Precision×Recall)/(Precision+Recall)
auc = roc_auc_score(y_true, y_prob)         # AUC:用的是y_prob,AUC需要模型输出的概率
#横轴是假正率（FPR），纵轴是真正率（TPR）。
#通过改变分类阈值（比如从 0.1 到 0.9），得到不同的 (FPR, TPR) 点，连成曲线
#AUC：ROC 曲线下的面积
#AUC = 1：完美分类器。
#AUC = 0.5：和随机猜差不多。
#AUC < 0.5：比随机猜还差（可能标签反了）
#AUC的含义：随机取一个正样本（真实为 1）和一个负样本（真实为 0），模型给正样本的预测概率高于负样本的概率的可能性
#模型区分正负样本的能力
print("\n分类评估指标：")
print(f"准确率 Accuracy: {acc:.4f}")
print(f"精确率 Precision: {precision:.4f}")
print(f"召回率 Recall: {recall:.4f}")
print(f"F1分数: {f1:.4f}")
print(f"AUC: {auc:.4f}")
