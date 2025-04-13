import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from preprocess import load_and_preprocess


def train_svm():
    # 加载数据：假设返回 (x_train, y_train), (x_val, y_val), (x_test, y_test)
    (_, _), (_, _), (x_test, y_test) = load_and_preprocess("single_patient_dataset/full_dataset_chb15.h5")
    # 对于 SVM，将多维数据展平
    X = x_test.reshape(x_test.shape[0], -1)
    y = np.array(y_test)

    # 创建 SVM 模型（启用概率估计用于计算 AUC）
    svm = SVC(probability=True, kernel='rbf', random_state=42)
    svm.fit(X, y)

    # 预测与评估
    y_pred = svm.predict(X)
    y_prob = svm.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc_val = roc_auc_score(y, y_prob)
    print("SVM Metrics:")
    print("Accuracy: {:.4f}, F1: {:.4f}, AUC: {:.4f}".format(acc, f1, auc_val))
    return acc, f1, auc_val


if __name__ == "__main__":
    train_svm()
