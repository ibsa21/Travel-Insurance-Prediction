import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm
from scipy.stats import ttest_ind

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

def perform_t_test(data, group_column, feature_column):
    group1 = data[data[group_column] == "Yes"][feature_column]
    group2 = data[data[group_column] == "No"][feature_column]

    t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
    return t_stat, p_value, group1

def perform_proportion_test(data, group_column, feature_column, value_mapping):
    group1 = data[data[group_column] == 1][feature_column]
    group2 = data[data[group_column] == 0][feature_column]

    count_group1 = group1.map(value_mapping).sum()
    count_group2 = group2.map(value_mapping).sum()

    nobs_group1 = len(group1)
    nobs_group2 = len(group2)

    prop_group1 = count_group1 / nobs_group1
    prop_group2 = count_group2 / nobs_group2

    p_pool = (count_group1 + count_group2) / (nobs_group1 + nobs_group2)

    z_stat = (prop_group1 - prop_group2) / np.sqrt(
        p_pool * (1 - p_pool) * ((1 / nobs_group1) + (1 / nobs_group2))
    )
    p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))

    return z_stat, p_value

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1


def plot_classification_report(
    y_true, y_pred, class_names, title="Classification Report"
):
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.imshow(
        [
            [
                report[label]["precision"],
                report[label]["recall"],
                report[label]["f1-score"],
            ]
            for label in class_names
        ],
        interpolation="nearest",
        cmap=plt.cm.Blues,
    )
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.ylabel("Metrics")
    plt.xlabel("Classes")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def plot_roc_curve(y_true, y_pred_prob, title="ROC Curve"):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()