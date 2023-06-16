import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score, f1_score


def calc_classification_metrics(pred_labels, true_labels, log_dir):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    # save targets
    labels_save_path = os.path.join(log_dir, "labels")
    os.makedirs(labels_save_path, exist_ok=True)
    np.save(os.path.join(labels_save_path, "predicted_labels.npy"), pred_labels)
    np.save(os.path.join(labels_save_path, "true_labels.npy"), true_labels)

    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)
    cm = confusion_matrix(true_labels, pred_labels)
    df = pd.DataFrame(r)
    df["cohen"] = cohen_kappa_score(true_labels, pred_labels)
    df["accuracy"] = accuracy_score(true_labels, pred_labels)
    df["macro_f1"] = f1_score(true_labels, pred_labels, average="macro")
    df["micro_f1"] = f1_score(true_labels, pred_labels, average="micro")
    df["weighted_f1"] = f1_score(true_labels, pred_labels, average="weighted")
    df = df * 100

    # save classification report
    exp_name = os.path.split(os.path.dirname(log_dir))[-1]  # 获取父文件夹名字
    training_mode = os.path.basename(log_dir)  # 获取路径叶结点名
    file_name = f"{exp_name}_{training_mode}_classification_report.xlsx"
    report_Save_path = os.path.join(log_dir, file_name)
    df.to_excel(report_Save_path)

    # save confusion matrix
    cm_file_name = f"{exp_name}_{training_mode}_confusion_matrix.torch"
    cm_Save_path = os.path.join(log_dir, cm_file_name)
    torch.save(cm, cm_Save_path)
    
    return df["accuracy"][0], df["macro_f1"][0], df["micro_f1"][0], df["weighted_f1"][0]
    