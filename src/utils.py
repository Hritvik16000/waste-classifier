import random, os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ConfusionMatrixPlotter:
    def __init__(self, class_names):
        self.class_names = class_names
    def save(self, y_true, y_pred, out_path="reports/cm.png"):
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(self.class_names))))
        cmn = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        plt.figure(figsize=(7,6))
        import seaborn as sns
        sns.heatmap(cmn, annot=True, fmt=".2f",
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
