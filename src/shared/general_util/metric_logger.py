import json
import os
from sklearn.metrics import confusion_matrix


class MetricLogger:
    def __init__(self, save_path="metrics.json"):
        self.save_path = save_path
        self.metrics = {
            "train": {"acc": [], "loss": [], "precision": [], "recall": [], "f1": []},
            "validation": {"acc": [], "loss": [], "precision": [], "recall": [], "f1": []},
            "test": {"acc": [], "loss": [], "precision": [], "recall": [], "f1": []},
            "confusion": [],
            "finally": {"acc": 0, "loss": 0, "precision": 0, "recall": 0, "f1": 0}
        }

    def log(self, phase, acc, loss, precision, recall, f1, y_true=None, y_pred=None):
        if phase not in self.metrics:
            raise ValueError("Phase must be 'train', 'validation' or 'test'")

        self.metrics[phase]["acc"].append(acc)
        self.metrics[phase]["loss"].append(loss)
        self.metrics[phase]["precision"].append(precision)
        self.metrics[phase]["recall"].append(recall)
        self.metrics[phase]["f1"].append(f1)

        # constructs and saves only when set...
        if y_true is not None and y_pred is not None:
            cm = confusion_matrix(y_true, y_pred, labels=None).tolist()
            self.metrics["confusion"].append(cm)

    def save_log(self):

        # calculate final metrics
        self.metrics["finally"]["acc"] = self.metrics["validation"]["acc"][-1]
        self.metrics["finally"]["loss"] = self.metrics["validation"]["loss"][-1]
        self.metrics["finally"]["precision"] = self.metrics["validation"]["precision"][-1]
        self.metrics["finally"]["recall"] = self.metrics["validation"]["recall"][-1]
        self.metrics["finally"]["f1"] = self.metrics["validation"]["f1"][-1]

        with open(self.save_path, "w") as f:
            json.dump(self.metrics, f, indent=4)

    def load_log(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, "r") as f:
                self.metrics = json.load(f)
        else:
            print("No previous log file found.")
