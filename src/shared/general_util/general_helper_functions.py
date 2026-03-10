import os
import tarfile
import numpy as np
import wandb

def create_tarball(output_filename, source_dir):
    """
    Use this function to wrap your results and save it to the projects folder on the cluster
    :param output_filename:
    :param source_dir:
    :return:
    """
    # Erstelle einen gzip-komprimierten Tarball
    with tarfile.open(output_filename, "w:gz") as tar:
        # Füge das Quellverzeichnis hinzu. arcname ändert den Pfad im Archiv.
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def log_cfm(y_true, y_pred):
    """
    Loggt die Confusion Matrix direkt in Weights & Biases.

    Parameter:
    - y_true: Echte Labels als NumPy-Array
    - y_pred: Vorhersagen als NumPy-Array
    """
    labels = np.unique(y_true)  # Klassenlabels ermitteln
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_true,
        preds=y_pred,
        class_names=labels
    )})
