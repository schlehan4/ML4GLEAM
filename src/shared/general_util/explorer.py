import numpy as np
import wandb

class Explorer():
    def __init__(self):
        self.actual_inputs = None  # Kein leeres Array, sondern `None` für Initialisierung
        self.actual_labels = None
        self.actual_predictions = None
        self.randomized_sample = []

    def setup_actual_batch(self, inputs, labels, predictions):
        # just saves the data and converts it to numpy, and set it to cpu if not there already
        self.actual_inputs = inputs.cpu().numpy()
        self.actual_labels = labels.cpu().numpy()
        self.actual_predictions = predictions.cpu().numpy()

    def add_randomized_sample(self, probability=0.2, num_samples=1):
        if self.actual_inputs is None:
            print("Warnung: Kein Batch gespeichert. `setup_actual_batch()` muss zuerst aufgerufen werden!")
            return

        if np.random.rand() < probability:  # Entscheidet, ob ein Sample gespeichert wird
            total_samples = len(self.actual_inputs)
            indices = np.arange(total_samples)

            # Zufällige Auswahl eines Samples aus dem aktuellen Batch
            selected_idx = np.random.choice(indices, size=num_samples, replace=False)

            # Speichere das zufällige Sample
            self.randomized_sample.append((
                self.actual_inputs[selected_idx],  # Bereits ein NumPy-Array
                self.actual_labels[selected_idx],
                self.actual_predictions[selected_idx]
            ))

    def log_sample_to_wandb(self):
        """
        Loggt die gespeicherten Samples mit Labels und Predictions zu wandb.
        """
        if not self.randomized_sample:
            print("Keine gesammelten Samples zum Loggen!")
            return

        images = []
        for img, label, pred in self.randomized_sample:
            # wandb take numpy array with shape (Height, Width, Channels)
            # but you have (Channels, Height, Width)
            # this transpose changes (C, H, W) to (H, W, C)
            img = img.squeeze(0)
            img = np.transpose(img, (1, 2, 0))
            # then you collect the wandb images
            images.append(wandb.Image(img, caption=f"True: {label}, Pred: {pred}"))

        # and log them to the cloud...
        wandb.log({"examples": images})
