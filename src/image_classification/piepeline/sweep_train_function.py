import os
import uuid
import numpy as np
import torch
import wandb
from shared.general_util.general_helper_functions import log_cfm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from shared.general_util.metric_logger import MetricLogger
from shared.general_util.explorer import Explorer
from shared.general_util.l1_regularization import l1_penalty

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, config, enable_logging=False):
    """
    Train the model for the specified number of epochs

    :param model: model to train
    :param train_loader: dataloader with training set
    :param val_loader: dataloader with validation set
    :param criterion: loss function for network (aka criterion)
    :param optimizer: optimizer for backpropagation
    :param scheduler: learning rate scheduler
    :param num_epochs: number of epochs to train the model
    :param config: whole wandb configuration, to set all the hyperparameters
    :return: final validation accuracy of the trained model
    """
    # val loss for scheduler
    validation_loss_sched = 0

    # actual uuid of the model and all the metrics
    actual_uuid = uuid.uuid4().hex[:8]

    # ground folder to save the model metrics and model itself
    dist_ground = ""

    if config.run_on_cluster:
        # path on cluster
        dist_ground = f"/scratch/artifacts/{actual_uuid}"
    else:
        # path local, on notebook
        dist_ground = f"./artifacts/{actual_uuid}"
    # subfolders to save the model metrics
    dist_metric = f"{dist_ground}/metrics"
    dist_model = f"{dist_ground}/model"

    if enable_logging:
        # generate the folders, if not there
        os.makedirs(dist_ground, exist_ok=True)
        os.makedirs(dist_metric, exist_ok=True)
        # os.makedirs(dist_confusion, exist_ok=True)
        os.makedirs(dist_model, exist_ok=True)

    metric_logger = MetricLogger(save_path=f"{dist_metric}/metrics_{actual_uuid}.json")

    # Train the model for the specified number of epochs
    for epoch in range(num_epochs):
        # Set the model to train mode
        model.train()

        # Initialize the running loss and accuracy
        running_loss = 0.0

        all_preds = []
        all_labels = []

        # Iterate over the batches of the train loader
        for inputs, labels in train_loader:
            # Move the inputs and labels to the device
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)

            # Zero the optimizer gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss = l1_penalty(loss=loss, model=model, l1_lambda=config.l1lambda) # with 0.001 good against overfitting, but too strong...
            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            # Speichere Vorhersagen, Labels & Wahrscheinlichkeiten für sklearn asd
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update the running loss and accuracy
            running_loss += loss.item() * inputs.size(0)

            torch.cuda.empty_cache()

        # Umwandlung in NumPy-Arrays für sklearn
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        # Berechnung der Metriken mit sklearn
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_labels, all_preds)
        train_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        train_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        train_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        # log metrics to wandb
        if enable_logging:
            # log locally
            metric_logger.log("train", train_acc, train_loss, train_precision, train_recall, train_f1)

            # log to cloud
            wandb.log({"epoch": epoch,
                       "train_acc": train_acc,
                       "train_loss": train_loss,
                       "train_precision": train_precision,
                       "train_recall": train_recall,
                       "train_f1_score": train_f1})

        # Set the model to evaluation mode
        model.eval()

        explorer = Explorer()

        # Iterate over the batches of the validation loader
        with torch.no_grad():
            running_loss = 0.0
            all_preds = []
            all_labels = []

            for inputs, labels in val_loader:
                # Move the inputs and labels to the device
                inputs = inputs.to(config.device)
                labels = labels.to(config.device)

                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                pred = preds.cpu().numpy()
                label = labels.cpu().numpy()
                # Store predictions, labels
                all_preds.extend(pred)
                all_labels.extend(label)

                # add stuff to explorer to sampel img, labels, preds
                # save batch temporary
                explorer.setup_actual_batch(inputs, labels, preds)

                # take a sample with probability of p
                explorer.add_randomized_sample(probability=0.2, num_samples=1)

                # Update the running loss and accuracy
                running_loss += loss.item() * inputs.size(0)

                torch.cuda.empty_cache()

            # Convert lists to NumPy arrays for sklearn
            all_labels = np.array(all_labels)
            all_preds = np.array(all_preds)

            # Compute validation metrics
            val_loss = running_loss / len(val_loader.dataset)
            validation_loss_sched = val_loss
            val_acc = accuracy_score(all_labels, all_preds)
            val_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            val_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

            # log metrics of validations set to wandb
            if enable_logging:
                # log locally, confusion is also saved
                metric_logger.log("validation", val_acc, val_loss, val_precision, val_recall, val_f1, all_labels,
                                  all_preds)

                # log to cloud
                wandb.log({"epoch": epoch,
                           "val_acc": val_acc,
                           "val_loss": val_loss,
                           "val_precision": val_precision,
                           "val_recall": val_recall,
                           "val_f1_score": val_f1})

                # log the collected samples in the explorer to wandb
                explorer.log_sample_to_wandb()

                # log confusion matrix
                log_cfm(all_labels, all_preds)

            final_val_acc_for_optim = val_acc

        # learning rate scheduler after each epoch, default is exponential, here plateau
        scheduler.step(validation_loss_sched)

        # Print the epoch results
        print('Epoch [{}/{}], train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}'
              .format(epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc))

    if enable_logging:
        # save model params
        actual_model_path = f"{dist_model}/model_{actual_uuid}.pth"
        torch.save(model.state_dict(), actual_model_path)

        artifact = wandb.Artifact(name="actual_model", type="model")
        artifact.add_file(actual_model_path)
        wandb.log_artifact(artifact)

    return final_val_acc_for_optim
