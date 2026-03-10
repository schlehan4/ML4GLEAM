from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet34_Weights
from torchvision.models import ResNet50_Weights

import os, argparse, glob

import wandb
from torchvision.transforms import InterpolationMode
from omegaconf import OmegaConf
import torch
from torchvision import models, transforms
from shared.dataset.dataset import preprocess
from shared.dataset.dataset import FitsSet

from pipeline.image_classification_pipeline.template.heads.resnet50_head import RGZ_ResNet50Head
from pipeline.image_classification_sweep.template.sweep_pipeline import Pipeline
from pipeline.image_classification_sweep.template.sweep_train_function import train as train_func

def objective(config):
    """
    um das Skript lokal auszuführen:

    1. pip install -r requirements.txt
    your path here -> set the path to the root of the repo / project
    2. export PYTHONPATH=$PYTHONPATH:/home/mercus/PycharmProjects/BA_ML4FGLEAM
    3. python3 single_experiment_script.py 

    """
    print("Starting new experiment...")

    # datapaths
    dest_path = f"/raid/persistent_scratch/schlehan/jobs/fits_test"  # here?
    anno_tar = f"{dest_path}/AugementedAnnotations.tar"
    img_tar = f"{dest_path}/AugmentedFITSImages.tar"
    anno_path = f"{dest_path}/annotations/"  # TODO: correct the schreibfehler when needed, is this a bug?
    img_path = f"{dest_path}/images/"
    json_path = f"{dest_path}/coco_annotation.json"

    # create experiment / pipeline
    exp = Pipeline(config=config, train_func=train_func)
    print("Experiment / pipeline created...")

    # transform function for dataset, needs to be done here
    # data augmentation can be done in this function
    transform = transforms.Compose([
        # TODO: just try the padding to avoid artifact in scify image data
        # transforms.Pad((46, 46, 46, 46), fill=0),  # Pads from 132×132 to 224×224 # this is not the best idea -> since you would need to transform the validation set also...
        transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Drei Kanäle normalisieren
    ])

    # load the dataset, here cifar 10
    # use the coco or fits dataloader here later -> schlehan's code
    preprocess(img_tar, anno_tar, anno_path, img_path, json_path, ".fits")
    dataset = FitsSet(img_path, json_path, transform=transform)

    # set up the dataloaders in the pipeline, make train / val split
    exp.setup_dataloaders(dataset=dataset, train_size=0.8, val_size=0.2)  # creates also a test set and test loader
    print("DataLoaders created...")

    # Load the pre-trained model, here you can change the model...
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = RGZ_ResNet50Head(model, in_features=2048, num_classes=6, dropout_rate=config.dropout)

    # log gradients...
    wandb.watch(model, log="all", log_freq=10)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum,
                                weight_decay=config.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=config.learning_rate,
    #                              betas=config.betas,
    #                              eps=config.eps,
    #                              weight_decay=config.weight_decay)

    # learning-rate scheduler for learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # setup model with all stuff in the pipeline
    exp.setup_model(model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    scheduler=scheduler,
                    )
    print("Model set  up...")

    # all setup is done, the experiment can now be started

    print("Starting the experiment... / starting the training...")
    # start the experiment
    score = exp.run()  # in this func. the train function is executed. -> you can adapt this in the pipeline.py

    print("Experiment finished...")

    # here the exp is finished

    return score

def main():
    wandb.init(project="hannahs-long-try")
    score = objective(wandb.config)
    wandb.log({"score": score})
#
# # 2: Define the search space
# sweep_configuration = {
#     "method": "bayes",
#     "metric": {"goal": "maximize", "name": "score"},
#     "parameters": {
#         "l1-lambda": {"max": 0.005, "min": 0.0005},
#         "weigth-decay-l2": {"max": 0.05, "min": 0.005},
#         "learning-rate": {"max": 0.05, "min": 0.001},
#     },
# }

# load base config
base_config = OmegaConf.load(
    "../resnet_sweep/omega_conf/sweep_configuration.yaml")

# load override config, to merge later
override_config = OmegaConf.load(
    "../resnet_sweep/omega_conf/sweep_override.yaml")

# merge both config together
# all changes from override_config will overwrite the base_config
config = OmegaConf.merge(base_config, override_config)
print("Config loaded...")

# Sweep-Parameter aus OmegaConf extrahieren
sweep_configuration = OmegaConf.to_container(config, resolve=True)

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="hannahs-long-try", entity="PABA-ML4GLEAM")

wandb.agent(sweep_id, function=main, count=20)
