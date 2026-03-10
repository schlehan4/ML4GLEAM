import wandb
from omegaconf import OmegaConf
import torch
from shared.dataset.dataset import preprocess
from exp_pipeline_full import Pipeline

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
    dest_path = f"/scratch/schlehan/jobs/sweep/"  # here?
    anno_tar = f"{dest_path}/AugementedAnnotations.tar"
    img_tar = f"{dest_path}/AugmentedFITSImages.tar"
    anno_path = f"{dest_path}/annotations/"  # TODO: correct the schreibfehler when needed, is this a bug?
    img_path = f"{dest_path}/images/"
    json_path = f"{dest_path}/coco_annotation.json"

    # create experiment / pipeline
    exp = Pipeline(config=config)
    print("Experiment / pipeline created...")

    preprocess(img_tar, anno_tar, anno_path, img_path, json_path, ".fits")
    
    # set up the dataloaders in the pipeline, make train / val split
    exp.setup_dataloaders(img_path, json_path, train_size=0.8, val_size=0.2)  # creates also a test set and test loader
    print("DataLoaders created...")

    # setup model with all stuff in the pipeline
    exp.setup_model()
    print("Model set  up...")

    # log gradients...
    wandb.watch(exp.get_model(), log="all", log_freq=10)

    # all setup is done, the experiment can now be started
    print("Starting the experiment... / starting the training...")
    # start the experiment
    score = exp.run()  # in this func. the train function is executed. -> you can adapt this in the pipeline.py

    print("Experiment finished...")

    # here the exp is finished

    return score

def main():
    wandb.init(project="Swin-small-preporcess", entity="PABA-ML4GLEAM", group="hep-min-zscale")
    score = objective(wandb.config)
    wandb.log({"score": score})

# load base config
base_config = OmegaConf.load(
    "../schlehans_piepeline/omega_conf/sweep_configuration.yaml")

# load override config, to merge later
override_config = OmegaConf.load(
    "../schlehans_piepeline/omega_conf/sweep_override.yaml")

# merge both config together
# all changes from override_config will overwrite the base_config
config = OmegaConf.merge(base_config, override_config)
print("Config loaded...")

# Sweep-Parameter aus OmegaConf extrahieren
sweep_configuration = OmegaConf.to_container(config, resolve=True)

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="Efficient-try", entity="PABA-ML4GLEAM")

wandb.agent(sweep_id, function=main, count=20)
