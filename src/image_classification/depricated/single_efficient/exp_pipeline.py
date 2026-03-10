import wandb
from omegaconf import OmegaConf
import torch
import uuid
import os
from torch.utils.data import random_split, DataLoader
from shared.general_util.general_helper_functions import create_tarball

class Pipeline:
    def __init__(self, config , train_func):
        """
        Just initialize the pipeline with the config and the training function
        Setup all the necessary stuff for training

        :param config: OmegaConf config object, contains all the configuration for the experiment
        :param train_func: training function for your network, can be adapted
        """
        self.config = config
        self.train = train_func

        self.dataset = None

        self.device = config.training.device

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # enable wandb config if logging is enabled
        if self.config.experiment.enable_logging:
            self.init_wandb()

    def init_wandb(self):
        """
        Initialize WandB for logging (wandb.init)
        Weights and Biases (wandb) uses a global object, just log to this object anywhere

        :return: None
        """
        # wandb.login()

        wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            # name=self.config.wandb.run_name, # run name could be defined here
            notes=self.config.wandb.description,
            config=OmegaConf.to_container(self.config, resolve=True)
        )
        print(f"WandB Run gestartet")

    def setup_model(self, model, optimizer, criterion, scheduler):
        """
        Set up the model, optimizer, criterion and scheduler for training
        :param model: Neural Network model which should be trained
        :param optimizer: Optimizer for training
        :param criterion: Loss function for training
        :param scheduler: learning rate scheduler, standard -> exponential
        :return:
        """
        # set model, and put it to device (default = cuda)
        self.model = model.to(self.device)

        # set optimizer and scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler

        # set loss function (aka criterion)
        self.criterion = criterion

    def collate_fn(self, batch):
        """ Entfernt None-Werte aus dem Batch """
        batch = [b for b in batch if b is not None]  # Filtert None-Werte heraus
        if len(batch) == 0:
            return None  # Falls der gesamte Batch leer ist

        return torch.utils.data.default_collate(batch)  # Standard `collate_fn`

    def setup_dataloaders(self, dataset, train_size, val_size):
        """
        Set up the dataloaders for training, validation and testing
        Makes a train / val / test split
        the train_size and val_size don't need to sum up to 1.
        When train and val are set, the rest (fraction) is for the test set

        :param dataset: dataset which should be trained on (model)
        :param train_size: size of the training set (float between 0 and 1)
        :param val_size: size of the validation set (float between 0 and 1)
        :return:
        """
        # set dataset
        self.dataset = dataset

        dataset_len = len(dataset)
        # compute fractions
        train_size = int(train_size * dataset_len)
        val_size = int(val_size * dataset_len)
        test_size = dataset_len - train_size - val_size

        # reproduce your results with random seed
        generator = torch.Generator().manual_seed(self.config.experiment.random_seed)

        # split up dataset
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=generator)

        # Create data loaders for the train and validation datasets (Test set only used in HP optimization)
        train_loader = DataLoader(train_set, batch_size=self.config.dataset.batch_size, shuffle=True, collate_fn=self.collate_fn) # todo. alter this later to shuffle = true again
        val_loader = DataLoader(val_set, batch_size=self.config.dataset.batch_size, shuffle=True, collate_fn=self.collate_fn)
        test_loader = DataLoader(test_set, batch_size=self.config.dataset.batch_size, shuffle=False, collate_fn=self.collate_fn)

        # set dataloaders in class
        self.train_loader = train_loader
        self.val_loader = val_loader
        # TODO: can we set this fraction to zero? then no test set is created, so when train and val sum up to 1, no test set should be created...?
        self.test_loader = test_loader  # TODO: do we need this in single train, -> no, only in sweeps...

    def run(self):
        """
        Main training loop to train the model

        :return: None
        """
        assert self.model is not None, "Model muss zuerst gesetzt werden!"
        assert self.optimizer is not None, "Optimizer muss zuerst gesetzt werden!"
        assert self.train_loader is not None, "Train dataset muss zuerst gesetzt werden!"
        assert self.val_loader is not None, "Validation dataset muss zuerst gesetzt werden! (für Logging)"

        # Freeze den EfficientNet Backbone
        for param in self.model.base_model.parameters():
            param.requires_grad = False

        # Unfreeze den Head
        for param in self.model.classifier.parameters():
            param.requires_grad = True



        # # unfreeze only last layer
        # for param in self.model.fc.parameters():
        #     param.requires_grad = True
        
        print("Train last layer")
        # Fine-tune the last layer for a few epochs
        self.train(model=self.model,
                   train_loader=self.train_loader,
                   val_loader=self.val_loader,
                   criterion=self.criterion,
                   optimizer=self.optimizer,
                   scheduler=self.scheduler,
                   num_epochs=self.config.training.num_epochs_lastlayer,
                   config=self.config,
                   enable_logging=False)

        print("Finetune whole Network")
        # Unfreeze all the layers and fine-tune the entire network for a few more epochs
        for param in self.model.parameters():
            param.requires_grad = True


        # make a new optimizer for the whole network
        optimizer = torch.optim.Adam(self.model.parameters(),
                             lr=self.config.training.learning_rate,
                             betas=self.config.training.betas,
                             eps=self.config.training.eps,
                             weight_decay=self.config.training.weight_decay)
        """self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.config.training.learning_rate,
                                         momentum=self.config.training.momentum,
                                         weight_decay=self.config.training.weight_decay)"""

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)

        final_acc = self.train(model=self.model,
                   train_loader=self.train_loader,
                   val_loader=self.val_loader,
                   criterion=self.criterion,
                   optimizer=self.optimizer,
                   scheduler=self.scheduler,
                   num_epochs=self.config.training.num_epochs_finetune,
                   config=self.config,
                   enable_logging=self.config.experiment.enable_logging,)

        print("Final Validation Accuracy: ", final_acc)

        print("Training finished ")

        if self.config.experiment.run_on_cluster and self.config.experiment.enable_logging:
            print("Make Tarball and save to projects folder on Cluster")
            actual_uuid = uuid.uuid4().hex[:8]
            tarball_save_path = f"/cluster/projects/ml4gleam/{self.config.model.name}_{self.config.dataset.name}"
            os.makedirs(tarball_save_path, exist_ok=True)
            tarball_name = f"{self.config.model.name}_{self.config.dataset.name}_train_{actual_uuid}.tar.gz"
            # create_tarball(f"{tarball_save_path}/{tarball_name}", "/scratch/dist")

        if self.config.experiment.enable_logging:
            wandb.finish()
            print("WandB Run finished")

        print("EXP Run function is done!")