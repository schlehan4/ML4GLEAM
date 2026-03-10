import wandb, torch, uuid, os
from omegaconf import OmegaConf
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet34_Weights
from torchvision.models import ResNet50_Weights
from efficientnet_pytorch import EfficientNet

from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup

from shared.dataset.dataset import preprocess

from shared.heads.resnet34_head import RGZ_ResNet34Head
from shared.heads.resnet50_head import RGZ_ResNet50Head 
from shared.heads.rgz_small_head import RGZ_SmallHead 


from torch.utils.data import random_split, DataLoader
from shared.general_util.general_helper_functions import create_tarball
from sweep_train_function_L1 import train as train_func
from dataset import FitsSet

class Pipeline:
    def __init__(self, config):
        """
        Just initialize the pipeline with the config and the training function
        Setup all the necessary stuff for training

        :param config: OmegaConf config object, contains all the configuration for the experiment
        :param train_func: training function for your network, can be adapted
        """     
        self.config = config
        self.train = None

        self.dataset = None
        self.device = config.device 

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # enable wandb config if logging is enabled
        if self.config.enable_logging:
            self.init_wandb()

    def init_wandb(self):
        """
        Initialize WandB for logging (wandb.init)
        Weights and Biases (wandb) uses a global object, just log to this object anywhere

        :return: None
        """
        # wandb.login()

        """wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            # name=self.config.wandb.run_name, # run name could be defined here
            notes=self.config.wandb.description,
            config=OmegaConf.to_container(self.config, resolve=True)
        )"""
        print(f"WandB Run gestartet")

    def build_optimizer(self, finetune=False):
        if self.config.reg_type == "both" or self.config.reg_type == "L2":
            if self.config.optimizer == "ADAM":
                optimizer = torch.optim.Adam(self.model.parameters(),
                                    lr=self.config.learning_rate,
                                    betas=self.config.betas,
                                    eps=self.config.eps,
                                    weight_decay=self.config.weight_decay)
            elif self.config.optimizer == "SGD":
                optimizer = torch.optim.SGD(self.model.parameters(),
                                            lr=self.config.learning_rate,
                                            momentum=self.config.momentum,
                                            weight_decay=self.config.weight_decay)
            elif self.config.optimizer == "ADAMW":
                if self.config.model_architecture== "Swin" and finetune:
                    optimizer = torch.optim.AdamW(self.model.parameters(),
                                    lr=(0.01*self.config.learning_rate),
                                    betas=self.config.betas,
                                    eps=self.config.eps,
                                    weight_decay=self.config.weight_decay)
                else:
                    optimizer = torch.optim.AdamW(self.model.parameters(),
                                        lr=self.config.learning_rate,
                                        betas=self.config.betas,
                                        eps=self.config.eps,
                                        weight_decay=self.config.weight_decay)
            else:
                raise ValueError("No optimizer defined")
            
        elif self.config.reg_type == "L1" or self.config.reg_type == "none":
            if self.config.optimizer == "ADAM":
                optimizer = torch.optim.Adam(self.model.parameters(),
                                    lr=self.config.learning_rate,
                                    betas=self.config.betas,
                                    eps=self.config.eps)
            elif self.config.optimizer == "SGD":
                optimizer = torch.optim.SGD(self.model.parameters(),
                                            lr=self.config.learning_rate,
                                            momentum=self.config.momentum)
                
            elif self.config.optimizer == "ADAMW":
                if self.config.model_architecture== "Swin" and finetune:
                    optimizer = torch.optim.AdamW(self.model.parameters(),
                                    lr=(0.01*self.config.learning_rate),
                                    betas=self.config.betas,
                                    eps=self.config.eps,
                                    weight_decay=self.config.weight_decay)
                else:
                    optimizer = torch.optim.AdamW(self.model.parameters(),
                                        lr=self.config.learning_rate,
                                        betas=self.config.betas,
                                        eps=self.config.eps,
                                        weight_decay=self.config.weight_decay)
            else:
                raise ValueError("No optimizer defined")
            
        else:
            raise ValueError("No optimizer defined, because no reg_type found")
        
        return optimizer
    
    def build_scheduler(self, finetune):
        if self.config.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
            
        elif self.config.scheduler == "cos":
            if not finetune:
                scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.num_epochs_lastlayer,eta_min=0.000001)
                print("train last layer", scheduler)
            elif finetune:
                scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.num_epochs_finetune,eta_min=0.000001)
                print("train finetune,", scheduler)
            else:
                raise ValueError("you didnt pass the finetune variable to the scheduler!")
            
        elif self.config.scheduler == "cosWarmUp":
            if not finetune:
                scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.num_epochs_lastlayer)
            elif finetune:
                assert self.train_loader is not None, "train_loader must be passed for warmup scheduler"

                num_training_steps = len(self.train_loader) * self.config.num_epochs_finetune
                num_warmup_steps = int(self.config.learingn_rate_div * num_training_steps)  # e.g., 10% warmup

                scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps
            )
            else:
                raise ValueError("you didnt pass the finetune variable to the scheduler!")
        else:
            raise ValueError("no scheduler defined")
        return scheduler
    
    def define_criterion(self):
        return torch.nn.CrossEntropyLoss()

    def setup_model(self):
        """
        Set up the model, optimizer, criterion and scheduler for training
        :param model: Neural Network model which should be trained
        :param optimizer: Optimizer for training
        :param criterion: Loss function for training
        :param scheduler: learning rate scheduler, standard -> exponential
        :return:
        """
        
        self.train=train_func
        
        #region Model
        # set model, and put it to device (default = cuda)
        if self.config.model_architecture == "EfficientNet":
            model = EfficientNet.from_pretrained(self.config.model_name)
            model = RGZ_SmallHead(model, in_features=self.config.in_features, num_classes=self.config.num_classes, dropout_rate=self.config.dropout)
        elif self.config.model_architecture == "ResNet":
            if self.config.model_name == "ResNet34":
                model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
                model = RGZ_ResNet34Head(model, in_features=512, num_classes=self.config.num_classes, dropout_rate=self.config.dropout)
            if self.config.model_name == "ResNet50":
                model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
                model = RGZ_ResNet50Head(model, in_features=2048, num_classes=self.config.num_classes, dropout_rate=self.config.dropout)
        elif self.config.model_architecture == "Swin":
            if self.config.model_name == "tiny":
                model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
            elif self.config.model_name == "small":
                model = models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1)
            else:
                raise ValueError(f"Unknown Swin model name: {self.config.model_name}")
        else:
            print("No Model specified")
        self.model = model.to(self.device)
        #endregion

        #region Optimizer, Scheduler, Criterion
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler(finetune=False)
        self.criterion = self.define_criterion()
        #endregion

    def collate_fn(self, batch):
        """ Entfernt None-Werte aus dem Batch """
        batch = [b for b in batch if b is not None]  # Filtert None-Werte heraus
        if len(batch) == 0:
            return None  # Falls der gesamte Batch leer ist

        return torch.utils.data.default_collate(batch)  # Standard `collate_fn`

    def setup_dataloaders(self, img_path, json_path, train_size, val_size):
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
        #region Transform
        transform = None
        size = self.config.img_size
        if self.config.model_architecture == "EfficientNet":
            transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Drei Kanäle normalisieren
            ])
            
        elif self.config.model_architecture == "ResNet":
            transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Drei Kanäle normalisieren
            ])
        
        elif self.config.model_architecture == "Swin":
            transform = transforms.Compose([
                transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        
        else:
            print("No transform fitting to model, setting to None")
        #endregion
        
        # load the dataset, here cifar 10
        # use the coco or fits dataloader here later -> schlehan's code
        dataset = FitsSet(img_path, json_path,order=self.config.tansform_first, gamma=[self.config.gamma1, self.config.gamma2, self.config.gamma3], transform_method=self.config.transform_method, glob_min=self.config.glob_min, glob_max=self.config.glob_max, transform=transform)
            
        # set dataset
        self.dataset = dataset

        dataset_len = len(dataset)
        # compute fractions
        train_size = int(train_size * dataset_len)
        val_size = int(val_size * dataset_len)
        test_size = dataset_len - train_size - val_size

        # reproduce your results with random seed
        generator = torch.Generator().manual_seed(self.config.random_seed)

        # split up dataset
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=generator)

        # Create data loaders for the train and validation datasets (Test set only used in HP optimization)
        train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True, collate_fn=self.collate_fn)
        val_loader = DataLoader(val_set, batch_size=self.config.batch_size, shuffle=True, collate_fn=self.collate_fn)
        test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False, collate_fn=self.collate_fn)

        # set dataloaders in class
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader  # TODO: do we need this in single train, -> no, only in sweeps...

    def get_model(self):
        return self.model

    def run(self):
        """
        Main training loop to train the model

        :return: None
        """
        assert self.model is not None, "Model muss zuerst gesetzt werden!"
        assert self.optimizer is not None, "Optimizer muss zuerst gesetzt werden!"
        assert self.train_loader is not None, "Train dataset muss zuerst gesetzt werden!"
        assert self.val_loader is not None, "Validation dataset muss zuerst gesetzt werden! (für Logging)"

        if self.config.model_architecture == "EfficientNet":
            # Freeze den EfficientNet Backbone
            for param in self.model.base_model.parameters():
                param.requires_grad = False
            # Unfreeze den Head
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        elif self.config.model_architecture == "Swin":
            # Freeze alles außer den Head
            for name, param in self.model.named_parameters():
                if "head" not in name:
                    param.requires_grad = False
                else:
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
                   num_epochs=self.config.num_epochs_lastlayer,
                   config=self.config,
                   enable_logging=False)

        print("Finetune whole Network")
        # Unfreeze all the layers and fine-tune the entire network for a few more epochs
        for param in self.model.parameters():
            param.requires_grad = True


        # make a new optimizer for the whole network
        self.optimizer = self.build_optimizer(finetune=True)
        self.scheduler = self.build_scheduler(finetune=True)

        final_acc = self.train(model=self.model,
                   train_loader=self.train_loader,
                   val_loader=self.val_loader,
                   criterion=self.criterion,
                   optimizer=self.optimizer,
                   scheduler=self.scheduler,
                   num_epochs=self.config.num_epochs_finetune,
                   config=self.config,
                   enable_logging=self.config.enable_logging,)

        print("Final Validation Accuracy: ", final_acc)

        print("Training finished ")

        if self.config.run_on_cluster and self.config.enable_logging:
            print("Make Tarball and save to projects folder on Cluster")
            #actual_uuid = uuid.uuid4().hex[:8]
            #tarball_save_path = f"/cluster/projects/ml4gleam/{self.config.model_name}_{self.config.name}"
            #os.makedirs(tarball_save_path, exist_ok=True)
            #tarball_name = f"{self.config.model_name}_{self.config.name}_train_{actual_uuid}.tar.gz"
            # create_tarball(f"{tarball_save_path}/{tarball_name}", "/scratch/dist")

        if self.config.enable_logging:
            wandb.finish()
            print("WandB Run finished")

        print("EXP Run function is done!")
        return final_acc