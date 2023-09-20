from functools import partial
import json
from pathlib import Path
import shutil
import torchvision
import torch
# import torch.nn as nn
from typing import Any, Dict, List, NamedTuple, Optional

import typer

from active_learning import PARAM_ERROR, SUCCESS, MODEL_ERROR, DATASET_ERROR, DEVICE_ERROR
from active_learning.cli import _register
from active_learning.utils import generate_heuristic_sample, get_train_val, create_weighted_sampler, filter_dict, train_epoch


class ActiveLearningIter(NamedTuple):
    model: str # Nazwa modelu sieci

    labeled_dataset: str # Ścieżka do folderu ze zbiorem etykietowanym
    unlabeled_dataset: str # Ścieżka do folderu ze zbiorem nieetykietowanym
    to_annotate: str #Ścieżka do folderu ze zbiorem do oznaczenia 

    valid_n: int | float # Wielkość zbioru walidacyjnego (ułamek lub procent)

    loss: Dict[str, Any] # Moduł straty
    optimizer: Dict[str, Any] # Moduł optymalizatora
    
    batch_size: Dict[str, int] # Liczba próbek ładowanych w procesie uczenia
    epochs: int # Liczba epok treningu
    
    checkpoint_path: str # Ścieżka do pliku z parametrami modułów w punktach kontrolnych
    checkpoint_every: int # Liczba epok między kolejnymi punktami kontrolnymi

    device: Optional[str] = None # Urządzenie na którym przeprowadzane są obliczenia
    num_workers: Optional[int] = 0 # Liczba wątków używanych podczas ładowania próbek 
    
    #One of these has to be filled
    weights: Optional[str] = None # Nazwa modułu wag, gdy wykorzystywany jest transfer uczenia
    input_size: Optional[int|tuple[int, int]] = None # Rozmiar wejścia sieci
    transforms: Optional[str] = None # Ścieżka do pliku .pt w którym przechowywany jest moduł przekształceń obrazów, używany podczas ładowania
    
    scheduler: Optional[Dict[str, Any]] = None # Moduł planowania wartości współczynnika uczenia
    classes: Optional[int] = None # Liczba klas potrzebna do odpowiedniej modyfikacji modelu
    
    copy: Optional[bool] = False
    


class ActiveLearningHandler:
    def __init__(self, param_path) -> None:
        self.al_iter = ActiveLearningIter(**json.loads(param_path.read_text()))

    def load_device(self):
        try:
            if not self.al_iter.device:
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device(self.al_iter.device)
        except RuntimeError:
            return DEVICE_ERROR
        return SUCCESS

    def load_model(self) -> int:
        try:
            if self.al_iter.weights is not None:
                w_class, w_version = self.al_iter.weights.split(".")
                weights = getattr(getattr(torchvision.models, w_class), w_version)
                self.default_transforms = weights.transforms
            else:
                weights = None
            model = getattr(torchvision.models, self.al_iter.model)(weights=weights)
            
            self.net = model
        except AttributeError:
            return MODEL_ERROR
        return SUCCESS

    def adjust_model(self) -> int:
        #adjust the last layer for the number of classes
        for param in self.net.features.parameters():
            param.requires_grad = False
        input_lastLayer = self.net.classifier[1].in_features
        if self.al_iter.classes:
            self.net.classifier[1] = torch.nn.Linear(input_lastLayer,self.al_iter.classes)
        else:
            return PARAM_ERROR
        self.net = self.net.to(self.device)
        return SUCCESS

    def get_transforms(self):
        if not self.al_iter.transforms:
            if not self.default_transforms:
                from torchvision.transforms._presets import ImageClassification
                return ImageClassification(crop_size=self.al_iter.input_size)
            else:
                return self.default_transforms()
        else:
            return torch.load(self.al_iter.transforms)
    
    def load_data(self, labeled, resume=False) -> int:
        try:
            transforms = self.get_transforms()
            if labeled:
                self.labeled_dataset = torchvision.datasets.ImageFolder(
                    root=self.al_iter.labeled_dataset, transform=transforms)
                if not resume:
                    train_idx, validation_idx = get_train_val(self.labeled_dataset, self.al_iter.valid_n)
                    dest = Path(self.al_iter.checkpoint_path)
                    dest.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "train": train_idx,
                        "val":validation_idx
                        }, dest.parent.joinpath("indices.pt"))
                else:
                    checkpoint = torch.load(Path(self.al_iter.checkpoint_path.replace("epoch-nn.pth","indices.pt")))
                    train_idx, validation_idx = checkpoint["train"], checkpoint["val"]
                self.trainset, self.valset = torch.utils.data.Subset(self.labeled_dataset, train_idx),torch.utils.data.Subset(self.labeled_dataset, validation_idx)
                if not self.al_iter.classes:
                    self.al_iter = self.al_iter._replace(classes=len(self.labeled_dataset.classes))
                    _register([f"classes={self.al_iter.classes}"])
                elif self.al_iter.classes != len(self.labeled_dataset.classes):
                    return PARAM_ERROR
            else:
                self.unlabeled_dataset = torchvision.datasets.ImageFolder(
                    root=self.al_iter.unlabeled_dataset, transform=transforms)
        except FileNotFoundError:
            return DATASET_ERROR
        return SUCCESS

    def load_modules(self) -> int:
        loss_module = getattr(torch.nn, self.al_iter.loss.get("name"))
        self.loss = loss_module(**filter_dict(loss_module, self.al_iter.loss))
        
        optim_module = getattr(torch.optim, self.al_iter.optimizer.get("name"))
        self.optimizer = optim_module(self.net.parameters(), **filter_dict(optim_module, self.al_iter.optimizer))

        if self.optimizer and self.al_iter.scheduler:
            sched_module = getattr(
                torch.optim.lr_scheduler, self.al_iter.scheduler.get("name"))
            self.scheduler = sched_module(
                self.optimizer, **filter_dict(sched_module, self.al_iter.scheduler))
        else:
            self.scheduler = None

        return SUCCESS

    def load_dataloaders(self) -> int:
        if hasattr(self, "trainset") and hasattr(self, "valset"):
            self.train_loader = torch.utils.data.DataLoader(
                self.trainset, batch_size=self.al_iter.batch_size.get("train"), num_workers=self.al_iter.num_workers, sampler=create_weighted_sampler(self.trainset))
            self.val_loader = torch.utils.data.DataLoader(
                self.valset, batch_size=self.al_iter.batch_size.get("valid",self.al_iter.batch_size.get("train")), num_workers=self.al_iter.num_workers)
        if hasattr(self, "unlabeled_dataset"):
            self.unlabeled_loader = torch.utils.data.DataLoader(
                self.unlabeled_dataset, batch_size=self.al_iter.batch_size.get("unlab", self.al_iter.batch_size.get("train")), num_workers=self.al_iter.num_workers)
        return SUCCESS

    def checkpoint(self, epoch):
        Path(self.al_iter.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'model': self.net.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None
        }, self.al_iter.checkpoint_path.replace("nn",str(epoch)))
    
    def resume(self, epoch):
        checkpoint = torch.load(self.al_iter.checkpoint_path.replace("nn",str(epoch)))
        self.net.load_state_dict(checkpoint['model'])
        if epoch != "final":
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        return SUCCESS
    
    def run_training(self, resume_epoch, cont) -> int:
        if resume_epoch > 0:
            self.resume(resume_epoch-1)
        elif cont:
            self.resume("final")
            resume_epoch += 1

        with typer.progressbar(range(resume_epoch-1, self.al_iter.epochs), label="Training") as progress:
            for epoch in progress:
                log_str = train_epoch(self.net, self.device, self.optimizer, self.scheduler, self.loss, self.train_loader, self.val_loader)
                # typer.echo(log_str)
                if (epoch + 1) % self.al_iter.checkpoint_every == 0:
                    self.checkpoint(epoch)
        self.checkpoint("final")
        return SUCCESS
    
    def generate_subset(self, fast, random, quantity):
        heuristic = "largest_margin" if fast else "mc_dropout"
        heuristic = None if random else heuristic
        if quantity < 1:
            quantity = int(quantity * len(self.unlabeled_dataset))
        elif quantity > len(self.unlabeled_dataset):
            return PARAM_ERROR
        self.chosen, _ = generate_heuristic_sample(self.unlabeled_loader,quantity,self.net,self.device, heuristic)
        return SUCCESS
    
    def move_files(self):
        if hasattr(self, "chosen"):
            self.chosen = [Path(p) for p, _ in self.chosen]
        dirmade = False
        i = 1
        if self.al_iter.to_annotate:
            while not dirmade:
                try: 
                    Path(self.al_iter.to_annotate).mkdir(parents=True, exist_ok=False)
                except FileExistsError:
                    self.al_iter.to_annotate = self.al_iter.to_annotate + f"({i})"
                else:
                    dirmade = True
            if self.al_iter.copy:
                for f in self.chosen:
                    shutil.copy2(f, Path(self.al_iter.to_annotate).joinpath(f.name))
            else:
                for f in self.chosen:
                    shutil.move(f, Path(self.al_iter.to_annotate).joinpath(f.name))
            return SUCCESS
        else:
            return PARAM_ERROR
    
    def train(self, resume_epoch, cont) -> int:
        steps = [
            (self.load_device, "Device"),
            (self.load_model, "Model"),
            (partial(self.load_data, True, resume_epoch!=0), "Dataset"),
            (self.load_dataloaders, "Dataloaders"),
            (self.adjust_model, "Model"),
            (self.load_modules, "Modules"),
            (partial(self.run_training, resume_epoch, cont), "Train"),
        ]

        for step_func, step_name in steps:
            code = step_func()
            if code != SUCCESS:
                print(f"Error in {step_name} step. Error code: {code}")
                return code

        return SUCCESS
    
    def generate(self, fast, random, quantity) -> int:
        steps = [
            (self.load_device, "Device"),
            (self.load_model, "Model"),
            (partial(self.load_data, False), "Dataset"),
            (self.load_dataloaders, "Dataloaders"),
            (self.adjust_model, "Model"),
            (self.load_modules, "Modules"),
            (partial(self.resume, "final"), "Checkpoint"),
            (partial(self.generate_subset, fast, random, quantity), "Generating"),
            (self.move_files, "Moving files")
        ]
        for step_func, step_name in steps:
            code = step_func()
            if code != SUCCESS:
                print(f"Error in {step_name} step. Error code: {code}")
                return code
        return SUCCESS

