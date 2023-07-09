import torch
import logging as log 

from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score

def get_evaluation_metrics(model, device, validation_loader, test_loader, loss_module, sub_epoch,
                           training_loss, training_preds, training_targets):
    training_loss = training_loss/training_targets.shape[0]
    training_accuracy = multiclass_accuracy(training_preds, training_targets) # global
    training_f1_score = multiclass_f1_score(training_preds, training_targets) # global
    
    validation_loss, validation_accuracy, validation_f1_score = validate(model, device, validation_loader, loss_module)
    test_loss, test_accuracy, test_f1_score, per_class_accuracy, per_class_f1_score, weighted_f1_score = validate(model, device, test_loader, loss_module, per_class=True)
    log_str = f"Sub epoch {sub_epoch} train acc: {training_accuracy:.3f} train loss: {training_loss:.4f} train f1: {training_f1_score:.4f} "\
                f"val acc: {validation_accuracy:.3f} val loss: {validation_loss:.4f} val f1: {validation_f1_score:.4f} "\
                f"test acc: {test_accuracy:.3f} test loss: {test_loss:.4f} test f1: {test_f1_score:.4f}"
    log.info(log_str)
    for classname, accuracy, f1_score in zip(test_loader.dataset.classes, per_class_accuracy, per_class_f1_score):
        log_str = f"Class: {classname:5s} test accuracy: {accuracy:.3f} test f1: {f1_score:.4f}"
        log.info(log_str)
    log.info(f"Weighted f1: {weighted_f1_score:.4f}")
    metric_names = ["training_loss", "training_accuracy", "training_f1_score", 
                        "validation_accuracy", "validation_loss", "validation_f1_score", 
                        "test_accuracy", "test_loss", "test_f1_score",
                        "per_class_accuracy", "per_class_f1_score", "weighted_f1_score"]
    metrics = {}
    for name in metric_names:
        metrics[name] = eval(name)
    # metrics = dict((name, eval(name)) for name in metric_names)
    return metrics

def validate(model, device, dataloader, loss_module, per_class=False):
    running_loss = 0.0
    all_preds = torch.Tensor().to(device)
    all_targets = torch.Tensor().to(device)
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            data, target = data[0].to(device), data[1].to(device)
            output = model(data)
            loss = loss_module(output, target)
            running_loss += loss.item()
            _, preds = torch.max(output.data, 1)
            all_preds = torch.cat((all_preds, preds), -1)
            all_targets = torch.cat((all_targets, target), -1)
        loss = running_loss/all_targets.shape[0]
        accuracy = multiclass_accuracy(all_preds, all_targets) # global
        f1_score = multiclass_f1_score(all_preds, all_targets) # global
        if not per_class:
            return loss, accuracy, f1_score
        else:
            accuracy_per_class = multiclass_accuracy(all_preds.to(torch.int64), all_targets.to(torch.int64), num_classes=len(dataloader.dataset.classes), average=None) # per class
            f1_score_per_class = multiclass_f1_score(all_preds.to(torch.int64), all_targets.to(torch.int64), num_classes=len(dataloader.dataset.classes), average=None) # per class
            f1_score_weighted = multiclass_f1_score(all_preds.to(torch.int64), all_targets.to(torch.int64), num_classes=len(dataloader.dataset.classes), average='weighted') # weighted sum
            return loss, accuracy, f1_score, accuracy_per_class, f1_score_per_class, f1_score_weighted
