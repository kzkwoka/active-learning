import torch
import logging as log 

def validate(model, device, dataloader, loss_module):
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            data, target = data[0].to(device), data[1].to(device)
            output = model(data)
            loss = loss_module(output, target)
            val_running_loss += loss.item()
            _, preds = torch.max(output.data, 1)
            val_running_correct += (preds == target).sum().item()
        val_loss = val_running_loss/len(dataloader.dataset)
        val_accuracy = 100. * val_running_correct/len(dataloader.dataset)
        # val_f1score = 
    return val_loss, val_accuracy

def get_acc_per_class(model, device, testloader, classes):
    model.eval()
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data    
            images = images.to(device)
            outputs = model(images).cpu()   
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    # print accuracy for each class
    all_acc = []
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        all_acc.append(accuracy)
        log.info("Accuracy for class {:5s} is: {:.1f} %".format(classname, 
                                                       accuracy))
    log.info(f"Average accuracy is {sum(all_acc)/len(all_acc)}")
    return all_acc