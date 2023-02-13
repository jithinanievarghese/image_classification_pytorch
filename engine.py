from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import logging
logger = logging.getLogger("image_classification.engine")
logger.setLevel(logging.DEBUG)


def train_model(data_loader, model, optimizer, epoch, device, loss_criterion):
    """
    This is the main training function that trains model
    for one epoch and does back propagation and
    return the average training loss
    args:
    data_loader :  pytorch data loader
    model: neural network model class
    optimizer: neural network optimizer
    epoch: epoch  at which training occurs
    device: cpu or cuda
    loss_criterion: function that compares the target and output values
    """
    # set model to training mode
    training_losses = []
    model.train()
    # go through batches of data in data loader
    logger.info(f'training for epoch {epoch}')
    for data in tqdm(data_loader):
        images, targets = get_data(data, device)
        # pass through the model
        optimizer.zero_grad()
        predictions = model(images)
        # calculate the loss
        loss = loss_criterion(
            predictions,
            targets.view(-1, 1).float())
        # compute gradient of loss w.r.t.
        # all parameters of the model that are trainable
        loss.backward()
        # single optimization step
        optimizer.step()
        training_losses.append(loss.item())
    avg_training_loss = np.array(training_losses).mean()
    return avg_training_loss


def evaluate(data_loader, model, device, loss_criterion):
    """
    validation loop to test the model for one epoch and
    return the validation loss and validation accuracy
    model with best validation accuracy is saved
    args:
    data_loader :  pytorch data loader
    model: neural network model class
    device: cpu or cuda
    loss_criterion: function that compares the target and predicted values
    """
    # initialize empty lists to store predictions , targets and losses
    validation_losses = []
    final_targets = []
    final_predictions = []
    # put the model in eval mode
    model.eval()
    # disable gradient calculation
    with torch.no_grad():
        for data in data_loader:
            images, targets = get_data(data, device)
            # make predictions
            outputs = model(images)
            valid_loss = loss_criterion(
                outputs,
                targets.view(-1, 1).float())
            # calculate validtion loss
            validation_losses.append(valid_loss.item())
            # move predictions and targets to list
            # we need to move predictions and targets to cpu too
            targets = targets.cpu().detach()
            targets = targets.numpy().tolist()
            final_targets.extend(targets)
            # convert outputs to cpu and extend the final list
            outputs = torch.sigmoid(outputs).cpu().detach()
            outputs = outputs.numpy().tolist()
            final_predictions.extend(outputs)
    outputs = np.array(final_predictions) >= 0.5
    validation_accuracy = accuracy_score(final_targets, outputs)
    avg_valid_loss = np.array(validation_losses).mean()
    return avg_valid_loss, validation_accuracy


def get_data(data, device):
    """
    helper function to return image tensors and targets
    from the dataset object
    """
    images = data['images']
    targets = data['targets']
    # move everything to specified device
    images = images.to(device, dtype=torch.float)
    targets = targets.to(device, dtype=torch.long)
    return images, targets
