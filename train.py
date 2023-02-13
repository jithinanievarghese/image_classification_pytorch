from sklearn.model_selection import train_test_split
from dataset import ClassificationDataset
from engine import train_model, evaluate
from model import Net
from utils import EarlyStopper
from os.path import join
import torch.nn as nn
import pandas as pd
import torch
import config
import logging

logger = logging.getLogger("image_classification")
c_handler = logging.StreamHandler()
c_format = logging.Formatter('%(name)s- %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)
logger.setLevel(logging.DEBUG)


def run():
    """
    to read data and implement training.
    save the model with best accuracy and early stop
    if there is no further improvment in validation loss
    """
    df = pd.read_csv(config.IMAGE_META_DIR)
    df = df.sample(frac=1)
    logger.warning(f'total not of training data is {df.shape[0]}')
    df['image_path'] = df.image_name.apply(lambda x: join(config.IMAGE_DATA_DIR, x))
    X_train, X_test, y_train, y_test = train_test_split(
        df.image_path.to_list(), df.target.to_list(), test_size=0.10, random_state=42)
    train_dataset = ClassificationDataset(
        image_paths=X_train,
        targets=y_train
    )
    validation_dataset = ClassificationDataset(
        image_paths=X_test,
        targets=y_test,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
    )
    available_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(available_device)
    logger.info(f'available device is {available_device}')
    model = Net()
    # send model to device
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    loss_criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=4, verbose=True
    )
    logger.info(f'Total no of epochs for training = {config.EPOCHS}')
    best_accuracy = 0
    early_stopper = EarlyStopper(patience=5, min_delta=0.05)
    model_history = {"training": {}, "validation": {}}
    for epoch in range(config.EPOCHS):
        # train one epoch
        avg_training_loss = train_model(
            train_loader, model, optimizer, epoch,
            device, loss_criterion
        )
        # validate
        avg_validation_loss, accuracy = evaluate(
            validation_loader,
            model,
            device,
            loss_criterion
        )
        # Note that scheduler step should be called after validation
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau
        scheduler.step(avg_validation_loss)
        # save the loss history for every epoch for plotting
        model_history['validation'].update({epoch: avg_validation_loss})
        model_history['training'].update({epoch: avg_training_loss})
        logger.info(f'epoch: {epoch}, average:- trainig loss: {avg_training_loss} | validation loss {avg_validation_loss}')
        logger.info(f"epoch:{epoch} | Validation Accuracy Score = {accuracy}.")
        # save the model with best accuracy
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy
        # stop training if there is no further improvement in validation loss
        if early_stopper.early_stop(avg_validation_loss):
            logger.warning(f'no further improvement in validation loss, stopping at epoch {epoch}')
            break
    logger.warning(f'training completed and model with best accuracy {best_accuracy} saved to outputs')
    return model_history


if __name__ == "__main__":
    run()
