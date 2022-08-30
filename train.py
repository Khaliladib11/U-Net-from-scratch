import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from Unet.model import UNET

from utils import *

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 600
IMAGE_WIDTH = 600
PIN_MEMORY = True
LOAD_MODEL = False
IMG_DIR = "../data/cityscapes/leftImg8bit"
MASK_DIR = "../data/cityscapes/gtFine"


def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for idx, batch in enumerate(loop):
        images, masks = batch
        images = images.to(DEVICE)
        masks = masks.float().unsqueeze(1).to(DEVICE)

        with torch.cuda.amp.autocast():
            predicted_mask = model(images)
            loss = loss_fn(predicted_mask, masks)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())


@torch.no_grad
def validate(loader, model, loss_fn):
    model.eval()
    loop = tqdm(loader)
    for idx, batch in enumerate(loop):
        images, masks = batch
        images = images.to(DEVICE)
        masks = masks.float().unsqueeze(1).to(DEVICE)
        predicted_mask = model(images)
        loss = loss_fn(predicted_mask, masks)


def main():
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0]
        ),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0]
        ),
        ToTensorV2()
    ])

    model = UNET(input_channels=3, output_channels=2).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loader(IMG_DIR,
                                          MASK_DIR,
                                          BATCH_SIZE,
                                          train_transform,
                                          val_transform,
                                          PIN_MEMORY,
                                          NUM_WORKERS)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train(train_loader, model, optimizer, loss_fn, scaler)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)


if __name__ == '__main__':
    main()
