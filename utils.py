import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as transforms
from torch.utils.data import DataLoader
from dataset.CityScapes import CityScapes

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


def save_checkpoint(state, file_name='my_checkpoint.pt'):
    print("Saving model...")
    torch.save(state, file_name)
    print("Model saved.")


def load_checkpoint(check_point, model):
    print("Loading checkpoint....")
    model.load_state_dict(check_point["state_checkpoint"])





def get_loader(imgs_path,
               masks_path,
               batch_size,
               train_transform,
               val_transform,
               pin_memory=4,
               num_worker=True):
    train_params = {
        "img_root": imgs_path,
        "masks_root": masks_path,
        "list_of_classes": ['__bgr__', 'car', 'person'],
        "stage": 'train',
        "transform": train_transform
    }

    train_dataset = CityScapes(**train_params)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=pin_memory,
                              num_worker=num_worker)

    val_params = {
        "img_root": imgs_path,
        "masks_root": masks_path,
        "list_of_classes": ['__bgr__', 'car', 'person'],
        "stage": 'val',
        "transform": val_transform
    }

    val_dataset = CityScapes(**val_params)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=pin_memory,
                            num_worker=num_worker)

    return train_loader, val_loader


def predict(model, img_path, predicted_image_name="predicted image", device='cuda'):
    model.eval()
    im = Image.open(img_path)
    t_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.457, 0.407],
                             std=[0.229, 0.224, 0.225])])
    img = t_(im)
    img.unsqueeze_(0)
    if device == torch.device("cuda"):
        img = img.to(device)
    # get the output from the model
    model.eval()
    bgr_img = np.array(Image.open(img_path))
    output = model(img)['out']
    out = output.argmax(1).squeeze_(0).detach().clone().cpu().numpy()
    color_array = np.zeros([out.shape[0], out.shape[1], 3], dtype=np.uint8)
    print("Detected:")
    for id in np.unique(out):
        if id == 1:
            color_array[out == id] = [255, 0, 0]
        elif id == 2:
            color_array[out == id] = [0, 255, 0]
    added_image = cv2.addWeighted(bgr_img, 0.5, color_array, 0.6, 0)
    plt.imsave(f'{predicted_image_name}.png', added_image)
