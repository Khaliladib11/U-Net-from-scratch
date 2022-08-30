import os
import numpy as np
from PIL import Image
from collections import deque
import torch
from torch.utils import data
import torchvision.transforms as transforms

from CityScapes_labels import label2id, id2label

class CityScapes(data.Dataset)

    def __init__(self, img_root, masks_root, list_of_classes, stage):
        super(CityScapes, self).__init__()

        assert stage in ['train', 'test', 'val'], "Please the 'stage' parameter must be from ['train', 'test', 'val']"



        self.img_root = os.path.join(img_root, stage)
        self.masks_root = os.path.join(masks_root, stage)

        self.list_of_classes = list_of_classes
        self.stage = stage

        self.label2id = label2id
        self.id2label = id2label

        self.cities = os.listdir(self.img_root)

        self.mean_transform = [0.407, 0.457, 0.485]
        self.std = [0.229,0.224,0.225]

        self.images = self.__load_images()
        self.masks = self.__load_semantic_mask()


    def __create_classes_helper(self):
        self.class_to_idx = {}
        self.idx_to_class = {}
        for idz, c in enumerate(self.list_of_classes[1:]):
            assert c in self.label2id.keys()
            self.class_to_idx[self.label2id[c]] = idz + 1
            self.idx_to_class[idz + 1] = self.label2id[c]

    def __load_images(self):
        cities = os.listdir(self.img_root)
        images = deque()

        for city in self.cities:
            city_images = os.listdir(os.path.join(self.img_root, city))
            images.extend(city_images)

        return images


    def __load_semantic_mask(self):
        semantic_masks = deque()
        for city in self.cities:
            masks = os.listdir(self.masks_root, city)
            for file in masks:
                if file.endswith('labelIds.png'):
                    semantic_masks.append(file)

        return semantic_masks

    def apply_transform(self, image):
        t_ = transforms.Compose([
            transforms.Resize(640, 640),
            transforms.ToTensor(),
            transforms.Normalize(mean = self.mean_transform,
                                 std = self.std_transform)
        ])
        img = t_(image)
        return img

    def load_image(self, idx, trasform=True):
        image = self.images[idx]
        city_name = image.split('_')[0]
        image = Image.open(os.path.join(self.img_root, city_name, image))
        if trasform:
            image = self.apply_transform(image)

        return image

    # Get Semantic Mask
    def load_semantic_mask(self, idx):
        img_file_name = self.images[idx]
        img_file_name = img_file_name.split("_")
        semantic_mask_file = img_file_name[0]+'_'+img_file_name[1]+'_'+img_file_name[2]+'_gtFine_labelIds.png'
        #semantic_mask_file = self.semantic_masks[idx]
        city_name = semantic_mask_file.split('_')[0]
        mask = np.array(Image.open(os.path.join(self.masks_root, city_name, semantic_mask_file)))
        unique_values = np.unique(mask)
        for label in unique_values:
            _l = self.id2label[label]
            if _l not in self.list_of_classes:
                mask[mask == label] = 0

        for value in np.unique(mask)[1:]:
            mask[mask == value] = self.class_to_idx[value]

        mask = torch.tensor(mask, dtype=torch.uint8)
        return mask

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        image = self.load_image(idx)
        mask = self.load_semantic_mask(idx)

        return idx, image, mask



