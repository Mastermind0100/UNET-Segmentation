from torch.utils.data import Dataset
import torch.nn as nn
import torch
import cv2
import json
import numpy as np
from typing import Tuple
import random

def center_crop(layer, target_height, target_width):
    _, _, h, w = layer.shape
    delta_h = h - target_height
    delta_w = w - target_width
    top = delta_h // 2
    left = delta_w // 2
    return layer[:, :, top:top+target_height, left:left+target_width]

class Compose:
    def __init__(self, transforms:list) -> None:
        self.transforms = transforms
    
    def __call__(self, img:np.ndarray) -> Tuple[np.ndarray]:
        for t in self.transforms:
            img = t(img)
        return img

class CropTransform:
    def __init__(self, crop_x: int = 497, crop_y: int = 497, target_width: int = 767):
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.target_width = target_width

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img_cropped = img[:self.crop_y, self.crop_x:]  # (497, W_cropped)

        # Pad right side if needed
        W = img_cropped.shape[1]
        pad_width = self.target_width - W
        if pad_width > 0:
            img_cropped = np.pad(
                img_cropped,
                ((0, 0), (0, pad_width)),  # pad right side
                mode='constant',
                constant_values=0
            )

        return img_cropped.copy()  # copy ensures contiguous array

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray]:
        if random.random() < self.p:
            img = img[:, ::-1].copy()
            return img
        return img.copy()

class OCTDataset(Dataset):
    def __init__(self, filepath:str='dataset.json', transform:Compose=None) -> None:
        super().__init__()
        with open(filepath, 'r', encoding='utf-8') as file:
            img_list = json.load(file)
            self.dataset = img_list['data']
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx:int) -> Tuple[np.ndarray, np.ndarray]:
        img_path = self.dataset[idx]
        img = cv2.imread(img_path, 0)

        if self.transform:
            img = self.transform(img.copy())    
        
        height, width = img.shape
        new_height, new_width = [int(1*x) for x in [height, width]]
        img = img[0:new_height, 0:new_width]

        denoised = cv2.fastNlMeansDenoising(img, None, h=5, templateWindowSize=7, searchWindowSize=21)
        ret, thresh = cv2.threshold(denoised, 75, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        blank_img = np.zeros_like(img)
        contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        max_idx = 0
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_idx = idx

        cv2.drawContours(blank_img, contours, max_idx, (255,255,255), -1)
        mask = np.zeros_like(blank_img)

        for col in range(new_width):
            column = blank_img[:, col]
            white_pixels = np.where(column == 255)[0]
            if white_pixels.size > 0:
                top_index = white_pixels[0]
                mask[:top_index, col] = 255

        roi = cv2.bitwise_and(img, img, mask=mask) # TOP OF THE CONTOUR

        return torch.tensor(img, dtype=torch.float32).unsqueeze(0), torch.tensor(roi, dtype=torch.float32).unsqueeze(0)
    
class UNetBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)

class UNETAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = UNetBlock(1, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = UNetBlock(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = UNetBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = UNetBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = UNetBlock(128, 64)

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        e3_cropped = center_crop(e3, d3.shape[2], d3.shape[3])  # match size
        d3 = torch.cat([d3, e3_cropped], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e2_cropped = center_crop(e2, d2.shape[2], d2.shape[3])
        d2 = torch.cat([d2, e2_cropped], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1_cropped = center_crop(e1, d1.shape[2], d1.shape[3])
        d1 = torch.cat([d1, e1_cropped], dim=1)
        d1 = self.dec1(d1)

        out = torch.sigmoid(self.out_conv(d1))
        return out

if __name__ == "__main__":
    transforms = Compose([
        CropTransform()
    ])
    dataset = OCTDataset(transform=transforms)

    while True:
        choice = random.randint(0,100)
        img = dataset[choice][0]
        roi = dataset[choice][1]

        cv2.imshow("image", img)
        cv2.imshow("roi", roi)
        
        if cv2.waitKey(0) == ord('q'):
            break

    cv2.destroyAllWindows()