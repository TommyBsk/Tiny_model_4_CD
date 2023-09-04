from typing import List, Tuple
from collections.abc import Sized
from os.path import join
import albumentations as alb
from torchvision.transforms import Normalize
from torchvision.utils import save_image
import numpy as np
import torch
from matplotlib.image import imread
# from cv2 import imread
from torch.utils.data import Dataset
from torch import Tensor
from utils.mask_deformation import apply_deformation, create_circle
import random
from PIL import Image 


class MyDataset(Dataset, Sized):
    def __init__(
        self,
        data_path: str,
        txt_data: str,
        mode: str,
    ) -> None:
    
        # Store the path data path + mode (train,val,test):
        self._mode = mode
        self._img_path = join(data_path,"data")
        self._mask_path = join(data_path,"seg")

        # In all the dirs, the files share the same names:
        self._list_images = self._read_images_list(txt_data)
        #ritorna la lista di tutte le immagini

        # Initialize augmentations:
        if mode == 'train':
            self._augmentation = _create_shared_augmentation()
            self._aberration = _create_aberration_augmentation()
        else:
            self._resize_eval_images = _resize_eval_images()
        
        # Initialize normalization:
        self._normalize = Normalize(mean=[0.45],
                                 std=[0.225])

    def _imgname2maskname(self, img_name:str)->str:
        if "data" in img_name:
            return img_name.replace("data","seg")
    
    def __getitem__(self, indx):
        # Current image set name:
        imgname = self._list_images[indx].strip('\n')

        # Loading the images:
        x_img = imread(join(self._img_path, imgname))
        x_mask = _binarize(imread(join(self._mask_path, self._imgname2maskname(imgname))))
        
        # Create deformed image and mask for 50% of the image
        if random.uniform(0,1)>0.5:
            x_deformed_image, x_deformed_mask = apply_deformation(x_img, x_mask) 
        else: 
            x_deformed_image, x_deformed_mask = x_img, x_mask
        # create prothesis in 50% of images:
        if random.uniform(0,1)>0.5:
            x_deformed_image = create_circle(x_deformed_image, x_deformed_mask) 
        # create gt for change detection
        x_mask_cd = (x_deformed_mask + x_mask) % 2

        # Data augmentation in case of training:
        if self._mode == "train":
            x_img, x_deformed_image, x_mask_cd = self._augment(x_img, x_deformed_image, x_mask_cd)
        else:
            x_img, x_deformed_image, x_mask_cd = self._resize_eval(x_img, x_deformed_image, x_mask_cd)

        # Trasform data from HWC to CWH:
        x_img, x_deformed_image, x_mask_cd = self._to_tensors(np.expand_dims(x_img, axis=2), np.expand_dims(x_deformed_image, axis=2), x_mask_cd)
        # save_image(x_img_t.float(),"/home/ramat/code/Tiny_CD/Tiny_model_4_CD/utils/immagini/" + imgname + "img.jpeg")
        # save_image(x_deformed_image_t.float(),"/home/ramat/code/Tiny_CD/Tiny_model_4_CD/utils/immagini/" + imgname + "defomred_img.jpeg")
        # save_image(x_mask_cd_t.float(),"/home/ramat/code/Tiny_CD/Tiny_model_4_CD/utils/immagini/" + imgname + "mask.jpeg")
        
        return {"image":x_img.repeat(3,1,1), "deformed_image":x_deformed_image.repeat(3,1,1) ,"mask":x_mask_cd,"img_name":imgname}

    def __len__(self):
        return len(self._list_images)

    def _read_images_list(self, txt_file: str) -> List[str]:
        with open(txt_file, "r") as f:
            return f.readlines()
    
    def _augment(
        self, x_ref: np.ndarray, x_test: np.ndarray, x_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # First apply augmentations in equal manner to test/ref/x_mask:
        transformed = self._augmentation(image=x_ref, image0=x_test, x_mask0=x_mask)
        x_ref = transformed["image"]
        x_test = transformed["image0"]
        x_mask = transformed["x_mask0"]

        # Then apply augmentation to single test ref in different way:
        x_ref = self._aberration(image=x_ref)["image"]
        x_test = self._aberration(image=x_test)["image"]

        return x_ref, x_test, x_mask
    
    def _resize_eval(
        self, x_img: np.ndarray,x_test:np.ndarray, x_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Resize image and mask:
        transformed = self._resize_eval_images(image=x_img, image0=x_test, x_mask0=x_mask)
        x_img = transformed["image"]
        x_test = transformed["image0"]
        x_mask = transformed["x_mask0"]

        return x_img, x_test, x_mask
    
    def _to_tensors(
        self, x_ref: np.ndarray, x_test: np.ndarray, x_mask: np.ndarray
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return (
            self._normalize(torch.tensor(x_ref).permute(2, 0, 1)),
            self._normalize(torch.tensor(x_test).permute(2, 0, 1)),
            torch.tensor(x_mask),
        )


def _create_shared_augmentation():
    return alb.Compose(
        [
            alb.Resize(256, 256),
            alb.Flip(p=0.5),
            alb.Rotate(limit=5, p=0.5),
        ],
        additional_targets={"image0": "image", "x_mask0": "mask"},
    )

def _resize_eval_images():
    return alb.Compose(
        [
            alb.Resize(256, 256)
        ],
        additional_targets={"image0": "image", "x_mask0": "mask"},
    )


def _create_aberration_augmentation():
    return alb.Compose([
        # alb.RandomBrightnessContrast(
        #     brightness_limit=0.2, contrast_limit=0.2, p=0.5
        # ),
        alb.GaussianBlur(blur_limit=[3, 5], p=0.5),
    ])

def _binarize(mask: np.ndarray) -> np.ndarray:
    return np.clip(mask * 255, 0, 1).astype(int)
