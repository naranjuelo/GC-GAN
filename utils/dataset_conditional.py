from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch
import cv2
import lmdb
import albumentations
import albumentations.augmentations as A

class MaskDataset_conditional(Dataset):
    def __init__(self, path, transform=None, resolution=256, label_size=0, aug=False):

        self.path = path
        self.transform = transform
        self.resolution = resolution
        self.label_size = label_size

        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('image-length'.encode('utf-8')).decode('utf-8'))

        self.labels = np.load(path+"/labels_gaze.npy", allow_pickle=True).item()
        #print(self.labels)
        if self.transform is None:
            self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5), inplace=True)
                    ])

        self.aug = False
        print(" -- augs deactivated --")
        if self.aug == True:
            self.aug_t = albumentations.Compose([
                            #A.transforms.HorizontalFlip(p=0.5),
                            A.geometric.transforms.ShiftScaleRotate(shift_limit=0.05,
                                                scale_limit=0.06,
                                                rotate_limit=0,
                                                border_mode=cv2.BORDER_CONSTANT,
                                                value=0,
                                                mask_value=0,
                                                p=0.5),
                    ])
        

    def _onehot_mask(self, mask):
        label_size = self.label_size
        labels = np.zeros((label_size, mask.shape[0], mask.shape[1]))
        for i in range(label_size):
            labels[i][mask==i] = 1.0
        
        return labels
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):

        with self.env.begin(write=False) as txn:
            label = self.labels[idx]
            img = Image.open(BytesIO(txn.get(f'image-{str(idx).zfill(7)}-{label}'.encode('utf-8')))).convert('RGB')
            # label = label_data[4] + '_' + label_data[5]                
            #    txn.put(f"{prefix}-{str(i).zfill(7)}-{label}".encode("utf-8"), img)
            if img.size[0] != self.resolution:
                img = img.resize((self.resolution, self.resolution), resample=Image.LANCZOS)
                
            mask = Image.open(BytesIO(txn.get(f'label-{str(idx).zfill(7)}-{label}'.encode('utf-8')))).convert('L')
            if mask.size[0] != self.resolution:
                mask = mask.resize((self.resolution, self.resolution), resample=Image.NEAREST)

        if self.aug:
            augmented = self.aug_t(image=np.array(img), mask=np.array(mask))
            img = Image.fromarray(augmented['image'])
            mask = augmented['mask']
        
        img = self.transform(img)
        mask = self._onehot_mask(np.array(mask))
        mask = torch.tensor(mask, dtype=torch.float) * 2 - 1
        labelx = np.float32(label.split('_')[0])
        labely = np.float32(label.split('_')[1])
        #print(type(labelx))
        return {"image": img, "mask": mask, "gaze_x": labelx, "gaze_y": labely}
    
def draw_gaze(img, pitchyaw, thickness=2, color=(0, 255, 255)):
    """Draw gaze angle on given image with a given eye positions."""

    image_out = img
    (h, w) = img.shape[:2]
    length = w / 2.0
    pos = (int(h / 2.0), int(w / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    im_g = np.array(image_out).copy()
    cv2.arrowedLine((im_g), tuple(np.round(pos).astype(np.int32)),
                tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                thickness, cv2.LINE_AA, tipLength=0.2)
    return im_g