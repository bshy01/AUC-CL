from PIL import Image
import numpy as np
import torch
import cv2
import os

def cv2_imread(fns_img, color=cv2.IMREAD_UNCHANGED):
    img_array = np.fromfile(fns_img, np.uint8)
    img = cv2.imdecode(img_array, color)
    return img

class DatasetAcne04Class(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, color=cv2.IMREAD_COLOR_RGB, colorspace=None, training_mode='simclr',
                 train=True, albu_transform_used=False, **kwargs):
        self.path_src = root
        self.color = color
        self.cs = colorspace
        self.transform = transform
        self.training_mode = training_mode
        self.albu_transform_used = albu_transform_used
        self.examples = []

        split_name = 'Tr' if train else 'Te'
        split_path = os.path.join(self.path_src, split_name)

        if not os.path.isdir(split_path):
            split_name_lower = split_name.lower()
            split_path_lower = os.path.join(self.path_src, split_name_lower)
            if os.path.isdir(split_path_lower):
                split_path = split_path_lower
            else:
                raise FileNotFoundError(f"Dataset split not found at {split_path} or {split_path_lower}")

        image_files = sorted([f for f in os.listdir(split_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

        for str_img in image_files:
            try:
                label_char = str_img[5]
                level = int(label_char)

                example = {}
                example["fns_img"] = os.path.join(split_path, str_img)
                example["lbl"] = level
                example["str_img"] = str_img
                self.examples.append(example)
            except (IndexError, ValueError):
                print(f"Warning: Skipping file with unexpected name format: {str_img}")
                continue

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        fns_img = example['fns_img']
        inp_lbl = example['lbl']

        img_src = cv2_imread(fns_img, self.color)  # cv2_imread가 정의되어 있다고 가정
        img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
        if self.cs is not None:
            img_src = cv2.cvtColor(img_src, self.cs)

        if len(img_src.shape) == 2:
            img_src = cv2.cvtColor(img_src, cv2.COLOR_GRAY2RGB)

        # [수정된 부분] auc_cl 모드도 simclr 처럼 이미지 쌍(Pair)이 필요합니다.
        if self.training_mode in ['simclr', 'auc_cl']:
            if self.albu_transform_used:
                q_numpy_hwc = self.transform(image=img_src)["image"]
                k_numpy_hwc = self.transform(image=img_src)["image"]
                pos_1 = torch.from_numpy(q_numpy_hwc).permute(2, 0, 1).contiguous().float()
                pos_2 = torch.from_numpy(k_numpy_hwc).permute(2, 0, 1).contiguous().float()
            else:
                img_pil = Image.fromarray(img_src)
                pos_1 = self.transform(img_pil)
                pos_2 = self.transform(img_pil)
            return pos_1, pos_2, inp_lbl

        else:  # supervised mode
            if self.albu_transform_used:
                img_numpy_hwc = self.transform(image=img_src)["image"]
                img_tensor_chw = torch.from_numpy(img_numpy_hwc).permute(2, 0, 1).contiguous().float()
                return img_tensor_chw, inp_lbl
            else:
                img_pil = Image.fromarray(img_src)
                img_tensor_chw = self.transform(img_pil)
                return img_tensor_chw, inp_lbl

    def __len__(self):
        return self.num_examples

    def get_class_weights(self, num_classes):
        # Calculate class frequencies
        class_counts = {}
        for ex in self.examples:
            label = ex['lbl']
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

        # Formula: weight = total_samples / (num_classes * count)
        total_samples = sum(class_counts.values())

        weights = [0] * num_classes
        for class_id, count in class_counts.items():
            if count > 0 and class_id < num_classes:
                weights[class_id] = total_samples / (num_classes * count)

        return torch.tensor(weights, dtype=torch.float)