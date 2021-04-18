import torch
import os
from os.path import join
from torchvision.datasets.vision import VisionDataset
import PIL
import pandas
import numpy as np
import zipfile
from functools import partial
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity, verify_str_arg


class CelebA(VisionDataset):
    base_folder = "celeba"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                         MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(self, root, split="train", target_type="attr", transform=None,
                 target_transform=None, download=False, target_attr='Attractive', labelwise=False):
        super(CelebA, self).__init__(root, transform=transform,
                                     target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        # SELECT the features
        self.sensitive_attr = 'Male'
        self.target_attr = target_attr       
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split = split_map[verify_str_arg(split.lower(), "split",
                                         ("train", "valid", "test", "all" ))]

        fn = partial(join, self.root, self.base_folder)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)

        mask = slice(None) if split is None else (splits[1] == split)

        self.filename = splits[mask].index.values
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)
        print(self.attr_names)
        self.target_idx = self.attr_names.index(self.target_attr)
        self.sensi_idx = self.attr_names.index(self.sensitive_attr)
        self.feature_idx = [i for i in range(len(self.attr_names)) if i != self.target_idx and i!=self.sensi_idx]
        self.num_classes = 2
        self.num_groups =2 
        print('num classes is {}'.format(self.num_classes))
        self.num_data = self._data_count()
        if self.split == "test":
            self._balance_test_data()
        self.labelwise = labelwise
        if self.labelwise:
            self.idx_map = self._make_idx_map()
            
    def _make_idx_map(self):
        idx_map = [[] for i in range(self.num_groups * self.num_classes)]
        for j, i in enumerate(self.attr):
            y = self.attr[j, self.target_idx]
            s = self.attr[j, self.sensi_idx]
            pos = s*self.num_classes + y
            idx_map[pos].append(j)
        final_map = []
        for l in idx_map:
            final_map.extend(l)
        return final_map            
        
    def _check_integrity(self):
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r") as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    def __getitem__(self, index):
        if self.labelwise:
            index = self.idx_map[index]
        img_name = self.filename[index]
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", img_name))

        target = self.attr[index, self.target_idx]
        sensitive = self.attr[index, self.sensi_idx]
        feature = self.attr[index, self.feature_idx]
        if self.transform is not None:
            X = self.transform(X)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return X, feature, sensitive, target, (index, img_name)

    def __len__(self):
        return len(self.attr)
    
    def _data_count(self):
        data_count = np.zeros((self.num_groups, self.num_classes), dtype=int)
        print('<data_count> %s mode'%self.split)
        for index in range(len(self.attr)):
            target = self.attr[index, self.target_idx]
            sensitive = self.attr[index, self.sensi_idx]
            data_count[sensitive, target] += 1
        for i in range(self.num_groups):
            print('# of %d groups data : '%i, data_count[i, :])
        return data_count

    def _balance_test_data(self):
        num_data_min = np.min(self.num_data)
        print('min : ', num_data_min)
        data_count = np.zeros((self.num_groups, self.num_classes), dtype=int)
        new_filename = []
        new_attr = []
        print(len(self.attr))        
        for index in range(len(self.attr)):
            target=self.attr[index, self.target_idx]
            sensitive = self.attr[index, self.sensi_idx]
            if data_count[sensitive, target] < num_data_min:
                new_filename.append(self.filename[index])
                new_attr.append(self.attr[index])
                data_count[sensitive, target] += 1
            
        for i in range(self.num_groups):
            print('# of balanced %d\'s groups data : '%i, data_count[i, :])
            
        self.filename = new_filename
        self.attr = torch.stack(new_attr)
