from os.path import join
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from utils import list_files
from natsort import natsorted
import random
import numpy as np

class UTKFaceDataset(VisionDataset):
    
    label = 'age'
    sensi = 'race'
    fea_map = {
        'age' : 0,
        'gender' : 1,
        'race' : 2
    }
    num_map = {
        'age' : 100,
        'gender' : 2,
        'race' : 4
    }

    def __init__(self, root, split='train', transform=None, target_transform=None,
                 labelwise=False):
        
        super(UTKFaceDataset, self).__init__(root, transform=transform,
                                             target_transform=target_transform)
        
        self.split = split
        self.filename = list_files(root, '.jpg')
        self.filename = natsorted(self.filename)
        self._delete_incomplete_images()
        self._delete_others_n_age_filter()
        self.num_groups = self.num_map[self.sensi]
        self.num_classes = self.num_map[self.label]        
        self.labelwise = labelwise
        
        random.seed(1)
        random.shuffle(self.filename)
        
        self._make_data()
        self.num_data = self._data_count()

        if self.labelwise:
            self.idx_map = self._make_idx_map()

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, index):
        if self.labelwise:
            index = self.idx_map[index]
        img_name = self.filename[index]
        s, l = self._filename2SY(img_name)
        
        image_path = join(self.root, img_name)
        image = Image.open(image_path, mode='r').convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 1, np.float32(s), np.int64(l), (index, img_name)
    
    def _make_idx_map(self):
        idx_map = [[] for i in range(self.num_groups * self.num_classes)]
        for j, i in enumerate(self.filename):
            s, y = self._filename2SY(i)
            pos = s*self.num_classes + y
            idx_map[pos].append(j)
            
        final_map = []
        for l in idx_map:
            final_map.extend(l)
        return final_map

    def lg_filter(self, l, g):
        tmp = []
        for i in self.filename:
            g_, l_ = self._filename2SY(i)
            if l == l_ and g == g_:
                tmp.append(i)
        return tmp

    def _delete_incomplete_images(self):
        self.filename = [image for image in self.filename if len(image.split('_')) == 4]

    def _delete_others_n_age_filter(self):

        self.filename = [image for image in self.filename
                         if ((image.split('_')[self.fea_map['race']] != '4'))]
        ages = [self._transform_age(int(image.split('_')[self.fea_map['age']])) for image in self.filename]
        self.num_map['age'] = len(set(ages))

    def _filename2SY(self, filename):        
        tmp = filename.split('_')
        sensi = int(tmp[self.fea_map[self.sensi]])
        label = int(tmp[self.fea_map[self.label]])
        if self.sensi == 'age':
            sensi = self._transform_age(sensi)
        if self.label == 'age':
            label = self._transform_age(label)
        return int(sensi), int(label)
        
    def _transform_age(self, age):
        if age<20:
            label = 0
        elif age<40:
            label = 1
        else:
            label = 2
        return label 

    def _make_data(self):
        import copy
        min_cnt = 100
        data_count = np.zeros((self.num_groups, self.num_classes), dtype=int)
        if self.split == 'train':
            tmp = copy.deepcopy(self.filename)
        else:
            tmp = []
            
        for i in reversed(self.filename):
            s, l = self._filename2SY(i)
            data_count[s, l] += 1
            if data_count[s, l] <= min_cnt:
                if self.split =='train':
                    tmp.remove(i)
                else:
                    tmp.append(i)
                    
        self.filename = tmp
        
    def _data_count(self):
        data_count = np.zeros((self.num_groups, self.num_classes), dtype=int)
        data_set = self.filename

        for img_name in data_set:
            s, l = self._filename2SY(img_name)
            data_count[s, l] += 1
        
        for i in range(self.num_groups):
            print('# of %d groyp data : '%i, data_count[i, :])
        return data_count
