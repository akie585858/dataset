from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import pandas as pd
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import math
import torch

class LoadIter():
    def __init__(self, data_array:np.ndarray, root_path='mini-imagenet', transform:transforms=None) -> None:
        self.data_array = data_array
        self.root_path = root_path
        self.transform = transform
        self.index = -1
        
    def __iter__(self):
        return self
    
    def __next__(self):
        self.index += 1
        if self.index < len(self.data_array):
            img_name, label = self.data_array[self.index]
            img_path = self.root_path + '/images/' + img_name
            img = Image.open(img_path)
            if not (self.transform is None):
                img = self.transform(img)
            return img, label
        else:
            self.index = -1
            raise StopIteration


class MiniImageNet(IterableDataset):
    label = ['n02108915', 'n03207743', 'n03980874', 'n02981792', 'n02687172', 'n02219486',
            'n01843383', 'n03047690', 'n04146614', 'n02108551', 'n13133613', 'n03924679',
            'n06794110', 'n03417042', 'n04509417', 'n04296562', 'n03998194', 'n02105505',
            'n09256479', 'n02113712', 'n02111277', 'n04418357', 'n03527444', 'n01749939',
            'n03146219', 'n03220513', 'n09246464', 'n01930112', 'n03676483', 'n04258138',
            'n01770081', 'n02110341', 'n04251144', 'n03337140', 'n03400231', 'n02950826',
            'n03272010', 'n02129165', 'n03908618', 'n02114548', 'n02091831', 'n01558993',
            'n07747607', 'n02747177', 'n04443257', 'n02074367', 'n03017168', 'n02138441',
            'n02089867', 'n07697537', 'n02116738', 'n03773504', 'n01532829', 'n01981276',
            'n03775546', 'n04389033', 'n02971356', 'n03854065', 'n03888605', 'n03535780',
            'n03075370', 'n02966193', 'n07613480', 'n04275548', 'n04515003', 'n02457408',
            'n01704323', 'n03476684', 'n03770439', 'n04612504', 'n03347037', 'n04604644',
            'n01855672', 'n04149813', 'n02165456', 'n07584110', 'n04067472', 'n02091244',
            'n02101006', 'n04522168', 'n03544143', 'n02795169', 'n03127925', 'n02823428',
            'n02871525', 'n02099601', 'n02110063', 'n13054560', 'n02120079', 'n01910747',
            'n02443484', 'n03584254', 'n02174001', 'n04435653', 'n02108089', 'n03062245',
            'n04596742', 'n03838899', 'n02606052', 'n04243546']
    
    def __init__(self, is_train:bool=True, root_path='/home/akie/workplace/mylib/datas/DataSets/mini-imagenet') -> None:
        super().__init__()

        # csv文件预处理
        self.root_path = root_path
        if is_train:
            path = pd.read_csv(os.path.join(self.root_path, 'train.csv'))
        else:
            path = pd.read_csv(os.path.join(self.root_path, 'test.csv'))
        path = path.sample(frac=1).reset_index(drop=True)

        label_map = {}
        for idx, label_s in enumerate(MiniImageNet.label):
            label_map[label_s] = idx

        path['label'] = path['label'].map(label_map)
        self.data = path.to_numpy()  #关键总路路径数据

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.Pad((16, 16)),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4733, 0.4490, 0.4031),
                                 std=(0.2795, 0.2713, 0.2850))
        ])

    def shuffle(self):
        np.random.shuffle(self.data)

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            return LoadIter(self.data, self.root_path, self.transform)
        else:
            per_worker = int(math.ceil(len(self.data)/float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start+per_worker, len(self.data)-1)
            iter_data = self.data[iter_start:iter_end]

            return LoadIter(iter_data, self.root_path, self.transform)
            
    def __len__(self):
        return len(self.data)
    

if __name__ == '__main__':
    dataset = MiniImageNet()
    dataset = MiniImageNet(False)
    exit() 
    loader = DataLoader(dataset, batch_size=16, drop_last=False, num_workers=4)

    # mean = 0.4733, 0.4490, 0.4031
    # std = 0.2795, 0.2713, 0.2850

    x_mean = torch.Tensor([0.4733, 0.4490, 0.4031])
    x_std = 0
    for x, _ in loader:
        x_std += torch.sum(torch.mean(torch.square(x-x_mean.reshape(1, 3, 1, 1).repeat(x.shape[0], 1, 224, 224)), dim=[2, 3]), dim=0)

    x_std /= float(len(dataset))
    x_std = torch.sqrt(x_std)
    print(x_std)
        