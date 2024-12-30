import os
import re
from PIL import Image
from torchvision import transforms
from torch.utils.data import IterableDataset, get_worker_info, Dataset
import torch
import numpy as np
import math
from torch.nn import functional as F
import cv2
from mylib.datas.utils.VOC import process_origin_data, get_y

class SquarePad():
	def __call__(self, image):
		_, h, w = image.shape
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (vp, vp, hp, hp)
		return F.pad(image, padding, 'constant', 0)


class LoadIter():
    def __init__(self, path_list:np.ndarray, y:torch.Tensor, transform:transforms) -> None:
        self.path_list = path_list
        self.y = y
        self.transform = transform

        self.index = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        if self.index < len(self.path_list):
            img_path = self.path_list[self.index]
            img = Image.open(img_path)
            img = self.transform(img)

            y = self.y[self.index]
            return img, y
        else:
            self.index = -1
            raise StopIteration


class VOCDateSet(Dataset):
    def __init__(self, is_train:bool=True, img_size:int=224, s:int=7, b:int=2) -> None:
        super().__init__()

        self.imgs_root = '/home/akie/workplace/mylib/datas/DataSets/VOCdevkit/VOC2012/JPEGImages'
        self.annotations_root = '/home/akie/workplace/mylib/datas/DataSets/VOCdevkit/VOC2012/Annotations'
        self.split_root = '/home/akie/workplace/mylib/datas/DataSets/VOCdevkit/VOC2012/ImageSets/Main'

        self.s = s
        self.b = b

        # 统一转换器
        if is_train:
            self.unify_transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                SquarePad(),
                transforms.Resize((img_size, img_size)),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.unify_transform = transforms.Compose([
                transforms.ToTensor(),
                SquarePad(),
                transforms.Resize((img_size, img_size)),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


        # 读取原数据：图片、锚框信息、标签类型
        self.path_list = self.get_origin_data(is_train)
        # self.index = np.arange(len(self.path_list))

    ## 获取原数据函数------------------------------------------------------------------------------
    def get_origin_data(self, is_train:bool) -> tuple[np.ndarray, torch.Tensor, list]:
        '''
        根据is_train划分数据集
        返回结果列表，类别列表

        平均时间: 1.2164332389831543 s
        '''
        # 对图片路径进行划分
        if is_train:
            splite_path = os.path.join(self.split_root, 'train.txt')
        else:
            splite_path = os.path.join(self.split_root, 'val.txt')
        with open(splite_path, 'r') as file:
            splite = file.readlines()

        # 获取类别列表
        # for _, _, files in os.walk(self.split_root):
        #     file_list = files
        #     break
        
        # patten = r'(.+)_train.txt'
        # patten = re.compile(patten)

        # # 从文件名匹配类名
        # def match_class(file_name, patten):
        #     result = patten.match(file_name)
        #     if result is None:
        #         return None
        #     else:
        #         return result.group(1)
    
        # # 过滤未匹配到的元素
        # def filter_fun(x):
        #     if x is None:
        #         return False
        #     else:
        #         return True
        
        # file_list = list(map(lambda x:match_class(x, patten), file_list))
        # file_list = list(filter(filter_fun, file_list))         #类名列表

        # # 获取标注信息同时完成路径处理
        # patten = r'(\d+_\d+)[^\d]*?'
        # patten = re.compile(patten)

        path_list = []
        # target_list = []
        while len(splite) > 0:
            img_path = splite.pop()
            
            # path, target = process_origin_data(img_path, file_list, patten, self.s, self.b, len(file_list), self.imgs_root, self.annotations_root)
            path_list.append(img_path[:-1])
            # target_list.append(target)
    
        return path_list

    def __len__(self) -> int:
        return len(self.path_list)

    def __getitem__(self, index):
        img_path, y = get_y(self.path_list[index])
        img = cv2.imread(img_path)
        x = self.unify_transform(img)

        return x, y


    # def __iter__(self):
    #     self.update()
    #     worker_info = get_worker_info()
    #     if worker_info is None:
    #         iter_index = self.index
    #     else:
    #         per_worker = int(math.ceil(len(self.path_list)/float(worker_info.num_workers)))
    #         worker_id = worker_info.id
    #         iter_start = worker_id * per_worker
    #         iter_end = min(iter_start+per_worker, len(self.path_list)-1)
    #         iter_index = self.index[iter_start:iter_end]

    #     return LoadIter(self.path_list[iter_index], self.y[iter_index], self.unify_transform)

    # 更新数据集----------------------------------------------------------------------------------
    # def update(self) -> None:
    #     np.random.shuffle(self.index)


if __name__ == '__main__':
    s = 7
    b = 2
    p = 224/s
    xcoodmask = torch.arange(0, s*s, 1, requires_grad=False).reshape(s, s)
    xcoodmask = xcoodmask % s
    xcoodmask = xcoodmask.reshape(s, s)
    ycoodmask = xcoodmask.transpose(0, 1)
    xcoodmask = xcoodmask.reshape(-1)
    ycoodmask = ycoodmask.reshape(-1)

    data = VOCDateSet(is_train=False)
    data = iter(data)

    x, y = next(data)
    img = np.array(transforms.ToPILImage()(x))

    y = y[..., :5*b]
    y = y.reshape(-1, 2, 5)[:, 0, :]

    # 获取长度
    w = y[:, 2]
    h = y[:, 3]
    w = w * 224
    h = h * 224
    w = w*w/2
    h = h*h/2

    # 获取坐标
    xc = y[:, 0]
    yc = y[:, 1]
    xc = xc + xcoodmask
    yc = yc + ycoodmask
    xc = xc * p
    yc = yc * p    

    # 获取框
    c = y[:, -1]
    w = w[c==1]
    h = h[c==1]
    xc = xc[c==1]
    yc = yc[c==1]


    xmin = xc-w
    xmax = xc+w
    ymin = yc-h
    ymax = yc+h

    for idx in range(len(xmax)):
        img = cv2.rectangle(img, (int(xmin[idx]), int(ymin[idx])), (int(xmax[idx]), int(ymax[idx])), (0, 255, 0), 1)

    cv2.imwrite('test.jpg', img)



    
        
    
   