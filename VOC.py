import os
import torch
from torch.utils.data import Dataset
from mylib.datas.utils.VOC import xml2annotation
from torchvision.transforms import v2
from torchvision.ops import box_convert
from torchvision import tv_tensors
from torch import nn
from typing import List
from PIL import Image 
import tqdm

class SquarePad(nn.Module):
    def forward(self, image:torch.Tensor, box:tv_tensors.BoundingBoxes) -> tuple[torch.Tensor, tv_tensors.BoundingBoxes]:
        h, w = image.shape[-2:]
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        image = v2.functional.pad(image, padding, 0, 'constant')
        box, _ = v2.functional.pad_bounding_boxes(box, 'cxcywh', (w, h), padding)
        box = tv_tensors.BoundingBoxes(
            data=box,
            format='cxcywh',
            canvas_size=(max_wh, max_wh)
        )

        return image, box


class YoloDataSet(Dataset):
    def __init__(self, data: List[str] | str = ['VOC2012', 'VOC2007'], dataset: List[str] | str = ['train', 'val', 'trainval']) -> None:
        '''
        输入：data(VOC2012, VOC2007)、dataset(train, val, trainval)

        获取VOC数据原始数据并进行封装，提供数个只读变量接口

        main：
            img_path_list: 图片路径列表
            object:标注信息列表 [img(bndbox tv_tensor)], 
            cls_index:标注类别索引列表 [img[cls_index]],

        other：
            annotations_path_list: 标注路径列表
            imgs_root_list: 图片根目录列表
            annotations_root_list: 标注根目录列表
            split_root_list: 划分根目录列表

        并提供虚函数__process_target()，用于处理目标信息，需要根据具体任务重写
        '''
        super().__init__()

        data = [data] if isinstance(data, str) else data
        dataset = [dataset] if isinstance(dataset, str) else dataset

        self._imgs_root_list = [f'/home/akie/workplace/mylib/datas/DataSets/VOCdevkit/{y}/JPEGImages' for y in data]
        self._annotations_root_list = [f'/home/akie/workplace/mylib/datas/DataSets/VOCdevkit/{y}/Annotations' for y in data]
        self._split_root_list = [f'/home/akie/workplace/mylib/datas/DataSets/VOCdevkit/{y}/ImageSets/Main' for y in data]

        # 整合所有图片和标注的路径信息
        self._img_path_list = []
        self._annotations_path_list = []

        for data_index in range(len(data)):
            img_root = self._imgs_root_list[data_index]
            annotations_root = self._annotations_root_list[data_index]
            split_root = self._split_root_list[data_index]

            for dataset_index in range(len(dataset)):
                split_path = os.path.join(split_root, f'{dataset[dataset_index]}.txt')
                with open(split_path, 'r') as file:
                    split = file.readlines()
                img_paths = [os.path.join(img_root, f'{img_name.strip()}.jpg') for img_name in split]
                annotations_paths = [os.path.join(annotations_root, f'{img_name.strip()}.xml') for img_name in split]
                self._img_path_list.extend(img_paths)
                self._annotations_path_list.extend(annotations_paths)

        # 处理标注信息
        self._size:torch.Tensor 
        self._objects = []
        self._cls_index = []

        self.__process_annotations()
    
    def __process_annotations(self) -> None:
        '''
        处理标注信息
        path_list[annotation_path]->info_dict{
        'size':Tensor shape(b, w, h),
        'object':list[img(bndbox tv_tensor)], 
        'cls_index':list[img[cls_index]],
        }
        '''
        sizes = []
        tbar = tqdm.tqdm(self._annotations_path_list)
        tbar.set_description('Processing origin Annotations')
        for annotation_path in tbar:
            size, objects, cls_indexs = xml2annotation(annotation_path)

            # objects = tv_tensors.BoundingBoxes(
            #     data=objects,
            #     format=str('xyxy'),
            #     canvas_size=size
            # )

            objects = v2.functional.convert_bounding_box_format(objects, 'xyxy', 'cxcywh')

            sizes.append(size)
            self._objects.append(objects)
            self._cls_index.append(cls_indexs)
        self._size = torch.Tensor(sizes)

    def __process_target(self) -> None:...

    # 只读变量--------------------------------------------------------------------------------------
    @property
    def img_path_list(self):
        return self._img_path_list
    
    @property
    def size(self):
        return self._size
    
    @property
    def objects(self):
        return self._objects

    @property
    def cls_index(self):
        return self._cls_index

    @property
    def annotations_path_list(self):
        return self._annotations_path_list
    
    @property
    def imgs_root_list(self):
        return self._imgs_root_list
    
    @property
    def annotations_root_list(self):
        return self._annotations_root_list
    
    @property
    def split_root_list(self):
        return self._split_root_list


class YoloV1DataSet(YoloDataSet):
    def __init__(self, data: List[str] | str = ['VOC2012', 'VOC2007'], dataset: List[str] | str = ['train', 'val', 'trainval'], s=7) -> None:
        '''
        yolov1数据集 特殊处理：目标值处理
        目标值：
            cls_index_list: [tensor(c, cls_num)]
            coord_list: [tensor(c, 4(cx, cy, w, h))]

        其中c为每张图片的目标数量，cls_num为类别数量，
        cx, cy为网格中的相对位置，归一化到0-1，
        w, h为目标框根据图片大小标准化宽高的开方，

        额外辅助数据:
            bndbox_index_list: [[index(0, s*s-1)]]
        '''
        super().__init__(data, dataset)

        self._cls_index_list = []
        self._coord_list = []
        self._bndbox_index_list = []
        self.s = s

        self._train_transform = v2.Compose([
            v2.ToImage(),
            SquarePad(),
            v2.Resize((480, 480)),
            v2.RandomAffine(degrees=(0, 0), translate=(0.1, 0.1)),
            v2.RandomHorizontalFlip(),
            v2.RandomCrop((448, 448), pad_if_needed=True)
        ])

        self._val_transform = v2.Compose([
            v2.ToImage(),
            SquarePad(),
            v2.Resize((448, 448))
        ])

    def __process_target(self) -> None:
        '''
        处理目标信息
        '''

        for img_idx in tqdm.trange(len(self.objects), desc='Processing target'):
            objects = self.objects[img_idx]
            img_w, img_h = self.size[img_idx]
            
            cls_index = []
            coord = []
            # bndbox_index = []

            cell_w = img_w / self.s
            cell_h = img_h / self.s
            for obj in objects:
                cls_index.append(obj[-1])
                x1, y1, x2, y2 = obj[:4]
                w = torch.sqrt((x2 - x1) / img_w)
                h = torch.sqrt((y2 - y1) / img_h)

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                # grid_x = int(cx / cell_w)
                # grid_y = int(cy / cell_h)
                # cx = (cx - grid_x * cell_w) / cell_w
                # cy = (cy - grid_y * cell_h) / cell_h

                coord.append([cx, cy, w, h])
                # bndbox_index.append(grid_y * self.s + grid_x)
            
            cls_index = torch.Tensor(cls_index)
            cls_index = torch.nn.functional.one_hot(cls_index.to(torch.int64), num_classes=20).to(torch.bool)
            self._cls_index_list.append(torch.Tensor(cls_index))
            self._coord_list.append(torch.Tensor(coord))
            # self._bndbox_index_list.append(bndbox_index)

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img = Image.open(self.img_path_list[index])
        cood = self.objects[index]

        img, cood = self._train_transform(img, cood)

        return img, cood,  self.cls_index[index]

    # 只读变量--------------------------------------------------------------------------------------
    @property
    def cls_index_list(self):
        return self._cls_index_list

    @property
    def coord_list(self):
        return self._coord_list

    @property
    def bndbox_index_list(self):
        return self._bndbox_index_list


if __name__ == '__main__':
    from utils.VOC import file_list
    import cv2
    import numpy as np

    dataset = YoloV1DataSet(data='VOC2012', dataset=['train', 'val', 'trainval'])
    img, objs, cls_index = dataset[0]

    img:np.ndarray = img.permute(1, 2, 0).numpy()
    img = img.copy().astype(np.uint8)
    objs = box_convert(objs, 'cxcywh', 'xyxy')

    print(img.shape)

    for obj in objs:
        cv2.rectangle(img, (int(obj[0]), int(obj[1])), (int(obj[2]), int(obj[3])), (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey(-1)

    
    




    
        
    
   