import os
from math import sqrt
from xml.etree import ElementTree as ET
from torchvision import transforms
import torch
import cv2
import numpy as np
from torch.nn import functional as F

img_root = '/home/akie/workplace/mylib/datas/DataSets/VOCdevkit/VOC2012/JPEGImages'
annotation_root = '/home/akie/workplace/mylib/datas/DataSets/VOCdevkit/VOC2012/Annotations'
file_list = ['boat', 'car', 'bottle', 'diningtable', 'dog', 'cat', 'person', 
            'aeroplane', 'motorbike', 'pottedplant', 'train', 'bird', 'sheep', 
            'bicycle', 'chair', 'cow', 'bus', 'tvmonitor', 'sofa', 'horse']

class SquarePad():
	def __call__(self, image):
		_, h, w = image.shape
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, hp, vp, vp)
		return F.pad(image, padding, 'constant', 0)


def process_origin_data(img_id, file_list, patten, s, b, c_nums, imgs_root, annotations_root) -> tuple[str, torch.Tensor]:
    '''
    单一原数据处理函数，返回图片相关信息
    '''
    img_id = patten.match(img_id)
    img_name = img_id.group(1)

    img_path = os.path.join(imgs_root, img_name+'.jpg')
    annotation_path = os.path.join(annotations_root, img_name+'.xml')

    # 获取图片、锚框信息
    annotation_tree = ET.parse(annotation_path)
    anno_tree_root = annotation_tree.getroot()

    img_size = anno_tree_root.find('size')
    img_w = int(img_size.find('width').text)    #图片宽度
    img_h = int(img_size.find('height').text)   #图片高度

    objs = anno_tree_root.findall('object')
    objs_list = []
    for obj in objs:
        detail = []
        name = obj.find('name').text
        box = obj.find('bndbox')
        detail.append(int(box.find('xmin').text))
        detail.append(int(box.find('ymin').text))
        detail.append(int(box.find('xmax').text))
        detail.append(int(box.find('ymax').text))
        detail.append(file_list.index(name))

        objs_list.append(detail)

    # 处理获得目标值--------------------------------------------------------------------------
    # target处理
    y_target = torch.zeros(s, s, 5*b+c_nums)
    max_wh = max(img_h, img_w)
    grid_w = max_wh / s
    hp = int((max_wh - img_w) / 2)
    vp = int((max_wh - img_h) / 2)

    for obj in objs_list:
        # 值获取
        xmin = obj[0] + hp
        ymin = obj[1] + vp
        xmax = obj[2] + hp
        ymax = obj[3] + vp
        c_index = obj[4]

        w = sqrt(xmax - xmin) / img_w
        h = sqrt(ymax - ymin) / img_h

        # 获取窗口位置
        w_index = int((xmin + xmax)/(2*grid_w))
        h_index = int((ymin + ymax)/(2*grid_w))

        # 相对格子归一化
        x = ((xmin+xmax)/2-(w_index*grid_w)) / grid_w
        y = ((ymin+ymax)/2-(h_index*grid_w)) / grid_w


        box_target = torch.Tensor([x, y, w, h, 1]).repeat(b)
        c = torch.zeros(c_nums)
        c[c_index] = 1
        box_target = torch.cat([box_target, c])

        y_target[w_index][h_index] = box_target

    return img_path, y_target


def get_y(file_name:str, s:int=7, b:int=2):
    global pt_list
    img_path = os.path.join(img_root, file_name+'.jpg')
    annotation_path = os.path.join(annotation_root, file_name+'.xml')
    c_nums = len(file_list)

    # 获取图片、锚框信息
    annotation_tree = ET.parse(annotation_path)
    anno_tree_root = annotation_tree.getroot()

    img_size = anno_tree_root.find('size')
    img_w = int(img_size.find('width').text)    #图片宽度
    img_h = int(img_size.find('height').text)   #图片高度

    objs = anno_tree_root.findall('object')
    objs_list = []
    for obj in objs:
        detail = []
        name = obj.find('name').text
        box = obj.find('bndbox')
        detail.append(int(box.find('xmin').text))
        detail.append(int(box.find('ymin').text))
        detail.append(int(box.find('xmax').text))
        detail.append(int(box.find('ymax').text))
        detail.append(file_list.index(name))

        objs_list.append(detail)

    # 处理获得目标值--------------------------------------------------------------------------
    # target处理
    y_target = torch.zeros(s, s, 5*b+c_nums)
    max_wh = max(img_h, img_w)
    grid_w = max_wh / s
    hp = int((max_wh - img_w) / 2)
    vp = int((max_wh - img_h) / 2)

    for obj in objs_list:
        # 值获取
        xmin = obj[0] + hp
        ymin = obj[1] + vp
        xmax = obj[2] + hp
        ymax = obj[3] + vp
        c_index = obj[4]

        # pt_list.append([(xmin, ymin), (xmax, ymax)])

        # pt_list.append([(int(xmin/max_wh*224), int(ymin/max_wh*224)), (int(xmax/max_wh*224), int(ymax/max_wh*224))])
        # print(int((xmax+xmin)/(2*img_w)*224))
        # print(int((ymax+ymin)/(2*img_h)*224))
        # print()

        w = sqrt((xmax - xmin) / max_wh)
        h = sqrt((ymax - ymin) / max_wh)

        # 获取窗口位置
        w_index = int((xmin + xmax)/(2*grid_w))
        h_index = int((ymin + ymax)/(2*grid_w))

        # 相对格子归一化
        x = ((xmin+xmax)/2-(w_index*grid_w)) / grid_w
        y = ((ymin+ymax)/2-(h_index*grid_w)) / grid_w


        box_target = torch.Tensor([x, y, w, h, 1]).repeat(b)
        c = torch.zeros(c_nums)
        c[c_index] = 1
        box_target = torch.cat([box_target, c])

        y_target[w_index][h_index] = box_target

    return img_path, y_target


def draw_rec(img:torch.Tensor, y:torch.Tensor, s=7, b=2, img_size=224):
    img = np.array(transforms.ToPILImage()(img))

    p = img_size/s
    ycoodmask = torch.arange(0, s*s, 1, requires_grad=False).reshape(s, s)
    ycoodmask = ycoodmask % s
    ycoodmask = ycoodmask.reshape(s, s)
    xcoodmask = ycoodmask.transpose(0, 1)
    xcoodmask = xcoodmask.reshape(-1)
    ycoodmask = ycoodmask.reshape(-1)

    y = y[..., :5*b]
    y = y.reshape(-1, 2, 5)[:, 0, :]

    # 获取长度
    w = y[:, 2]
    h = y[:, 3]
    w = w*w * img_size
    h = h*h * img_size
    w = w/2
    h = h/2

    # 获取坐标
    xc = y[:, 0]
    yc = y[:, 1]
    xc = xc + xcoodmask
    yc = yc + ycoodmask

    c = y[:, -1]

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
    
    return img


if __name__ == '__main__':
    pt_list = []

    img_path, y = get_y('2007_000121')
    img = cv2.imread(img_path)

    unify_transform = transforms.Compose([
                transforms.ToTensor(),
                SquarePad(),
                transforms.Resize((448, 448))
                ])

    rimg = unify_transform(img)
    # rimg = np.array(transforms.ToPILImage()(rimg))

    rimg = draw_rec(rimg, y)

    # for pt in pt_list:
    #     rimg = cv2.rectangle(rimg, pt[0], pt[1], (255, 0, 0), 1)
    cv2.imwrite('test.jpg', rimg)
    
