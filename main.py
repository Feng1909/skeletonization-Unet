from sched import scheduler
import time
from core.model import UNet
from solver.loss import Loss
from solver.dataset import build_transforms, SkeletonDataset

from tqdm import tqdm

import yaml
from sklearn.metrics import f1_score
from easydict import EasyDict as edict
import os
# 构建数据集
import cv2
import numpy as np
import paddle
from paddle.io import Dataset, DataLoader
from paddleseg.transforms import Compose, Resize

import paddle
from paddleseg.models import UNetPlusPlus
from paddleseg.models.losses import BCELoss
from paddleseg.core import train

def creat_data_list(dataset_path, mode='train'):
    with open(os.path.join(dataset_path, (mode+'/'+mode + '_pair.lst')), 'w') as f:
        A_path = os.path.join(os.path.join(dataset_path, mode), 'im')
        A_imgs_name = os.listdir(A_path)  # 获取文件夹下的所有文件名
        A_imgs_name.sort()
        for A_img_name in A_imgs_name:
            A_img = os.path.join(A_path, A_img_name)
            # B_img = os.path.join(A_path.replace('im', 'im'), A_img_name)
            label_img = os.path.join(A_path.replace('im', 'gt'), A_img_name[:-4]+'.png')
            f.write(A_img + ' ' + label_img + '\n')  # 写入list.txt
    print(mode + '_data_list generated')


class ChangeDataset(Dataset):
    # 这里的transforms、num_classes和ignore_index需要，避免PaddleSeg在Eval时报错
    def __init__(self, cfg, type):
        self.cfg = cfg
        self.data_folder = cfg['data_folder']
        self.indexes = []
        self.deal_annotation(cfg['ann_file'], type)
    
    def deal_annotation(self, ann_file, type):
        with open(ann_file, 'r') as f:
            ann_file = f.readlines()
        for i in ann_file:
            i = i.replace('\n', '').split(' ')
            self.indexes.append([i[0], i[1]])
    
    def __getitem__(self, idx):
        image, label = self.indexes[idx]
        image = cv2.imread(f'{self.data_folder}{image}')
        image = cv2.resize(image, [256,256])
        image = image.transpose(2,0,1)
        label = cv2.imread(f'{self.data_folder}{label}', cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, [256,256])

        return image, label

    def __len__(self):
        return len(self.indexes)


if __name__ == "__main__":
    with open('configs/unet_att.yaml') as yamlfile:
        cfgs = yaml.load(yamlfile, Loader=yaml.FullLoader)
        cfgs = edict(cfgs)


    dataset_path = 'dataset'  # data的文件夹

    # 完成三个数据的创建
    # train_data = ChangeDataset(dataset_path, 'train', transforms)
    test_data = ChangeDataset(cfgs['dataloader']['dataset'], 'test')
    # val_data = ChangeDataset(dataset_path, 'val', transforms)
    train_tranforms = build_transforms(is_train=True, cfg=cfgs)
    test_loader = DataLoader(dataset=SkeletonDataset(test_data, train_tranforms),
                    batch_size=cfgs['dataloader']['batch_size'],
                    drop_last=True,
                    num_workers=cfgs['dataloader']['num_workers'],
                    # pin_memory=True,
                    shuffle=True,
                    )

    # 分别创建三个list.txt
    # creat_data_list(dataset_path, mode='train')
    creat_data_list(dataset_path, mode='test')
    # creat_data_list(dataset_path, mode='val')
    
    
    model = UNet()


    # 参数、优化器及损失
    epochs = 20
    batch_size = 4
    iters = epochs * 445 // batch_size
    base_lr = 2e-1

    loss_fn = Loss()
    # paddle.summary(model, (1, 3, 1024, 1024))  # 可查看网络结构
    lr = paddle.optimizer.lr.CosineAnnealingDecay(base_lr, T_max=(iters // 3), last_epoch=0.5)  # 余弦衰减
    optimizer = paddle.optimizer.SGD(0.03, parameters=model.parameters())  # Adam优化器
    model.train()
    acc = []
    for epoch in range(epochs):
        print(epoch)
        running_dice_loss = 0.0
        running_bce_loss = 0.0
        running_acc = 0.0
        for input, target, label_128, label_64, label_32 in tqdm(test_loader):
            model.train()
            input = input.astype("float32")
            output, aux_128, aux_64, aux_32 = model(input)

            # compute loss
            soft_dice_loss, bce_loss = loss_fn(output, target)
            running_dice_loss += soft_dice_loss.item()
            print(soft_dice_loss.item(), bce_loss.item())
            running_bce_loss += bce_loss.item()

            # compute accuracy
            pred = output.clone()
            pred[pred >= 0.1] = 1
            pred[pred < 0.1] = 0
            running_acc += f1_score(paddle.flatten(pred).numpy(), 
                                        paddle.flatten(target).numpy())

            # auxiliary losses
            soft_dice_loss_128, bce_loss_128 = loss_fn(aux_128, label_128)
            soft_dice_loss_64, bce_loss_64 = loss_fn(aux_64, label_64)
            soft_dice_loss_32, bce_loss_32 = loss_fn(aux_32, label_32)

            loss = 0.5*(soft_dice_loss + bce_loss) \
                    + 0.3*(soft_dice_loss_128 + bce_loss_128) \
                    + 0.2*(soft_dice_loss_64 + bce_loss_64) \
                    + 0.1*(soft_dice_loss_32 + bce_loss_32) 
            print(loss.numpy())
            print(type(loss))
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            
        epoch_dice_loss = running_dice_loss / len(test_loader)
        epoch_bce_loss = running_bce_loss / len(test_loader)
        epoch_acc = running_acc / len(test_loader)
        print(epoch_acc)
        acc.append(epoch_acc)
        # break

    obj = {'model': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': epochs}
    path = 'model.pdparams'
    paddle.save(obj, path)
    print(acc)

    # model.load_dict('model.pdparams')
    load_layer_state_dict = paddle.load("model.pdparams")
    # print(load_layer_state_dict.keys())
    model.load_dict(load_layer_state_dict['model'])
    model.eval()

    for input, target, label_128, label_64, label_32 in tqdm(test_loader):
        input = input.astype("float32")
        output = model(input)
        break
    # print(output)
    print(len(output))
    print(output[1][0].numpy())
    # with open('out.jpg', 'w') as f:
        