import paddle
import cv2

class Normalize(object):
    def __call__(self, sample):
        image, label = sample
        image = image / 255.
        label = label / 255.
        label[label >= 0.5] = 1
        label[label < 0.5] = 0

        label_128 = cv2.resize(label, (128, 128), interpolation=cv2.INTER_AREA)
        label_64 = cv2.resize(label, (64, 64), interpolation=cv2.INTER_AREA)
        label_32 = cv2.resize(label, (32, 32), interpolation=cv2.INTER_AREA)

        return image, label, label_128, label_64, label_32

class MyDataset(paddle.io.Dataset):
    def __init__(self, ann_path):
        super(MyDataset,self).__init__()
        self.normalize = Normalize()
        self.dataset = []
        self.indexes = []
        with open(ann_path, 'r') as f:
            ann_file = f.readlines()
        for i in ann_file:
            i = i.replace('\n', '').split(' ')
            self.indexes.append([i[0], i[1]])
            img = cv2.imread(i[0])
            label = cv2.imread(i[1], cv2.IMREAD_GRAYSCALE)
            ret, label = cv2.threshold(label, 10, 1, cv2.THRESH_BINARY, label)
            # cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
            # print(label.shape)
            # ma = 0
            # for i in range(label.shape[0]):
            #     for j in range(label.shape[1]):
            #         ma = max(ma, label[i][j])
            # print(ma)
            # sdfs
            self.dataset.append([img, label])
        

    def __getitem__(self,index):
        image, label =  self.dataset[index]
        image = cv2.resize(image, (256, 256))
        image = image.transpose(2,0,1)
        label = cv2.resize(label, (256, 256))
        image, label, label_128, label_64, label_32 = self.normalize((image, label))

        return [image.astype("float32")], [label, label_128, label_64, label_32]
        # return [image.astype("float32")], label
    
    def __len__(self):
        return len(self.indexes)