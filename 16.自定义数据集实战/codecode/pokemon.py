import torch
import os, glob
import random, csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

root = '/home/zfk/jupyterProjects/16.自定义数据集实战/data/pokemon'

class Pokemon(Dataset):
    def __init__(self, root, resize, mode):
        super(Pokemon, self).__init__()

        self.root = root
        self.resize = resize

        self.name2label = {}
        for name in sorted(os.listdir((os.path.join(root)))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())
        # print(self.name2label)

        # image, label
        self.images, self.labels = self.load_csv('images.csv')  # 要么创建要么读取，创建文件可通过load_csv方法实现

        if mode == 'train':  # 60%
            self.images = self.images[: int(0.6 * len(self.images))]
            self.labels = self.labels[: int(0.6 * len(self.labels))]
        elif mode == 'val':  # 20%
            self.images = self.images[int(0.6 * len(self.images)): int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)): int(0.8 * len(self.labels))]
        else:  # 20%
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):  # 如果文件不存在则需要创建它并保存在csv文件中.
            images = []
            for name in self.name2label.keys():
                # 'pokemon\\mewteo\\0.0001.png
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))

            # 1167, 'pokemon\\bulbasaur\\00000000.png'
            print(len(images), images)

            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:  # 'pokemon\\bulbasaur\\00000000.png'
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    # 'pokemon\\bulbasaur\\00000000.png', 0
                    writer.writerow([img, label])
                print('writen into csv file:', filename)

        # read from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            # 'pokemon\\bulbasaur\\00000000.png', 0
            for row in reader:
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)

            assert len(images) == len(labels)

            return images, labels

    def __len__(self):
        return len(self.images)

    def denormalize(self, x_hat):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x_hat = (x-mean)/std
        # x = x_hat*std = mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean

        return x

    def __getitem__(self, idx):
        # idx-[0-len(images)]
        # self.images, self.labels
        # img: 'pokemon\\bulbasaur\\00000000.png' img的格式还是路径格式的
        # label: 0 label的类型还是一个整数
        img, label = self.images[idx], self.labels[idx]
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # string path ->  image data
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        ])

        img = tf(img)
        label = torch.tensor(label)
        return img, label


def main():
    import torchvision
    import visdom
    import time

    viz = visdom.Visdom()

    # =================================================================== #
    # 使用更加简便的API实现
    # tf = transforms.Compose([
    #     transforms.Resize((64, 64)),
    #     transforms.ToTensor(),
    # ])
    # db = torchvision.datasets.ImageFolder(root=root, transform=tf)
    # loader = DataLoader(db, batch_size=64, shuffle=True, num_workers=4)
    # print(db.class_to_idx)  # 打印表格编码 print(self.name2label)
    # for x, y in loader:
    #     viz.images(x, nrow=8, win='batch', opts=dict(title='batch'))
    #     viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
        # time.sleep(10)
    # ==================================================================== #

    db = Pokemon(root, 224, 'train')

    x, y = next(iter(db))
    print('sample:', x.shape, y.shape, y)

    viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))

    loader = DataLoader(db, batch_size=32, shuffle=True)

    for x, y in loader:
        viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
        # time.sleep(10)


if __name__ == "__main__":
    main()
