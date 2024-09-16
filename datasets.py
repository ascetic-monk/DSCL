from __future__ import print_function

import os.path
import sys

from torchvision import datasets

# import datasets

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torchvision
from torchvision import transforms
from utils import RandAugmentMC
import math
import os
import os.path

import numpy as np
from utils import TransformTwice_w_s_s
import torch
import warnings
from PIL import Image, ImageOps, ImageFilter
import random

class Solarize(object):
    def __init__(self, p=0.2):
        self.prob = p

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        v = torch.rand(1) * 256
        return ImageOps.solarize(img, v)


class Equalize(object):
    def __init__(self, p=0.2):
        self.prob = p

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        return ImageOps.equalize(img)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def getdataset(args):
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        train_label_set, train_unlabel_set, val_set, val_l_set, test_set = getcifar(args)
        args.num_classes = 10 if args.dataset == 'cifar10' else 100
        print(args.num_classes)
        args.num_workers_train = 2
        args.num_workers_val = 0
    elif args.dataset == 'imagenet100':
        train_label_set, train_unlabel_set, val_set, val_l_set, test_set = getimagenet100(args)
        args.num_classes = 100
        args.num_workers_train = 8
        args.num_workers_val = 8
    elif args.dataset == 'tinyimagenet':
        train_label_set, train_unlabel_set, val_set, val_l_set, test_set = gettinyimagenet(args)
        args.num_classes = 200
        args.num_workers_train = 8
        args.num_workers_val = 8
    elif args.dataset == 'oxford' or args.dataset == 'cars' or args.dataset == 'aircraft':
        train_label_set, train_unlabel_set, val_set, val_l_set, test_set = getgeneric(args)
        classes_dic = {'oxford': 37, 'cars': 196, 'aircraft': 100}
        args.num_classes = classes_dic[args.dataset]
        args.num_workers_train = 8
        args.num_workers_val = 8
    else:
        warnings.warn('No registered dataset!')
        return

    labeled_len = len(train_label_set)
    unlabeled_len = len(train_unlabel_set)
    labeled_batch_size = int(args.batch_size * labeled_len / (labeled_len + unlabeled_len))

    # Initialize the splits
    train_label_loader = torch.utils.data.DataLoader(train_label_set, batch_size=labeled_batch_size, shuffle=True,
                                                     num_workers=args.num_workers_train, drop_last=True)
    train_unlabel_loader = torch.utils.data.DataLoader(train_unlabel_set,
                                                       batch_size=args.batch_size - labeled_batch_size, shuffle=True,
                                                       num_workers=args.num_workers_train, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=100, shuffle=False, num_workers=args.num_workers_val)
    val_l_loader = torch.utils.data.DataLoader(val_l_set, batch_size=100, shuffle=False, num_workers=args.num_workers_val)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=args.num_workers_val)
    return train_label_loader, train_unlabel_loader, val_loader, val_l_loader, test_loader


def get_labeled_index(targets, labeled_classes, labeled_ratio, class_wise=False):
    labeled_idxs = []
    unlabeled_idxs = []
    if class_wise:
        for label in range(max(targets) + 1):
            idx = np.where(np.array(targets) == label)[0]
            n_lbl_sample = math.ceil(len(idx) * labeled_ratio)
            np.random.shuffle(idx)
            if label in labeled_classes:
                labeled_idxs.extend(idx[:n_lbl_sample])
                unlabeled_idxs.extend(idx[n_lbl_sample:])
            else:
                unlabeled_idxs.extend(idx)
    else:
        for idx, label in enumerate(targets):
            if label in labeled_classes and np.random.rand() < labeled_ratio:
                labeled_idxs.append(idx)
            else:
                unlabeled_idxs.append(idx)
    # print(labeled_idxs)
    return labeled_idxs, unlabeled_idxs


def shrink_data(data, targets, idxs):
    targets = np.array(targets)
    targets = targets[idxs].tolist()
    data = data[idxs, ...]
    return data, targets


class OPENWORLDCIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, labeled=True, labeled_num=50, labeled_ratio=0.5, rand_number=0, transform=None,
                 target_transform=None,
                 download=False, unlabeled_idxs=None, train=True, neighbor=None, nei_transform=None,
                 transform_backup=None):
        super(OPENWORLDCIFAR100, self).__init__(root, train, transform, target_transform, download)

        if train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list
        self.data = []
        self.targets = []
        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        labeled_classes = range(labeled_num)
        np.random.seed(rand_number)

        self.data_raw = self.data
        self.neighbor = neighbor
        self.neighbor_dist = None
        self.nei_transform = nei_transform
        self.transform_backup = transform_backup

        self.indice = np.arange(len(self.targets))
        if train and labeled:
            self.labeled_idxs, self.unlabeled_idxs = get_labeled_index(self.targets, labeled_classes, labeled_ratio,
                                                                       True)
            self.data, self.targets = shrink_data(self.data, self.targets, self.labeled_idxs)
            self.indice = self.labeled_idxs
        elif train and not labeled:
            assert unlabeled_idxs
            self.data, self.targets = shrink_data(self.data, self.targets, unlabeled_idxs)
            self.indice = unlabeled_idxs

    def update_neighbor(self, distance, neighbor):
        self.neighbor = neighbor
        self.neighbor_dist = distance

    def transform_swi(self):
        self.transform, self.transform_backup = self.transform_backup, self.transform

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if not (self.neighbor is None):
            img_nei = self.data_raw[self.neighbor[self.indice[index]]]
            img_nei = Image.fromarray(img_nei)
            if self.nei_transform is not None:
                img_nei = self.nei_transform(img_nei)
            return img, (img_nei, self.neighbor_dist[self.indice[index]]), target, self.indice[index]
        else:
            return img, target, self.indice[index]

    def __len__(self):
        return len(self.data)


class OPENWORLDCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, labeled=True, labeled_num=50, labeled_ratio=0.5, rand_number=0, transform=None,
                 target_transform=None,
                 download=False, unlabeled_idxs=None, train=True, neighbor=None, nei_transform=None,
                 transform_backup=None):
        super(OPENWORLDCIFAR10, self).__init__(root, train, transform, target_transform, download)

        if train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list
        self.data = []
        self.targets = []
        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        labeled_classes = range(labeled_num)
        np.random.seed(rand_number)

        self.data_raw = self.data
        self.neighbor = neighbor
        self.neighbor_dist = None
        self.nei_transform = nei_transform
        self.transform_backup = transform_backup

        self.indice = np.arange(len(self.targets))
        if train and labeled:
            self.labeled_idxs, self.unlabeled_idxs = get_labeled_index(self.targets, labeled_classes, labeled_ratio,
                                                                       True)
            self.data, self.targets = shrink_data(self.data, self.targets, self.labeled_idxs)
            self.indice = self.labeled_idxs
        elif train and not labeled:
            assert unlabeled_idxs
            self.data, self.targets = shrink_data(self.data, self.targets, unlabeled_idxs)
            self.indice = unlabeled_idxs

    def update_neighbor(self, distance, neighbor):
        self.neighbor = neighbor
        self.neighbor_dist = distance

    def transform_swi(self):
        self.transform, self.transform_backup = self.transform_backup, self.transform

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if not (self.neighbor is None):
            img_nei = self.data_raw[self.neighbor[self.indice[index]]]
            img_nei = Image.fromarray(img_nei)
            if self.nei_transform is not None:
                img_nei = self.nei_transform(img_nei)
            return img, (img_nei, self.neighbor_dist[self.indice[index]]), target, self.indice[index]
        else:
            return img, target, self.indice[index]

    def __len__(self):
        return len(self.data)


class ImageNetDataset(datasets.ImageFolder):
    def __init__(self, root, labeled=True, labeled_num=50, labeled_ratio=0.5, rand_number=0, transform=None,
                 target_transform=None,
                 download=False, unlabeled_idxs=None, train=True, neighbor=None, nei_transform=None,
                 transform_backup=None):
        super().__init__(os.path.join(root, 'train') if train else os.path.join(root, 'val'),
                         transform=transform, target_transform=target_transform)
        self.imgs = np.array(self.imgs)
        self.targets = self.imgs[:, 1]
        self.targets = list(map(int, self.targets.tolist()))
        self.data = np.array(self.imgs[:, 0])
        self.targets = np.array(self.targets)

        labeled_classes = range(labeled_num)
        np.random.seed(rand_number)

        self.data_raw = self.data
        self.neighbor = neighbor
        self.neighbor_dist = None
        self.nei_transform = nei_transform
        self.transform_backup = transform_backup

        self.indice = np.arange(len(self.targets))
        if train and labeled:
            self.labeled_idxs, self.unlabeled_idxs = get_labeled_index(self.targets, labeled_classes, labeled_ratio,
                                                                       True)
            self.data, self.targets = shrink_data(self.data, self.targets, self.labeled_idxs)
            self.indice = self.labeled_idxs
        elif train and not labeled:
            assert unlabeled_idxs
            self.data, self.targets = shrink_data(self.data, self.targets, unlabeled_idxs)
            self.indice = unlabeled_idxs

    def update_neighbor(self, distance, neighbor):
        self.neighbor = neighbor
        self.neighbor_dist = distance

    def transform_swi(self):
        self.transform, self.transform_backup = self.transform_backup, self.transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.data[index]  # os.path.join(self.root, self.samples[index])
        target = self.targets[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if not (self.neighbor is None):
            img_nei = self.loader(self.data_raw[self.neighbor[self.indice[index]]])
            # img_nei = Image.fromarray(img_nei)
            if self.nei_transform is not None:
                img_nei = self.nei_transform(img_nei)
            return sample, (img_nei, self.neighbor_dist[self.indice[index]]), target, self.indice[index]
        else:
            return sample, target, self.indice[index]

        # return sample, target

    def __len__(self):
        return len(self.data)


def getcifar(args):
    # Dictionary of transforms
    dict_transform = {
        'cifar_weak': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]),
        'cifar_strong1': transforms.Compose([
            transforms.RandomChoice([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomResizedCrop(32, (0.5, 1.0)),
            ]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
            Solarize(p=0.1),
            Equalize(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]),
        'cifar_strong2': transforms.Compose([
            transforms.RandomChoice([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomResizedCrop(32, (0.5, 1.0)),
            ]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
            Solarize(p=0.1),
            Equalize(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]),
        # 'cifar_strong1': transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     RandAugmentMC(n=2, m=10),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        # ]),
        # 'cifar_strong2': transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     RandAugmentMC(n=2, m=10),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        # ]),
        'cifar_test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    }

    if args.dataset == 'cifar100':
        cifardataset = OPENWORLDCIFAR100
    elif args.dataset == 'cifar10':
        cifardataset = OPENWORLDCIFAR10
    else:
        warnings.warn('No not find the correct dataset!')
        return

    train_label_set = cifardataset(root='./datasets', labeled=True, labeled_num=args.labeled_num,
                                   labeled_ratio=args.labeled_ratio, rand_number=args.datseed, download=True,
                                   transform=TransformTwice_w_s_s(dict_transform['cifar_weak'],
                                                                  dict_transform['cifar_strong1'],
                                                                  dict_transform['cifar_strong2']),
                                   nei_transform=dict_transform['cifar_strong1'])
    train_unlabel_set = cifardataset(root='./datasets', labeled=False, labeled_num=args.labeled_num,
                                     labeled_ratio=args.labeled_ratio, download=True,
                                     transform=TransformTwice_w_s_s(dict_transform['cifar_weak'],
                                                                    dict_transform['cifar_strong1'],
                                                                    dict_transform['cifar_strong2']),
                                     unlabeled_idxs=train_label_set.unlabeled_idxs,
                                     nei_transform=dict_transform['cifar_strong1'])
    val_set = cifardataset(root='./datasets', labeled=False, labeled_num=args.labeled_num,
                           labeled_ratio=args.labeled_ratio, download=True,
                           transform=dict_transform['cifar_test'],
                           unlabeled_idxs=train_label_set.unlabeled_idxs,
                           transform_backup=dict_transform['cifar_strong1'])
    val_l_set = cifardataset(root='./datasets', labeled=False, labeled_num=args.labeled_num,
                             labeled_ratio=args.labeled_ratio, download=True,
                             transform=dict_transform['cifar_test'],
                             unlabeled_idxs=train_label_set.labeled_idxs,
                             transform_backup=dict_transform['cifar_strong1'])
    test_set = cifardataset(root='./datasets', train=False, transform=dict_transform['cifar_test'])
    return train_label_set, train_unlabel_set, val_set, val_l_set, test_set


def getimagenet100(args):
    dict_transform = {
        'imagenet100_weak': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'imagenet100_strong1': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'imagenet100_strong2': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        # 'imagenet100_strong1': transforms.Compose([
        #     transforms.RandomResizedCrop(224, (0.2, 1.0)), #stronger augmnetation
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.2),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ]),
        # 'imagenet100_strong2': transforms.Compose([
        #     transforms.RandomResizedCrop(224, (0.2, 1.0)), #stronger augmnetation
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.2),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ]),
        'imagenet100_test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])}

    root = os.path.join('./datasets', 'imagenet100_scan')
    train_label_set = ImageNetDataset(root=root, labeled=True, labeled_num=args.labeled_num,
                                      labeled_ratio=args.labeled_ratio, rand_number=args.datseed, download=True,
                                      transform=TransformTwice_w_s_s(dict_transform['imagenet100_weak'],
                                                                     dict_transform['imagenet100_strong1'],
                                                                     dict_transform['imagenet100_strong2']),
                                      nei_transform=dict_transform['imagenet100_strong1'])
    train_unlabel_set = ImageNetDataset(root=root, labeled=False, labeled_num=args.labeled_num,
                                        labeled_ratio=args.labeled_ratio, download=True,
                                        transform=TransformTwice_w_s_s(dict_transform['imagenet100_weak'],
                                                                       dict_transform['imagenet100_strong1'],
                                                                       dict_transform['imagenet100_strong2']),
                                        nei_transform=dict_transform['imagenet100_strong1'],
                                        unlabeled_idxs=train_label_set.unlabeled_idxs)
    val_set = ImageNetDataset(root=root, labeled=False, labeled_num=args.labeled_num,
                              labeled_ratio=args.labeled_ratio, download=True,
                              transform=dict_transform['imagenet100_test'],
                              unlabeled_idxs=train_label_set.unlabeled_idxs)
    val_l_set = ImageNetDataset(root=root, labeled=False, labeled_num=args.labeled_num,
                                labeled_ratio=args.labeled_ratio, download=True,
                                transform=dict_transform['imagenet100_test'],
                                unlabeled_idxs=train_label_set.labeled_idxs)
    test_set = ImageNetDataset(root=root, train=False, transform=dict_transform['imagenet100_test'])
    return train_label_set, train_unlabel_set, val_set, val_l_set, test_set


def gettinyimagenet(args):
    dict_transform = {
        'tinyimagenet_weak': transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        ]),
        'tinyimagenet_strong1': transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        ]),
        'tinyimagenet_strong2': transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]),
        ]),
        'tinyimagenet_test': transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        ])}

    root = os.path.join('./datasets', 'tiny-imagenet-200')
    train_label_set = ImageNetDataset(root=root, labeled=True, labeled_num=args.labeled_num,
                                      labeled_ratio=args.labeled_ratio, rand_number=args.datseed, download=True,
                                      transform=TransformTwice_w_s_s(dict_transform['tinyimagenet_weak'],
                                                                     dict_transform['tinyimagenet_strong1'],
                                                                     dict_transform['tinyimagenet_strong2']))
    train_unlabel_set = ImageNetDataset(root=root, labeled=False, labeled_num=args.labeled_num,
                                        labeled_ratio=args.labeled_ratio, download=True,
                                        transform=TransformTwice_w_s_s(dict_transform['tinyimagenet_weak'],
                                                                       dict_transform['tinyimagenet_strong1'],
                                                                       dict_transform['tinyimagenet_strong2']),
                                        unlabeled_idxs=train_label_set.unlabeled_idxs)
    val_set = ImageNetDataset(root=root, labeled=False, labeled_num=args.labeled_num,
                              labeled_ratio=args.labeled_ratio, download=True,
                              transform=dict_transform['tinyimagenet_test'],
                              unlabeled_idxs=train_label_set.unlabeled_idxs)
    val_l_set = ImageNetDataset(root=root, labeled=False, labeled_num=args.labeled_num,
                                labeled_ratio=args.labeled_ratio, download=True,
                                transform=dict_transform['tinyimagenet_test'],
                                unlabeled_idxs=train_label_set.labeled_idxs)
    test_set = ImageNetDataset(root=root, train=False, transform=dict_transform['tinyimagenet_test'])
    return train_label_set, train_unlabel_set, val_set, val_l_set, test_set


def getgeneric(args):
    imgnet_mean, imgnet_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    dict_transform = {
        'generic_weak': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        ]),
        'generic_strong1': transforms.Compose([
            transforms.RandomResizedCrop(224, (0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(imgnet_mean, imgnet_std),
        ]),
        'generic_strong2': transforms.Compose([
            transforms.RandomResizedCrop(224, (0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(imgnet_mean, imgnet_std),
        ]),
        'generic_test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imgnet_mean, std=imgnet_std)
        ])}
    if args.dataset == 'oxford':
        root = os.path.join('./datasets', 'oxford_trans')
    elif args.dataset == 'cars':
        root = os.path.join('./datasets', 'cars_trans')
    elif args.dataset == 'aircraft':
        root = os.path.join('./datasets', 'aircraft_trans')
    else:
        return

    train_label_set = GenericDataset(root=root, labeled=True, labeled_num=args.labeled_num,
                                     labeled_ratio=args.labeled_ratio, rand_number=args.datseed, download=True,
                                     transform=TransformTwice_w_s_s(dict_transform['generic_weak'],
                                                                    dict_transform['generic_strong1'],
                                                                    dict_transform['generic_strong2']),
                                     nei_transform=dict_transform['generic_strong1'])
    train_unlabel_set = GenericDataset(root=root, labeled=False, labeled_num=args.labeled_num,
                                       labeled_ratio=args.labeled_ratio, download=True,
                                       transform=TransformTwice_w_s_s(dict_transform['generic_weak'],
                                                                      dict_transform['generic_strong1'],
                                                                      dict_transform['generic_strong2']),
                                       nei_transform=dict_transform['generic_strong1'],
                                       unlabeled_idxs=train_label_set.unlabeled_idxs)
    val_set = GenericDataset(root=root, labeled=False, labeled_num=args.labeled_num,
                             labeled_ratio=args.labeled_ratio, download=True,
                             transform=dict_transform['generic_test'],
                             unlabeled_idxs=train_label_set.unlabeled_idxs)
    val_l_set = GenericDataset(root=root, labeled=False, labeled_num=args.labeled_num,
                                labeled_ratio=args.labeled_ratio, download=True,
                                transform=dict_transform['generic_test'],
                                unlabeled_idxs=train_label_set.labeled_idxs)
    test_set = GenericDataset(root=root, train=False, transform=dict_transform['generic_test'])
    return train_label_set, train_unlabel_set, val_set, val_l_set, test_set


class GenericDataset(datasets.ImageFolder):
    def __init__(self, root, labeled=True, labeled_num=50, labeled_ratio=0.5, rand_number=0, transform=None,
                 target_transform=None,
                 download=False, unlabeled_idxs=None, train=True, neighbor=None, nei_transform=None,
                 transform_backup=None):
        super().__init__(os.path.join(root, 'train') if train else os.path.join(root, 'val'),
                         transform=transform, target_transform=target_transform)
        self.imgs = np.array(self.imgs)
        self.targets = self.imgs[:, 1]
        self.targets = list(map(int, self.targets.tolist()))
        self.data = np.array(self.imgs[:, 0])
        self.targets = np.array(self.targets)

        labeled_classes = range(labeled_num)
        np.random.seed(rand_number)

        self.data_raw = self.data
        self.neighbor = neighbor
        self.neighbor_dist = None
        self.nei_transform = nei_transform
        self.transform_backup = transform_backup

        self.indice = np.arange(len(self.targets))
        if train and labeled:
            self.labeled_idxs, self.unlabeled_idxs = get_labeled_index(self.targets, labeled_classes, labeled_ratio,
                                                                       True)
            self.data, self.targets = shrink_data(self.data, self.targets, self.labeled_idxs)
            self.indice = self.labeled_idxs
        elif train and not labeled:
            assert unlabeled_idxs
            self.data, self.targets = shrink_data(self.data, self.targets, unlabeled_idxs)
            self.indice = unlabeled_idxs

    def update_neighbor(self, distance, neighbor):
        self.neighbor = neighbor
        self.neighbor_dist = distance

    def transform_swi(self):
        self.transform, self.transform_backup = self.transform_backup, self.transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.data[index]  # os.path.join(self.root, self.samples[index])
        target = self.targets[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if not (self.neighbor is None):
            img_nei = self.loader(self.data_raw[self.neighbor[self.indice[index]]])
            # img_nei = Image.fromarray(img_nei)
            if self.nei_transform is not None:
                img_nei = self.nei_transform(img_nei)
            return sample, (img_nei, self.neighbor_dist[self.indice[index]]), target, self.indice[index]
        else:
            return sample, target, self.indice[index]

        # return sample, target

    def __len__(self):
        return len(self.data)



if __name__ == '__main__':
    # dataset = ImageNetDataset('./datasets/imagenet100_scan/', transform=transform_test)
    # for (x, y) in dataset:
    #     # pass
    pass
