import argparse
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter

import models
# import open_world_cifar as datasets
from datasets import getdataset
from utils import cluster_acc, accuracy, TransformTwice, memory_monitor

# memory_monitor(memory_gpu=23000)
#
os.environ['CUDA_VISIBLE_DEVICES'] = '5'


def valid(args, model, labeled_num, device, val_loader, epoch, tf_writer):
    val_label_loader, val_unlabel_loader = val_loader
    model.eval()
    preds = np.array([])
    targets = np.array([])
    confs = np.array([])
    features = np.zeros((0, 512))
    outputs = np.zeros((0, 100))
    with torch.no_grad():
        for batch_idx, (x, label, _) in enumerate(val_label_loader):
            x, label = x.to(device), label.to(device)
            output, _ = model(x)
            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
            features = np.append(features, _.cpu().numpy(), axis=0)
            outputs = np.append(outputs, output.cpu().numpy(), axis=0)
        for batch_idx, (x, label, _) in enumerate(val_unlabel_loader):
            x, label = x.to(device), label.to(device)
            output, _ = model(x)
            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
            features = np.append(features, _.cpu().numpy(), axis=0)
            outputs = np.append(outputs, output.cpu().numpy(), axis=0)
    targets = targets.astype(int)
    preds = preds.astype(int)

    # feat_normalize = features/(features ** 2).sum(1)[:, np.newaxis]
    # save_dict = {'features': feat_normalize, 'targets': targets, 'preds': preds, }
    # np.save('./results_feature/fver_debug_ema_model_cifar100.npy', save_dict)
    pass

    seen_mask = targets < labeled_num
    unseen_mask = ~seen_mask
    overall_acc = cluster_acc(preds, targets)
    seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
    unseen_nmi = metrics.normalized_mutual_info_score(targets[unseen_mask], preds[unseen_mask])
    mean_uncert = 1 - np.mean(confs)
    print('{:>3d}-th epoch, Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}'.format(epoch, overall_acc,
                                                                                                seen_acc, unseen_acc))

    return mean_uncert


def mask_data(mask, datas):
    datas_copy = []
    for i, data in enumerate(datas):
        datas_copy.append(data[mask])
    return datas_copy


def test(args, model, labeled_num, device, test_loader, epoch, tf_writer):
    model.eval()
    preds = np.array([])
    targets = np.array([])
    confs = np.array([])
    confs_output = np.array([])
    features = np.zeros((0, 512)) if args.dataset != 'imagenet100' else np.zeros((0, 2048))
    cls_num = 10 if args.dataset == 'cifar10' else 100
    outputs = np.zeros((0, cls_num))
    outes = np.zeros(0,)
    with torch.no_grad():
        for batch_idx, (x, label, _) in enumerate(test_loader):
            x, label = x.to(device), label.to(device)
            output, _ = model(x)
            prob = F.softmax(output, dim=1)
            outes = np.append(outes, torch.log(torch.exp(output).sum(1)).cpu().numpy())
            conf, pred = prob.max(1)
            conf_output, __ = output.max(1)
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
            confs_output = np.append(confs_output, conf_output.cpu().numpy())
            features = np.append(features, _.cpu().numpy(), axis=0)
            outputs = np.append(outputs, output.cpu().numpy(), axis=0)
    targets = targets.astype(int)
    preds = preds.astype(int)
    #
    # save_dict = {'features': features, 'targets': targets, 'preds': preds}

    seen_mask = targets < labeled_num
    unseen_mask = ~seen_mask
    overall_acc = cluster_acc(preds, targets)
    seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
    unseen_nmi = metrics.normalized_mutual_info_score(targets[unseen_mask], preds[unseen_mask])
    mean_uncert = 1 - np.mean(confs)
    print('{:>3d}-th epoch, Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}'.format(epoch, overall_acc,
                                                                                                seen_acc, unseen_acc))

    return mean_uncert


def test_process(args, device):
    args.savedir = os.path.join(args.exp_root, args.name)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    args.labeled_num=5 if args.dataset == 'cifar10' else 50
    args.labeled_ratio=0.5
    args.seed=1
    args.datseed=0

    train_label_loader, train_unlabel_loader, val_loader, val_l_loader, test_loader = getdataset(args)

    # First network intialization: pretrain the RotNet network
    if args.dataset == 'imagenet100':
        model = models.resnet50(num_classes=args.num_classes)
    else:
        model = models.resnet18(num_classes=args.num_classes)
    model = model.to(device)
    state_dict = torch.load('./checkpoints/'+args.name+'_ema_model_'+args.dataset+'.pth.tar')
    # state_dict = torch.load('./checkpoints/'+args.name+'_model_'+args.dataset+'.pth.tar')

    # state_dict_new = model.state_dict()
    # for k, v in state_dict_new.items():
    #     if 'module.' + k in state_dict.keys():
    #         state_dict_new[k] = state_dict['module.' + k]
    # state_dict = state_dict_new
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    tf_writer = SummaryWriter(log_dir=args.savedir)

    mean_uncert = test(args, model, args.labeled_num, device, val_loader, 0, tf_writer)
    # mean_uncert = valid(args, model, args.labeled_num, device, (train_label_loader, train_unlabel_loader), 0, tf_writer)



def main():
    parser = argparse.ArgumentParser(description='orca')
    parser.add_argument('--milestones', nargs='+', type=int, default=[140, 180])
    parser.add_argument('--dataset', default='cifar100', help='dataset setting')
    parser.add_argument('--labeled-num', default=5, type=int)
    parser.add_argument('--labeled-ratio', default=0.5, type=float)
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    parser.add_argument('--exp_root', type=str, default='./results/')
    parser.add_argument('--epochs', type=int, default=200)

    parser.add_argument('--name', type=str, default='fver_cifar10_5_12_kdur5_ic2_5')

    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        metavar='N',
                        help='mini-batch size')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    test_list = [
        ['cifar100', 'fver_aba_cifar100_aba0'],
    ]
    for dataset, name in test_list:
        args.dataset = dataset
        args.name = name
        print(args.name)
        try:
            test_process(args, device)
        except:
            print('error')

if __name__ == '__main__':
    main()


