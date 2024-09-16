import argparse
import logging
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter

from bulid_model import getmodels
from contrastive_loss import Intra_p_to_p, Intra_f_to_f, inter_f_to_p, inter_p_to_f
from datasets import getdataset
from memory import MemoryBank, load_neighbor
from utils import cluster_acc, AverageMeter, entropy, MarginLoss, \
    accuracy, memory_monitor, WeightEMA, Log

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# memory_monitor(memory_gpu=13000)


# os.environ['CUDA_VISIBLE_DEVICES'] = '5'


def selection_record(args, memory, index_lu, conf, feats, labeled_len):
    feats_1, feats_2, feats_3 = feats
    preds_mem_lu = memory.preds_mem.cuda()[index_lu]

    conf_msk = preds_mem_lu >= 0
    conf_msk[:labeled_len] = True

    conf, indcls, feats_1, feats_2, feats_3 = conf[conf_msk], preds_mem_lu[conf_msk], \
                                              feats_1[conf_msk], feats_2[conf_msk], feats_3[conf_msk]
    conf[:labeled_len] = 1.

    return conf_msk, conf, (feats_1, feats_2, feats_3), indcls


def train(args, model, device, train_label_loader, train_unlabel_loader,
          optimizer, m, epoch, tf_writer, mlp, ema_optimizer,
          ema_model, memory):
    model.train()
    mlp.train()
    m = min(m, args.mb)
    ce = MarginLoss(m=-1 * m)
    p_to_p_loss_func = Intra_p_to_p(class_num=args.num_classes, temperature=1.0, device=device, args=args)
    f_to_f_loss_func = Intra_f_to_f(batch_size=args.batch_size, temperature=0.5, device=device)

    losses = AverageMeter('loss', ':.4e')
    for batch_idx, (
    ((x, x2, x3), (x_nei, _), target, index_l), ((ux, ux2, ux3), (ux_nei, _), _, index_u)) in enumerate(
            zip(train_label_loader, train_unlabel_loader)):
        labeled_len = len(target)
        x, x2, x3, x_nei, target = torch.cat((x, ux), dim=0).to(device), torch.cat((x2, ux2), dim=0).to(device), \
                                   torch.cat((x3, ux3), dim=0).to(device), torch.cat((x_nei, ux_nei), dim=0).to(device), \
                                   target.to(device)
        index_lu = torch.cat((index_l, index_u), dim=0).to(device)
        optimizer.zero_grad()

        (output, feat), (output2, feat2), (output3, feat3), (output_nei, feat_nei) = model(x), model(x2), model(
            x3), model(x_nei)
        prob, prob2, prob3, prob_nei = F.softmax(output, dim=1), F.softmax(output2, dim=1), \
                                       F.softmax(output3, dim=1), F.softmax(output_nei, dim=1)

        feat1_mlp, feat2_mlp, feat3_mlp = mlp(feat), mlp(feat2), mlp(feat3)
        feat1_n, feat2_n, feat3_n = F.normalize(feat1_mlp, dim=1), F.normalize(feat2_mlp, dim=1), F.normalize(feat3_mlp, dim=1)
        feats_raw = F.normalize(feat, dim=1), F.normalize(feat2, dim=1), F.normalize(feat3, dim=1)

        output_ema, feat_ema = ema_model(x)
        prob_ema = F.softmax(output_ema, dim=1)

        conf_ema, _ = torch.max(prob_ema, dim=1)

        conf_msk, _, feats, indcls = selection_record(args, memory, index_lu, conf_ema, feats_raw, len(target))

        # intra-space loss
        loss_p_to_p, loss_ent = (p_to_p_loss_func(prob, prob2))
        loss_f_to_f = (f_to_f_loss_func(feat1_n, feat2_n) + f_to_f_loss_func(feat2_n, feat3_n)) / 2

        # inter-space loss
        loss_p_to_f = inter_p_to_f(indcls, feats, device)
        loss_f_to_p = inter_f_to_p(args, feat, target, prob, prob2, prob_nei, conf_msk, indcls)

        ce_loss = ce(output[:labeled_len], target)

        loss = ce_loss + args.weight_entropy * loss_ent + args.weight_f_to_p * loss_f_to_p + args.weight_p_to_p * loss_p_to_p + \
               args.weight_p_to_f * loss_p_to_f + args.weight_f_to_f * loss_f_to_f

        losses.update(loss.item(), args.batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

    tf_writer.add_scalar('loss/losses', losses.avg, epoch)


    return losses.avg


def validate(args, model, ema_model, mlp, labeled_num, device, val_loader, val_l_loader,
             epoch, tf_writer, memory):
    model.eval()
    ema_model.eval()
    mlp.eval()
    preds = np.array([])
    targets = np.array([])
    confs = np.array([])
    feats = torch.zeros(0, args.feat_channel).cuda()
    outputs = torch.zeros(0, args.num_classes).cuda()
    indexs = torch.zeros(0, )

    preds_mem = torch.zeros(0, ).cuda()

    with torch.no_grad():
        # if epoch == 0:
        for batch_idx, (x, label, index) in enumerate(val_l_loader):
            x, label = x.to(device), label.to(device)
            output, feat = model(x)
            _, pred = output.max(1)
            feats = torch.cat((feats, feat), dim=0)
            outputs = torch.cat((outputs, output), dim=0)
            indexs = torch.cat((indexs, index))
            preds_mem = torch.cat((preds_mem, label))

        for batch_idx, (x, label, index) in enumerate(val_loader):
            x, label = x.to(device), label.to(device)

            output, _ = ema_model(x)

            _, feat = model(x)
            feats = torch.cat((feats, feat), dim=0)
            outputs = torch.cat((outputs, output), dim=0)
            indexs = torch.cat((indexs, index))

            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)

            preds_mem = torch.cat((preds_mem, pred))

            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())

    targets = targets.astype(int)
    cls_num = args.num_classes
    preds = preds.astype(int)

    # progressive learning paradigm
    pls_thres = [1] * cls_num
    duration = min(1, int(args.kdur * (epoch/args.epochs) * 10)/10)
    for i in range(cls_num):
        conf_i = np.sort(confs[preds == i])
        if len(conf_i)>0:
            c_i = conf_i[-max(int(len(conf_i)*duration), 1)]
            pls_thres[i] = max(0.9, c_i)

    pls_thres = np.array(pls_thres)
    preds_confs_mem = preds_mem.clone().cpu().numpy()

    preds_confs_mem[-len(preds):] = (confs >= pls_thres[preds]) * preds + (confs < pls_thres[preds]) * -1
    memory.update(indexs.long(), feats.cpu().detach(), torch.tensor(preds_confs_mem))


    seen_mask = targets < labeled_num
    unseen_mask = ~seen_mask
    overall_acc = cluster_acc(preds, targets)
    seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
    unseen_nmi = metrics.normalized_mutual_info_score(targets[unseen_mask], preds[unseen_mask])
    mean_uncert = 1 - np.mean(confs)

    tf_writer.add_scalar('acc/overall', overall_acc, epoch)
    tf_writer.add_scalar('acc/seen', seen_acc, epoch)
    tf_writer.add_scalar('acc/unseen', unseen_acc, epoch)
    tf_writer.add_scalar('nmi/unseen', unseen_nmi, epoch)
    tf_writer.add_scalar('uncert/test', mean_uncert, epoch)
    print('{} epochs, overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}'.format(epoch, overall_acc,
                                                                                                  seen_acc, unseen_acc))
    return mean_uncert


def test(args, model, mlp, labeled_num, device, test_loader, logger, logger_en=False):
    model.eval()
    mlp.eval()
    preds = np.array([])
    targets = np.array([])

    with torch.no_grad():
        for batch_idx, (x, label, _) in enumerate(test_loader):
            x, label = x.to(device), label.to(device)
            output, feat = model(x)
            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)

            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())

    targets = targets.astype(int)
    preds = preds.astype(int)

    seen_mask = targets < labeled_num
    unseen_mask = ~seen_mask
    overall_acc = cluster_acc(preds, targets)
    seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
    unseen_nmi = metrics.normalized_mutual_info_score(targets[unseen_mask], preds[unseen_mask])

    print('During test time, Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}'.format(overall_acc,
                                                                                                 seen_acc, unseen_acc))

    if logger_en:
        logger.info(
            'Dataset: {}, Note: {}, Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}'.format(args.dataset,
                                                                                                        args.name,
                                                                                                        overall_acc,
                                                                                                        seen_acc,
                                                                                                        unseen_acc))



def load_resume(model, ema_model, optimizer, mlp, ema_mlp, epoch, args):
    ema_model.load_state_dict(torch.load('./checkpoints/' + str(epoch) + args.name + '_ema_model_' + args.dataset + '.pth.tar'))
    model.load_state_dict(torch.load('./checkpoints/' + str(epoch) + args.name + '_model_' + args.dataset + '.pth.tar'))
    optimizer.load_state_dict(torch.load('./checkpoints/' + str(epoch) + args.name + '_model_optimizer' + args.dataset + '.pth.tar'))
    ema_mlp.load_state_dict(torch.load('./checkpoints/' + str(epoch) + args.name + '_ema_mlp_' + args.dataset + '.pth.tar'))
    mlp.load_state_dict(torch.load('./checkpoints/' + str(epoch) + args.name + '_mlp_' + args.dataset + '.pth.tar'))
    return model, ema_model, optimizer, mlp, ema_mlp

def main(logger):
    parser = argparse.ArgumentParser(description='orca')
    # parser.add_argument('--milestones', nargs='+', type=int, default=[140, 180])
    parser.add_argument('--milestones', nargs='+', type=int, default=[170])
    parser.add_argument('--dataset', default='cifar100', help='dataset setting')
    parser.add_argument('--labeled-num', default=50, type=int)
    parser.add_argument('--labeled-ratio', default=0.5, type=float)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--datseed', type=int, default=0, metavar='S', help='random data seed (default: 0)')
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--exp_root', type=str, default='./results/')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        metavar='N',
                        help='mini-batch size')
    parser.add_argument('--default', action='store_true')
    # the argument below are hyperparam.
    parser.add_argument('--weight_entropy', default=2.0, type=float, help='the hyperparameter $\lambda_{ent}$ in the paper')
    parser.add_argument('--weight_p_to_p', default=1.0, type=float, help='the hyperparameter $\lambda_p$ in the paper')
    parser.add_argument('--weight_f_to_f', default=2.0, type=float, help='the hyperparameter $\lambda_f$ in the paper')
    parser.add_argument('--weight_p_to_f', default=1.0, type=float, help='the hyperparameter $\lambda_{p\to f}$ in the paper')
    parser.add_argument('--weight_f_to_p', default=1.0, type=float, help='the hyperparameter $\lambda_{f\to p}$ in the paper')
    parser.add_argument('--kdur', default=1., type=float, help='the hyperparameter controlling the speed, in the paper, delta t = epochs / (kdur * 10)')
    parser.add_argument('--resume', default=-1, type=int)
    parser.add_argument('--no_pretrained', action='store_true')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = False
    args.default = True
    args.mb = 0.3
    if args.default:
        if args.dataset == 'cifar100':
            args.labeled_num, args.weight_decay, args.weight_entropy = 50, 5e-4, 2
            if not args.no_pretrained:
                args.milestones, args.epochs = [450], 450
            else:
                args.milestones, args.epochs = [1200], 1200
        elif args.dataset == 'cifar10':
            args.labeled_num, args.weight_decay, args.weight_entropy, args.weight_p_to_p, weight_f_to_f = 5, 5e-4, 5, 0.25, 2.5
            args.kdur = 1.5
            if not args.no_pretrained:
                args.milestones, args.epochs = [450], 450
            else:
                args.milestones, args.epochs = [1200], 1200
        elif args.dataset == 'imagenet100':
            args.no_pretrained = False
            args.kdur = 3
            args.labeled_num, args.milestones, args.epochs, args.weight_decay, args.weight_entropy = 50, [230], 230, 1e-4, 2

    args.savedir = os.path.join(args.exp_root, args.name)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    # Initialize the splits
    train_label_loader, train_unlabel_loader, val_loader, val_l_loader, test_loader = getdataset(args)

    N = len(train_unlabel_loader.dataset.data_raw)
    # First network intialization
    model, ema_model, mlp, ema_mlp = getmodels(args)

    # Set the optimizer
    optimizer = optim.SGD([{'params': model.parameters()}, {'params': mlp.parameters()}], lr=1e-1,
                          weight_decay=args.weight_decay, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    ema_optimizer = WeightEMA(args, model, ema_model, mlp, ema_mlp)

    tf_writer = SummaryWriter(log_dir=args.savedir)
    memory = MemoryBank(N=N, c=args.feat_channel, cls_num=args.num_classes)

    if args.resume>0:
        model, ema_model, optimizer, mlp, ema_mlp = load_resume(model, ema_model, optimizer,
                                                                mlp, ema_mlp, args.resume, args)

    for epoch in range(args.resume+1, args.epochs):
        # assign and select the pseudo label to the unlabeled data
        mean_uncert = validate(args, model, ema_model, ema_mlp, args.labeled_num, device, val_loader,
                               val_l_loader, epoch, tf_writer, memory)
        # neighborhood search in the entire dataset
        load_neighbor(args, memory, train_label_loader, train_unlabel_loader, epoch / args.epochs)

        loss_avg = train(args, model, device, train_label_loader, train_unlabel_loader,
                         optimizer, mean_uncert, epoch, tf_writer, mlp, ema_optimizer, ema_model, memory)
        scheduler.step()

        if epoch == args.epochs - 1:
            torch.save(ema_model.state_dict(), './checkpoints/' + args.name + '_ema_model_' + args.dataset + '.pth.tar')
            torch.save(optimizer.state_dict(),
                       './checkpoints/' + args.name + '_model_optimizer' + args.dataset + '.pth.tar')
            torch.save(model.state_dict(), './checkpoints/' + args.name + '_model_' + args.dataset + '.pth.tar')
            torch.save(ema_mlp.state_dict(), './checkpoints/' + args.name + '_ema_mlp_' + args.dataset + '.pth.tar')
            torch.save(mlp.state_dict(), './checkpoints/' + args.name + '_mlp_' + args.dataset + '.pth.tar')

    test(args, ema_model, mlp, args.labeled_num, device, val_loader, logger, logger_en=True)


if __name__ == '__main__':
    main(logger)
