import warnings

import torch
import torch.nn as nn

import models


def getmodels(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    # First network intialization: pretrain the SimCLR network
    if args.dataset == 'cifar100' or args.dataset == 'cifar10' or args.dataset == 'tinyimagenet':
        model = models.resnet18(num_classes=args.num_classes).to(device)
        ema_model = models.resnet18(num_classes=args.num_classes).to(device)
        args.feat_channel = 512
        # for name, param in model.named_parameters():
        #     if 'linear' not in name and 'layer4' not in name:
        #         param.requires_grad = False
    elif args.dataset == 'imagenet100':
        model = models.resnet50(num_classes=args.num_classes).to(device)
        ema_model = models.resnet50(num_classes=args.num_classes).to(device)
        args.feat_channel = 2048

        for name, param in model.named_parameters():
            if 'fc' not in name and 'layer4' not in name:
                param.requires_grad = False

    mlp = nn.Sequential(nn.Linear(args.feat_channel, 2048),
                        nn.ReLU(),
                        nn.Linear(2048, 128)).to(device)
    ema_mlp = nn.Sequential(nn.Linear(args.feat_channel, 2048),
                            nn.ReLU(),
                            nn.Linear(2048, 128)).to(device)

    if args.dataset == 'cifar10':
        state_dict = torch.load('./pretrained/simclr_cifar_10.pth.tar')
    elif args.dataset == 'cifar100':
        state_dict = torch.load('./pretrained/simclr_cifar_100.pth.tar')
    elif args.dataset == 'imagenet100':
        state_dict = torch.load('./pretrained/simclr_imagenet_100.pth.tar')
    elif args.dataset == 'tinyimagenet':
        state_dict = None
    else:
        warnings.warn('No registered dataset!')
        return


    if not args.no_pretrained and not (state_dict is None):
        model.load_state_dict(state_dict, strict=False)
        ema_model.load_state_dict(state_dict, strict=False)
        ema_mlp.load_state_dict(mlp.state_dict())
        print('SimCLR weight would be loaded!')
    else:
        ema_model.load_state_dict(model.state_dict())
        ema_mlp.load_state_dict(mlp.state_dict())
        print('No weight would be loaded!')

    for param in ema_model.parameters():
        param.detach_()
        param.requires_grad = False

    for param in ema_mlp.parameters():
        param.detach_()
        param.requires_grad = False

    if torch.cuda.device_count() > 1:
        model, ema_model = nn.DataParallel(model), nn.DataParallel(ema_model)
        mlp, ema_mlp = nn.DataParallel(mlp), nn.DataParallel(ema_mlp)
    # model, ema_model, mlp, ema_mlp = model.to(device), ema_model.to(device), mlp.to(device), ema_mlp.to(device)

    return model, ema_model, mlp, ema_mlp

