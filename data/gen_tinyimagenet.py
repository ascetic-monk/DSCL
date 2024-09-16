import os
import shutil

# with open('../datasets/tiny-imagenet-200/val/val_annotations.txt') as f:
#     s = f.read().split('\n')

# dict_c = {}
# for c in s:
#     if len(c)>2:
#         dict_c[c.split('\t')[0]] = c.split('\t')[1]

# for pth in os.listdir('../datasets/tiny-imagenet-200/val/images'):
#     if not os.path.exists(os.path.join('../datasets/tiny-imagenet-200/val_folders/', dict_c[pth])):
#         os.makedirs(os.path.join('../datasets/tiny-imagenet-200/val_folders/', dict_c[pth]))
#     shutil.copyfile(os.path.join('../datasets/tiny-imagenet-200/val/images', pth),
#                     os.path.join(os.path.join('../datasets/tiny-imagenet-200/val_folders/', dict_c[pth]), pth))
#
for pth in os.listdir('../datasets/tiny-imagenet-200/train'):
    if not os.path.exists(os.path.join('../datasets/tiny-imagenet-200/train_folders/', pth)):
        os.makedirs(os.path.join('../datasets/tiny-imagenet-200/train_folders/', pth))
    for subpth in os.listdir('../datasets/tiny-imagenet-200/train/'+pth+'/images'):
        shutil.copyfile(os.path.join('../datasets/tiny-imagenet-200/train/'+pth+'/images/'+subpth),
                        os.path.join(os.path.join('../datasets/tiny-imagenet-200/train_folders/', pth), subpth))
