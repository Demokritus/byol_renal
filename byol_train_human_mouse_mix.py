#!/usr/bin/env python3

# from audioop import avg
import os
import sys
import re
import numpy as np
from numpy import int16
from numpy import Inf
from typing import List, Tuple
# from pyrsistent import T
import torch
from torch import nn
# import torchvision
import copy
import argparse

import torchvision.transforms as transforms
from roi_tensor_extract import ROIExtract

from lightly.data import LightlyDataset
from lightly.data import BaseCollateFunction
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum

from resnet_monochrome import resnet18
from resnet_monochrome import resnet34
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from datetime import datetime

import torch.distributed as dist
import torch.multiprocessing as mp

from shuffle_crops import ShuffleCrops


class BYOL(nn.Module):
    def __init__(self, backbone, hidden_dim : int = 1024,
                middle_dim : int = 256,
                output_dim : int = 256,
                dropout_p : float = 0.2,):
        '''
        It creates an instance of BYOL made of ResNet as a backbone
        and a projection head which comprises of 4 layers
        Args:
            hidden_dim : int
                tha layer in the middle of projection,
            middle_dim : int
                layer in the beginning of projection,
            output_dim : int
                the last layer in the proj. head
        '''
        super().__init__() # inherit properties from torch.nn.Module

        self.backbone = backbone # passing backbone (by default it is ResNet 18)
        self.dropout = nn.Dropout(p=dropout_p) # dropout for the backbone
        self.projection_head = BYOLProjectionHead(512, hidden_dim, middle_dim) # 2 layers ofr projection head
        self.prediction_head = BYOLProjectionHead(middle_dim, hidden_dim, output_dim) # two layers ofr prediction head

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        # dropout for the backbone
        y = self.dropout(y)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z
    

class BaseCollateFunctionDropNan(nn.Module):
    """Base class for other collate implementations.

    Takes a batch of images as input and transforms each image into two 
    different augmentations with the help of random transforms. The images are
    then concatenated such that the output batch is exactly twice the length 
    of the input batch.

    Attributes:
        transform:
            A set of torchvision transforms which are randomly applied to
            each image.

    """

    def __init__(self, transform: transforms.Compose):
        super(BaseCollateFunctionDropNan, self).__init__()
        self.transform = transform

    def forward(self, batch: List[tuple]):
        """Turns a batch of tuples into a tuple of batches.

            Args:
                batch:
                    A batch of tuples of images, labels, and filenames which
                    is automatically provided if the dataloader is built from 
                    a LightlyDataset.

            Returns:
                A tuple of images, labels, and filenames. The images consist of 
                two batches corresponding to the two transformations of the
                input images.

            Examples:
                >>> # define a random transformation and the collate function
                >>> transform = ... # some random augmentations
                >>> collate_fn = BaseCollateFunction(transform)
                >>>
                >>> # input is a batch of tuples (here, batch_size = 1)
                >>> input = [(img, 0, 'my-image.png')]
                >>> output = collate_fn(input)
                >>>
                >>> # output consists of two random transforms of the images,
                >>> # the labels, and the filenames in the batch
                >>> (img_t0, img_t1), label, filename = output

        """
        batch_size = len(batch)
        
        # check that ROIExtract function is inside the transform list self.transform.transforms
        checkROI = any(list(map(lambda x: isinstance(x, ROIExtract), self.transform.transforms)))
            
        if checkROI:
            # transorms before and including ROI extraction 
            transform_with_roi = [self.transform.transforms[i] 
                                for i in range(len(self.transform.transforms)) 
                                if not isinstance(self.transform.transforms[i], ROIExtract)]
            
            transform_with_roi.append(ROIExtract())
            
                
            transform_with_roi = transforms.Compose(transform_with_roi)
            
            # transforms after ROI extraction
            transform_after_roi = [self.transform.transforms[i] 
                                    for i in range(len(transform_with_roi.transforms), len(self.transform.transforms))]
            transform_after_roi = transforms.Compose(transform_after_roi)

            # list of transformed images
            transforms_ = [transform_with_roi(batch[i % batch_size][0]) # .unsqueeze_(0)
                        for i in range(2 * batch_size)]
            
            transforms_filtered = list(filter(lambda x: x is not None, transforms_))
            # apply transform_after_roi to all entries in transforms_filtered
            transforms_filtered = [transform_after_roi(transforms_filtered[i]) for i in range(len(transforms_filtered))]
        else:
            transforms_ = [self.transform(batch[i % batch_size][0]) # .unsqueeze_(0)
                        for i in range(2 * batch_size)]
            # BEGIN MOD 1 : drop nan images - 2022.11.14
            # remove all None values from transforms
            transforms_filtered = list(filter(lambda x: x is not None, transforms_))
            
        transforms_filtered = [x.unsqueeze_(0) for x in transforms_filtered]
        
        checks = [True if x is not None else False for x in transforms_]
        batch2 = [batch[i // 2] for i in range(len(checks)) if checks[i]]
        # END MOD
        
        # list of labels
        labels = torch.LongTensor([item[1] for item in batch2])
        # list of filenames
        fnames = [item[2] for item in batch2]

        # tuple of transforms
        transforms_ = (
            torch.cat(transforms_filtered[:batch_size], 0),
            torch.cat(transforms_filtered[batch_size:], 0)
        )
        return transforms_, labels, fnames


def find_last_checkpoint(dir_checkpoint):
    '''Find the last checkpoint in a given directory'''
    fls = list(filter(lambda x: re.match(r'cp_([0-9]+).pth', x), os.listdir(dir_checkpoint)))
    if len(fls) > 0:
        nbs = list(map(lambda x: int(re.match(r'cp_([0-9]+).pth', x).group(1)), fls))
        return np.max(nbs)
    else:
        return 0


def get_args():
    '''
    The function to retrieve args passed in command line.
    '''
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dims', metavar='D', type=int, default=32,
                        help='Dimensionality of representations', dest='n_dims')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=1000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=int, default=0,
                         help='Load model from a checkpoint')
    parser.add_argument('-g', '--gpus', dest='gpus', type=int, default=2,
                        help='Number of GPUs')
    # parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
    #                     help='Downscaling factor of the images')
    return parser.parse_args()


def main_train(gpu, args):
    # resnet = torchvision.models.resnet18()
    # backbone = nn.Sequential(*list(resnet.children())[:-1])

    dist.init_process_group(
    backend='nccl',
    #init_method='env://',
    world_size=args.gpus,
    rank=gpu
    )

    resnet = resnet34()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model = BYOL(backbone)

    # device : str = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device)
    model.cuda(gpu)

    # cifar10 = torchvision.datasets.CIFAR10("datasets/cifar10", download=True)
    path_to_data = "/home/gsergei/data/binary_dataset_mix"

    dataset = LightlyDataset(path_to_data + "/train")
    # or create a dataset from a folder containing images or videos:
    # dataset = LightlyDataset("path/to/folder")

    torch.manual_seed(0)

    if args.load < 0:
        args.load = find_last_checkpoint(args.dir_checkpoint)
    if args.load > 0:
        fl = args.dir_checkpoint + 'cp_%i.pth' % args.load
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
        model.load_state_dict(torch.load(fl, map_location=map_location))
        #net.load_state_dict(torch.load(fl))
        print('(%i) Model loaded from %s' % (gpu, fl), flush=True)

    transform_train = transforms.Compose([
        transforms.RandomRotation(90),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),
        ShuffleCrops(),
        transforms.ToTensor(),
        # resize the images to 224x224
        transforms.Resize((224, 224)),
        # apply gaussian blur
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])
    
    transform_valid = transforms.Compose([
        # transforms.RandomRotation(90),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1)
        # resize the images to 224x224
        # transforms.Resize((224, 224)),
        # apply gaussian blur
        # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])

    # BEGIN MOD 2 : drop nan images - 2022.11.14
    # collate_fn = BaseCollateFunction(transform)
    collate_train = BaseCollateFunctionDropNan(transform_train)
    collate_valid = BaseCollateFunctionDropNan(transform_valid)
    # END MOD 2
    # collate_fn = SimCLRCollateFunction(input_size=32)

    # train_dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=4,
    #     collate_fn=collate_fn,
    #     # shuffle=True,
    #     drop_last=True,
    #     num_workers=4,
    # )


    # create a validation dataloader for the validation step
    # using files from "../mix_mouse_hoyer_imgs/validation"
    validation_dataset = LightlyDataset(path_to_data + "/validation")
    # validation_dataloader = torch.utils.data.DataLoader(
    #     validation_dataset,
    #     batch_size=4,
    #     collate_fn=collate_fn,
    #     # shuffle=True,
    #     drop_last=True,
    #     num_workers=4,
    # )

    # distribute datasets over gpus :parallelize:
    # dataset_train = InstDataset(img_dir, True, False, scale=args.scale)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.gpus, rank=gpu)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                        shuffle=False, num_workers=0, pin_memory=True, sampler=train_sampler,
                                        collate_fn=collate_train,
                                        drop_last=True)

    # dataset_val = InstDataset(os.path.join(img_dir, "test"), False, True, scale=args.scale)
    val_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset, num_replicas=args.gpus, rank=gpu)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, 
                                            shuffle=False, num_workers=0, pin_memory=True, sampler=val_sampler,
                                            collate_fn=collate_valid,
                                            drop_last=True)

    # distribute over several gpus :parallelize:
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], output_device=gpu)

    try:
        train(model, gpu, train_dataloader, validation_dataloader, lr = args.lr, max_n_epochs = args.epochs)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


# write the function for validation step
def validation_step(model, dataloader, criterion, device):
    # model.eval()
    val_loss = 0
    for (x0, x1), _, _ in dataloader:
        x0 = x0.to(device)
        x1 = x1.to(device)
        with torch.no_grad():
            p0 = model(x0)
            p1 = model(x1)
            loss = criterion(p0, p1)
            val_loss += loss.item()
    val_loss /= len(dataloader)
    return val_loss
    

def train(model,
        gpu: int,
        train_dataloader: torch.utils.data.DataLoader,
        valid_dataloader: torch.utils.data.DataLoader,
        max_n_epochs: int16 = 1000,
        patience: int = 30,
        criterion = NegativeCosineSimilarity(),
        ckpt_save_dir: str = "/scratch/gsergei/checkpoints/byol_renal_mouse_human_mix_V3", 
        log_dir: str = "/scratch/gsergei/tb_logs_mouse_human_mix_V3/",
        valid_freq: int = 10,
        lambda_: float = 0.001,
        lr: float = 0.001,
        lr_decay: float = 0.1,
        weight_decay: float = 0.0001,
        momentum: float = 0.9,):
    
    print("Training on GPU %i" % gpu, flush=True)

    log_name = datetime.now().strftime('%y-%h_%d-%H-%M') + "_byol_renal"
    tb_writer = SummaryWriter(log_dir=log_dir + log_name)

    # criterion = NegativeCosineSimilarity()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

    # create an SGD optimizer with Nesterov momentum 0f 0.9 and weight decay
    # of 0.0001
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # max_n_epochs : int16 = 1000
    prev_loss : float = Inf
    prev_val_loss: float = Inf
    # patience : int = 10
    pat_count : int = 0

    print("Starting Training")
    for epoch in range(max_n_epochs):
        total_loss : float = 0

        for (x0, x1), _, _ in train_dataloader:
            # skip the iteration x0 or x1 is filled with zeros
            if torch.sum(x0) == 0 or torch.sum(x1) == 0:
                continue
            # update_momentum(model.backbone, model.backbone_momentum, m=0.99)
            # update_momentum(model.projection_head, model.projection_head_momentum, m=0.99)
            update_momentum(model.module.backbone, model.module.backbone_momentum, m=0.99)
            update_momentum(model.module.projection_head, model.module.projection_head_momentum, m=0.99)
            x0 = x0.cuda(gpu)
            x1 = x1.cuda(gpu)
            p0 = model.module(x0)
            # z0 = model.forward_momentum(x0)
            z0 = model.module.forward_momentum(x0)
            # p1 = model(x1)
            p1 = model.module(x1)
            z1 = model.module.forward_momentum(x1)
            loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0))
            # regularize loss with Ridge loss that is proportional to the difference between the two projections
            loss += lambda_ * (torch.norm(p0 - p1, p=2) ** 2)

            # prev_loss = total_loss
            total_loss += loss.detach()
            loss.backward()

            # apply gradient clipping
            clipping_value = 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)

            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_dataloader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

        if epoch % valid_freq == 0:
            val_loss = validation_step(model, valid_dataloader, criterion, gpu)
            
            print("Control step is started...")
            if pat_count >= patience:
                print("Patience limit has been exceeded. \n \
                        Finishing training.")
                break
            if avg_loss < prev_loss and val_loss < prev_val_loss:
                print("Saving the checkpoint of the model...")
                torch.save(model.state_dict(), os.path.join(ckpt_save_dir, 
                "byol_resnet18_"+str(epoch)+".ckpt"))
                pat_count = 0
                print("Patience counter is set to 0: {}".format(pat_count))
            else:
                pat_count += 1
                print("Patience counter increased by 1: {}".format(pat_count))

            tb_writer.add_scalar('Loss/train', avg_loss, epoch)
            tb_writer.add_scalar('Loss/validation', val_loss, epoch)
            # tb_writer.add_image("image_views_x0", make_grid(x0, train_dataloader.batch_size), epoch)
            # tb_writer.add_image("image_views_x1", make_grid(x1, train_dataloader.batch_size), epoch)
            # add the image views to the tensorboard taking into account the range of pixel intensities (0-255)
            tb_writer.add_image("image_views_x0", make_grid(x0, train_dataloader.batch_size, normalize=True), epoch)
            tb_writer.add_image("image_views_x1", make_grid(x1, train_dataloader.batch_size, normalize=True), epoch)

        prev_loss = copy.deepcopy(avg_loss)
        prev_val_loss = copy.deepcopy(val_loss)


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    dir_checkpoint = "/scratch/gsergei/checkpoints/byol_renal_mouse_human_mix_V3"

    if not os.path.exists(dir_checkpoint):
        os.mkdir(dir_checkpoint)
        print('Created checkpoint directory %s' % dir_checkpoint, flush=True)
    args.dir_checkpoint = dir_checkpoint

    os.environ['MASTER_ADDR'] = 'localhost' #'127.0.0.1'  #
    os.environ['MASTER_PORT'] = '8900'  # 
    mp.spawn(main_train, nprocs=args.gpus, args=(args,))