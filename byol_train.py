_human_mouse_mix.py#!/usr/bin/env python3

# from audioop import avg
import os
from numpy import int16
from numpy import Inf
# from pyrsistent import T
import torch
from torch import nn
# import torchvision
import copy

import torchvision.transforms as transforms
from roi_tensor_extract import ROIExtract

from lightly.data import LightlyDataset
from lightly.data import BaseCollateFunction
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum

from resnet_monochrome import resnet18
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from datetime import datetime


class BYOL(nn.Module):
    def __init__(self, backbone, hidden_dim : int = 1024,
                middle_dim : int = 256,
                output_dim : int = 256):
        super().__init__()

        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(512, hidden_dim, middle_dim)
        self.prediction_head = BYOLProjectionHead(middle_dim, hidden_dim, output_dim)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z


# resnet = torchvision.models.resnet18()
# backbone = nn.Sequential(*list(resnet.children())[:-1])

resnet = resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = BYOL(backbone)

device : str = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# cifar10 = torchvision.datasets.CIFAR10("datasets/cifar10", download=True)
path_to_data = "/home/gsergei/imgs_linus/alles"

dataset = LightlyDataset(path_to_data)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

transform = transforms.Compose([
    transforms.RandomRotation(90),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ROIExtract()
])

collate_fn = BaseCollateFunction(transform)
# collate_fn = SimCLRCollateFunction(input_size=32)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    collate_fn=collate_fn,
    # shuffle=True,
    drop_last=True,
    num_workers=4,
)

criterion = NegativeCosineSimilarity()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

max_n_epochs : int16 = 1000
prev_loss : float = Inf
patience : int = 10
pat_count : int = 0

ckpt_save_dir : str = "/scratch/gsergei/checkpoints/byol_renal"
log_dir : str = "/scratch/gsergei/tb_logs/"

log_name = datetime.now().strftime('%y-%h_%d-%H-%M') + "_byol_renal_v2"
tb_writer = SummaryWriter(log_dir=log_dir + log_name)

print("Starting Training")
for epoch in range(max_n_epochs):
    total_loss : float = 0

    for (x0, x1), _, _ in dataloader:
        update_momentum(model.backbone, model.backbone_momentum, m=0.99)
        update_momentum(model.projection_head, model.projection_head_momentum, m=0.99)
        x0 = x0.to(device)
        x1 = x1.to(device)
        p0 = model(x0)
        z0 = model.forward_momentum(x0)
        p1 = model(x1)
        z1 = model.forward_momentum(x1)
        loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0))
        # prev_loss = total_loss
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

    if epoch % 5 == 0:
        print("Control step is started...")
        if pat_count >= patience:
            print("Patience limit has been exceeded. \n \
                    Finishing training.")
            break
        if avg_loss < prev_loss:
            print("Saving the checkpoint of the model...")
            torch.save(model.state_dict(), os.path.join(ckpt_save_dir, 
            "byol_resnet18_"+str(epoch)+".ckpt"))
            pat_count = 0
            print("Patience counter is set to 0: {}".format(pat_count))
        else:
            pat_count += 1
            print("Patience counter increased by 1: {}".format(pat_count))

        tb_writer.add_scalar('Loss/train', avg_loss, epoch)
        tb_writer.add_image("image_views_x0", make_grid(x0, dataloader.batch_size), epoch)
        tb_writer.add_image("image_views_x1", make_grid(x1, dataloader.batch_size), epoch)

    prev_loss = copy.deepcopy(avg_loss)