import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.optim as optim
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from PIL import Image
import os.path as osp
from tqdm import tqdm
import mmcv
from mmengine import Config
from tqdm import tqdm
import mmengine
from mmengine.runner import Runner
from mmseg.utils import register_all_modules
from matplotlib import colors as mcolors
import random
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torchvision.transforms as transforms
import torch
from torch.nn import functional as F
import torchvision.transforms.functional as TF
from torch.optim import lr_scheduler
from pytorch_wavelets import DWTForward
import torch.optim as optim
import time
print("cuda版本为：")
print(torch.version.cuda)
print("cudnn版本为：")
print(torch.backends.cudnn.version())

L1=False
Dice=False

def keep_image_size_open(path, size=(512, 512)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('P', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask
def keep_image_size_open_rgb(path, size=(512, 512)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask

class CustomDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_dir = os.path.join(root_dir, f'img_dir/{split}')
        self.label_dir = os.path.join(root_dir, f'ann_dir/{split}')
        self.images = os.listdir(self.image_dir)
        self.labels = os.listdir(self.label_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = os.path.join(self.image_dir, self.images[idx])
        label_name = os.path.join(self.label_dir, self.labels[idx])

        image = keep_image_size_open_rgb(image_name, size=(512, 512))
        label = keep_image_size_open(label_name, size=(512, 512))
        if self.transform:
            image = self.transform(image)
        # 将标签数据类型转换为LongTensor
        label = torch.from_numpy(np.array(label)).long()
        return {'image': image, 'label': label}

class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace=True),
                                    )
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)

        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class PHAUNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=6, deep_supervision=False):
        super(PHAUNet, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        nb_filter = [32, 64, 128, 256, 512]
        self.nb_filter = nb_filter
        self.conv00_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv0_0 = Down_wt(input_channels, nb_filter[0])
        self.conv1_0 = Down_wt(nb_filter[0] * 2, nb_filter[1])
        self.conv2_0 = Down_wt(nb_filter[1] * 2, nb_filter[2])
        self.conv3_0 = Down_wt(nb_filter[2] * 2, nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3] * 2, nb_filter[4])

        self.conv11_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv12_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv13_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv14_0 = DoubleConv(input_channels, nb_filter[0])

        self.conv22_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv23_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv24_0 = DoubleConv(nb_filter[0], nb_filter[1])

        self.conv33_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv34_0 = DoubleConv(nb_filter[1], nb_filter[2])

        self.conv44_0 = DoubleConv(nb_filter[2], nb_filter[3])


        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])

        self.Att4 = Attention_block(F_g=nb_filter[4], F_l=nb_filter[3], F_int=nb_filter[2])
        self.Att3 = Attention_block(F_g=nb_filter[3], F_l=nb_filter[2], F_int=nb_filter[1])
        self.Att2 = Attention_block(F_g=nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Att1 = Attention_block(F_g=nb_filter[1], F_l=nb_filter[0], F_int=int(nb_filter[0] / 2))

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        image_size = input.shape
        Images = []
        divsize = [2, 4, 8, 16]
        for i in range(len(self.nb_filter) - 1):
            Images.append(TF.resize(input, size=[int(image_size[2] / divsize[i]), int(image_size[3] / divsize[i])]))

        x0_0 = self.conv0_0.forward(input)
        x00_0 = self.conv00_0.forward(input)
        x11_0 = self.conv11_0.forward(Images[0])
        x1_0 = self.conv1_0.forward(torch.cat((x11_0, x0_0), dim=1))
        x12_0 = self.conv12_0.forward(Images[1])
        x22_0 = self.conv22_0.forward(x12_0)
        x2_0 = self.conv2_0.forward(torch.cat((x22_0, x1_0), dim=1))

        x13_0 = self.conv13_0.forward(Images[2])
        x23_0 = self.conv23_0.forward(x13_0)
        x33_0 = self.conv33_0.forward(x23_0)
        x3_0 = self.conv3_0.forward(torch.cat((x33_0, x2_0), dim=1))

        x14_0 = self.conv14_0.forward(Images[3])
        x24_0 = self.conv24_0.forward(x14_0)
        x34_0 = self.conv34_0.forward(x24_0)
        x44_0 = self.conv44_0.forward(x34_0)
        x4_0 = self.conv4_0.forward(torch.cat((x44_0, x3_0), dim=1))

        x3_1 = self.up(x4_0)
        con=torch.cat((x33_0, x2_0), dim=1)
        x3_0 = self.Att4.forward(g=x3_1, x=con)
        x3_1 = self.conv3_1.forward(torch.cat((x3_0, x3_1), dim=1))

        x2_2 = self.up(x3_1)
        con=torch.cat((x22_0, x1_0), dim=1)
        x2_0 = self.Att3.forward(g=x2_2, x=con)
        x2_2 = self.conv2_2.forward(torch.cat((x2_0, x2_2), dim=1))

        x1_3 = self.up(x2_2)
        con = torch.cat((x11_0, x0_0), dim=1)
        x1_0 = self.Att2.forward(g=x1_3, x=con)
        x1_3 = self.conv1_3.forward(torch.cat((x1_0, x1_3), dim=1))

        x0_4 = self.up(x1_3)
        x0_0 = self.Att1.forward(g=x0_4, x=x00_0)
        x0_4 = self.conv0_4.forward(torch.cat((x0_0, x0_4), dim=1))
        output = self.final(x0_4)
        return output
class ALRS:
    """
    refer to
    Bootstrap Generalization Ability from Loss Landscape Perspective
    """
    def __init__(self, optimizer, loss_threshold=0.01, loss_ratio_threshold=0.01, decay_rate=0.97):
        self.optimizer = optimizer
        self.loss_threshold = loss_threshold
        self.decay_rate = decay_rate
        self.loss_ratio_threshold = loss_ratio_threshold

        self.last_loss = 999

    def step(self, loss):
        delta = self.last_loss - loss
        if delta < self.loss_threshold and delta / self.last_loss < self.loss_ratio_threshold:
            for group in self.optimizer.param_groups:
                group["lr"] *= self.decay_rate
                now_lr = group["lr"]
                print(f"now lr = {now_lr}")

        self.last_loss = loss

def dice_coefficient(pred, target, smooth=1e-6):
    pred = torch.argmax(pred, dim=1)
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

def main():
    if __name__ == '__main__':
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)

    weight_path = 'Pre-train the weight file'
    model = PHAUNet()
    pretrained_weights = torch.load(weight_path, map_location=lambda storage, loc: storage)
    pretrained_state_dict = {k: v for k, v in pretrained_weights.items() if k != 'final.weight' and k != 'final.bias'}
    model.load_state_dict(pretrained_state_dict, strict=False)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    alrs = ALRS(optimizer)
    """
    #Custom category weights: {"_background_": 0,"buckling": 1,"crack": 2,"crush": 3,"spall": 3,"steel": 4,"stirrup": 4,"fragment": 5}
        """
    class_weights = torch.tensor([XX, XX, XX, XX, XX, XX])
    class_weights = class_weights.to('cuda:0')
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    # 打印输入图片和标签尺寸信息
    print("size：")
    for batch in train_loader:
        inputs = batch['image']
        labels = batch['label']
        print("img size：", inputs.shape)
        print("label size：", labels.shape)
        break

    unique_labels = set()
    for batch in train_loader:
        labels = batch['label']
        unique_labels.update(np.unique(labels.numpy()))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    epoch_times = []
    num_epochs = 300
    model.train()
    list_train_error = []
    list_test_error = []
    lr_list = []
    error_df = pd.DataFrame(columns=['Epoch', 'Train Error', 'Test Error', 'Learning Rate'])

    with pd.ExcelWriter('epoch_times.xlsx', engine='openpyxl') as writer:
        for epoch in range(num_epochs):
            torch.cuda.empty_cache()
            start_time = time.time()
            for batch in train_loader:
                inputs = batch['image'].to(device)
                labels = batch['label'].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            end_time = time.time()
            epoch_time = end_time - start_time
            print(epoch_time)
            epoch_times.append(epoch_time)

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
            list_train_error.append(loss.item())
            A = loss.item()
            alrs.step(loss.item())

            current_lr = optimizer.param_groups[0]['lr']
            lr_list.append(current_lr)

            if (epoch + 1) % 25 == 0:
                torch.save(model.state_dict(), f'unet_segmentation_model_epoch_{epoch + 1}.pth')

            model.eval()
            valid_loss = 0.0
            for batch in valid_loader:
                inputs = batch['image'].to(device)
                labels = batch['label'].to(device)
                with torch.no_grad():
                    outputs = model(inputs)
                loss = criterion(outputs, labels.long())
                valid_loss += loss.item()

            B = valid_loss / len(valid_loader)
            print(f'Validation Loss: {valid_loss / len(valid_loader)}')
            list_test_error.append(valid_loss / len(valid_loader))

            error_df = pd.concat([error_df, pd.DataFrame(
                {'Epoch': [epoch + 1], 'Train Error': [A], 'Test Error': [B], 'Learning Rate': [current_lr]})],
                                 ignore_index=True)

            error_df.to_excel(writer, sheet_name='Errors and LRs', index=False)

            df = pd.DataFrame({'Epoch': range(1, epoch + 2), 'Time (s)': epoch_times})
            df.to_excel(writer, sheet_name='Times', index=False)
            writer.save()

            model.train()

    x_values = range(1, len(list_train_error) + 1)
    plt.plot(x_values, list_train_error, marker='o')
    plt.title('Errors for train')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'unet-train_error_plot.png'))



    x_values = range(1, len(list_test_error) + 1)
    plt.plot(x_values, list_test_error, marker='o')
    plt.title('Errors for test')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'unet-test_error_plot.png'))


if __name__ == '__main__':
    mp.freeze_support()
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])
    }



    train_dataset = CustomDataset(root_dir=r"X:\XX\XX\Column-dataset",
                                  split='train', transform=data_transforms['train'])
    valid_dataset = CustomDataset(root_dir=r"X:\XX\XX\Column-dataset",
                                  split='val', transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

    main()
