import torch.nn as nn
import torch
from torch.nn.utils import spectral_norm


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class input_gate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel = channels
        self.t1_conv = nn.Conv2d(3*channels, channels, 1, 1, 0)
        self.flair_conv = nn.Conv2d(3*channels, channels, 1, 1, 0)
        self.t2_conv = nn.Conv2d(3*channels, channels, 1, 1, 0)

    def forward(self, x):
        t1, t2, flair = x[:, :self.channel], x[:, self.channel:2*self.channel], x[:, 2*self.channel:]
        t1_c = self.t1_conv(x)
        t2_c = self.t2_conv(x)
        flair_c = self.flair_conv(x)
        weight = torch.softmax(torch.cat([t1_c, t2_c, flair_c], dim=1), dim=1)
        t1_weight, t2_weight, flair_weight = weight[:, :self.channel], weight[:, self.channel:2*self.channel], weight[:, 2*self.channel:]
        out = t1_weight*t1 + t2_weight*t2 + flair_weight*flair
        return out
    

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout=0.1):
        super().__init__()
        self.d = dropout
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # self.relu = nn.Tanh()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, middle_channels, 3, padding=1))
        self.bn1 = nn.InstanceNorm2d(middle_channels)
        self.conv2 = spectral_norm(nn.Conv2d(middle_channels, out_channels, 3, padding=1))
        self.bn2 = nn.InstanceNorm2d(out_channels)
        if dropout:
            self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.d:
            out = self.drop(out)

        return out
    
# UNet++
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        self.input_gate = input_gate(out_channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
		
        # 1
        self.conv0_0 = VGGBlock(in_channels, nb_filter[0], nb_filter[0], dropout=False)
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
		
       	# 2
        self.conv0_1 = VGGBlock(nb_filter[0] * 1 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] * 1 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] * 1 + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] * 1 + nb_filter[4], nb_filter[3], nb_filter[3])
		
        # 3
        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])
		
        # 4
        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        # 5
        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])
		
        self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        self.end = nn.Sigmoid()
        # self.end = nn.Tanh()

    def forward(self, x):
        x = self.input_gate(x)
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.end(self.final(x0_4))
        # output = self.final(x0_4)
        return output



##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def same_conv(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=1, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *same_conv(256, 256),
            *discriminator_block(256, 512),
            *same_conv(512, 512),
            *discriminator_block(512, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )


    def forward(self, img_A):
        return self.model(img_A)    
