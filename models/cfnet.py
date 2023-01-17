from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
import math

class sparse_convolution(nn.Module):
    '''
        implement the sparse convolution defined in the paper "Sparse Invariant CNNs"
    '''
    def __init__(self):
        super(sparse_convolution, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=11, padding=5, stride=1, bias=False), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=7, padding=3, stride=1, bias=False), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=5, padding=2, stride=1, bias=False), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(16, 1, kernel_size=1, padding=0, stride=1, bias=False), nn.ReLU(inplace=True))

        self.maxpool1 = nn.MaxPool2d(11, stride = 1, padding = 5)
        self.maxpool2 = nn.MaxPool2d(7, stride = 1, padding = 3)
        self.maxpool3 = nn.MaxPool2d(5, stride = 1, padding = 2)
        self.maxpool4 = nn.MaxPool2d(3, stride = 1, padding = 1)
        self.maxpool5 = nn.MaxPool2d(3, stride = 1, padding = 1)
        self.maxpool6 = nn.MaxPool2d(1, stride = 1)

        self.small_value = 0.001

    def forward(self, sparse, mask):
        device = mask.get_device()
        mask_f = mask.to(torch.float)
        sparse_mask = sparse * mask_f
        feature11 = self.conv1(sparse_mask)
        mask11 = self.maxpool1(mask_f)
        mask11_norm = 1 / (F.conv2d(mask_f, torch.ones(1,mask_f.size()[1],11, 11, device = device), padding=5)  + self.small_value)
        #print(mask.size())
        #print(mask11.size())
        #print(feature11.size())
        #print(mask11_norm.size())
        feature11_norm = feature11 * mask11_norm
        #print(feature11_norm.size())
        #print(mask11.size())
        sparse_mask7 = feature11_norm * mask11
        #print(sparse_mask7.size())
        feature7 = self.conv2(sparse_mask7)
        mask7 = self.maxpool2(mask11)
        mask7_norm = 1 / (F.conv2d(mask11, torch.ones(1,mask11.size()[1],7, 7, device = device), padding=3) + self.small_value)
        feature7_norm = feature7 * mask7_norm

        sparse_mask5 = feature7_norm * mask7
        feature5 = self.conv3(sparse_mask5)
        mask5 = self.maxpool3(mask7)
        mask5_norm = 1 / (F.conv2d(mask7, torch.ones(1,mask7.size()[1],5, 5,device = device), padding=2)  + self.small_value)
        feature5_norm = feature5 * mask5_norm

        sparse_mask3_1 = feature5_norm * mask5
        feature3_1 = self.conv4(sparse_mask3_1)
        mask3_1 = self.maxpool4(mask5)
        mask3_1_norm = 1 / (F.conv2d(mask5, torch.ones(1,mask5.size()[1],3, 3,device = device), padding=1) + self.small_value)
        feature3_1_norm = feature3_1 * mask3_1_norm

        sparse_mask3_2 = feature3_1_norm * mask3_1
        feature3_2 = self.conv5(sparse_mask3_2)
        mask3_2 = self.maxpool5(mask3_1)
        mask3_2_norm = 1 / (F.conv2d(mask3_1, torch.ones(1,mask3_1.size()[1],3, 3,device = device), padding=1) + self.small_value)
        feature3_2_norm = feature3_2 * mask3_2_norm

        sparse_mask1 = feature3_2_norm * mask3_2
        feature1 = self.conv6(sparse_mask1)
        mask1_norm = 1 / (F.conv2d(mask3_2, torch.ones(1,mask3_2.size()[1],1, 1,device = device), padding=0) + self.small_value)
        feature1_norm = feature1 * mask1_norm

        return feature1_norm
        

class feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        # self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 1, 1, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 1, 2, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 192, 1, 2, 1, 1)
        self.layer5 = self._make_layer(BasicBlock, 256, 1, 2, 1, 1)
        self.layer6 = self._make_layer(BasicBlock, 512, 1, 2, 1, 1)
        self.pyramid_pooling = pyramidPooling(512, None, fusion_mode='sum', model_name='icnet')
        self.upconv6 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(512, 256, 3, 1, 1, 1),
                                     nn.ReLU(inplace=True))
        self.iconv5 = nn.Sequential(convbn(512, 256, 3, 1, 1, 1),
                                    nn.ReLU(inplace=True))
        self.upconv5 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(256, 192, 3, 1, 1, 1),
                                     nn.ReLU(inplace=True))
        self.iconv4 = nn.Sequential(convbn(384, 192, 3, 1, 1, 1),
                                    nn.ReLU(inplace=True))
        self.upconv4 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(192, 128, 3, 1, 1, 1),
                                     nn.ReLU(inplace=True))
        self.iconv3 = nn.Sequential(convbn(256, 128, 3, 1, 1, 1),
                                    nn.ReLU(inplace=True))
        self.upconv3 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(128, 64, 3, 1, 1, 1),
                                     nn.ReLU(inplace=True))
        self.iconv2 = nn.Sequential(convbn(128, 64, 3, 1, 1, 1),
                                    nn.ReLU(inplace=True))
        # self.upconv2 = nn.Sequential(nn.Upsample(scale_factor=2),
        #                              convbn(64, 32, 3, 1, 1, 1),
        #                              nn.ReLU(inplace=True))

        # self.gw1 = nn.Sequential(convbn(32, 40, 3, 1, 1, 1),
        #                          nn.ReLU(inplace=True),
        #                          nn.Conv2d(40, 40, kernel_size=1, padding=0, stride=1,
        #                                    bias=False))

        self.gw2 = nn.Sequential(convbn(64, 80, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(80, 80, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

        self.gw3 = nn.Sequential(convbn(128, 160, 3, 1, 1, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(160, 160, kernel_size=1, padding=0, stride=1,
                                           bias=False))

        self.gw4 = nn.Sequential(convbn(192, 160, 3, 1, 1, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(160, 160, kernel_size=1, padding=0, stride=1,
                                           bias=False))

        self.gw5 = nn.Sequential(convbn(256, 320, 3, 1, 1, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(320, 320, kernel_size=1, padding=0, stride=1,
                                           bias=False))

        self.gw6 = nn.Sequential(convbn(512, 320, 3, 1, 1, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(320, 320, kernel_size=1, padding=0, stride=1,
                                           bias=False))

        if self.concat_feature:
            # self.concat1 = nn.Sequential(convbn(32, 16, 3, 1, 1, 1),
            #                              nn.ReLU(inplace=True),
            #                              nn.Conv2d(16, concat_feature_channel // 4, kernel_size=1, padding=0, stride=1,
            #                                        bias=False))

            self.concat2 = nn.Sequential(convbn(64, 32, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(32, concat_feature_channel // 2, kernel_size=1, padding=0, stride=1,
                                                    bias=False))
            self.concat3 = nn.Sequential(convbn(128, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

            self.concat4 = nn.Sequential(convbn(192, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

            self.concat5 = nn.Sequential(convbn(256, 128, 3, 1, 1, 1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                   bias=False))

            self.concat6 = nn.Sequential(convbn(512, 128, 3, 1, 1, 1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                   bias=False))





    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        # x = self.layer1(x)
        l2 = self.layer2(x)     #1/2
        l3 = self.layer3(l2)    #1/4
        l4 = self.layer4(l3)    #1/8
        l5 = self.layer5(l4)    #1/16
        l6 = self.layer6(l5)    #1/32
        l6 = self.pyramid_pooling(l6)

        concat5 = torch.cat((l5, self.upconv6(l6)), dim=1)
        decov_5 = self.iconv5(concat5)
        concat4 = torch.cat((l4, self.upconv5(decov_5)), dim=1)
        # concat4 = torch.cat((l4, self.upconv5(l5)), dim=1)
        decov_4 = self.iconv4(concat4)
        concat3 = torch.cat((l3, self.upconv4(decov_4)), dim=1)
        decov_3 = self.iconv3(concat3)
        concat2 = torch.cat((l2, self.upconv3(decov_3)), dim=1)
        decov_2 = self.iconv2(concat2)
        # decov_1 = self.upconv2(decov_2)


        # gw1 = self.gw1(decov_1)
        gw2 = self.gw2(decov_2)
        gw3 = self.gw3(decov_3)
        gw4 = self.gw4(decov_4)
        gw5 = self.gw5(decov_5)
        gw6 = self.gw6(l6)

        if not self.concat_feature:
            return {"gw2": gw2, "gw3": gw3, "gw4": gw4}
        else:
            # concat_feature1 = self.concat1(decov_1)
            concat_feature2 = self.concat2(decov_2)
            concat_feature3 = self.concat3(decov_3)
            concat_feature4 = self.concat4(decov_4)
            concat_feature5 = self.concat5(decov_5)
            concat_feature6 = self.concat6(l6)
            return {"gw2": gw2, "gw3": gw3, "gw4": gw4, "gw5": gw5, "gw6": gw6,
                    "concat_feature2": concat_feature2, "concat_feature3": concat_feature3, "concat_feature4": concat_feature4,
                    "concat_feature5": concat_feature5, "concat_feature6": concat_feature6}

class sparse_feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(sparse_feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(1, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        # self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 1, 1, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 1, 2, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 192, 1, 2, 1, 1)
        self.layer5 = self._make_layer(BasicBlock, 256, 1, 2, 1, 1)
        self.layer6 = self._make_layer(BasicBlock, 512, 1, 2, 1, 1)
        self.pyramid_pooling = pyramidPooling(512, None, fusion_mode='sum', model_name='icnet')
        self.upconv6 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(512, 256, 3, 1, 1, 1),
                                     nn.ReLU(inplace=True))
        self.iconv5 = nn.Sequential(convbn(512, 256, 3, 1, 1, 1),
                                    nn.ReLU(inplace=True))
        self.upconv5 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(256, 192, 3, 1, 1, 1),
                                     nn.ReLU(inplace=True))
        self.iconv4 = nn.Sequential(convbn(384, 192, 3, 1, 1, 1),
                                    nn.ReLU(inplace=True))
        self.upconv4 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(192, 128, 3, 1, 1, 1),
                                     nn.ReLU(inplace=True))
        self.iconv3 = nn.Sequential(convbn(256, 128, 3, 1, 1, 1),
                                    nn.ReLU(inplace=True))
        self.upconv3 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(128, 64, 3, 1, 1, 1),
                                     nn.ReLU(inplace=True))
        self.iconv2 = nn.Sequential(convbn(128, 64, 3, 1, 1, 1),
                                    nn.ReLU(inplace=True))

        self.concat2 = nn.Sequential(convbn(64, 32, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(32, concat_feature_channel // 2, kernel_size=1, padding=0, stride=1,
                                                bias=False))
        self.concat3 = nn.Sequential(convbn(128, 128, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                bias=False))

        self.concat4 = nn.Sequential(convbn(192, 128, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                bias=False))

        self.concat5 = nn.Sequential(convbn(256, 128, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                bias=False))

        self.concat6 = nn.Sequential(convbn(512, 128, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        # x = self.layer1(x)
        l2 = self.layer2(x)     #1/2
        l3 = self.layer3(l2)    #1/4
        l4 = self.layer4(l3)    #1/8
        l5 = self.layer5(l4)    #1/16
        l6 = self.layer6(l5)    #1/32
        l6 = self.pyramid_pooling(l6)

        concat5 = torch.cat((l5, self.upconv6(l6)), dim=1)
        decov_5 = self.iconv5(concat5)
        concat4 = torch.cat((l4, self.upconv5(decov_5)), dim=1)
        # concat4 = torch.cat((l4, self.upconv5(l5)), dim=1)
        decov_4 = self.iconv4(concat4)
        concat3 = torch.cat((l3, self.upconv4(decov_4)), dim=1)
        decov_3 = self.iconv3(concat3)
        concat2 = torch.cat((l2, self.upconv3(decov_3)), dim=1)
        decov_2 = self.iconv2(concat2)
        
        # concat_feature1 = self.concat1(decov_1)
        concat_feature2 = torch.unsqueeze(self.concat2(decov_2),dim=2)
        concat_feature3 = torch.unsqueeze(self.concat3(decov_3),dim=2)
        concat_feature4 = torch.unsqueeze(self.concat4(decov_4),dim=2)
        concat_feature5 = torch.unsqueeze(self.concat5(decov_5),dim=2)
        concat_feature6 = torch.unsqueeze(self.concat6(l6),dim=2)
        return {"s2": concat_feature2, "s3": concat_feature3, "s4": concat_feature4,
                "s5": concat_feature5, "s6": concat_feature6}

class hourglassup(nn.Module):
    def __init__(self, in_channels):
        super(hourglassup, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels * 2, kernel_size=3, stride=2,
                                   padding=1, bias=False)

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Conv3d(in_channels * 2, in_channels * 4, kernel_size=3, stride=2,
                               padding=1, bias=False)

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv8 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.combine1 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.combine2 = nn.Sequential(convbn_3d(in_channels * 6, in_channels * 4, 3, 1, 1),
                                      nn.ReLU(inplace=True))
        self.combine3 = nn.Sequential(convbn_3d(in_channels * 6, in_channels * 4, 3, 1, 1),
                                      nn.ReLU(inplace=True))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)
        self.redir3 = convbn_3d(in_channels * 4, in_channels * 4, kernel_size=1, stride=1, pad=0)


    def forward(self, x, feature4, feature5):
        conv1 = self.conv1(x)          #1/8
        conv1 = torch.cat((conv1, feature4), dim=1)   #1/8
        conv1 = self.combine1(conv1)   #1/8
        conv2 = self.conv2(conv1)      #1/8

        conv3 = self.conv3(conv2)      #1/16
        conv3 = torch.cat((conv3, feature5), dim=1)   #1/16
        conv3 = self.combine2(conv3)   #1/16
        conv4 = self.conv4(conv3)      #1/16

        # conv8 = FMish(self.conv8(conv4) + self.redir2(conv2))
        # conv9 = FMish(self.conv9(conv8) + self.redir1(x))

        conv8 = F.relu(self.conv8(conv4) + self.redir2(conv2), inplace=True)
        conv9 = F.relu(self.conv9(conv8) + self.redir1(x), inplace=True)



        return conv9

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        # conv5 = FMish(self.conv5(conv4) + self.redir2(conv2))
        # conv6 = FMish(self.conv6(conv5) + self.redir1(x))

        return conv6

class cfnet(nn.Module):
    def __init__(self, maxdisp, use_concat_volume=False):
        super(cfnet, self).__init__()
        self.maxdisp = maxdisp
        self.use_concat_volume = use_concat_volume
        self.v_scale_s1 = 1
        self.v_scale_s2 = 2
        self.v_scale_s3 = 3
        self.sample_count_s1 = 6
        self.sample_count_s2 = 10
        self.sample_count_s3 = 14
        self.num_groups = 40
        self.sparse_feature_on = True
        self.uniform_sampler = UniformSampler()
        self.spatial_transformer = SpatialTransformer()

        self.sparse_convolution = sparse_convolution()
        if self.sparse_feature_on:
            self.sparse_channel = 12
            #self.sparse_channel = 1
        else:
            self.sparse_channel = 0

        self.sparse_feature_extraction = sparse_feature_extraction(concat_feature=True, concat_feature_channel=12)

        if self.use_concat_volume:
            self.concat_channels = 12
            self.feature_extraction = feature_extraction(concat_feature=True,
                                                         concat_feature_channel=self.concat_channels)
        else:
            self.concat_channels = 0
            self.feature_extraction = feature_extraction(concat_feature=False)

        self.dres0 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels*2+self.sparse_channel, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))


        self.dres0_5 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels*2+self.sparse_channel, 64, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(64, 64, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1_5 = nn.Sequential(convbn_3d(64, 64, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(64, 64, 3, 1, 1))

        self.dres0_6 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels*2+self.sparse_channel, 64, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(64, 64, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1_6 = nn.Sequential(convbn_3d(64, 64, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(64, 64, 3, 1, 1))

        self.combine1 = hourglassup(32)

        # self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        # self.dres4 = hourglass(32)

        self.confidence0_s3 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels*2 + 1+self.sparse_channel , 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.confidence1_s3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.confidence2_s3 = hourglass(32)

        self.confidence3_s3 = hourglass(32)

        self.confidence0_s2 = nn.Sequential(convbn_3d(self.num_groups//2 + self.concat_channels + 1+self.sparse_channel//2, 16, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(16, 16, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.confidence1_s2 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(16, 16, 3, 1, 1))

        self.confidence2_s2 = hourglass(16)

        self.confidence3_s2 = hourglass(16)


        # self.confidence0_s1 = nn.Sequential(convbn_3d(self.num_groups // 4 + self.concat_channels // 2 + 1, 16, 3, 1, 1),
        #                                     nn.ReLU(inplace=True),
        #                                     convbn_3d(16, 16, 3, 1, 1),
        #                                     nn.ReLU(inplace=True))
        #
        # self.confidence1_s1 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
        #                                     nn.ReLU(inplace=True),
        #                                     convbn_3d(16, 16, 3, 1, 1))
        #
        # self.confidence2_s1 = hourglass(16)

        # self.confidence3 = hourglass(32)
        #
        # self.confidence4 = hourglass(32)

        self.confidence_classif0_s3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                                    nn.ReLU(inplace=True),
                                                    nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.confidence_classif1_s3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.confidence_classifmid_s3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                                    nn.ReLU(inplace=True),
                                                    nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.confidence_classif0_s2 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                                    nn.ReLU(inplace=True),
                                                    nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))


        self.confidence_classif1_s2 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                                    nn.ReLU(inplace=True),
                                                    nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.confidence_classifmid_s2 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                                    nn.ReLU(inplace=True),
                                                    nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))

        # self.confidence_classif1_s1 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
        #                                             nn.ReLU(inplace=True),
        #                                             nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.gamma_s3 = nn.Parameter(torch.zeros(1))
        self.beta_s3 = nn.Parameter(torch.zeros(1))
        self.gamma_s2 = nn.Parameter(torch.zeros(1))
        self.beta_s2 = nn.Parameter(torch.zeros(1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
                # m.bias.data.zero_()

    def generate_search_range(self, sample_count, input_min_disparity, input_max_disparity, scale):
        """
        Description:    Generates the disparity search range.

        Returns:
            :min_disparity: Lower bound of disparity search range
            :max_disparity: Upper bound of disaprity search range.
        """
        # have been modified, origin: max=self.maxdisp
        min_disparity = torch.clamp(input_min_disparity - torch.clamp((
                sample_count - input_max_disparity + input_min_disparity), min=0) / 2.0, min=0, max=self.maxdisp // (2**scale) - 1)
        max_disparity = torch.clamp(input_max_disparity + torch.clamp(
                sample_count - input_max_disparity + input_min_disparity, min=0) / 2.0, min=0, max=self.maxdisp // (2**scale) - 1)

        return min_disparity, max_disparity

    def generate_disparity_samples(self, min_disparity, max_disparity, sample_count=12):
        """
        Description:    Generates "sample_count" number of disparity samples from the
                                                            search range (min_disparity, max_disparity)
                        Samples are generated by uniform sampler

        Args:
            :min_disparity: LowerBound of the disaprity search range.
            :max_disparity: UpperBound of the disparity search range.
            :sample_count: Number of samples to be generated from the input search range.

        Returns:
            :disparity_samples:
        """
        disparity_samples = self.uniform_sampler(min_disparity, max_disparity, sample_count)

        disparity_samples = torch.cat((torch.floor(min_disparity), disparity_samples, torch.ceil(max_disparity)),
                                      dim=1).long()                   # disparity level = sample_count + 2
        return disparity_samples

    def cost_volume_generator(self, left_input, right_input, disparity_samples, model = 'concat', num_groups = 40):
        """
        Description: Generates cost-volume using left image features, disaprity samples
                                                            and warped right image features.
        Args:
            :left_input: Left Image fetaures
            :right_input: Right Image features
            :disparity_samples: Disaprity samples
            :model : concat or group correlation

        Returns:
            :cost_volume:
            :disaprity_samples:
        """

        right_feature_map, left_feature_map = self.spatial_transformer(left_input,
                                                                       right_input, disparity_samples)
        disparity_samples = disparity_samples.unsqueeze(1).float()
        if model == 'concat':
             cost_volume = torch.cat((left_feature_map, right_feature_map), dim=1)
        else:
             cost_volume = groupwise_correlation_4D(left_feature_map, right_feature_map, num_groups)

        return cost_volume, disparity_samples
    
    def sparse_downsample(self, sparse, sparse_mask):
        '''
           try the downsample by sample the sparse result by fixed step 
        '''
        sparse_out = {}
        sparse_out["sparse2"] = sparse[:,:,0:-1:2,0:-1:2]/2
        sparse_out["sparse3"] = sparse[:,:,0:-1:4,0:-1:4]/4
        sparse_out["sparse4"] = sparse[:,:,0:-1:8,0:-1:8]/8
        sparse_out["sparse5"] = sparse[:,:,0:-1:16,0:-1:16]/16
        sparse_out["sparse6"] = sparse[:,:,0:-1:32,0:-1:32]/32

        sparse_mask_out = {}
        sparse_mask_out["sparse_mask2"] = sparse_mask[:,:,0:-1:2,0:-1:2]
        sparse_mask_out["sparse_mask3"] = sparse_mask[:,:,0:-1:4,0:-1:4]
        sparse_mask_out["sparse_mask4"] = sparse_mask[:,:,0:-1:8,0:-1:8]
        sparse_mask_out["sparse_mask5"] = sparse_mask[:,:,0:-1:16,0:-1:16]
        sparse_mask_out["sparse_mask6"] = sparse_mask[:,:,0:-1:32,0:-1:32]

        return sparse_out, sparse_mask_out
    
    def sparse_expand(self, sparse, sparse_mask, left_img):
        '''
            RGB guieded sparse expand
            input: 
                sparse_mask: N*1*W*H
                sparse:      N*1*W*H
                left_img:    N*3*W*H
            output:
                sparse_mask_exp: N*1*W*H
                sparse_exp:      N*1*W*H
        '''
        threshold = 10
        img_center = left_img[:,:,1:-1,1:-1]
        img_up_left = left_img[:,:,0:-2,0:-2]
        img_up = left_img[:,:,1:-1,0:-2]
        img_up_right = left_img[:,:,2:,0:-2]
        img_left = left_img[:,:,0:-2,1:-1]
        img_right = left_img[:,:,2:,1:-1]
        img_down_left = left_img[:,:,0:-2,2:]
        img_down = left_img[:,:,1:-1,2:]
        img_down_right = left_img[:,:,2:,2:]
        
        mask_up_left = torch.unsqueeze(torch.sum(torch.abs(img_up_left - img_center),1) < threshold, dim=1) & (sparse_mask[:,:,0:-2,0:-2] < 1) & (sparse_mask[:,:,1:-1,1:-1] > 0)
        sparse[:,:,0:-2,0:-2][mask_up_left] = sparse[:,:,1:-1,1:-1][mask_up_left]
        sparse_mask[:,:,0:-2,0:-2][mask_up_left] = sparse_mask[:,:,1:-1,1:-1][mask_up_left]

        mask_up = torch.unsqueeze(torch.sum(torch.abs(img_up - img_center),1) < threshold, dim=1) & (sparse_mask[:,:,1:-1,0:-2] < 1)& (sparse_mask[:,:,1:-1,1:-1] > 0)
        sparse[:,:,1:-1,0:-2][mask_up] = sparse[:,:,1:-1,1:-1][mask_up]
        sparse_mask[:,:,1:-1,0:-2][mask_up] = sparse_mask[:,:,1:-1,1:-1][mask_up]

        mask_up_right = torch.unsqueeze(torch.sum(torch.abs(img_up_right - img_center),1) < threshold, dim=1) & (sparse_mask[:,:,2:,0:-2] < 1)& (sparse_mask[:,:,1:-1,1:-1] > 0)
        sparse[:,:,2:,0:-2][mask_up_right] = sparse[:,:,1:-1,1:-1][mask_up_right]
        sparse_mask[:,:,2:,0:-2][mask_up_right] = sparse_mask[:,:,1:-1,1:-1][mask_up_right]

        mask_left = torch.unsqueeze(torch.sum(torch.abs(img_left - img_center),1) < threshold, dim=1) & (sparse_mask[:,:,0:-2,1:-1] < 1)& (sparse_mask[:,:,1:-1,1:-1] > 0)
        sparse[:,:,0:-2,1:-1][mask_left] = sparse[:,:,1:-1,1:-1][mask_left]
        sparse_mask[:,:,0:-2,1:-1][mask_left] = sparse_mask[:,:,1:-1,1:-1][mask_left]

        mask_right = torch.unsqueeze(torch.sum(torch.abs(img_right - img_center),1) < threshold, dim=1) & (sparse_mask[:,:,2:,1:-1] < 1)& (sparse_mask[:,:,1:-1,1:-1] > 0)
        sparse[:,:,2:,1:-1][mask_right] = sparse[:,:,1:-1,1:-1][mask_right]
        sparse_mask[:,:,2:,1:-1][mask_right] = sparse_mask[:,:,1:-1,1:-1][mask_right]

        mask_down_left = torch.unsqueeze(torch.sum(torch.abs(img_down_left - img_center),1) < threshold, dim=1) & (sparse_mask[:,:,0:-2,2:] < 1)& (sparse_mask[:,:,1:-1,1:-1] > 0)
        sparse[:,:,0:-2,2:][mask_down_left] = sparse[:,:,1:-1,1:-1][mask_down_left]
        sparse_mask[:,:,0:-2,2:][mask_down_left] = sparse_mask[:,:,1:-1,1:-1][mask_down_left]

        mask_down = torch.unsqueeze(torch.sum(torch.abs(img_down - img_center),1) < threshold, dim=1) & (sparse_mask[:,:,1:-1,2:] < 1)& (sparse_mask[:,:,1:-1,1:-1] > 0)
        sparse[:,:,1:-1,2:][mask_down] = sparse[:,:,1:-1,1:-1][mask_down]
        sparse_mask[:,:,1:-1,2:][mask_down] = sparse_mask[:,:,1:-1,1:-1][mask_down]

        mask_down_right = torch.unsqueeze(torch.sum(torch.abs(img_down_right - img_center),1) < threshold, dim=1) & (sparse_mask[:,:,2:,2:] < 1)& (sparse_mask[:,:,1:-1,1:-1] > 0)
        sparse[:,:,2:,2:][mask_down_right] = sparse[:,:,1:-1,1:-1][mask_down_right]
        sparse_mask[:,:,2:,2:][mask_down_right] = sparse_mask[:,:,1:-1,1:-1][mask_down_right]

        return sparse, sparse_mask

    def sparse_expand_round(self, sparse, sparse_mask, left_img):
        '''
            RGB guieded sparse expand
            input: 
                sparse_mask: N*1*W*H
                sparse:      N*1*W*H
                left_img:    N*3*W*H
            output:
                sparse_mask_exp: N*1*W*H
                sparse_exp:      N*1*W*H
        '''
        threshold = 10
        img_center = left_img[:,:,1:-1,1:-1]
        img_up_left = left_img[:,:,0:-2,0:-2]
        img_up = left_img[:,:,1:-1,0:-2]
        img_up_right = left_img[:,:,2:,0:-2]
        img_left = left_img[:,:,0:-2,1:-1]
        img_right = left_img[:,:,2:,1:-1]
        img_down_left = left_img[:,:,0:-2,2:]
        img_down = left_img[:,:,1:-1,2:]
        img_down_right = left_img[:,:,2:,2:]
        
        mask_up_left = torch.unsqueeze(torch.sum(torch.abs(img_up_left - img_center),1) < threshold, dim=1) & (sparse_mask[:,:,0:-2,0:-2] < 1) & (sparse_mask[:,:,1:-1,1:-1] > 0)
        
        mask_up = torch.unsqueeze(torch.sum(torch.abs(img_up - img_center),1) < threshold, dim=1) & (sparse_mask[:,:,1:-1,0:-2] < 1)& (sparse_mask[:,:,1:-1,1:-1] > 0)
        
        mask_up_right = torch.unsqueeze(torch.sum(torch.abs(img_up_right - img_center),1) < threshold, dim=1) & (sparse_mask[:,:,2:,0:-2] < 1)& (sparse_mask[:,:,1:-1,1:-1] > 0)
        
        mask_left = torch.unsqueeze(torch.sum(torch.abs(img_left - img_center),1) < threshold, dim=1) & (sparse_mask[:,:,0:-2,1:-1] < 1)& (sparse_mask[:,:,1:-1,1:-1] > 0)
        
        mask_right = torch.unsqueeze(torch.sum(torch.abs(img_right - img_center),1) < threshold, dim=1) & (sparse_mask[:,:,2:,1:-1] < 1)& (sparse_mask[:,:,1:-1,1:-1] > 0)
        
        mask_down_left = torch.unsqueeze(torch.sum(torch.abs(img_down_left - img_center),1) < threshold, dim=1) & (sparse_mask[:,:,0:-2,2:] < 1)& (sparse_mask[:,:,1:-1,1:-1] > 0)
        
        mask_down = torch.unsqueeze(torch.sum(torch.abs(img_down - img_center),1) < threshold, dim=1) & (sparse_mask[:,:,1:-1,2:] < 1)& (sparse_mask[:,:,1:-1,1:-1] > 0)
        
        mask_down_right = torch.unsqueeze(torch.sum(torch.abs(img_down_right - img_center),1) < threshold, dim=1) & (sparse_mask[:,:,2:,2:] < 1)& (sparse_mask[:,:,1:-1,1:-1] > 0)
        

        sparse_mask1 = torch.zeros_like(sparse_mask)
        sparse1 = torch.zeros_like(sparse)

        sparse1[:,:,1:-1,2:][mask_down] = sparse[:,:,1:-1,1:-1][mask_down]
        sparse_mask1[:,:,1:-1,2:][mask_down] = sparse_mask[:,:,1:-1,1:-1][mask_down]

        sparse1[:,:,0:-2,2:][mask_down_left] = sparse[:,:,1:-1,1:-1][mask_down_left]
        sparse_mask1[:,:,0:-2,2:][mask_down_left] = sparse_mask[:,:,1:-1,1:-1][mask_down_left]

        sparse1[:,:,2:,1:-1][mask_right] = sparse[:,:,1:-1,1:-1][mask_right]
        sparse_mask1[:,:,2:,1:-1][mask_right] = sparse_mask[:,:,1:-1,1:-1][mask_right]

        sparse1[:,:,0:-2,1:-1][mask_left] = sparse[:,:,1:-1,1:-1][mask_left]
        sparse_mask1[:,:,0:-2,1:-1][mask_left] = sparse_mask[:,:,1:-1,1:-1][mask_left]

        sparse1[:,:,2:,0:-2][mask_up_right] = sparse[:,:,1:-1,1:-1][mask_up_right]
        sparse_mask1[:,:,2:,0:-2][mask_up_right] = sparse_mask[:,:,1:-1,1:-1][mask_up_right]

        sparse1[:,:,1:-1,0:-2][mask_up] = sparse[:,:,1:-1,1:-1][mask_up]
        sparse_mask1[:,:,1:-1,0:-2][mask_up] = sparse_mask[:,:,1:-1,1:-1][mask_up]

        sparse1[:,:,0:-2,0:-2][mask_up_left] = sparse[:,:,1:-1,1:-1][mask_up_left]
        sparse_mask1[:,:,0:-2,0:-2][mask_up_left] = sparse_mask[:,:,1:-1,1:-1][mask_up_left]

        sparse1[:,:,2:,2:][mask_down_right] = sparse[:,:,1:-1,1:-1][mask_down_right]
        sparse_mask1[:,:,2:,2:][mask_down_right] = sparse_mask[:,:,1:-1,1:-1][mask_down_right]
        
        mask_up_left = torch.unsqueeze(torch.sum(torch.abs(img_up_left - img_center),1) < threshold, dim=1)& (sparse_mask[:,:,0:-2,0:-2] < 1) & (sparse_mask1[:,:,0:-2,0:-2] < 1) & (sparse_mask1[:,:,1:-1,1:-1] > 0)
        
        mask_up = torch.unsqueeze(torch.sum(torch.abs(img_up - img_center),1) < threshold, dim=1)& (sparse_mask[:,:,1:-1,0:-2] < 1) & (sparse_mask1[:,:,1:-1,0:-2] < 1)& (sparse_mask1[:,:,1:-1,1:-1] > 0)
        
        mask_up_right = torch.unsqueeze(torch.sum(torch.abs(img_up_right - img_center),1) < threshold, dim=1)& (sparse_mask[:,:,2:,0:-2] < 1) & (sparse_mask1[:,:,2:,0:-2] < 1)& (sparse_mask1[:,:,1:-1,1:-1] > 0)
        
        mask_left = torch.unsqueeze(torch.sum(torch.abs(img_left - img_center),1) < threshold, dim=1) & (sparse_mask[:,:,0:-2,1:-1] < 1)& (sparse_mask1[:,:,0:-2,1:-1] < 1)& (sparse_mask1[:,:,1:-1,1:-1] > 0)
        
        mask_right = torch.unsqueeze(torch.sum(torch.abs(img_right - img_center),1) < threshold, dim=1)& (sparse_mask[:,:,2:,1:-1] < 1) & (sparse_mask1[:,:,2:,1:-1] < 1)& (sparse_mask1[:,:,1:-1,1:-1] > 0)
        
        mask_down_left = torch.unsqueeze(torch.sum(torch.abs(img_down_left - img_center),1) < threshold, dim=1)& (sparse_mask[:,:,0:-2,2:] < 1) & (sparse_mask1[:,:,0:-2,2:] < 1)& (sparse_mask1[:,:,1:-1,1:-1] > 0)
        
        mask_down = torch.unsqueeze(torch.sum(torch.abs(img_down - img_center),1) < threshold, dim=1)& (sparse_mask[:,:,1:-1,2:] < 1) & (sparse_mask1[:,:,1:-1,2:] < 1)& (sparse_mask1[:,:,1:-1,1:-1] > 0)
        
        mask_down_right = torch.unsqueeze(torch.sum(torch.abs(img_down_right - img_center),1) < threshold, dim=1)& (sparse_mask[:,:,2:,2:] < 1) & (sparse_mask1[:,:,2:,2:] < 1)& (sparse_mask1[:,:,1:-1,1:-1] > 0)

        sparse1[:,:,1:-1,2:][mask_down] = sparse1[:,:,1:-1,1:-1][mask_down]
        sparse_mask1[:,:,1:-1,2:][mask_down] = sparse_mask1[:,:,1:-1,1:-1][mask_down]

        sparse1[:,:,0:-2,2:][mask_down_left] = sparse1[:,:,1:-1,1:-1][mask_down_left]
        sparse_mask1[:,:,0:-2,2:][mask_down_left] = sparse_mask1[:,:,1:-1,1:-1][mask_down_left]

        sparse1[:,:,2:,1:-1][mask_right] = sparse1[:,:,1:-1,1:-1][mask_right]
        sparse_mask1[:,:,2:,1:-1][mask_right] = sparse_mask1[:,:,1:-1,1:-1][mask_right]

        sparse1[:,:,0:-2,1:-1][mask_left] = sparse1[:,:,1:-1,1:-1][mask_left]
        sparse_mask1[:,:,0:-2,1:-1][mask_left] = sparse_mask1[:,:,1:-1,1:-1][mask_left]

        sparse1[:,:,2:,0:-2][mask_up_right] = sparse1[:,:,1:-1,1:-1][mask_up_right]
        sparse_mask1[:,:,2:,0:-2][mask_up_right] = sparse_mask1[:,:,1:-1,1:-1][mask_up_right]

        sparse1[:,:,1:-1,0:-2][mask_up] = sparse1[:,:,1:-1,1:-1][mask_up]
        sparse_mask1[:,:,1:-1,0:-2][mask_up] = sparse_mask1[:,:,1:-1,1:-1][mask_up]

        sparse1[:,:,0:-2,0:-2][mask_up_left] = sparse1[:,:,1:-1,1:-1][mask_up_left]
        sparse_mask1[:,:,0:-2,0:-2][mask_up_left] = sparse_mask1[:,:,1:-1,1:-1][mask_up_left]

        sparse1[:,:,2:,2:][mask_down_right] = sparse1[:,:,1:-1,1:-1][mask_down_right]
        sparse_mask1[:,:,2:,2:][mask_down_right] = sparse_mask1[:,:,1:-1,1:-1][mask_down_right]
        
        return sparse1, sparse_mask1


    def cost_volum_modulation(self, sparse, sparse_mask, sampler, cost_volum, max_disp = None):
        '''
            input: 
                sparse_mask: N*1*W*H
                sparse:      N*1*W*H
                sampler:     N*D*W*H
                cost_volum:  N*2F*D*W*H
            media_data:
                gaussian_modulation_element: N*D*W*H -->N*1*D*W*H
            output:
                modulated_cost_volum: N*2F*D*W*H

            process:
              sparse_mask
                            + sampler --> gaussian_modulation_element * cost_volum --> modulated_cost_volum
              sparse
        '''
        # for test 
        '''print('cf-net: sparse:', sparse.size())
        
        print('cf-net: sparse mask :', sparse_mask.size())
        print('cf-net: sampler', sampler)
        print('cf-net: cost volum', cost_volum.size())
        print(sparse.type())
        print(sparse_mask.type())
        print(cost_volum.type())'''
        # superparameter k=10,c=1 is set by GSM
        k = 10
        c = 1
        # N*W*H -->N*1*W*H
        #sparse = torch.unsqueeze(sparse, dim=1)
        #sparse_mask = torch.unsqueeze(sparse_mask, dim=1)
        # calculate the modulation element
        if max_disp is None:
            gaussian_modulation_element = 1 - sparse_mask + sparse_mask*k*torch.exp(-torch.square(sparse - sampler)/(2*pow(c,2)))
            #print('sampler: ', sampler.size())
        else:
            device = cost_volum.get_device()
            sampler_gen = torch.arange(0, max_disp, 1, device=device).view(1, max_disp, 1, 1) 
            gaussian_modulation_element = 1 - sparse_mask + sparse_mask*k*torch.exp(-torch.square(sparse - sampler_gen)/(2*pow(c,2)))
            #print('sampler_gen: ', sampler_gen.size())
        #print('gaussian_modulation_element: ', gaussian_modulation_element.size())
        #print('cost_volum', cost_volum.size())
        gaussian_modulation_element = torch.unsqueeze(gaussian_modulation_element, dim = 1)
        #print('gaussian_modulation_element2: ', gaussian_modulation_element.size())
        modulated_cost_volum = gaussian_modulation_element * cost_volum

        return modulated_cost_volum
    

    def cost_volum_modulation1(self, sparse, sparse_mask, sampler, cost_volum, max_disp = None):
        '''
            input: 
                sparse_mask: N*1*W*H
                sparse:      N*1*W*H
                sampler:     N*D*W*H
                cost_volum:  N*2F*D*W*H
            media_data:
                gaussian_modulation_element: N*D*W*H -->N*1*D*W*H
            output:
                modulated_cost_volum: N*2F*D*W*H

            process:
              sparse_mask
                            + sampler --> gaussian_modulation_element * cost_volum --> modulated_cost_volum
              sparse
        '''
        # for test 
        '''print('cf-net: sparse:', sparse.size())
        
        print('cf-net: sparse mask :', sparse_mask.size())
        print('cf-net: sampler', sampler)
        print('cf-net: cost volum', cost_volum.size())
        print(sparse.type())
        print(sparse_mask.type())
        print(cost_volum.type())'''
        # superparameter k=10,c=1 is set by GSM
        k = 2
        c = 8
        # N*W*H -->N*1*W*H
        #sparse = torch.unsqueeze(sparse, dim=1)
        #sparse_mask = torch.unsqueeze(sparse_mask, dim=1)
        # calculate the modulation element
        if max_disp is None:
            gaussian_modulation_element = 1 - sparse_mask + sparse_mask*k*torch.exp(-torch.square(sparse - sampler)/(2*pow(c,2)))
            #print('sampler: ', sampler.size())
        else:
            device = cost_volum.get_device()
            sampler_gen = torch.arange(0, max_disp, 1, device=device).view(1, max_disp, 1, 1) 
            gaussian_modulation_element = 1 - sparse_mask + sparse_mask*k*torch.exp(-torch.square(sparse - sampler_gen)/(2*pow(c,2)))
            #print('sampler_gen: ', sampler_gen.size())
        #print('gaussian_modulation_element: ', gaussian_modulation_element.size())
        #print('cost_volum', cost_volum.size())
        gaussian_modulation_element = torch.unsqueeze(gaussian_modulation_element, dim = 1)
        #print('gaussian_modulation_element2: ', gaussian_modulation_element.size())
        modulated_cost_volum = gaussian_modulation_element * cost_volum

        return modulated_cost_volum
    
    #''' extract sparse feature by sparse convolution '''
    '''
    def sparse_feature_extraction(self, sparse_out, sparse_mask_out):
        
        feature_sparse = {}
        feature_sparse['s2'] = torch.unsqueeze(self.sparse_convolution(sparse_out['sparse2'], sparse_mask_out['sparse_mask2']), dim=2)
        feature_sparse['s3'] = torch.unsqueeze(self.sparse_convolution(sparse_out['sparse3'], sparse_mask_out['sparse_mask3']), dim=2)
        feature_sparse['s4'] = torch.unsqueeze(self.sparse_convolution(sparse_out['sparse4'], sparse_mask_out['sparse_mask4']), dim=2)
        feature_sparse['s5'] = torch.unsqueeze(self.sparse_convolution(sparse_out['sparse5'], sparse_mask_out['sparse_mask5']), dim=2)
        feature_sparse['s6'] = torch.unsqueeze(self.sparse_convolution(sparse_out['sparse6'], sparse_mask_out['sparse_mask6']), dim=2)

        return feature_sparse
    '''
    def forward(self, left, right, sparse, sparse_mask):
        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)
        # RGB guided sparse expanded
        sparse1, sparse_mask1 = self.sparse_expand_round(sparse, sparse_mask, left)
        sparse2 = sparse1
        sparse2[sparse_mask>0] = sparse[sparse_mask>0]
        #sparse_mask2 = sparse_mask1
        #sparse_mask2[sparse_mask>0] = sparse_mask[sparse_mask>0]
        #sparse2, sparse_mask2 = self.sparse_expand(sparse, sparse_mask, left)

        sparse_out, sparse_mask_out = self.sparse_downsample(sparse, sparse_mask)
        sparse_out1, sparse_mask_out1 = self.sparse_downsample(sparse1, sparse_mask1)

        #sparse_out2, sparse_mask_out2 = self.sparse_downsample(sparse2, sparse_mask2)
        #sparse_out2, sparse_mask_out2 = sparse_out1, sparse_mask_out1
        #sparse_out2[sparse_mask_out] = sparse_out[sparse_mask_out]
        #feature_sparse = self.sparse_feature_extraction(sparse_out2, sparse_mask_out2)

        feature_sparse = self.sparse_feature_extraction(sparse2)

        gwc_volume4 = build_gwc_volume(features_left["gw4"], features_right["gw4"], self.maxdisp // 8,
                                       self.num_groups)

        gwc_volume5 = build_gwc_volume(features_left["gw5"], features_right["gw5"], self.maxdisp // 16,
                                       self.num_groups)

        gwc_volume6 = build_gwc_volume(features_left["gw6"], features_right["gw6"], self.maxdisp // 32,
                                       self.num_groups)
        if self.use_concat_volume:
            concat_volume4 = build_concat_volume(features_left["concat_feature4"], features_right["concat_feature4"],
                                                 self.maxdisp // 8)
            concat_volume5 = build_concat_volume(features_left["concat_feature5"], features_right["concat_feature5"],
                                                 self.maxdisp // 16)
            concat_volume6 = build_concat_volume(features_left["concat_feature6"], features_right["concat_feature6"],
                                                 self.maxdisp // 32)
            volume4 = torch.cat((gwc_volume4, concat_volume4), 1)
            volume5 = torch.cat((gwc_volume5, concat_volume5), 1)
            volume6 = torch.cat((gwc_volume6, concat_volume6), 1)

        else:
            volume4 = gwc_volume4
        
        if self.sparse_feature_on:
            volume4_s = torch.cat((volume4, feature_sparse['s4'].expand(-1,-1,volume4.size()[2],-1,-1)), 1)
            volume5_s = torch.cat((volume5, feature_sparse['s5'].expand(-1,-1,volume5.size()[2],-1,-1)), 1)
            volume6_s = torch.cat((volume6, feature_sparse['s6'].expand(-1,-1,volume6.size()[2],-1,-1)), 1)
        else:
            volume4_s = volume4
            volume5_s = volume5
            volume6_s = volume6

        volume6_m1 = self.cost_volum_modulation(sparse_out["sparse6"], sparse_mask_out["sparse_mask6"], None, volume6_s, self.maxdisp // 32)
        volume5_m1 = self.cost_volum_modulation(sparse_out["sparse5"], sparse_mask_out["sparse_mask5"], None, volume5_s, self.maxdisp // 16)
        volume4_m1 = self.cost_volum_modulation(sparse_out["sparse4"], sparse_mask_out["sparse_mask4"], None, volume4_s, self.maxdisp // 8)

        volume6_m = self.cost_volum_modulation1(sparse_out1["sparse6"], sparse_mask_out1["sparse_mask6"], None, volume6_m1, self.maxdisp // 32)
        volume5_m = self.cost_volum_modulation1(sparse_out1["sparse5"], sparse_mask_out1["sparse_mask5"], None, volume5_m1, self.maxdisp // 16)
        volume4_m = self.cost_volum_modulation1(sparse_out1["sparse4"], sparse_mask_out1["sparse_mask4"], None, volume4_m1, self.maxdisp // 8)

        cost0_4 = self.dres0(volume4_m)
        cost0_4 = self.dres1(cost0_4) + cost0_4

        cost0_5 = self.dres0_5(volume5_m)
        cost0_5 = self.dres1_5(cost0_5) + cost0_5
        cost0_6 = self.dres0_6(volume6_m)
        cost0_6 = self.dres1_6(cost0_6) + cost0_6
        out1_4 = self.combine1(cost0_4, cost0_5, cost0_6)
        out2_4 = self.dres3(out1_4)


        cost2_s4 = self.classif2(out2_4)
        cost2_s4 = torch.squeeze(cost2_s4, 1)
        pred2_possibility_s4 = F.softmax(cost2_s4, dim=1)
        pred2_s4 = disparity_regression(pred2_possibility_s4, self.maxdisp // 8).unsqueeze(1)
        pred2_s4_cur = pred2_s4.detach()
        #print('----pred2_possibility_s4',  pred2_possibility_s4.max(), pred2_possibility_s4.min())
        #print('----pred2_s4_cur',  pred2_s4_cur.max(), pred2_s4_cur.min())
        pred2_v_s4 = disparity_variance(pred2_possibility_s4, self.maxdisp // 8, pred2_s4_cur)  # get the variance
        pred2_v_s4 = pred2_v_s4.sqrt()
        mindisparity_s3 = pred2_s4_cur - (self.gamma_s3 + 1) * pred2_v_s4 - self.beta_s3
        maxdisparity_s3 = pred2_s4_cur + (self.gamma_s3 + 1) * pred2_v_s4 + self.beta_s3
        maxdisparity_s3 = F.upsample(maxdisparity_s3 * 2, [left.size()[2] // 4, left.size()[3] // 4], mode='bilinear', align_corners=True)
        mindisparity_s3 = F.upsample(mindisparity_s3 * 2, [left.size()[2] // 4, left.size()[3] // 4], mode='bilinear',
                                    align_corners=True)

        mindisparity_s3_1, maxdisparity_s3_1 = self.generate_search_range(self.sample_count_s3 + 1, mindisparity_s3, maxdisparity_s3, scale=2)
        disparity_samples_s3 = self.generate_disparity_samples(mindisparity_s3_1, maxdisparity_s3_1, self.sample_count_s3).float()
        confidence_v_concat_s3, _ = self.cost_volume_generator(features_left["concat_feature3"],
                                                            features_right["concat_feature3"], disparity_samples_s3, 'concat')
        confidence_v_gwc_s3, disparity_samples_s3 = self.cost_volume_generator(features_left["gw3"], features_right["gw3"],
                                                                         disparity_samples_s3, 'gwc', self.num_groups)
        confidence_v_s3 = torch.cat((confidence_v_gwc_s3, confidence_v_concat_s3, disparity_samples_s3), dim=1)
        # add gaussian modulation 
        if self.sparse_feature_on:
            confidence_v_s3_s = torch.cat((confidence_v_s3, feature_sparse['s3'].expand(-1,-1,confidence_v_s3.size()[2],-1,-1)), 1)
        else:
            confidence_v_s3_s = confidence_v_s3
        #confidence_v_s3_m = self.cost_volum_modulation(sparse_out["sparse3"], sparse_mask_out["sparse_mask3"], disparity_samples_s3, confidence_v_s3_s)
        disparity_samples_s3 = torch.squeeze(disparity_samples_s3, dim=1)
        confidence_v_s3_m = self.cost_volum_modulation(sparse_out["sparse3"], sparse_mask_out["sparse_mask3"], disparity_samples_s3, confidence_v_s3_s)
        confidence_v_s3_m1 = self.cost_volum_modulation1(sparse_out1["sparse3"], sparse_mask_out1["sparse_mask3"], disparity_samples_s3, confidence_v_s3_m)

        cost0_s3 = self.confidence0_s3(confidence_v_s3_m1)
        cost0_s3 = self.confidence1_s3(cost0_s3) + cost0_s3

        out1_s3 = self.confidence2_s3(cost0_s3)
        out2_s3 = self.confidence3_s3(out1_s3)

        cost1_s3 = self.confidence_classif1_s3(out2_s3).squeeze(1)
        cost1_s3_possibility = F.softmax(cost1_s3, dim=1)
        pred1_s3 = torch.sum(cost1_s3_possibility * disparity_samples_s3, dim=1, keepdim=True)
        pred1_s3_cur = pred1_s3.detach()
        pred1_v_s3 = disparity_variance_confidence(cost1_s3_possibility, disparity_samples_s3, pred1_s3_cur)

        #print('----cost1_s3_possibility',  cost1_s3_possibility.max(), cost1_s3_possibility.min())
        #print('----disparity_samples_s3',  disparity_samples_s3.max(), disparity_samples_s3.min())
        #print('----pred1_s2',  pred1_s3_cur.max(), pred1_s3_cur.min())
        pred1_v_s3 = pred1_v_s3.sqrt()
        mindisparity_s2 = pred1_s3_cur - (self.gamma_s2 + 1) * pred1_v_s3 - self.beta_s2
        maxdisparity_s2 = pred1_s3_cur + (self.gamma_s2 + 1) * pred1_v_s3 + self.beta_s2
        maxdisparity_s2 = F.upsample(maxdisparity_s2 * 2, [left.size()[2] // 2, left.size()[3] // 2], mode='bilinear',
                                     align_corners=True)
        mindisparity_s2 = F.upsample(mindisparity_s2 * 2, [left.size()[2] // 2, left.size()[3] // 2], mode='bilinear',
                                     align_corners=True)


        mindisparity_s2_1, maxdisparity_s2_1 = self.generate_search_range(self.sample_count_s2 + 1, mindisparity_s2, maxdisparity_s2, scale=1)
        disparity_samples_s2 = self.generate_disparity_samples(mindisparity_s2_1, maxdisparity_s2_1, self.sample_count_s2).float()
        confidence_v_concat_s2, _ = self.cost_volume_generator(features_left["concat_feature2"],
                                                            features_right["concat_feature2"], disparity_samples_s2, 'concat')
        confidence_v_gwc_s2, disparity_samples_s2 = self.cost_volume_generator(features_left["gw2"], features_right["gw2"],
                                                                         disparity_samples_s2, 'gwc', self.num_groups // 2)
        confidence_v_s2 = torch.cat((confidence_v_gwc_s2, confidence_v_concat_s2, disparity_samples_s2), dim=1)
        # add gaussian modulation 
        if self.sparse_feature_on:
            confidence_v_s2_s = torch.cat((confidence_v_s2, feature_sparse['s2'].expand(-1,-1,confidence_v_s2.size()[2],-1,-1)), 1)
        else:
            confidence_v_s2_s = confidence_v_s2
        #confidence_v_s2_m = self.cost_volum_modulation(sparse_out["sparse2"], sparse_mask_out["sparse_mask2"], disparity_samples_s2, confidence_v_s2_s)
        disparity_samples_s2 = torch.squeeze(disparity_samples_s2, dim=1)
        confidence_v_s2_m = self.cost_volum_modulation(sparse_out["sparse2"], sparse_mask_out["sparse_mask2"], disparity_samples_s2, confidence_v_s2_s)
        confidence_v_s2_m1 = self.cost_volum_modulation1(sparse_out1["sparse2"], sparse_mask_out1["sparse_mask2"], disparity_samples_s2, confidence_v_s2_m)

        cost0_s2 = self.confidence0_s2(confidence_v_s2_m1)
        cost0_s2 = self.confidence1_s2(cost0_s2) + cost0_s2

        out1_s2 = self.confidence2_s2(cost0_s2)
        out2_s2 = self.confidence3_s2(out1_s2)

        cost1_s2 = self.confidence_classif1_s2(out2_s2).squeeze(1)
        cost1_s2_possibility = F.softmax(cost1_s2, dim=1)
        pred1_s2 = torch.sum(cost1_s2_possibility * disparity_samples_s2, dim=1, keepdim=True)
        
        #print('----cost1_s2_possibility',  cost1_s2_possibility.max(), cost1_s2_possibility.min())
        #print('----disparity_samples_s2',  disparity_samples_s2.max(), disparity_samples_s2.min())
        #print('----pred1_s2',  pred1_s2.max(), pred1_s2.min())

        pred1_v_s2 = disparity_variance_confidence(cost1_s2_possibility, disparity_samples_s2, pred1_s2)
        pred1_v_s2 = pred1_v_s2.sqrt()

        if self.training:
            cost0_4 = self.classif0(cost0_4)
            cost1_4 = self.classif1(out1_4)

            cost0_4 = F.upsample(cost0_4, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear', align_corners=True)
            cost0_4 = torch.squeeze(cost0_4, 1)
            pred0_4 = F.softmax(cost0_4, dim=1)
            pred0_4 = disparity_regression(pred0_4, self.maxdisp)

            cost1_4 = F.upsample(cost1_4, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear', align_corners=True)
            cost1_4 = torch.squeeze(cost1_4, 1)
            pred1_4 = F.softmax(cost1_4, dim=1)
            pred1_4 = disparity_regression(pred1_4, self.maxdisp)

            pred2_s4 = F.upsample(pred2_s4 * 8, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)
            pred2_s4 = torch.squeeze(pred2_s4, 1)

            cost0_s3 = self.confidence_classif0_s3(cost0_s3).squeeze(1)
            cost0_s3 = F.softmax(cost0_s3, dim=1)
            pred0_s3 = torch.sum(cost0_s3 * disparity_samples_s3, dim=1, keepdim=True)
            pred0_s3 = F.upsample(pred0_s3 * 4, [left.size()[2], left.size()[3]], mode='bilinear',
                                     align_corners=True)
            pred0_s3 = torch.squeeze(pred0_s3, 1)

            costmid_s3 = self.confidence_classifmid_s3(out1_s3).squeeze(1)
            costmid_s3 = F.softmax(costmid_s3, dim=1)
            predmid_s3 = torch.sum(costmid_s3 * disparity_samples_s3, dim=1, keepdim=True)
            predmid_s3 = F.upsample(predmid_s3 * 4, [left.size()[2], left.size()[3]], mode='bilinear',
                                     align_corners=True)
            predmid_s3 = torch.squeeze(predmid_s3, 1)

            pred1_s3_up = F.upsample(pred1_s3 * 4, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)
            pred1_s3_up = torch.squeeze(pred1_s3_up, 1)

            cost0_s2 = self.confidence_classif0_s2(cost0_s2).squeeze(1)
            cost0_s2 = F.softmax(cost0_s2, dim=1)
            pred0_s2 = torch.sum(cost0_s2 * disparity_samples_s2, dim=1, keepdim=True)
            pred0_s2 = F.upsample(pred0_s2 * 2, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)
            pred0_s2 = torch.squeeze(pred0_s2, 1)

            costmid_s2 = self.confidence_classifmid_s2(out1_s2).squeeze(1)
            costmid_s2 = F.softmax(costmid_s2, dim=1)
            predmid_s2 = torch.sum(costmid_s2 * disparity_samples_s2, dim=1, keepdim=True)
            
            #  have been modified
            predmid_s2 = F.upsample(predmid_s2 * 2, [left.size()[2], left.size()[3]], mode='bilinear',
                                     align_corners=True)
            predmid_s2 = torch.squeeze(predmid_s2, 1)

            pred1_s2 = F.upsample(pred1_s2 * 2, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)
            pred1_s2 = torch.squeeze(pred1_s2, 1)

            #print('----pred1_v_s2',  pred1_v_s2.max(), pred1_v_s2.min())
            #print('----pred1_v_s3',  pred1_v_s3.max(), pred1_v_s3.min())
            #print('----pred1_v_s4',  pred2_v_s4.max(), pred2_v_s4.min())
            #return [pred0_4, pred1_4, pred2_s4, pred0_s3, predmid_s3, pred1_s3_up, pred0_s2, predmid_s2, pred1_s2, cost2_s4, pred2_v_s4, cost1_s3, pred1_v_s3,disparity_samples_s3, cost1_s2, pred1_v_s2, disparity_samples_s2]
            return [pred0_4, pred1_4, pred2_s4, pred0_s3, predmid_s3, pred1_s3_up, pred0_s2, predmid_s2, pred1_s2]

        else:
            pred2_s4 = F.upsample(pred2_s4 * 8, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)
            pred2_s4 = torch.squeeze(pred2_s4, 1)

            pred1_s3_up = F.upsample(pred1_s3 * 4, [left.size()[2], left.size()[3]], mode='bilinear',
                                     align_corners=True)
            pred1_s3_up = torch.squeeze(pred1_s3_up, 1)

            pred1_s2 = F.upsample(pred1_s2 * 2, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)
            pred1_s2 = torch.squeeze(pred1_s2, 1)


            return [pred1_s2], [pred1_s3_up], [pred2_s4]


def CFNet(d):
    return cfnet(d, use_concat_volume=True)
