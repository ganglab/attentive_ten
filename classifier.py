import torch
import torch.nn as nn
import torch.nn.functional as F

#用于0，1标签训练，不加注意力
class classifier( nn.Module ):
    def __init__( self, in_dim=1 ):
        super( classifier, self ).__init__()

        # self.downsample = nn.Sequential(
        #     nn.Conv2d( in_channels=in_dim, out_channels=128, kernel_size=(2, 7), padding=(0, 3), stride=(1, 1) ),  #[2,11]--[1,11]
        #     nn.BatchNorm2d(128), nn.ReLU(),
        #     nn.Conv2d( in_channels=128, out_channels=128, kernel_size=(1,15), padding=(0,7), stride=(1,1) ), #[1,11]--[1,11]
        #     nn.BatchNorm2d(128), nn.ReLU(),
        #     nn.Conv2d( in_channels=128, out_channels=128, kernel_size=(1, 15), padding=(0, 7), stride=(1, 1) ),  # 2,312--2,154
        #     nn.BatchNorm2d( 128 ), nn.ReLU(),
        #     nn.Conv2d( in_channels=128, out_channels=128, kernel_size=(1, 15), padding=(0, 7), stride=(1, 1) ), # 2,154--2,74
        #     nn.BatchNorm2d(128), nn.ReLU(),
        #     nn.Conv2d( in_channels=128, out_channels=128, kernel_size=(1, 15), padding=(0, 7), stride=(1, 1) ),
        #     nn.BatchNorm2d(128), nn.ReLU(),
        #     # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 15), padding=(0, 7), stride=(1, 1)),
        #     # nn.BatchNorm2d(128), nn.ReLU(),
        #     # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 15), padding=(0, 7), stride=(1, 1)),
        #     # nn.BatchNorm2d(128), nn.ReLU(),
        #     # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 15), padding=(0, 7), stride=(1, 1)),
        #     # nn.BatchNorm2d(128), nn.ReLU(),
        #     nn.Conv2d( in_channels=128, out_channels=1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1) ),
        # )
        # # self.downsample2 = nn.Conv2d( in_channels=128, out_channels=1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1) )

        self.conv1 = nn.Sequential(
            nn.Conv2d( in_channels=in_dim, out_channels=128, kernel_size=(2, 7), padding=(0, 3), stride=(1, 1) ),  #[2,11]--[1,11]
            nn.BatchNorm2d(128), nn.ReLU() )
        self.conv2 = nn.Sequential(
            nn.Conv2d( in_channels=128, out_channels=128, kernel_size=(1,15), padding=(0,7), stride=(1,1) ), #[1,11]--[1,11]
            nn.BatchNorm2d(128), nn.ReLU() )
        # self.conv3 = nn.Sequential(
            # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 15), padding=(0, 7), stride=(1, 1)),
            # nn.BatchNorm2d(128), nn.ReLU())
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 15), padding=(0, 7), stride=(1, 1)),
        #     nn.BatchNorm2d(128), nn.ReLU())
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 15), padding=(0, 7), stride=(1, 1)),
        #     nn.BatchNorm2d(128), nn.ReLU() )
        self.conv6 = nn.Conv2d( in_channels=128, out_channels=1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1) )





    def forward(self, inputs):

        # att_ = torch.unsqueeze(torch.unsqueeze(att, dim=1), dim=1)  # [bs, 1, 1, lenght]
        #
        # att[att > 0] = 1



        x = inputs   #[bs, channels, nodes, length ]
        # logit = self.downsample( x ) #[bs,1,1,l]
        # return torch.squeeze( torch.squeeze( logit,dim=1 ), dim=1 ) #[bs, 1, 1,l] - [bs,l]

        x = self.conv1( x )
        x = self.conv2( x )
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x) * att #[bs,1,1,l]
        # x = self.conv6(x) * att  # [bs,1,1,l]
        x = self.conv6( x )
        # print(x.shape, att.shape, x, att)
        # x = x[torch.where(att > 0) ]
        # x = x * att_

        logit = torch.squeeze(torch.squeeze(x, dim=1), dim=1)  # [bs, 1, 1,l] - [bs,l]
        logit = torch.mean(logit, dim=-1)  #[bs]
        # logit = logit.sum( dim=-1 ) / att.sum( dim= -1 )  #[bs] / [bs]
        return logit




    def forward2( self, inputs, att ):
        att[att > 0] = 1

        att_ = torch.unsqueeze(torch.unsqueeze(att, dim=1), dim=1)  # [bs, 1, 1, lenght]
        #

        # x = inputs
        x = inputs * att_    #[bs, channels, nodes, length ]
        # logit = self.downsample( x ) #[bs,1,1,l]
        # return torch.squeeze( torch.squeeze( logit,dim=1 ), dim=1 ) #[bs, 1, 1,l] - [bs,l]

        x = self.conv1( x )
        x = self.conv2( x )
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x) * att #[bs,1,1,l]
        # x = self.conv6(x) * att  # [bs,1,1,l]
        x = self.conv6( x )

        # x = x * att_

        logit = torch.squeeze( torch.squeeze(x, dim=1), dim=1 )  # [bs, 1, 1,l] - [bs,l]
        # logit = torch.mean( logit, dim=-1 )  #[bs]
        logit = torch.max(logit, dim=-1)[0]  # [bs]
        # print(logit.shape)
        # logit = logit.sum(dim=-1) / att.sum(dim=-1)  # [bs] / [bs]
        return logit











