from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VIDLoss(nn.Module):
    """Variational Information Distillation for Knowledge Transfer (CVPR 2019),
    code from author: https://github.com/ssahn0215/variational-information-distillation"""

    def __init__(self,
                 num_input_channels,
                 num_mid_channel,
                 num_target_channels,
                 init_pred_var=5.0,
                 eps=1e-5, device='cuda:0'):
        super(VIDLoss, self).__init__()

        # def conv1x1(in_channels, out_channels, stride=1):
        #     return nn.Conv1d(
        #         in_channels, out_channels,
        #         kernel_size=1, padding=0,
        #         bias=False, stride=stride)
        #
        # self.regressor = nn.Sequential(
        #     conv1x1(num_input_channels, num_mid_channel),
        #     nn.ReLU(),
        #     conv1x1(num_mid_channel, num_mid_channel),
        #     nn.ReLU(),
        #     conv1x1(num_mid_channel, num_target_channels),
        # )

        self.device = device
        # self.regressor.to(self.device)
        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(init_pred_var - eps) - 1.0) * torch.ones(num_target_channels)
        ).to(device=self.device)
        self.eps = eps
        self.pred_var = (torch.log(1.0 + torch.exp(self.log_scale)) + self.eps).to(device=self.device)

        self.pred_var = self.pred_var.view(1, -1, 1, 1)

    def forward(self, input, target, reduction_vid='none'):
        # pool for dimentsion match
        # s_H, t_H = input.shape[1], target.shape[1]
        # if s_H > t_H:
        #     input = F.adaptive_avg_pool2d(input, (t_H, t_H))
        # elif s_H < t_H:
        #     target = F.adaptive_avg_pool2d(target, (s_H, s_H))
        # else:
        #     pass
        pred_mean = input
        # pred_mean = self.regressor(input)

        neg_log_prob = 0.5 * (
                (pred_mean - target) ** 2 / self.pred_var + torch.log(self.pred_var)
        )

        # print(self.pred_var)
        if reduction_vid == 'mean':
            loss = torch.mean(neg_log_prob.to(device=self.device))
            # print(loss)
            return loss.half()
        if reduction_vid == 'none':
            return neg_log_prob.half()


# if __name__ == '__main__':
#     vid_loss = VIDLoss(3, 3, 25)
#     y = torch.randn([5, 25]).to('cuda')
#     labels = torch.randn([5, 25]).to('cuda')
#     print(vid_loss(y, labels, 'mean'))
