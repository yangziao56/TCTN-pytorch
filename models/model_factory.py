__author__ = 'ziao'
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from models import TCTN

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        networks_map = {
            'TCTN':TCTN.TCTN
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            if (configs.model_name == 'TCTN'):
                self.network = Network(model_depth=configs.model_depth, num_dec_frames=configs.dec_frames, num_heads=configs.n_heads,
                            num_layers=configs.n_layers, with_residual=configs.w_res, with_pos=configs.w_pos,
                            pos_kind=configs.pos_kind, mode=configs.model_type, config=configs).to(configs.device)

            else:
                self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
            #device_ids = [0, 1]
            self.network = torch.nn.DataParallel(self.network)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)
        self.MSE_criterion = nn.MSELoss()
        self.L1_criterion = nn.L1Loss()
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,100,eta_min=0,last_epoch=-1,verbose=True)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=configs.T_0, T_mult=configs.T_mult, eta_min=0, last_epoch=-1)
        #torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=configs.n_steps, gamma=configs.gamma)

    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        stats['optimizer_param'] = self.optimizer.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt'+'-'+str(itr))
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path)
        self.network.load_state_dict(stats['net_param'])
        # self.optimizer.load_state_dict(stats['optimizer_param'])

    # frames.shape : [batch, seq, height, width, channel]
    # that is : (batch_size, seq_length, height / patch, width / patch, patch_size * patch_size * num_channels)
    def train(self, frames, itr):
        frames_tensor = torch.FloatTensor(frames)
        frames_tensor = frames_tensor.permute(0, 1, 4, 2, 3).contiguous().to(self.configs.device)#[batch=seq, channel, height, width]
        #print(frames_tensor.shape)
        self.network.train()
        next_frames = self.network(frames_tensor[:, :-1],self.configs.de_train_type)
        loss_per_accumulation = self.MSE_criterion(next_frames, frames_tensor[:, 1:])# + self.L1_criterion(next_frames, frames_tensor[1:])
        loss = loss_per_accumulation/self.configs.accumulation_steps
        loss.backward()
        if(itr%self.configs.accumulation_steps)==0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.scheduler.step()  # 更新学习率
        if itr % self.configs.n_steps == 0:#100
            print('learning rate: ' + str(self.optimizer.param_groups[0]['lr']))
            print(itr)
        return loss_per_accumulation.detach().cpu().numpy()

    def test(self, frames):
        frames_tensor = torch.FloatTensor(frames)
        frames_tensor = frames_tensor.permute(0, 1, 4, 2, 3).contiguous().to(self.configs.device)
        self.network.eval()
        #print(frames_tensor.shape)
        with torch.no_grad():
            next_frames = self.network(frames_tensor[:, :-1],1).permute(0, 1, 3, 4, 2).contiguous()
        #print(next_frames.shape)
        return next_frames.detach().cpu().numpy()
