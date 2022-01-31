__author__ = 'ziao'
import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.sparse import Embedding
from TCTN_modules.TCTN_module import *

import argparse


class TCTN(nn.Module):
    def __init__(self, num_layers, num_dec_frames, model_depth, num_heads, with_residual,
                 with_pos, pos_kind, mode, config):
        super(TCTN, self).__init__()
        self.configs = config
        
        self.decoder_embedding = DecoderEmbedding(model_depth)
        self.decoder = Decoder(num_layers=config.de_layers, model_depth=model_depth, num_heads=num_heads,
                                              num_frames=num_dec_frames, with_residual=with_residual,
                                              with_pos=with_pos, pos_kind=pos_kind)

        self.conv_last = nn.Conv3d(model_depth, config.img_channel*config.patch_size*config.patch_size,
                                   kernel_size=1, stride=1, padding=0, bias=False)

        self.task = mode
        self.num_dec_frames = num_dec_frames

    def forward(self, input_img, val_signal=1):#[batch, seq-1, channel, height, width]
        # decoder
        if val_signal == 0:
            dec_init = self.decoder_embedding(input_img)
            decoderout = self.decoder(dec_init)
            if self.configs.w_pffn == 1:
                out = self.prediction(decoderout)
            else:
                out = self.conv_last(decoderout.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
            
        else:
            for i in range(self.configs.test_total_length - self.configs.test_input_length):
                if i  == 0 :
                    img_last = input_img[:, 0:self.configs.test_input_length]
                    #print(img_last.shape)
                    dec_init = self.decoder_embedding(img_last)
                else:
                    #dec_init = torch.cat((dec_init,new_embedding),1)
                    dec_init = new_embedding
                decoderout = self.decoder(dec_init)
                #print(decoderout.shape)

                if i < self.configs.test_total_length - self.configs.test_input_length - 1:
                    nex_img = decoderout[:,-1].unsqueeze(1)
                    if self.configs.w_pffn == 1:
                        img = self.prediction(nex_img)
                    else:
                        img = self.conv_last(nex_img.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
                    #print(img.shape)
                    img_last = torch.cat((img_last,img),1)
                    #print(img_last.shape)
                    new_embedding = self.decoder_embedding(img_last)
                else:
                    if self.configs.w_pffn == 1:
                        out = self.prediction(decoderout)
                    else:
                        out = self.conv_last(decoderout.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)                   
        
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_depth', type=int, default=64, help='depth of the model')
    parser.add_argument('--n_layers', type=int, default=1, help='number of layers in encoder and decoder')
    parser.add_argument('--en_layers', type=int, default=0, help='number of layers in encoder and decoder')
    parser.add_argument('--de_layers', type=int, default=2, help='number of layers in encoder and decoder')
    parser.add_argument('--n_heads', type=int, default=4, help='number of heads in conv mult-ihead attention')
    parser.add_argument('--dec_frames', type=int, default=19, help='nummber of output frames')
    parser.add_argument('--w_res', type=bool, default=True, help='using residual connect or not')
    parser.add_argument('--w_pos', type=bool, default=True, help='using positional encoding or not')
    parser.add_argument('--pos_kind', type=str, default='sine', help='kind of positional encoding,two choice: sine,learned')
    parser.add_argument('--model_type', type=int, default=1, help='type of the model, 0 for interpoation model and 1 for extrapolation model')

    parser.add_argument('--input_length', type=int, default=10)
    parser.add_argument('--total_length', type=int, default=20)
    parser.add_argument('--test_input_length', type=int, default=10)
    parser.add_argument('--test_total_length', type=int, default=20)
    parser.add_argument('--img_width', type=int, default=64)
    parser.add_argument('--img_channel', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--filter_size', type=int, default=3)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--w_pffn', type=int, default=0)

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=1)
    #parser.add_argument('--test', type=int, default=1, help='')


    args = parser.parse_args()
    model = TCTN(model_depth=args.model_depth, num_dec_frames=args.dec_frames, num_heads=args.n_heads,
                            num_layers=args.n_layers, with_residual=args.w_res, with_pos=args.w_pos,
                            pos_kind=args.pos_kind, mode=args.model_type, config=args).to(args.device)
    #model.train()
    #model.eval()
    #[batch, seq, channel, height, width]
    img = torch.randn(args.batch_size, args.total_length - 1, args.patch_size ** 2 * args.img_channel, 
    args.img_width // args.patch_size, args.img_width // args.patch_size).to(args.device)
    #print(img.shape)
    #for i in range (10):
    out = model(img, 1)
   

    print(out.shape)
