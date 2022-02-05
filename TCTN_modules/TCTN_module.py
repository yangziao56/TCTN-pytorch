__author__ = 'ziao'
import torch
import torch.nn as nn
from TCTN_modules.TCTN_module_utils import *

####################################################################################
######################### definition for feature embedding #########################
####################################################################################
class DecoderEmbedding(nn.Module):
    def __init__(self, depth=256):
        super(DecoderEmbedding, self).__init__()
        self.pad = (3 - 1) *1
        self.conv0 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=depth, kernel_size=(3,7,7), stride=1, padding=(self.pad,3,3), bias=True),
            #nn.GroupNorm(num_groups=1, num_channels=depth),
            nn.LeakyReLU(0.2, inplace=True)
            # inplace=True的意思是进行原地操作，例如x=x+5, 对tensor直接进行修改，好处就是可以节省运行内存，不用多存储变量
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=depth, out_channels=depth, kernel_size=(3,5,5), stride=1, padding=(self.pad,2,2), bias=True),
            #nn.GroupNorm(num_groups=1, num_channels=depth),
            nn.LeakyReLU(0.2, inplace=True)
            # inplace=True的意思是进行原地操作，例如x=x+5, 对tensor直接进行修改，好处就是可以节省运行内存，不用多存储变量
        )
      
        self.depth = depth
        self.dropout = nn.Dropout3d(0.1)
       

    def forward(self, input_img):
        #[batch, seq, channel, height, width]
        img_ = input_img.permute(0, 2, 1, 3, 4).clone()
        
        feature_0 = self.conv0(img_)
        feature_0 = feature_0[:, :, :-self.pad]
        feature_1 = self.conv1(feature_0)
        feature_1 = feature_1[:, :, :-self.pad]
        feature_1 = feature_0 + feature_1
        '''
        b, c, s, h, w = feature_2.shape
        pos = positional_encoding(s, self.depth, h, w)
        pos = pos.unsqueeze(0).expand(b, -1, -1, -1, -1)
        feature_2 = feature_2 + pos.permute(0, 2, 1, 3, 4)
        '''
        out = self.dropout(feature_1).permute(0, 2, 1, 3, 4)
        return out

####################################################################################
#########################   definition for decoder   ###############################
####################################################################################
class Decoder(nn.Module):
    def __init__(self, num_layers=5, num_frames=1, model_depth=128, num_heads=4,
                 with_residual=True, with_pos=True, pos_kind='sine'):
        super(Decoder, self).__init__()
        self.depth = model_depth
        self.decoderlayer = DecoderLayer(model_depth, num_heads, with_pos=with_pos)
        self.num_layers = num_layers
        self.decoder = self.__get_clones(self.decoderlayer, self.num_layers)
        self.positionnet = PositionalEmbeddingLearned(int(model_depth/num_heads))
        self.num_frames = num_frames
        self.pos_kind = pos_kind
        self.GN = nn.GroupNorm(num_groups=1, num_channels=model_depth)

    def __get_clones(self, module, n):
        return nn.ModuleList([copy.deepcopy(module) for i in range(n)])

    def forward(self, dec_init):    
        b, s, c, h, w = dec_init.shape
        out = dec_init
        if self.pos_kind == 'sine':
            pos_dec = positional_encoding(s, self.depth, h, w)
            pos_dec = pos_dec.unsqueeze(0).expand(b, -1, -1, -1, -1)
        elif self.pos_kind == 'learned':
            pos_dec = self.positionnet(out.shape)
            #pos_enc = self.positionnet(encoderin.shape)
        else:
            print('Positional Encoding is wrong')
            return
        for layer in self.decoder:
            out = layer(out, pos_dec)
        return self.GN(out.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)


class DecoderLayer(nn.Module):
    def __init__(self, model_depth=128, num_heads=4, with_pos=True):
        super(DecoderLayer, self).__init__()
        self.depth = model_depth
        self.depth_perhead = int(model_depth / num_heads)
        self.attention = self.__get_clones(MultiHeadAttention(self.depth_perhead, num_heads, with_pos=with_pos), 1)
        self.out = out(self.depth)
        self.feedforward = FeedForwardNet(self.depth)
        self.GN1 = nn.GroupNorm(num_groups=1, num_channels=model_depth)
        self.GN2 = nn.GroupNorm(num_groups=1, num_channels=model_depth)
        self.dropout1 = nn.Dropout3d(0.1)
        self.dropout2 = nn.Dropout3d(0.1)

    def __get_clones(self, module, n):
        return nn.ModuleList([copy.deepcopy(module) for i in range(n)])

    def forward(self, input_tensor, pos_decoding):
        # sequence mask query self-attention
        att_layer_in = self.GN1(input_tensor.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        i = 0
        for layer in self.attention:
            att_out = layer(att_layer_in, pos_decoding, type=0)
        att_layer_out = self.dropout1(self.out(att_out).permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4) + input_tensor
        
        # feedforward
        ff_in = self.GN2(att_layer_out.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        out = self.dropout2(self.feedforward(ff_in).permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)+att_layer_out

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, head_depth=32, num_heads=4,with_pos=True):
        super(MultiHeadAttention, self).__init__()
        self.depth_perhead = head_depth
        self.num_heads = num_heads
        self.qkv =QKVNet(self.depth_perhead*self.num_heads)
        self.with_pos = with_pos
        #self.time_weighting = nn.Parameter(torch.ones(batch, height, width, self.num_heads, seq, seq))
        self.dropout1 = nn.Dropout3d(0.1)

    def forward(self, input_tensor, pos_decoding, type=0):  # encoderin换用model方式描述， 比如model = 0 or 1
        if type == 0:  # deocder--query self attention
            batch, seq, channel, height, width = input_tensor.shape
            input_tensor = input_tensor.permute(0, 2, 1, 3, 4)
            qkvconcat = self.qkv(input_tensor)
            q_feature, k_feature, v_feature =torch.split(qkvconcat, self.depth_perhead*self.num_heads, dim=1)
            if self.with_pos:
                q_feature = (q_feature + pos_decoding.permute(0, 2, 1, 3, 4))
                k_feature = (k_feature + pos_decoding.permute(0, 2, 1, 3, 4))
            q_feature = q_feature.view(batch, self.num_heads, self.depth_perhead, seq, height, width)
            k_feature = k_feature.view(batch, self.num_heads, self.depth_perhead, seq, height, width)
            v_feature = v_feature.view(batch, self.num_heads, self.depth_perhead, seq, height, width)
            
            # scaled dot product attention
            q = q_feature.permute(0, 4, 5, 1, 3, 2)#[batch, height, width, heads, seq, channel/head]
            k = k_feature.permute(0, 4, 5, 1, 2, 3)
            v = v_feature.permute(0, 4, 5, 1, 3, 2)
            attention_map = torch.matmul(q, k)/math.sqrt(self.depth_perhead)#[batch, height, width, heads seq, seq]
            #print(attention_map[0][0][0][0])
            #sequence mask
            mask = 1- torch.triu(torch.ones((seq, seq)),diagonal=1)
            mask = mask.unsqueeze(0).expand(batch*height*width*self.num_heads, -1, -1).view(batch, height, width, self.num_heads, seq, seq).cuda()
            attention_map = attention_map * mask
            attention_map = attention_map.masked_fill(attention_map==0, -1e9)
            attention_map = nn.Softmax(dim=-1)(attention_map)

            #[batch, heads, seq_k, seq_q, height, width]
            attention_map_ = attention_map.permute(0, 3, 5, 4, 1, 2).contiguous().view(batch, -1, seq, height, width)
            attention_map_ = self.dropout1(attention_map_) 
            attention_map = attention_map_.view(batch, self.num_heads, seq, seq, height, width).permute(0, 4, 5, 1, 3, 2)
            attentioned_v_Feature = torch.matmul(attention_map,v).permute(0, 4, 3, 5, 1, 2).reshape(batch, seq, self.num_heads*self.depth_perhead, height, width)
        return attentioned_v_Feature

