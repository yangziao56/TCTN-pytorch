__author__ = 'yunbo'
__revised__ = 'shuxin'

import numpy as np
import torch


def reshape_patch(img_tensor, patch_size):
    # img_tensor.shape: (batch_size, seq_length, img_height, img_width, num_channels)
    assert 5 == img_tensor.ndim
    batch_size = np.shape(img_tensor)[0]
    seq_length = np.shape(img_tensor)[1]
    img_height = np.shape(img_tensor)[2]
    img_width = np.shape(img_tensor)[3]
    num_channels = np.shape(img_tensor)[4]
    # reshape the tensor into: (batch_size, seq_length, height/patch, patch_size, width/patch, patch_size, num_channels)
    a = np.reshape(img_tensor, [batch_size, seq_length,
                                img_height // patch_size, patch_size,
                                img_width // patch_size, patch_size,
                                num_channels])
    # transpose into: (batch_size, seq_length, height/patch, width/patch, patch_size, patch_size, num_channels)
    b = np.transpose(a, [0, 1, 2, 4, 3, 5, 6])
    # reshape into: (batch_size, seq_length, height/patch, width/patch, patch_size*patch_size*num_channels)
    patch_tensor = np.reshape(b, [batch_size, seq_length,
                                  img_height // patch_size,
                                  img_width // patch_size,
                                  patch_size * patch_size * num_channels])
    return patch_tensor


def reshape_patch_back(patch_tensor, patch_size):
    # patch_tensor.shape: (batch_size, seq_length, patch_height, patch_width, channels)
    assert 5 == patch_tensor.ndim
    batch_size = np.shape(patch_tensor)[0]
    seq_length = np.shape(patch_tensor)[1]
    patch_height = np.shape(patch_tensor)[2]
    patch_width = np.shape(patch_tensor)[3]
    channels = np.shape(patch_tensor)[4]
    # calculate the img_channels
    img_channels = channels // (patch_size * patch_size)
    # reshape into: (batch_size, seq_length, height/patch, width/patch, patch_size, patch_size, num_channels)
    a = np.reshape(patch_tensor, [batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  img_channels])
    # transpose into: (batch_size, seq_length, height/patch, patch_size, width/patch, patch_size, num_channels)
    b = np.transpose(a, [0, 1, 2, 4, 3, 5, 6])
    # reshape into: (batch_size, seq_length, height, width, num_channels)
    img_tensor = np.reshape(b, [batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                img_channels])
    return img_tensor


def reshape_patch_tensor(img_tensor, patch_size):
    # img_tensor.shape: (batch_size, seq_length, num_channels, img_height, img_width)
    # img_tensor's type: tensor
    assert 5 == len(img_tensor.shape)
    batch_size = img_tensor.shape[0]
    seq_length = img_tensor.shape[1]
    num_channels = img_tensor.shape[2]
    img_height = img_tensor.shape[3]
    img_width = img_tensor.shape[4]
    # torch view is not share memory, transpose does.
    # reshape the tensor into: (batch_size, seq_length, height/patch, patch_size, width/patch, patch_size, num_channels)
    # first, transpose the tensor shape into (batch, seq_len, height, width, channels)
    a = img_tensor.permute(0, 1, 3, 4, 2)
    a = a.contiguous()
    a = a.view(batch_size, seq_length,
               img_height // patch_size, patch_size,
               img_width // patch_size, patch_size,
               num_channels)
    # transpose into: (batch_size, seq_length, height/patch, width/patch, patch_size, patch_size, num_channels)
    b = a.permute(0, 1, 2, 4, 3, 5, 6)
    # reshape into: (batch_size, seq_length, height/patch, width/patch, patch_size*patch_size*num_channels)
    b = b.contiguous()
    patch_tensor = b.view(batch_size, seq_length,
                          img_height // patch_size,
                          img_width // patch_size,
                          patch_size * patch_size * num_channels)
    return patch_tensor


#  patch_tensor shape: batch_size, num_channels, img_height, img_width
def reshape_patch_back_tensor(patch_tensor, patch_size):
    # img_tensor.shape: (batch_size, seq_length, num_channels, img_height, img_width)
    # img_tensor's type: tensor
    assert 4 == len(patch_tensor.shape)
    batch_size = patch_tensor.shape[0]
    num_channels = patch_tensor.shape[1]
    patch_height = patch_tensor.shape[2]
    patch_width = patch_tensor.shape[3]
    # calculate the img_channels
    img_channels = num_channels // (patch_size * patch_size)
    # reshape into the shape: b, h, w, c
    a = patch_tensor.permute(0, 2, 3, 1)
    a = a.contiguous()
    # torch view is not share memory, transpose does.
    a = a.view(batch_size, patch_height, patch_width,
               patch_size, patch_size, img_channels)
    # transpose into: (batch_size, height, patch_size, width, patch_size, img_channels)
    b = a.permute(0, 5, 1, 3, 2, 4)
    # reshape into: (batch_size, seq_length, height/patch, width/patch, patch_size*patch_size*num_channels)
    b = b.contiguous()
    # b, c, h, w
    img_tensor = b.view(batch_size, img_channels,
                        patch_height * patch_size,
                        patch_width * patch_size)
    return img_tensor


def reshape_back_batch_tensor(patch_tensor, patch_size):
    # img_tensor.shape: (batch_size, seq_length, num_channels, img_height, img_width)
    # img_tensor's type: tensor
    assert 5 == len(patch_tensor.shape)

    batch_size = patch_tensor.shape[0]
    seq_len = patch_tensor.shape[1]
    num_channels = patch_tensor.shape[2]
    patch_height = patch_tensor.shape[3]
    patch_width = patch_tensor.shape[4]
    # calculate the img_channels
    img_channels = num_channels // (patch_size * patch_size)
    # reshape into the shape: b, s, h, w, c
    a = patch_tensor.permute(0, 1, 3, 4, 2)
    a = a.contiguous()
    # torch view is not share memory, transpose does.
    a = a.view(batch_size, seq_len, patch_height, patch_width,
               patch_size, patch_size, img_channels)
    # transpose into: (batch_size, height, patch_size, width, patch_size, img_channels)
    b = a.permute(0, 1, 6, 2, 4, 3, 5)
    # reshape into: (batch_size, seq_length, height/patch, width/patch, patch_size*patch_size*num_channels)
    b = b.contiguous()
    # b, c, h, w
    img_tensor = b.view(batch_size, seq_len, img_channels,
                        patch_height * patch_size,
                        patch_width * patch_size)
    # b, s, c, h, w
    return img_tensor


if __name__ == '__main__':
    tensor = torch.randn(2, 4, 80, 16, 16)
    patch_size = 4
    re_tensor = reshape_back_batch_tensor(tensor, patch_size)
    print(re_tensor.shape)
