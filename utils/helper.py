import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ListAverageMeter(object):
    """Computes and stores the average and current values of a list"""

    def __init__(self):
        self.len = 10000  # set up the maximum length
        self.reset()
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def reset(self):
        self.val = [0] * self.len
        self.avg = [0] * self.len
        self.sum = [0] * self.len
        self.count = 0

    def set_len(self, n):
        self.len = n
        self.reset()

    def update(self, vals, n=1):
        assert len(vals) == self.len, 'length of vals not equal to self.len'
        self.val = vals
        for i in range(self.len):
            self.sum[i] += self.val[i] * n
        self.count += n
        for i in range(self.len):
            self.avg[i] = self.sum[i] / self.count


def save_checkpoint(state, save_dir, is_best=False, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_dir, filename), os.path.join(save_dir, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


# cross entropy
# thresholds = [0.25, 0.375, 0.5, 0.625]
# weights = torch.FloatTensor([2, 3, 6, 10, 30])
# midlevalue = [0.1875, 0.3125, 0.4375, 0.5625, 0.8125]
# Initialize: WeightedCrossEntropyLoss(thresholds, weights, 5),  prob = ProbToPixel(midlevalue)
class WeightedCrossEntropyLoss(nn.Module):
    # weight should be a 1D Tensor assigning weight to each of the classes.
    def __init__(self, thresholds, weight=None, LAMBDA=None):
        super().__init__()
        # 每个类别的权重，使用原文章的权重。
        self._weight = weight
        # 每一帧 Loss 递进参数
        self._lambda = LAMBDA
        # thresholds: 雷达反射率
        self._thresholds = thresholds

    # input: output prob, b*s*C*H*W
    # target: b*s*1*H*W, original data, range [0, 1]
    # mask: S*B*1*H*W
    def forward(self, inputs, targets):
        # assert input.size(0) == cfg.HKO.BENCHMARK.OUT_LEN

        # F.cross_entropy should be B*C*S*H*W
        inputs = inputs.permute((0, 2, 1, 3, 4))

        # B*S*H*W
        targets = targets.squeeze(2)
        class_index = torch.zeros_like(targets).long()

        thresholds = self._thresholds
        class_index[...] = 0

        for i, threshold in enumerate(thresholds):
            i = i + 1
            class_index[targets >= threshold] = i

        if self._weight is not None:
            self._weight = self._weight.to(targets.device)
            error = F.cross_entropy(inputs, class_index, self._weight, reduction='none')
        else:
            error = F.cross_entropy(inputs, class_index, reduction='none')

        if self._lambda is not None:
            B, S, H, W = error.size()
            w = torch.arange(1.0, 1.0 + S * self._lambda, self._lambda)

            if torch.cuda.is_available():
                w = w.to(targets.device)

            # B, H, W, S
            error = (w * error.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # S*B*1*H*W
        error = error.permute(0, 1, 2, 3).unsqueeze(2)

        return torch.mean(error.float())


class ProbToPixel(object):
    # 转换类别为预测值
    def __init__(self, middle_value):
        self._middle_value = middle_value

    def __call__(self, prediction):
        """

        prediction: 输入的类别预测值，b*s*C*H*W

        ground_truth: 实际值，像素/255.0, [0, 1]

        lr: 学习率

        :param prediction:

        :return:

        """
        # 分类结果，0 到 classes - 1
        # prediction: b*s*C*H*W
        # self._middle_value = self._middle_value.to(prediction.device)
        # print(prediction)

        result = torch.argmax(prediction, dim=2, keepdim=True)
        # print(result)
        prediction_result = torch.ones_like(result, dtype=torch.float32)

        for i in range(len(self._middle_value)):
            prediction_result[result == i] = self._middle_value[i]

            # 如果需要更新替代值

            # 更新替代值
        # print(prediction_result.dtype)
        return prediction_result.detach().cpu().numpy()

