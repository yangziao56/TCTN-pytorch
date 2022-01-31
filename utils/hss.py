__author__ = 'zsx'

import numpy as np
import copy
from utils.table import contingency_table_multicategory


def single_hss(truths, predictions, classes=None, threshold=None):
    """Determines the Heidke Skill Score, returning a scalar. See
    `here <http://www.cawcr.gov.au/projects/verification/#Methods_for_multi-category_forecasts>`_
    for more details on the Peirce Skill Score
    :param truths: The true labels of these data
    :param predictions: The predictions of the model
    :param classes: an ordered set for the label possibilities. If not given,
        will be deduced from the truth values
    :param threshold: for radar classify
    :returns: a single value for the heidke skill score
    """
    table = _get_contingency_table(truths, predictions, classes, threshold)
    return _peirce_skill_score(table)


def _get_contingency_table(truths, predictions, classes=None, threshold=None):
    """Uses the truths and predictions to compute a contingency matrix
    :param truths: The true labels of these data
    :param predictions: The predictions of the model
    :param classes: an ordered set for the label possibilities. If not given,
        will be deduced from the truth values if possible
    :returns: a numpy array of shape (num_classes, num_classes)
    """
    if len(truths) != len(predictions):
        raise ValueError("The shape of predictions and truths are not the same!")
    # Truths and predictions are now both deterministic
    if classes is None:
        classes = np.unique(np.append(np.unique(truths), np.unique(predictions)))
        preds = predictions
        labels = truths
    else:
        preds = _radar_classify(predictions, threshold)
        labels = _radar_classify(truths, threshold)
    """
    table =[]
    for i, c1 in enumerate(classes):
        c = Counter(preds[np.where(labels == c1)])
        single_row = []
        for j in range(len(classes)):
            single_row.append(c[j])
        table.append(single_row)
    table = np.array(table).astype(np.float32)
    """

    table = np.zeros((len(classes), len(classes)), dtype=np.float32)
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            for p, t in zip(preds, labels):
                table[i, j] += [p1 == c1 and t1 == c2 for p1, t1 in zip(p, t)].count(True)

    return table


def _radar_classify(inputs, threshold=None):
    # classify into 5 classes
    # threshold = [0, 20, 30, 40, 50, 70]
    if threshold is None:
        threshold = [0, 20, 30, 40, 50, 70]
    inputs_logit = copy.deepcopy(inputs)
    inputs_logit[inputs_logit <= threshold[1]] = 1
    for i in range(1, len(threshold) - 1):
        inputs_logit[(threshold[i] < inputs) & (inputs <= threshold[i + 1])] = i + 1
    return inputs_logit


def _peirce_skill_score(table):
    """This function is borrowed with modification from the hagelslag repository
    MulticlassContingencyTable class. It is used here with permission of
    David John Gagne II <djgagne@ou.edu>
    Multiclass Peirce Skill Score (also Hanssen and Kuipers score, True Skill Score)
    """
    n = float(table.sum())
    nf = table.sum(axis=1)
    no = table.sum(axis=0)
    correct = float(table.trace())
    no_squared = (no * no).sum()
    if n ** 2 == no_squared:
        return correct / n
    else:
        return (n * correct - (nf * no).sum()) / (n ** 2 - no_squared)


def batch_hss(truths, predictions, classes=None, threshold=None):
    # truths's shape: [batch, width, height]
    # batch_hss calculate the same time step
    bat_num = truths.shape[0]
    bat_hss = 0
    for i in range(bat_num):
        hss = single_hss(truths[i], predictions[i], classes, threshold)
        bat_hss += hss
    return bat_hss


def prep_clf(obs, pre, threshold=0.5):
    """
    func: 计算二分类结果-混淆矩阵的四个元素
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

    returns:
        hits, misses, falsealarms, correctnegatives
        #aliases: TP, FN, FP, TN
    """
    # 根据阈值分类为 0, 1
    obs = np.where(obs >= threshold, 1, 0)
    pre = np.where(pre >= threshold, 1, 0)

    # True positive (TP)
    hits = np.sum((obs == 1) & (pre == 1))

    # False negative (FN)
    misses = np.sum((obs == 1) & (pre == 0))

    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (pre == 1))

    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (pre == 0))

    return hits, misses, falsealarms, correctnegatives


def HSS(obs, pre, threshold=0.5):
    """
    HSS - Heidke skill score
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): pre
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: HSS value
    """
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre, threshold=threshold)

    HSS_num = 2 * (hits * correctnegatives - misses * falsealarms)
    HSS_den = (misses ** 2 + falsealarms ** 2 + 2 * hits * correctnegatives +
               (misses + falsealarms) * (hits + correctnegatives))

    return HSS_num / HSS_den


# github: github.com/nmcdev/meteva/blob/7524be059bdbae58fe46a7bf37b306b6fef0b5b3/meteva/method/multi_category/score.py
# usage document: https://www.showdoc.cc/meteva?page_id=3975612394328262
def meteva_tc(ob, fo, grade_list=None):
    """
    accuracy 求多分类预报准确率
    :param ob: 实况数据 不确定维numpy
    :param fo:  预测数据 不确定维numpy
    :param grade_list:等级，如果grade_list= None则ob和fo里的值代表类别，否则，根据grade_list来进行分类
    :return: 返回一维数组，包括（总样本数，正确数）
    """
    # 多分类预报准确率
    if grade_list is None:
        # ob 和fo是不确定维的numpy数组，其中每个值是代表分类的类别
        ob1 = ob.reshape((-1))
        fo1 = fo.reshape((-1))
    else:
        ob1 = np.zeros_like(ob)
        fo1 = np.zeros_like(fo)
        # ob 和fo 是连续的变量，通过 threshold_list 将ob 和fo划分成连续的等级之后再计算等级准确性
        for index in range(len(grade_list) - 1):
            ob_index_list = np.where((grade_list[index] <= ob) & (ob < grade_list[index + 1]))
            ob1[ob_index_list] = index + 1
            fo_index_list = np.where((grade_list[index] <= fo) & (fo < grade_list[index + 1]))
            fo1[fo_index_list] = index + 1
        ob_index_list = np.where(grade_list[-1] <= ob)
        ob1[ob_index_list] = len(grade_list)
        fo_index_list = np.where(grade_list[-1] <= fo)
        fo1[fo_index_list] = len(grade_list)
    correct_num = np.sum(fo1 == ob1)
    return np.array([ob.size, correct_num])


def meteva_accuracy(ob, fo, grade_list=None):
    """
    accuracy 求多分类预报准确率
    :param ob: 实况数据 任意维numpy数组
    :param fo: 预测数据 任意维numpy数组,Fo.shape 和Ob.shape一致
    :param grade_list: 如果该参数为None，观测或预报值出现过的值都作为分类标记.
    如果该参数不为None，它必须是一个从小到大排列的实数，以其中列出的数值划分出的多个区间作为分类标签。
    对于预报和观测值不为整数的情况，grade_list 不能设置为None。
    :return: 0-1的实数，0代表无技巧，最优预报为1
    """
    tc1 = meteva_tc(ob, fo, grade_list)
    total_count = tc1[0]
    correct_count = tc1[1]
    accuracy_score = correct_count / total_count
    return accuracy_score


def meteva_hss(ob, fo, grade_list=None):
    """
    hss heidke技能得分,它表现实际的预报的分类准确性相对于随机分类达到的准确性的技巧
    :param ob: 实况数据 任意维numpy数组
    :param fo: 预测数据 任意维numpy数组,Fo.shape 和Ob.shape一致
    :param grade_list: 如果该参数为None，观测或预报值出现过的值都作为分类标记.
    如果该参数不为None，它必须是一个从小到大排列的实数，以其中列出的数值划分出的多个区间作为分类标签。
    对于预报和观测值不为整数的情况，grade_list 不能设置为None。
    :return:
    """
    IV = 999999
    conf_mx = contingency_table_multicategory(ob, fo, grade_list)
    accuracy_score = meteva_accuracy(ob, fo, grade_list)
    total_num = ob.size
    NF_array = conf_mx[0:-1, -1]
    NO_array = conf_mx[-1, 0:-1]
    random_score = np.dot(NF_array, NO_array) / (total_num * total_num)
    if random_score == 1:
        HSS = IV  # IV
    else:
        HSS = (accuracy_score - random_score) / (1 - random_score)
    return HSS
