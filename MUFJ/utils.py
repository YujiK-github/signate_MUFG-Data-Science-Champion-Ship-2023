import os
import math
import time
import random
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

def seed_everything(seed):
    """fix random factors"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    
    
def get_score(y_true, y_pred, step: float = 0.01, return_threshold:bool = False, disable:bool = True):
    """
    評価関数の入力となる検証用データ、及び学習に使用する学習用データの目的変数について、
    1: 不正利用あり, 0: 不正利用なしとします。
    """
    best_score = -np.inf
    j = 0
    for i in tqdm(np.arange(0, 1, step), leave=False, disable=disable):
        y_pred_class = [1 if y > i else 0 for y in y_pred]
        score = f1_score(y_true, y_pred_class)
        if best_score < score:
            best_score = score
            best_threshold = i
            j = 0
        else:
            j += 1
        if j > 20:
            break
    if return_threshold:
        return best_score, best_threshold
    else:
        return best_score
    
    
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))