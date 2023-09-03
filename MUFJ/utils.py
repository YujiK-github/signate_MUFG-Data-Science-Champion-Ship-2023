import os
import math
import time
import random
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import f1_score


def seed_everything(seed: int):
    """fix seed

    Args:
        seed (int): シード
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    
def get_score(y_true, y_pred, step: float = 0.01, return_threshold:bool = False, disable:bool = True) -> tuple[float, float] | float:
    """f1_scoreを計算する

    Args:
        y_true (pl.Series): 正解の値(バイナリー)
        y_pred (pl.Series): 予測値(確率)
        step (float, optional): 閾値を計算するときの数字の間隔. Defaults to 0.01.
        return_threshold (bool, optional): スコアが最も良いときの閾値を返すか否か. Defaults to False.
        disable (bool, optional): progress barを表示するか否か. Defaults to True.

    Returns:
        tuple[float, float] | float: ベストスコアとそのときの閾値 (return_threshold=Trueのとき) | ベストスコア (return_threshold=Falseのとき)
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
    
    
def asMinutes(s: int) -> str:
    """秒数を〇分〇秒に変換する

    Args:
        s (int): 秒数

    Returns:
        str: 〇分〇秒
    """
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since: float, percent: float) -> str:
    """sinceから経過した時間と進捗状況から計算して、経過時間と残り時間を表示する
    Args:
        since (float): time.time()
        percent (float): 進捗状況

    Returns:
        str: 経過時間 (remain 残り時間)
    """
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))