import os
import random
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import f1_score


def seed_everything(seed: int):
    """fix seed

    Args:
        seed (int): seed
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    
def get_score(y_true, y_pred, step: float = 0.01, return_threshold:bool = False, disable:bool = True) -> tuple[float, float] | float:
    """calculate f1_score

    Args:
        y_true (pl.Series): Correct values(binary)
        y_pred (pl.Series): Predictions(probably)
        step (float, optional): Interval between numbers when calculating thresholds. Defaults to 0.01.
        return_threshold (bool, optional): Whether to return the threshold when the score is best. Defaults to False.
        disable (bool, optional): Whether to show progress bar. Defaults to True.

    Returns:
        tuple[float, float] | float: best score and threshold (if return_threshold=True) | best score (if return_threshold=False)
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