# %%
import pandas as pd
from typing import Dict, List
import numpy as np
import os
import time
from typing import Union

AUTHOR = 'author'
THEME_SINGER = 'theme_singer'
STUDIO_STAFF_NUM = 'studio_staff_num'
IS_AIRING = 'is_airing'
IN_SEIYUU_RANK = 'in_seiyuu_rank'
SEASONS = '#seasons'
STYLE = 'style'
SCREENPLAY_WRITER_NUM = 'screenplay_writer_num'
AVG_RATING = 'avg_rating'
AVG_EP_LENGTH = 'avg_ep_length'
CHARACTER_NUM = 'character_num'


MASTERPIECE = 'masterpiece'
REFRESHING = 'refreshing'
BLAND = 'bland'
Classes = (MASTERPIECE, REFRESHING, BLAND)


LABEL = 'class'
ATTRIBUTES = {
    #  discrete; numericals
    STUDIO_STAFF_NUM: range(50, 300+1),  # discrete (numerical)
    SCREENPLAY_WRITER_NUM: range(1, 6+1),  # discrete  (numerical)
    SEASONS: range(1, 5+1),              # discrete  (numerical)

    # boolean
    IS_AIRING: [0, 1],           # 0, 1 binary
    IN_SEIYUU_RANK: [0, 1],      # 0, 1 binary

    # categorical
    AUTHOR: [hash(x) for x in range(10)],  # categorical
    # categorical
    THEME_SINGER: ['Lisa', 'Sawano Hiroyuki', 'Hoshino Gen', 'Kenshi Yonezu', 'Mukai Taichi', 'Eve', 'Chico with HoneyWorks'],
    STYLE: ['運動', '戰鬥', '致鬱', '奇幻', '搞笑', '懸疑', '戀愛'],  # categorical

    # continuous
    AVG_RATING: [3.5, 0.5],              # continuous, gaussian
    AVG_EP_LENGTH: [25, 5],              # continuous, gaussian
    # (numerical), but sampled from continuous, gaussian (30, 10) -> integer
    CHARACTER_NUM: [30, 5],

}
DESIGNED_PROBS = {
    THEME_SINGER: [0.3, 0.3, 0.1, 0.1, 0.1, 0.05, 0.05],
    STYLE: [0.3, 0.3, 0.1, 0.1, 0.1, 0.05, 0.05]}


def _gen_data(attributes: Dict[str, List[str]]) -> Dict[str, Union[int, float]]:
    """Generating data

    Args:
        attributes (Dict[str, List[str]]): the attribute dictionary
    Returns:
        Dict[str, Union(int, float)]: the generated data dictionary; will be a row in csv
    """
    data_row = {}
    global lisacount

    for attr in attributes:
        numericals = [STUDIO_STAFF_NUM, SCREENPLAY_WRITER_NUM, SEASONS]
        categoricals = [AUTHOR, THEME_SINGER, STYLE]
        bools = [IS_AIRING, IN_SEIYUU_RANK]
        if attr in numericals + categoricals + bools:
            data_row[attr] = np.random.choice(attributes[attr])
            if attr in DESIGNED_PROBS:

                data_row[attr] = np.random.choice(
                    attributes[attr], p=DESIGNED_PROBS[attr])

        else:
            mu, sigma = attributes[attr]
            data_row[attr] = np.random.normal(mu, sigma)
            if attr == CHARACTER_NUM:
                data_row[attr] = int(data_row[attr])
            if attr == AVG_RATING:
                data_row[attr] = min(5, max(0, data_row[attr]))

    if is_masterpiece(data_row):
        data_row[LABEL] = MASTERPIECE
    elif is_refreshing(data_row):
        data_row[LABEL] = REFRESHING
    else:
        data_row[LABEL] = BLAND
    return data_row


def is_masterpiece(d: Dict[str, Union[int, float]]):
    """
    Determine if a given anime is a masterpeice
    ===============================================
    - 鉅作：
    IN_SEIYUU_RANK = True and STUDIO_STAFF_NUM >= 120
    且滿足至少以下其中一個條件
    (1)
    AVG_RATING > 3.8
    CHARACTER_NUM > 35
    SCREENPLAY_WRITER_NUM < 3
    SEASONS > 2

    (2)
    THEME_SINGER = 'Lisa' or 'Sawano Hiroyuki'
    STYLE = '運動' or '戰鬥'
    AVG_RATING > 3.5
    CHARACTER_NUM > 40
    """

    if d[IN_SEIYUU_RANK] == 0 or d[STUDIO_STAFF_NUM] < 120:
        return False
    # condition (1)
    if d[AVG_RATING] > 4.0 and d[CHARACTER_NUM] > 35 \
            and d[SCREENPLAY_WRITER_NUM] > 3 and d[SEASONS] > 2:
        return True
    # condition (2)
    if (d[THEME_SINGER] in ('Lisa', 'Sawano Hiroyuki')) and d[STYLE] in ('運動', '戰鬥') \
            and d[AVG_RATING] > 3.8:
        return True
    return False


def is_refreshing(d: Dict[str, Union[int, float]]):
    """_summary_
    Args:
        d (_type_): _description_

    Returns:
        _type_: _description_
    ============================
    - 小品：
    不符合鉅作規定者中，
    IN_SEIYUU_RANK = True, STUDIO_STAFF_NUM >= 50
    且滿足至少以下其中一個條件

    (1)
    AVG_RATING > 3.5
    CHARACTER_NUM > 20
    SEASONS > 1
    AUTHOR = 1 or 2  一、二號作家特別喜歡寫清新小品

    (2)
    THEME_SINGER = 'Hoshino Gen', 'Kenshi Yonezu', 'Mukai Taichi',  'Chico with HoneyWorks'
    STYLE = '致鬱', '奇幻', '搞笑', '懸疑', '戀愛'
    AVG_RATING > 3.8
    AVG_EP_LENGTH > 23
    """
    if d[IN_SEIYUU_RANK] == 0 or d[STUDIO_STAFF_NUM] < 50:
        return False
    # condition (1)
    if d[AVG_RATING] > 3.5 and d[CHARACTER_NUM] > 20 \
            and d[SEASONS] > 1 and d[AUTHOR] in [1, 2]:
        return True
    # condition (2)
    if d[THEME_SINGER] in ['Hoshino Gen', 'Kenshi Yonezu', 'Mukai Taichi', 'Chico with HoneyWorks'] \
        and d[STYLE] in ['致鬱', '奇幻', '搞笑', '懸疑', '戀愛'] \
            and d[AVG_RATING] > 3.8 and d[AVG_EP_LENGTH] > 23:
        return True
    return False


def gen_dataset(N: int,
                messup_ratio: float = 0.05):
    data_rows = []
    print('========== Gen Dataset ============= ')
    for i in range(N):
        data = _gen_data(attributes=ATTRIBUTES)
        data_rows.append(data)
    messup_count = int(N * messup_ratio)
    messup_ids = np.random.choice(range(N), messup_count)
    for xid in messup_ids:
        data_rows[xid][LABEL] = np.random.choice(
            [x for x in Classes if x != data_rows[xid][LABEL]])
    print(f'Hyperprameters: data count: {N}')
    print(f'Messup data count: {len(messup_ids)}')
    df = pd.DataFrame(data_rows)
    return df

# %%


named_tuple = time.localtime()  # get struct_time
time_suffix = time.strftime("%m-%d-%Y-%H", named_tuple)

INPUTDIR = 'input'
os.makedirs(INPUTDIR, exist_ok=True)

# The gold-rule dataset
tweak_ratio = 0
N = 10000
abs_df = gen_dataset(N=N, messup_ratio=0)
abs_df.to_csv(os.path.join(
    INPUTDIR, f'anime_dataset_{N}-{tweak_ratio}.csv'), index=False)
print(abs_df['class'].value_counts())
# Delibrately messing up the gold-rule category
tweak_ratio = 0.05
N = 10000
tweaked_df = gen_dataset(N=N, messup_ratio=0.05)
tweaked_df.to_csv(os.path.join(
    INPUTDIR, f'anime_dataset_{N}-{tweak_ratio}.csv'), index=False)
print(tweaked_df['class'].value_counts())
print('==================================')
print('Output Success.')

# %%
