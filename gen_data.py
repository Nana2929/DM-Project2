#%%
import pandas as pd
from typing import Dict, List
import numpy as np
import os
import time


AUTHOR = 'author'
THEME_SINGER = 'theme_singer'
ANIME_STUDIO = 'anime_studio'
IS_ONGOING = 'is_ongoing'
IN_SEIYUU_RANK = 'in_seiyuu_rank'
SEASONS = '#seasons'
MAIN_SEIYUU = 'main_seiyuu'
STYLE = 'style'
SCREENPLAY_WRITER = 'screenplay_writer'

MASTERPIECE = 'masterpiece'
REFRESHING = 'refreshing'
BLAND = 'bland'
Classes = (MASTERPIECE, REFRESHING, BLAND)
# Hyperparameters
LisaProb = 0.2
SeiyuuRankProbs = [0.4, 0.6]


LABEL = 'class'
ATTRIBUTES = {
    AUTHOR: [hash(x) for x in range(10)],
    THEME_SINGER: ['Lisa', 'Hoshino Gen', 'Kenshi Yonezu', 'Mukai Taichi', 'Sawano Hiroyuki', 'Eve', 'Chico with HoneyWorks'],
    ANIME_STUDIO: ['KyotoAni', 'ufotable', 'Madhouse', 'Mappa', 'Production I.G.', 'Seven Arcs'],
    IS_ONGOING: [True, False],
    IN_SEIYUU_RANK: [True, False],
    SEASONS: range(1, 5+1),
    MAIN_SEIYUU: ['花澤香菜', '石川界人', '早見沙織', '村瀬歩', '櫻井孝宏', '宮野真守'],
    STYLE: ['致鬱', '奇幻', '搞笑', '懸疑', '戀愛', '運動', '戰鬥'],
    SCREENPLAY_WRITER: ['虛淵玄', '花田十輝', '橋本潤', '浦澤義雄', '橫手美智子', '沖方丁']}


def _get_theme_singer_prob(LisaProb, singer_count):
    other_singer_count = singer_count - 1
    return [LisaProb] + [(1-LisaProb)/(other_singer_count)]*(other_singer_count)


def _gen_data(attributes: Dict[str, List[str]]):
    data_row = {}
    for attr in attributes:
        candidates = attributes[attr]
        rand = int(np.random.uniform(0, len(candidates)))
        select_attr = candidates[rand]
        if attr == IN_SEIYUU_RANK:
            select_attr = np.random.choice(candidates, p=SeiyuuRankProbs)
        elif attr == THEME_SINGER:
            select_attr = np.random.choice(candidates,
                                           p=_get_theme_singer_prob(LisaProb, len(candidates)))
        elif attr == SEASONS:
            # makes #seasons longer
            n = len(candidates)
            q = 2 / (n * (n + 1))
            select_attr = np.random.choice(
                candidates, p=[i*q for i in range(1, n+1)])
        data_row[attr] = select_attr

    mp = is_masterpiece(data_row)
    rf = is_refreshing(data_row)
    if mp and rf:
        # print('Invalid data (detect conflicts).')
        return
    if not (mp or rf):
        data_row[LABEL] = BLAND
    elif mp:
        data_row[LABEL] = MASTERPIECE
    else:
        data_row[LABEL] = REFRESHING

    return data_row


def is_masterpiece(d):
    if not d[IN_SEIYUU_RANK]:
        return False
    #  ======= MASTERPIECE ==========

    if d[SEASONS] >= 2:
        if d[SCREENPLAY_WRITER] == '虛淵玄' and d[STYLE] == '致鬱':
            return True
        if d[ANIME_STUDIO] == 'Mappa':
            if d[STYLE] == '戰鬥' and d[THEME_SINGER] == 'Lisa':
                return True
            elif d[STYLE] in ['戰鬥', '奇幻'] and d[THEME_SINGER] in ['Sawano Hiroyuki', 'Eve']:
                return True
        elif d[ANIME_STUDIO] == 'Production I.G.' and d[STYLE] == '運動':
            return True
        elif d[ANIME_STUDIO] == 'ufotable' and d[STYLE] == '奇幻':
            return True
    return False


def is_refreshing(d):
    if not d[IN_SEIYUU_RANK]:
        return False
    if d[ANIME_STUDIO] == 'KyotoAni':
        return True
    elif d[THEME_SINGER] == 'Chico with HoneyWorks' and d[STYLE] == '戀愛' and d[MAIN_SEIYUU] == '花澤香菜':
        return True
    elif d[THEME_SINGER] == 'Hoshino Gen' and d[STYLE] == '搞笑':
        return True
    elif d[THEME_SINGER] == 'Mukai Taichi' and d[STYLE] == '運動':
        return True
    return False


def gen_dataset(N: int,
                messup_ratio: float = 0.05):
    data_rows = []
    valid_count = 0
    print('========== Gen Dataset ============= ')
    for i in range(N):
        data = _gen_data(attributes=ATTRIBUTES)
        if data:
            data_rows.append(data)
            valid_count += 1
    messup_count = int(valid_count * messup_ratio)
    global messup_ids
    messup_ids = np.random.choice(range(valid_count), messup_count)
    for xid in messup_ids:
        data_rows[xid][LABEL] = np.random.choice(
            [x for x in Classes if x != data_rows[xid][LABEL]])
    print(f'Hyperprameters: {N}')
    print(f'Valid data count: {valid_count}/{N}')
    print(f'Messup data count: {len(messup_ids)}')

    df = pd.DataFrame(data_rows)
    return df

#%%
# Delibrately messing up the gold-rule category

named_tuple = time.localtime() # get struct_time
time_suffix = time.strftime("%m-%d-%Y-%H", named_tuple)

INPUTDIR = 'input'
os.makedirs(INPUTDIR, exist_ok=True)

tweak_ratio = 0
abs_df = gen_dataset(N = 1000, messup_ratio=tweak_ratio)
abs_df.to_csv(os.path.join(INPUTDIR, f'anime_data_{tweak_ratio}_{time_suffix}.csv'), index=False)

tweak_ratio = 0.05
tweaked_df = gen_dataset(N = 1000, messup_ratio=tweak_ratio)
tweaked_df.to_csv(os.path.join(INPUTDIR, f'anime_data_{tweak_ratio}_{time_suffix}.csv'), index=False)
print('Output Success.')
# %%
