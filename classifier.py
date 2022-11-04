from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance
import pydotplus
import numpy as np
import pandas as pd
from typing import Final, Dict, Union, Tuple
import matplotlib.pyplot as plt
import os

OUTDIR = 'output/'
SINGER_MAP: Final[Dict] = {0: 'Lisa',
                           1: 'Sawano Hiroyuki',
                           2: 'Kenshi Yonezu',
                           3: 'Chico with HoneyWorks',
                           4: 'Eve',
                           5: 'Hoshino Gen',
                           6: 'Mukai Taichi'}

STYLE_MAP: Final[Dict] = {0: '搞笑', 1: '戀愛',
                          2: '致鬱', 3: '懸疑', 4: '奇幻', 5: '戰鬥',  6: '運動'}
CLASS_MAP: Final[Dict] = {0: 'bland', 1: 'refreshing', 2: 'masterpiece'}
colors = ('lightcyan1', 'lightblue1', 'lightskyblue', 'white')
MIN_SAMEPLES_LEAF = 50
MAX_DEPTH = 5

# https://stackoverflow.com/questions/43214350/color-of-the-node-of-tree-with-graphviz-using-class-names
# black box issue: https://stackoverflow.com/questions/71867657/an-empty-block-always-appears-on-my-decision-tree-created-by-python


def inverse_map(d: Dict) -> Dict:
    return {v: k for k, v in d.items()}


def readin(filename:
           str = 'anime_dataset_10000-0.csv') -> Tuple:
    """Reading in anime dataset

    Args:
        filename (str, optional): Dataset name in form f'anime_dataset_{datacount}-{tweak_ratio}'.
        datacount: number of data points
        tweak_ratio: ratio of tweaked data (with randomly flipped class)
        Defaults to 'anime_dataset_10000-0.csv'.

    Returns:
        Tuple: (train/test data), class names
    """
    INPUTDIR = 'input/'

    df = pd.read_csv(os.path.join(INPUTDIR, filename))
    print(f'Reading {filename}...')
    print(f"class counts:{df['class'].value_counts()}")
    # encoding string features to integers
    df['theme_singer'] = df['theme_singer'].apply(
        lambda x: inverse_map(SINGER_MAP)[x])
    df['style'] = df['style'].apply(lambda x: inverse_map(STYLE_MAP)[x])
    df['class'] = df['class'].apply(lambda x: inverse_map(CLASS_MAP)[x])
    feature_names = df.columns[:-1]
    # normalizing features to [0, 1]?
    anime_X = df.iloc[:, :-1].values
    anime_y = df.iloc[:, -1].values
    train_X, test_X, train_y, test_y = train_test_split(
        anime_X, anime_y, test_size=0.2)

    return train_X, test_X, train_y, test_y, feature_names


def save_dt(clf, graph, nodes, tratio):
    for node in nodes:
        try:
            values = clf.tree_.value[int(node.get_name())][0]
            # color only nodes where only one class is present
            if max(values) == sum(values):
                node.set_fillcolor(colors[np.argmax(values)])
            # mixed nodes get the default color
            else:
                node.set_fillcolor(colors[-1])
        except:
            pass  # print(f'invalid node name {node.get_name()}')
    graph.write_png(f'{OUTDIR}tree_{tratio}.png')


def save_cfm(clf, tratio, test_X, test_y):
    cm = confusion_matrix(test_y,
                          clf.predict(test_X),
                          labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=clf.classes_)
    disp.plot()
    plt.savefig(f'{OUTDIR}cfm_{tratio}.png')


# Decision Tree
def dt_classify(tratio=0.05):
    print('=========== Decision Tree ============')
    filename = f'anime_dataset_10000-{tratio}.csv'
    train_X, test_X, train_y, test_y, feature_names = readin(filename)
    dt_clf = tree.DecisionTreeClassifier(min_samples_leaf=MIN_SAMEPLES_LEAF,
                                         max_depth=MAX_DEPTH)
    dt_clf = dt_clf.fit(train_X, train_y)

    dot_data = tree.export_graphviz(dt_clf,
                                    feature_names=feature_names,
                                    out_file=None,
                                    filled=True,
                                    rounded=True)

    graph = pydotplus.graph_from_dot_data(dot_data)
    nodes = graph.get_node_list()
    dt_score = dt_clf.score(test_X, test_y)
    print(f'Score: {dt_score:.3f}')
    feature_analysis(dt_clf, test_X, test_y, feature_names)
    save_dt(dt_clf, graph, nodes, tratio)
    save_cfm(dt_clf, tratio, test_X=test_X, test_y=test_y)


# Naive Bayes
def bayes_classify(tratio):

    filename = f'anime_dataset_10000-{tratio}.csv'
    train_X, test_X, train_y, test_y, feature_names = readin(filename)
    bayes_clf = GaussianNB()
    bayes_clf = bayes_clf.fit(train_X, train_y)
    bayes_score = bayes_clf.score(test_X, test_y)
    print(f'Score: {bayes_score:.3f}')
    save_cfm(bayes_clf, tratio, test_X=test_X, test_y=test_y)
    feature_analysis(bayes_clf, test_X, test_y, feature_names)


def feature_analysis(model, X, y, feature_names):
    r = permutation_importance(model, X, y,
                               n_repeats=30,
                               random_state=0)

    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{feature_names[i]:<8}\t"
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")
