import pandas as pd
from sklearn import model_selection
import math

def kfoldify(X, y, n_splits=5, shuffle=True, random_state=12345):
    X_train, X_test = {}, {}
    y_train, y_test = {}, {}
    kf = model_selection.KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    i=0
    for train_index, test_index in kf.split(X,y):
        print("TRAIN:", len(train_index), "TEST:", len(test_index))
        X_train[i], X_test[i] = X[train_index], X[test_index]
        y_train[i], y_test[i] = y[train_index], y[test_index]
        i +=1


    return X_train, X_test, y_train, y_test


def splitify_detailed(*arrays, test_size=0.2, stratify=None, random_state=None):

    if stratify is not None:
        CVClass = model_selection.StratifiedShuffleSplit
    else:
        CVClass = model_selection.ShuffleSplit

    cv = CVClass(test_size=test_size, random_state=random_state)

    train, test = next(cv.split(X=arrays[0], y=stratify))

    # X_train_idxs, X_test_idxs, y_train_idxs, y_test_idxs = model_selection.train_test_split( list(range(len(X))), y, test_size=0.2, random_state=random_state, stratify=y)

    return train, test


def group_based_stratified_split(df, 
group_by=['current_passage_code', 'UTTERANCE'], 
stratify_key_func=(lambda groupKey, group: groupKey), 
test_size=0.3, random_state=1212 ):
    """Do a stratified split of a dataframe but keeping groups together so their rows
    do not end in different splits

    Parameters
    ----------
    df : Dataframe
        Dataframe to be split
    group_by : list, optional
        List of Dataframe columns to be used to group rows, by default ['current_passage_code', 'UTTERANCE']
    stratify_key_func : [type], optional
        a function that return the value to be used as stratify label, by default use the GroupKey as stratify label
    test_size : float, optional
        [description], by default 0.3
    random_state : int, optional
        [description], by default 1212

    Returns
    -------
    Tuple of Dataframes
        (train_split, test_split)
    """
    rs3 = []
    # generate groups so they do not get splitted during split
    for groupKey, group in df.groupby(group_by):
        #key = str(group[0]) + "-" + passage.iloc[0]['label'][:2]
        key = stratify_key_func(groupKey, group) # key representative of the group for the stratification split
        rs3.append({'groupkey': key, 'group': group})
    rs3 = pd.DataFrame(rs3)
    
    # do split over groups
    rs3 = model_selection.train_test_split(rs3, test_size=test_size, random_state=random_state, stratify=rs3[['groupkey']])
    
    train_split = pd.DataFrame().append([p for p in rs3[0].group])
    test_split = pd.DataFrame().append([p for p in rs3[1].group])
    
    return (train_split, test_split)

def splitify_dataframe(df, splits=[0.8,0.1,0.1], stratify=None, random_state=1212):
    """Extract splits from a dataframe according to the specified split sizes.

    Parameters
    ----------
    df : Dataframe
        Data to be splitted
    splits : list, optional
        Size of desired splits, can be in percentage or absolute numbers, by default [0.8,0.1,0.1]
    stratify : Str|list[str], optional
        Field(s) to stratify the splits, by default None
    random_state : int, optional
        Random state used on the splitting, by default 1212

    Returns
    -------
    list[dataframe]
        resulting splits, len() == len(splits)
    """
    print(splits)
    assert sum(splits) == 1.0 or sum(splits)==len(df), f"Splits whould add to 1.0 or to the size of the dataset [{len(df)}]"
    
    _splits_sizes = [ math.floor(len(df)*ss)  for ss in splits] if sum(splits) == 1.0 else splits # convert from percentage to numbs
    _splits_sizes[0] += (len(df) - sum(_splits_sizes)) # adjust discrepancies due to roundings (add to the first split)    

    # define strats
    _strats = None
    if type(stratify) is str:
        _strats = [stratify]
    if type(stratify) is list:
        _strats = stratify

    rs = []
    _sremaining = df.copy(deep=False)
    for ss in _splits_sizes[:-1]: # last split is computed in the second-last
        _split, _sremaining = model_selection.train_test_split(_sremaining, train_size=ss, stratify= _sremaining[_strats] if _strats else None  )
        rs.append(_split)    
    rs.append(_sremaining)    

    return rs