import pandas as pd 
import numpy as np 

def edgelist2snapshots(filepath, g):
    """Convert a edgelist with timestamps into g snapshots.

    Arguments:
        filepath (str): the filepath of the edgelist
        g (int): the number of graph snapshots
    """
    df = pd.read_csv(filepath, sep='\s+', header=None, names=['from', 'to', 'weight', 'timestamp'], dtype={'from':'str', 'to':'str', 'weight':np.float64, 'timestamp':int})
    ecount = len(df.index)
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()
    interval = (end_time - start_time) / g
    
    snapshots = dict()
    for i in range(g):
        snapshots[i] = []

    dfs = df.sort_values(by=['timestamp'])
    for i in range(ecount):
        idx = np.floor((dfs.iloc[i]['timestamp'] - start_time) / interval)
        if int(idx) == g:
            snapshots[g - 1].append(i)
        else:
            snapshots[int(idx)].append(i)

    filename = filepath.split('/')[1].split('.')[0]
    for i in range(g):
        dfs.iloc[snapshots[i]].to_csv(f'datasets/temporal-{filename}/{filename}-{i}.txt', sep=' ', header=False, index=False)