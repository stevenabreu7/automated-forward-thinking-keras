from autoforwardthinking import AutoForwardThinking
from datasets import Datasets
import keras.backend as K
import os
import pandas as pd


def run_aft(possible_widths, pool_size, max_layers, cand_epochs, final_epochs,
            iteration=''):
    data = Datasets.mnist()
    comp = 0.0

    model = AutoForwardThinking(possible_widths, pool_size, max_layers, data)
    stats = model.train(final_epochs=final_epochs, cand_epochs=cand_epochs,
                        batch_size=128, stopping_comp=comp)

    for key in stats.keys():
        print(key, len(stats[key]))

    # save training stats
    df = pd.DataFrame(stats)
    comments = list(df['comments'])[0]
    df['min_width'] = min(possible_widths)
    df['max_width'] = max(possible_widths)
    df['pool_size'] = pool_size
    df['max_layers'] = max_layers
    df['cand_epochs'] = cand_epochs
    df['final_epochs'] = final_epochs
    df['compensation'] = comp
    layers = comments.split(']')[0].replace('[', '')
    df['n_layers'] = layers.count(',') + 1
    df['layers'] = layers
    comments = comments.split(']')[1].strip(',').strip()
    df['hid_act'] = comments.split(',')[0].strip()
    df['out_act'] = comments.split(',')[1].strip()
    df['optimizer'] = comments.split(',')[-1].replace('prop', '').strip()
    df = df.drop(columns='comments')
    fname = 'aft_min{}_max{}_p{}_maxl{}_canep{}_finep{}_comp{}_{}.csv'.format(
        min(possible_widths), max(possible_widths), pool_size,
        max_layers, cand_epochs, final_epochs,
        str(comp).replace('.', '-'),
        iteration
    )
    while os.path.exists(fname):
        fname = fname.replace('.csv', '_.csv')
    df.to_csv(fname)

    K.clear_session()


possible_widths = list(range(100, 1050, 100))
for i in range(20):
    run_aft(possible_widths, pool_size=8, max_layers=10, cand_epochs=2,
            final_epochs=-3, iteration=i)
