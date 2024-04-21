import numpy as np
from hftbacktest.data.utils import databento
from hftbacktest.data.utils import create_last_snapshot

import os 
base_path = os.getcwd()
path = os.path.join(base_path, 'data/mbp.parquet')
data = databento.convert(path)


# Set numba NUMBA_DEBUG environment variable to 1 to get more information
# os.environ['NUMBA_DEBUG'] = '1'
# os.environ['NUMBA_OPT'] = '0'

np.savez('test_dbn', data=data)
data = create_last_snapshot("/Users/danny/projects/hftbacktest/test_dbn.npz", tick_size=0.01, lot_size=0.001)
np.savez(path + 'bn_eod.npz', data=data)

# create_last_snapshot(
#     '/Users/coopergamble/Desktop/usc/clubs/quant/hftbacktest/test_dbn.npz',
#     tick_size=0.01,
#     lot_size=0.001,
#     initial_snapshot='btcusdt_20230404_eod.npz',
#     output_snapshot_filename='btcusdt_20230405_eod'
# )


