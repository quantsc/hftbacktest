import numpy as np
from hftbacktest.data.utils import databento
from hftbacktest.data.utils import create_last_snapshot
import faulthandler
import signal
import os 
base_path = os.getcwd()
# path = os.path.join(base_path, 'processed/GOOG/dbeq-basic-20231214.mbp-10.parquet')
# data = databento.convert(path)
# np.savez('test_dbn', data=data)
# Print the shape
# print(data.shape)
# Set numba NUMBA_DEBUG environment variable to 1 to get more information
# os.environ['NUMBA_DEBUG'] = '1'
# os.environ['NUMBA_OPT'] = '0'

data = create_last_snapshot(
    data=np.load(os.path.join(base_path, 'test_dbn.npz'))['data'],
    output_snapshot_filename="snapshot.npz", 
    compress=True,
     tick_size=0.01, lot_size=1)


# create_last_snapshot(
#     '/Users/coopergamble/Desktop/usc/clubs/quant/hftbacktest/test_dbn.npz',
#     tick_size=0.01,
#     lot_size=0.001,
#     initial_snapshot='btcusdt_20230404_eod.npz',
#     output_snapshot_filename='btcusdt_20230405_eod'
# )
