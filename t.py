import numpy as np
from hftbacktest.data.utils import databento
from hftbacktest.data.utils import create_last_snapshot
import faulthandler
import signal
faulthandler.register(signal.SIGUSR1.value, all_threads=True)
import os 
base_path = os.getcwd()
path = os.path.join(base_path, 'data/mbp.parquet')
data = databento.convert(path)
np.savez('test_dbn', data=data)
save_path = os.path.join(base_path, 'test_dbn.npz')

# Set numba NUMBA_DEBUG environment variable to 1 to get more information
# os.environ['NUMBA_DEBUG'] = '1'
# os.environ['NUMBA_OPT'] = '0'

data = create_last_snapshot(save_path, tick_size=0.01, lot_size=1)
np.savez(base_path + '/bn_eod.npz', data=data)

# create_last_snapshot(
#     '/Users/coopergamble/Desktop/usc/clubs/quant/hftbacktest/test_dbn.npz',
#     tick_size=0.01,
#     lot_size=0.001,
#     initial_snapshot='btcusdt_20230404_eod.npz',
#     output_snapshot_filename='btcusdt_20230405_eod'
# )


