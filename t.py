import numpy as np
from hftbacktest.data.utils import databento
from hftbacktest.data.utils import create_last_snapshot
import os

base_path = os.getcwd()
snapshot = os.path.join(base_path, "processed/GOOG/dbeq-basic-20231215.mbp-10.parquet")
databento.convert(snapshot, output_filename="mbp/GOOG_20231215_small", limit=20000)
# create_last_snapshot(
#     data=np.load(os.path.join(base_path, "mbp/GOOG_20231214.npz"))["data"],
#     output_snapshot_filename="mbp/GOOG_20231214_S.npz",
#     compress=True,
#     tick_size=0.01,
#     lot_size=1,
# )
# new = os.path.join(base_path, "processed/GOOG/dbeq-basic-20231215.mbp-10.parquet")
# databento.convert(new, output_filename="mbp/GOOG_20231215")
