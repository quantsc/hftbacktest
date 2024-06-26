from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..validation import convert_to_struct_arr
from ... import HftBacktest, Linear, ConstantLatency
from ...typing import DataCollection, Data
from ...reader import UNTIL_END_OF_DATA, DEPTH_SNAPSHOT_EVENT
from .databento import BID_FLAG, ASK_FLAG
from tqdm import tqdm

def create_last_snapshot(
        data: DataCollection,
        tick_size: float,
        lot_size: float,
        initial_snapshot: Optional[Data] = None,
        output_snapshot_filename: Optional[str] = None,
        compress: bool = False,
        structured_array: bool = False
) -> NDArray:
    r"""
    Creates a snapshot of the last market depth for the specified data, which can be used as the initial snapshot data
    for subsequent data.

    Args:
         data: Data to be processed to obtain the last market depth snapshot.
         tick_size: Minimum price increment for the given asset.
         lot_size: Minimum order quantity for the given asset.
         initial_snapshot: The initial market depth snapshot.
         output_snapshot_filename: If provided, the snapshot data will be saved to the specified filename in ``npz``
                                   format.
         compress: If this is set to True, the output file will be compressed.
         structured_array: If this is set to True, the output is converted into the new
                           format(currently only Rust impl).

    Returns:
        Snapshot of the last market depth compatible with HftBacktest.
    """
    # Just to reconstruct order book from the given snapshot to the end of the given data.
    hbt = HftBacktest(
        data,
        tick_size,
        lot_size,
        0,
        0,
        ConstantLatency(0, 0),
        Linear,
        snapshot=initial_snapshot
    )
    # Go to the end of the data.
    hbt.goto(UNTIL_END_OF_DATA)
    snapshot = []

    # For bid depth
    with tqdm(total=len(hbt.bid_depth.items()), desc="Processing bid depth") as pbar:
        for bid, qty in sorted(hbt.bid_depth.items(), key=lambda v: -float(v[0])):
            snapshot.append([
                DEPTH_SNAPSHOT_EVENT,
                hbt.last_timestamp,
                -1,
                BID_FLAG,
                float(bid * tick_size),
                float(qty)
            ])
            pbar.update(1)

    # For ask depth
    with tqdm(total=len(hbt.ask_depth.items()), desc="Processing ask depth") as pbar:
        for ask, qty in sorted(hbt.ask_depth.items(), key=lambda v: float(v[0])):
            snapshot.append([
                DEPTH_SNAPSHOT_EVENT,
                hbt.last_timestamp,
                -1,
                ASK_FLAG,
                float(ask * tick_size),
                float(qty)
            ])
            pbar.update(1)


    snapshot = np.asarray(snapshot, np.float64)

    if structured_array:
        snapshot = convert_to_struct_arr(snapshot)

    if output_snapshot_filename is not None:
        if compress:
            np.savez_compressed(output_snapshot_filename, data=snapshot)
        else:
            np.savez(output_snapshot_filename, data=snapshot)

    return snapshot
