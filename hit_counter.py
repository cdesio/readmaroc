import sys, os
import numpy as np
from collections import Counter

sys.path.append("../readmaroc")
from maroc_data_new_tboard import MarocData
from operator import add
from functools import reduce
import json

input_dat = sys.argv[1]
marocdata = MarocData(input_dat)

marocdata.fix_p1(debug=True)

offset = sys.argv[2]
thresholds = {
    b: offset + (mu + 5 * std) for b, (mu, std) in marocdata.noise_tot.items()
}
pedestals = marocdata.pedestals_tot

over_threshold_per_board = {}
for bid in marocdata.active_boards:
    timestamps = []
    board = marocdata.get_board(bid)
    for eid, signal in board.signals.items():
        if np.any(signal - pedestals[bid] > thresholds[bid]):
            event = board.get_event(eid)
            timestamps.append(event.TS_norm)
    over_threshold_per_board[bid] = timestamps

    all_ts = reduce(add, over_threshold_per_board.values())

    ts_to_plot = [ts for ts, occ in Counter(all_ts).items() if occ > 1]

counts_per_board = {bid: len(tss) for bid, tss in over_threshold_per_board.items()}

outfile = input_dat.split(".dat")[0] + "_counts_thresh{}.json".format(offset)
if not os.path.exists(outfile):
    os.system("touch {}".format(outfile))

d = {str(k): value for k, value in counts_per_board.items()}
with open(outfile, "r+") as file:
    json.dump(d, open(outfile, "w"))
