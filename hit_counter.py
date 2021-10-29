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

marocdata.fix_p1(debug=False)
print("processing fname: {}".format(input_dat))

sigma = int(sys.argv[2])

pedestals = marocdata.pedestals_tot
noise_tot = marocdata.noise_tot(sigma)


def over_threshold_per_board(marocdata, pedestals, noise_tot):
    over_threshold_per_board = {}
    for bid in marocdata.active_boards:
        timestamps = []
        board = marocdata.get_board(bid)
        for eid, signal in board.signals.items():
            if np.any((signal - pedestals[bid]) > noise_tot[bid]):
                event = board.get_event(eid)
                timestamps.append(event.TS_norm)
        over_threshold_per_board[bid] = timestamps
    return over_threshold_per_board


ts_over_threshold = over_threshold_per_board(marocdata, pedestals, noise_tot)

all_ts = reduce(add, ts_over_threshold.values())

ts_to_plot = [ts for ts, occ in Counter(all_ts).items() if occ >= 3]

out_dir = sys.argv[3]

counts_per_board = {bid: len(tss) for bid, tss in ts_over_threshold.items()}
out_dir = sys.argv[3]

outfile = out_dir + "/" + input_dat.split(".dat")[0].split("/")[-1] + "_counts.json"
if not os.path.exists(outfile):
    os.system("touch {}".format(outfile))

d = {str(k): value for k, value in counts_per_board.items()}
with open(outfile, "r+") as file:
    json.dump(d, open(outfile, "w"))
