import sys, os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import json

sys.path.append("../readmaroc")
from maroc_data_new_tboard import MarocData
from operator import add
from functools import reduce

import matplotlib.backends.backend_pdf

input_dat = sys.argv[1]
marocdata = MarocData(input_dat)

y_offset = [12000, 10000, 8000, 4000, 2000]
marocs = [(i, j) for i, j in zip(np.arange(0, 384, 64), np.arange(0, 384, 64)[1:])]

marocdata.fix_p1(debug=False)
print("processing fname: {}".format(input_dat))


def board_plot(ax, ts, marocdata, board_id, board_idx, triplet_idx, c="blue"):
    if board_id in marocdata.active_boards:
        board = marocdata.get_board(board_id)
        if ts in board.clean_timestamps.keys():
            evt = board.clean_timestamps[ts]
            if evt in board:
                signal = board.signals[evt]
                if np.max(signal) > 2000:
                    signal = signal * 0.45
                pedestal = pedestals_tot[board_id]
                noise = noise_tot[board_id]
                if np.any(signal - pedestal > noise):
                    over = np.where(signal - pedestal > noise)[0]
                    seed = np.max((signal - pedestal)[over])
                    over_x = np.where(signal - pedestal == seed)[0][0]
                    ax.scatter(
                        over_x + (board_idx * 320),
                        seed + y_offset[triplet_idx],
                        color="k",
                        marker="s",
                        s=400,
                        facecolors="none",
                        alpha=0.7,
                    )
                ax.plot(
                    np.arange(0 + (board_idx * 320), 320 + 320 * board_idx),
                    (signal - pedestal) + y_offset[triplet_idx],
                    color=c,
                    linewidth=1,
                )
                #                 ax.plot(
                #                     np.arange(0 + (board_idx * 320), 320 + 320 * board_idx),
                #                     (noise) + y_offset[triplet_idx],
                #                     color='green',
                #                     linewidth=1,
                #                 )
                ax.text(
                    (320 + 320 * board_idx - 0 + (board_idx * 320)) / 2 - 150,
                    y_offset[triplet_idx] - 300,
                    "board: {}".format(board_id),
                    size="small",
                )
                ax.axvline(
                    320 * (board_idx + 1),
                    linestyle="--",
                    linewidth=0.75,
                    color="grey",
                    alpha=0.5,
                )
    return ax


def plot_event_ts_new(ts, marocdata):
    print(ts)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10), sharey=True, sharex=True)
    evt = None
    for i, (triplet_y, triplet_x) in enumerate(
        zip(np.arange(1, 16).reshape(5, 3), np.arange(16, 31).reshape(5, 3))
    ):
        for j, (board_y, board_x) in enumerate(zip(triplet_y, triplet_x)):
            if board_y in marocdata.active_boards:
                yboard = marocdata.get_board(board_y)
                ax1 = board_plot(ax1, ts, marocdata, board_y, j, i)
            if board_x in marocdata.active_boards:
                xboard = marocdata.get_board(board_x)
                ax2 = board_plot(ax2, ts, marocdata, board_x, j, i, c="red")
    #    if (evt is None):
    #        return None # skip
    ax1.set_title("y layers", size="x-large")
    ax2.set_title("x layers", size="x-large")
    plt.yticks(
        y_offset,
        ["layer 0", "layer 1", "layer 2", "layer 3", "layer 4"],
        size="x-large",
    )
    plt.xticks([0, 320, 640, 960])
    fig.text(0.5, 0.05, "strips", size="large")
    fig.text(0.5, 0.95, "TS {}".format(ts), size="large")
    plt.ylim(1000, 13000)
    plt.xlim(-10, 970)
    return fig, ax1, ax2


# offset = int(sys.argv[2])
# thresholds = {
#    b: offset + (mu + 5 * std) for b, (mu, std) in marocdata.noise_tot.items()
# }
sigma = int(sys.argv[2])
pedestals_tot = marocdata.pedestals_tot
noise_tot = marocdata.noise_tot(sigma)


def take_consecutive(index_list):
    if len(index_list) < 3:
        return None
    else:
        consecutive = []
        index_list = np.sort(index_list)
        for el, elp1 in zip(index_list, index_list[1:]):
            if elp1 == el + 1:
                consecutive.append(el)
                consecutive.append(elp1)
        if len(consecutive) == 0:
            return None
        else:
            return np.unique(consecutive)


def over_threshold_per_board(marocdata, pedestals, noise):
    ts_over_threshold_per_board = {}
    for bid in marocdata.active_boards:
        timestamps = []
        board = marocdata.get_board(bid)
        for eid, signal in board.signals.items():
            if np.any((signal - pedestals[bid]) > noise[bid]):
                over = np.sort(np.where((signal - pedestals[bid]) > noise[bid])[0])
                consecutives = take_consecutive(over)
                if consecutives is None:
                    pass
                else:
                    if len(consecutives) <= 40:
                        event = board.get_event(eid)
                        # print(bid, eid, consecutives)
                        timestamps.append(event.TS_norm)
        ts_over_threshold_per_board[bid] = timestamps
    return ts_over_threshold_per_board


""" def over_threshold_per_board(marocdata, pedestals, noise_tot):
    over_threshold_per_board = {}
    for bid in marocdata.active_boards:
        timestamps = []
        board = marocdata.get_board(bid)
        for eid, signal in board.signals.items():
            if np.any((signal - pedestals[bid]) > noise_tot[bid]):
                event = board.get_event(eid)
                timestamps.append(event.TS_norm)
        over_threshold_per_board[bid] = timestamps
    return over_threshold_per_board """


ts_over_threshold = over_threshold_per_board(marocdata, pedestals, noise_tot)

all_ts = reduce(add, ts_over_threshold.values())
no_hits = int(sys.argv[3])
ts_to_plot = [ts for ts, occ in Counter(all_ts).items() if occ >= no_hits]

out_dir = sys.argv[4]

outfile_pdf = (
    out_dir
    + "/"
    + input_dat.split(".dat")[0].split("/")[-1]
    + "_output_ts_clean_fixed_p1_{}sigma_{}hits.pdf".format(sigma, no_hits)
)

pdf = matplotlib.backends.backend_pdf.PdfPages(outfile_pdf)

for ts in ts_to_plot:
    fig, ax1, ax2 = plot_event_ts_new(ts, marocdata)
    pdf.savefig(fig)
    plt.close(fig)
pdf.close()
plt.close()

counts_per_board = {bid: len(tss) for bid, tss in ts_over_threshold.items()}

outfile_json = (
    out_dir + "/" + input_dat.split(".dat")[0].split("/")[-1] + "_counts.json"
)

if not os.path.exists(outfile_json):
    os.system("touch {}".format(outfile_json))

d = {str(k): value for k, value in counts_per_board.items()}
with open(outfile_json, "r+") as file:
    json.dump(d, open(outfile_json, "w"))
