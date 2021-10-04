import sys
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

sys.path.append("../readmaroc")
from maroc_data_new_tboard import MarocData
from operator import add
from functools import reduce

import matplotlib.backends.backend_pdf

input_dat = sys.argv[1]
marocdata = MarocData(input_dat)

y_offset = [12000, 10000, 8000, 4000, 2000]

marocdata.fix_p1()


def board_plot(ax, ts, board_id, board_idx, triplet_idx, c="blue"):
    if board_id in marocdata.active_boards:
        board = marocdata.get_board(board_id)
        if ts in board.clean_timestamps.keys():
            evt = board.clean_timestamps[ts]
            if evt in board:
                signal = board.get_event(evt).signal
                if np.max(signal) > 2000:
                    signal = signal * 0.45
                ax.plot(
                    np.arange(0 + (board_idx * 320), 320 + 320 * board_idx),
                    (signal) + y_offset[triplet_idx],
                    color=c,
                    linewidth=1,
                )
                ax.text(
                    (320 + 320 * board_idx - 0 + (board_idx * 320)) / 2 - 150,
                    y_offset[triplet_idx] - 250,
                    "b: {}, TS: {} ".format(board_id, np.uint32(ts)),
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharey=True, sharex=True)
    evt = None
    for i, (triplet_y, triplet_x) in enumerate(
        zip(np.arange(1, 16).reshape(5, 3), np.arange(16, 31).reshape(5, 3))
    ):
        for j, (board_y, board_x) in enumerate(zip(triplet_y, triplet_x)):
            if board_y in marocdata.active_boards:
                yboard = marocdata.get_board(board_y)
                ax1 = board_plot(ax1, ts, board_y, j, i)
            if board_x in marocdata.active_boards:
                xboard = marocdata.get_board(board_x)
                ax2 = board_plot(ax2, ts, board_x, j, i, c="red")
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
    # fig.text(.5, .95, 'Evt {}'.format(evt), size='large')
    plt.ylim(1000, 13000)
    plt.xlim(-10, 970)
    return fig, ax1, ax2


offset = 200
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

outfile = input_dat.split(".dat")[0] + "_output_ts_clean_fixed_p1_thresh_{}.pdf".format(
    offset
)
pdf = matplotlib.backends.backend_pdf.PdfPages(outfile)
for ts in ts_to_plot:
    fig, ax1, ax2 = plot_event_ts_new(ts, marocdata)
    pdf.savefig(fig)
    plt.close(fig)
pdf.close()
plt.close()
