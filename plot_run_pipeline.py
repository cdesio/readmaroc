import sys, os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

sys.path.append("../readmaroc")
from maroc import MarocData
from operator import add
from functools import reduce

from pipeline import (
    Pipeline,
    CommonMode,
    Doubling,
    Reordering,
    Pedestal,
    find_hits,
    Fix_overflow,
    MAROC,
    random_noise,
)

import matplotlib.backends.backend_pdf

y_offset = [12000, 10000, 8000, 4000, 2000]


def board_plot(
    ax,
    ts,
    marocdata,
    corrected_signal,
    hits_per_board,
    board_id,
    board_idx,
    triplet_idx,
    c="blue",
    lw=0.75,
):
    if board_id in marocdata.active_boards:
        board = marocdata.get_board(board_id)
        hits = hits_per_board[board_id]
        ref_evt = board.reference_event.evt_id
        if ts in board.clean_timestamps.keys():
            evt = board.clean_timestamps[ts]
            if evt in board:
                signal = corrected_signal[evt]
                if np.max(signal) > 2000:
                    signal = signal * 0.45
                # pedestal = ped[board_id]
                # noise_board = noise[board_id]
                seeds = hits[evt].hit
                lefts = hits[evt].left
                rights = hits[evt].right
                if seeds:
                    seeds = np.asarray(seeds)
                    lefts = np.asarray(lefts)
                    rights = np.asarray(rights)
                    ax.scatter(
                        seeds + (board_idx * 320),
                        signal[seeds] + y_offset[triplet_idx],
                        color="k",
                        marker="s",
                        s=400,
                        facecolors="none",
                        alpha=0.7,
                    )
                    ax.scatter(
                        lefts + (board_idx * 320),
                        signal[seeds] + y_offset[triplet_idx],
                        color="green",
                        marker="x",
                        s=10,
                        facecolors="none",
                        alpha=0.7,
                    )
                    ax.scatter(
                        rights + (board_idx * 320),
                        signal[seeds] + y_offset[triplet_idx],
                        color="red",
                        marker="x",
                        s=10,
                        facecolors="none",
                        alpha=0.7,
                    )
                ax.plot(
                    np.arange(0 + (board_idx * 320), 320 + 320 * board_idx),
                    (signal) + y_offset[triplet_idx],
                    color=c,
                    linewidth=lw,
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
                    "board: {}, evt: {}".format(board_id, evt - ref_evt),
                    size="small",
                )
                ax.axvline(
                    320 * (board_idx + 1),
                    linestyle="--",
                    linewidth=0.75,
                    color="grey",
                    alpha=0.5,
                )
                for _, maroc in MAROC.items():
                    ax.axvline(
                        maroc.left + 320 * board_idx,
                        linewidth=0.5,
                        linestyle="--",
                        c="grey",
                        alpha=0.25,
                    )
    return ax


def plot_event_ts_new(
    ts, marocdata, corrected_signal, hits_per_board, fig=None, ax1=None, ax2=None
):
    print("Plotting TS: {}".format(ts))
    if fig is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10), sharey=True, sharex=True)
    for i, (triplet_y, triplet_x) in enumerate(
        zip(np.arange(1, 16).reshape(5, 3), np.arange(16, 31).reshape(5, 3))
    ):
        for j, (board_y, board_x) in enumerate(zip(triplet_y, triplet_x)):
            if board_y in marocdata.active_boards:
                # yboard = marocdata.get_board(board_y)
                corrected_signal = corrected_signals[board_y]
                ax1 = board_plot(
                    ax1, ts, marocdata, corrected_signal, hits_per_board, board_y, j, i
                )
            if board_x in marocdata.active_boards:
                # xboard = marocdata.get_board(board_x)
                corrected_signal = corrected_signals[board_x]
                ax2 = board_plot(
                    ax2,
                    ts,
                    marocdata,
                    corrected_signal,
                    hits_per_board,
                    board_x,
                    j,
                    i,
                    c="red",
                )
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
    plt.xlim(0, 960)
    return fig, ax1, ax2

    # def take_consecutive(index_list):
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

    # def check_faulty_ribbon(signal, delta_signal=100, n_oscillations=30):
    counts = np.zeros(5)
    for k, (mi, mj) in enumerate(marocs):
        count_maroc = 0
        for (i, si), (j, sj) in zip(
            enumerate(signal[mi:mj]), enumerate(signal[mi:mj][1:])
        ):
            if np.abs(sj - si) > delta_signal:
                count_maroc += 1
                # print(i, j, si, sj)
        if count_maroc > n_oscillations:
            # print("maroc {}, no. oscillations: {}".format(k, count_maroc))
            counts[k] = 1
    # print(counts)
    if np.any(counts == 1):
        return True
    else:
        return False


def over_threshold_per_board(marocdata, corrected_signals):
    ts_over_threshold_per_board = {}
    hits_per_board = {}
    for bid in marocdata.active_boards:
        timestamps = []
        board = marocdata.get_board(bid)
        corrected_signal = corrected_signals[bid]
        noise_final = random_noise(corrected_signal, sigma=4)
        hits = find_hits(corrected_signal, noise=noise_final)
        for eid, _ in corrected_signal.items():
            # if check_faulty_ribbon(signal):
            #    pass
            seeds = hits[eid].hit
            if seeds:
                event = board.get_event(eid)
                timestamps.append(event.TS_norm)
        ts_over_threshold_per_board[bid] = timestamps
        hits_per_board[bid] = hits
    return ts_over_threshold_per_board, hits_per_board


if __name__ == "__main__":

    input_dat = sys.argv[1]
    print("processing fname: {}".format(input_dat))

    marocdata = MarocData(input_dat)

    all_boards = np.arange(1, 31)
    non_active = [b for b in all_boards if b not in marocdata.active_boards]
    print("Boards {} not in this file".format(non_active))

    check_ts = bool(sys.argv[5])
    if check_ts:
        print("checking clean ts")
        marocdata.check_clean_ts()
    marocdata.fix_p1(debug=False)

    # offset = int(sys.argv[2])
    # thresholds = {
    #    b: offset + (mu + 5 * std) for b, (mu, std) in marocdata.noise_tot.items()
    # }
    sigma = int(sys.argv[2])
    # pedestals_tot = marocdata.pedestals_tot
    # noise_tot = marocdata.noise_tot(sigma)
    pipeline = Pipeline(
        [
            Reordering(),
            Fix_overflow(),
            Doubling(),
            p := Pedestal(sigma=sigma),
            CommonMode(),
        ]
    )
    print("Applying pipeline to data...")
    corrected_signals = {}
    for bid in marocdata.active_boards:
        board = marocdata.get_board(bid)
        corrected_signal = board.apply(pipeline)
        corrected_signals[bid] = corrected_signal

    print("Saving timestamps containing hits...")
    ts_over_threshold, hits_per_board = over_threshold_per_board(
        marocdata, corrected_signals
    )
    # print(ts_over_threshold)
    all_ts = reduce(add, ts_over_threshold.values())
    no_hits = int(sys.argv[3])
    ts_to_plot = [
        ts for ts, occ in Counter(all_ts).items() if occ >= no_hits
    ]  # and occ < 11]

    # print(len(ts_to_plot))

    ts_board_hit = {}
    for board_id in marocdata.active_boards:
        ts_hit = {
            ts: (evt, hits_per_board[board_id][evt].hit)
            for ts, evt in marocdata.get_board(board_id).clean_timestamps.items()
            if len(hits_per_board[board_id][evt].hit)
        }
        ts_board_hit[board_id] = ts_hit

    out_dir = os.path.abspath(sys.argv[4])
    print("out_dir:", format(out_dir))
    out_fname_pdf = input_dat.split(".dat")[0].split(os.path.sep)[
        -1
    ] + "_output_ts_clean_fixed_p1_{}sigma_{}hits_test_ts.pdf".format(sigma, no_hits)
    # print(ts_to_plot)
    outfile_pdf = os.path.join(out_dir, out_fname_pdf)
    print("out_fname:{}".format(outfile_pdf))
    if len(ts_to_plot) > 0:
        print("plotting and saving to file")
        pdf = matplotlib.backends.backend_pdf.PdfPages(outfile_pdf)
        for ts in ts_to_plot:
            fig, ax1, ax2 = plot_event_ts_new(
                ts, marocdata, corrected_signals, hits_per_board
            )
            pdf.savefig(fig)
            plt.close(fig)
        pdf.close()
        plt.close()
        print("Done.Closing file.")
    else:
        print("Nothing to plot. Quitting.")
    out_fname_npz = input_dat.split(".dat")[0].split(os.path.sep)[
        -1
    ] + "_ts_hits_per_board_{}sigma_{}hits.npz".format(sigma, no_hits)
    outfile_npz = os.path.join(out_dir, out_fname_npz)

    print(f"Saving hits to output npz file {outfile_npz}")

    np.savez_compressed(
        outfile_npz, hits=ts_board_hit, ts=ts_to_plot, allow_pickle=True
    )
    print("Bye.")
    # counts_per_board = {bid: len(tss) for bid, tss in ts_over_threshold.items()}
