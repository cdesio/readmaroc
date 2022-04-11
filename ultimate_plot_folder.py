import sys, os
import numpy as np
from collections import defaultdict
from maroc import MarocData
import numpy as np
import matplotlib.pyplot as plt

from pipeline import Pipeline, CommonMode, Doubling, Reordering, Pedestal, random_noise

OFFSET = [(l, h) for l, h in zip(range(0, 384, 64), range(64, 384, 64))]


def sn_per_board(marocdata, pipeline, sigma):
    sn_dict = {}
    for bid in marocdata.active_boards:
        print("Processing board {}".format(bid))
        board = marocdata.get_board(bid)
        corrected_signals = board.apply(pipeline)
        noise_arr = random_noise(corrected_signals, sigma=sigma)
        sn_ratio = np.ravel(np.asarray(list(corrected_signals.values()) / noise_arr))
        sn_dict[bid] = sn_ratio
    return sn_dict


if __name__ == "__main__":

    infolder = sys.argv[1]
    sigma = int(sys.argv[2])
    outdir = sys.argv[3]
    print("Reading from {}".format(infolder))
    files_list = [
        os.path.join(infolder, fname)
        for fname in os.listdir(infolder)
        if fname.endswith("dat")
    ]
    out_dict_sn = defaultdict(list)

    for fname in files_list:
        marocdata = MarocData(infolder)

        all_boards = np.arange(1, 31)
        non_active = [b for b in all_boards if b not in marocdata.active_boards]
        print("Boards {} not in this file".format(non_active))

        pipeline = Pipeline(
            [Doubling(), Reordering(), p := Pedestal(sigma=sigma), CommonMode()]
        )

        sn_dict = sn_per_board(marocdata, pipeline, sigma)
        # print(sn_dict)
    for board_id, sn in sn_dict.items():
        out_dict_sn[board_id.extend(sn)]

    outfile_npz = os.path.join(outdir, infolder, "sn.npz")

    if not os.path.exists(outfile_npz):
        open(outfile_npz, "w").close()

    print("Done. Saving to: {}".format(outdir + outfile_npz))

    d = {str(k): value for k, value in sn_dict.items()}
    np.savez_compressed(outfile_npz, out_dict_sn)
