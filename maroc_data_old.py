import os
import matplotlib
import numpy as np
import sys

import ipaddress

matplotlib.use("Qt4Agg", force=True)
from matplotlib import pyplot as plt

# from matplotlib.backends.backend_pdf import PdfPages


fname = sys.argv[1]
nevents = int(sys.argv[2])

if __name__ == "__main__":
    with open(fname, "r") as fb:
        dat = np.fromfile(fb, dtype=np.uint32)

    step_head = 39
    step_data = 320
    headers = []
    data = []
    for i in range(0, nevents):
        headers.append(
            dat[i * step_head + i * step_data : (i + 1) * step_head + step_data * i]
        )
        data.append(
            dat[
                (i + 1) * step_head
                + step_data * i : (i + 1) * step_head
                + (i + 1) * step_data
            ]
        )
    headers = np.asarray(headers, dtype=int)
    data = np.asarray(data, dtype=int)

    print(
        "number of active boards in file: {} \n".format(len(np.unique(headers[:, 0])))
    )

    ipaddr = np.load("../analysis_playground/ip_addrs.npz")["ipaddr"]
    dec = np.load("../analysis_playground/ip_addrs.npz")["dec"]

    print("Save signal plots per board")

    # pp = PdfPages("Signal_per_event.pdf")

    for i in range(0, nevents):
        print(ipaddress.IPv4Address(int(headers[:, 0][i])))
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(0, 320), data[i])
        plt.xlabel("strip")
        plt.title(
            "signal event {}, board {}".format(
                i, int(np.where(dec == int(headers[:, 0][i]))[0]) + 1
            )
        )
        plt.ylim(0, 2000)
        plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(0, 320), np.mean(data[:nevents], axis=0))
    plt.xlabel("strip")
    plt.title("avg signal {} events".format(nevents))
    plt.show()

    noise = 0
    for i in range(nevents):
        noise += np.sqrt((data[i] - np.mean(data[:nevents], axis=0)) ** 2)
    noise = noise / float(nevents - 1)

    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(0, 320), noise)
    plt.xlabel("strip")
    plt.title("avg noise {} events".format(nevents))
    plt.show()
