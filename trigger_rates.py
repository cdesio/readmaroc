import os, sys, time
import numpy as np
from datetime import datetime

folder = sys.argv[1]

filelist = [
    fname
    for fname in sorted(os.listdir(folder))
    if fname.endswith("logfile.txt") and fname.startswith("Run")
]


def get_max_evt_number(folder, fname):
    with open(os.path.join(folder, fname)) as f:
        event_numbers = []
        firstline = f.readline()
        print(fname)
        date = datetime.strptime(firstline, "%d/%m/%Y\t%H:%M\n")
        print(firstline, date)
        for line in f.readlines()[1:]:
            event_numbers.append(int(line.split("\t")[-1].split("\n")[0]))
    return date, np.max(event_numbers)


rates = []
for (i, ts), (j, tsp1) in zip(enumerate(filelist), enumerate(filelist[1:])):
    # delta = (timestamps[tsp1]-timestamps[ts]).seconds
    filetime1, events1 = get_max_evt_number(folder, ts)
    filetime2, events2 = get_max_evt_number(folder, tsp1)
    delta_ft = filetime2 - filetime1
    if delta_ft.seconds > 0:
        rates.append(np.ceil(events1 / delta_ft.seconds))
        # print(filetime1, filetime2, ts, tsp1, delta_ft.seconds, np.ceil(events1/delta_ft.seconds))
print(rates, np.mean(rates))
