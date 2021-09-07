import numpy as np
from maroc_ipaddress import boards_ip
import matplotlib.pyplot as plt

STEP_HEADER = 39
STEP_DATA = 320


class MarocData:
    def __init__(self, infile):
        self.data = self._read_data(infile)
        self.header, self.marocdata = self._split_data()

    def _read_data(self, fname):
        with open(fname, "r") as fp:
            data = np.fromfile(fp, dtype=np.uint32)
        return data

    def _split_data(self):
        headers = []
        marocdata = []
        for i in range(self.tot_events):
            headers.append(
                self.data[
                    i * STEP_HEADER
                    + i * STEP_DATA : (i + 1) * STEP_HEADER
                    + STEP_DATA * i
                ]
            )
            marocdata.append(
                self.data[
                    (i + 1) * STEP_HEADER
                    + STEP_DATA * i : (i + 1) * STEP_HEADER
                    + (i + 1) * STEP_DATA
                ]
            )
        headers = np.asarray(headers, dtype=int)
        marocdata = np.asarray(marocdata, dtype=int)
        return headers, marocdata

    @property
    def tot_events(self):
        return int(self.data.shape[0] / (STEP_DATA + STEP_HEADER))

    def nevents_per_board(self, board_id):
        if board_id in self.active_boards:
            return len(
                np.where(
                    self.header[:, 0]
                    == int(boards_ip[boards_ip["board_id"] == board_id]["dec"])
                )[0]
            )
        else:
            raise ValueError("Board not in data file")

    @property
    def active_boards(self):
        active_boards = []
        for dec in np.unique(self.header[:, 0]):
            active_boards.append(int(boards_ip[boards_ip["dec"] == dec]["board_id"]))
        return sorted(active_boards)

    @property
    def n_active_boards(self):
        return len(self.active_boards)

    def board_data(self, board_id):
        if board_id in self.active_boards:
            idx = np.where(
                self.header[:, 0]
                == int(boards_ip[boards_ip["board_id"] == board_id]["dec"])
            )[0]
            return self.marocdata[idx]
        else:
            raise ValueError("Board not in data file")
    def header_board(self, board_id):
        if board_id in self.active_boards:
            idx = np.where(
                self.header[:, 0]
                == int(boards_ip[boards_ip["board_id"] == board_id]["dec"])
            )[0]
            return self.header[idx]
        else:
            raise ValueError("Board not in data file")

    def _noise(self, board_id):
        n_events = self.nevents_per_board(board_id)
        board_data = self.board_data(board_id)
        avg_data = self._avg_data(board_id)
        noise = 0
        noise = np.sum(
            list((map(lambda dat: np.sqrt((dat - avg_data) ** 2), board_data))), axis=0
        ) / (n_events - 1)

        return noise

    def _avg_data(self, board_id):
        n_events = self.nevents_per_board(board_id)
        board_data = self.board_data(board_id)

        avg_board_data = np.mean(board_data[:n_events], axis=0)

        return avg_board_data

    def plot_avg_data(self, board_id, ylim=2000):
        avg_data = self._avg_data(board_id)
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(0, 320), avg_data)
        plt.xlabel("strip")
        plt.title(
            "avg signal {} events, board {}".format(
                self.nevents_per_board(board_id), board_id
            )
        )
        plt.ylim(0, ylim)
        plt.show()
        return

    def plot_noise(self, board_id, ylim=500):
        noise = self._noise(board_id)
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(0, STEP_DATA), noise)
        plt.xlabel("strip")
        plt.title(
            "avg noise {} events, board {}".format(
                self.nevents_per_board(board_id), board_id
            )
        )
        plt.ylim(0, ylim)
        plt.show()


class Header:
    def __init__(self, header):
        self.header = header
    @property
    def ip(self):
        return self.header[0]
    @property
    def trigg_n_TS(self):
        return self.header[1]
    @property
    def TS(self):
        return self.header[2]
    @property
    def TS_fine_n(self):
        return self.header[3]
    @property
    def TS_fine(self):
        return self.header[4:14]
    @property
    def orcounts_n(self):
        return self.header[14]
    @property
    def orcounts(self):
        return self.header[15:25]
    @property
    def trg_ADC_n(self):
        return self.header[25]
    @property
    def trg_ADC(self):
        return self.header[26:31]
    @property
    def timestamp_ADC_n(self):
        return self.header[31]
    @property
    def timestamp_ADC(self):
        return self.header[32:37]
    @property
    def ADC_samples(self):
        return self.header[37:39]