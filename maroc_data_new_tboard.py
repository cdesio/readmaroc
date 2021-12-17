import numpy as np
from numpy.lib import add_docstring
from maroc_ipaddress import BOARDS
import matplotlib.pyplot as plt

from dataclasses import dataclass
from dataclasses import InitVar, field
from typing import List
from scipy.stats import norm
from typing import Dict

STEP_HEADER = 39
STEP_DATA = 320
MAX_UINT32 = np.iinfo(np.uint32).max
marocs = [(i, j) for i, j in zip(np.arange(0, 384, 64), np.arange(0, 384, 64)[1:])]


@dataclass
class Event:
    metadata: InitVar[List[int]]
    evt_data: InitVar[List[int]]
    evt_id: int = field(init=False)
    TS: int = field(init=False)
    TS_fine_n: List[int] = field(init=False)
    TS_fine: int = field(init=False)
    orcounts_n: List[int] = field(init=False)
    orcounts: int = field(init=False)
    trg_ADC_n: int = field(init=False)
    trg_ADC: List[int] = field(init=False)
    timestamp_ADC_n: int = field(init=False)
    timestamp_ADC: List[int] = field(init=False)
    ADC_samples: List[int] = field(init=False)
    signal: List[int] = field(init=False)
    TS_norm: int = field(init=False)

    def __post_init__(self, metadata, evt_data):
        if metadata is not None:
            self.evt_id = metadata[1]
            self.TS = metadata[2]
            self.TS_fine_n = metadata[3]
            self.TS_fine = metadata[4:14]
            self.orcounts_n = metadata[14]
            self.orcounts = metadata[15:25]
            self.trg_ADC_n = metadata[25]
            self.trg_ADC = metadata[26:31]
            self.timestamp_ADC_n = metadata[31]
            self.timestamp_ADC = metadata[32:37]
            self.ADC_samples = metadata[37:39]
            self.TS_norm = None
        if evt_data is not None:
            self.signal = np.asarray(evt_data, dtype=int)

    def __str__(self) -> str:
        return f"Event({self.evt_id})"

    def __repr__(self) -> str:
        return str(self)


class Board:
    def __init__(self, bid: int, ipv4: int, dec: int):
        self.ip = ipv4
        self.bid = bid
        self.dec = dec
        self.events = {}
        self._pedestal_1st = None
        self._pedestal_2nd = None
        self._sign_ped_corr = None
        self._pedestal = None
        self._noise_1st = None
        self._noise_2nd = None
        self._noise_final = None
        self._noise = None
        self._signals = None
        self._ref_evtid = None
        self._second_evt_id = None
        self._timestamps = None

    def add(self, event: Event):
        self.events[event.evt_id] = event
        self._pedestal_1st = None
        self._pedestal_2nd = None
        self._noise = None
        self._noise_1st = None
        self._noise_2nd = None
        self._noise_final = None
        self._pedestal = None
        self._sign_ped_corr = None
        self._signals = None
        # self._timestamps = None

    def get_event(self, evt_id):
        return self.events.get(evt_id, None)

    @property
    def tot_events(self):
        return len(self.events)

    def __contains__(self, evt_id):
        return evt_id in self.events

    @property
    def event_ids(self):
        return np.asarray(list(self.events.keys()))

    @property
    def signals(self):
        if self._signals is None:
            self._signals = {
                # eid: self.do_common_mode(self.reorder_signal(evt.signal), marocs)
                eid: self.reorder_signal(self.fix_doubling(evt.signal))
                for eid, evt in self.events.items()
            }
        return self._signals

    @staticmethod
    def overflow_idx(ts):
        indices = []
        for i, (t, tp1) in enumerate(zip(ts, ts[1:])):
            if tp1 < t:
                indices.append(i + 1)
        indices.append(len(ts))
        return indices

    # @property
    # def timestamps(self):
    #     if self._timestamps is None:
    #         ts_eid_map = {evt.TS: eid for eid, evt in list(self.events.items())}

    #         # if self._ref_evtid>self._second_evt_id:

    #         ts = list(ts_eid_map.keys())
    #         ids = list(ts_eid_map.values())
    #         if ids[0] > ids[1]:
    #             self._ref_evtid = ids[1]
    #             ts_64 = np.asarray(ts[self._ref_evtid :], dtype=np.uint64)
    #             clean_ids = list(ts_eid_map.values())[1:]

    #         else:
    #             self._ref_evtid = ids[0]
    #             ts_64 = np.asarray(ts, dtype=np.uint64)
    #             clean_ids = ids
    #         overflow = self.overflow_idx(ts)
    #         restart = True if len(overflow) > 2 else False
    #         if restart:
    #             ts_0 = ts_64[0]
    #             ts_64[: overflow[0]] -= ts_0
    #             for i, (start, stop) in enumerate(zip(overflow, overflow[1:])):
    #                 # print(start)
    #                 ts_restart = ts_64[start]
    #                 ts_64[start:stop] += MAX_UINT32 * (i + 1)
    #                 ts_64[start:stop] -= ts_restart
    #         else:
    #             ts_0 = ts_64[0]
    #             ts_64 -= ts_0
    #         self._timestamps = {eid: ts_mod for eid, ts_mod in zip(clean_ids, ts_64)}

    #         for eid, ets in self._timestamps.items():
    #             self.set_tsnorm(eid, ets)
    #     return self._timestamps

    @property
    def timestamps(self):
        if self._timestamps is None:
            ts_eid_map = {evt.TS: eid for eid, evt in list(self.events.items())}
            # if self._ref_evtid>self._second_evt_id:
            ts = list(ts_eid_map.keys())
            ids = list(ts_eid_map.values())
            if ids[0] > ids[1]:
                self._ref_evtid = ids[1]
                ts_64 = np.asarray(ts[self._ref_evtid :], dtype=np.uint64)
                clean_ids = list(ts_eid_map.values())[1:]

            else:
                self._ref_evtid = ids[0]
                ts_64 = np.asarray(ts, dtype=np.uint64)
                clean_ids = ids
            ts_0 = ts_64[0]
            ts_64 -= ts_0
            self._timestamps = {eid: ts_mod for eid, ts_mod in zip(clean_ids, ts_64)}

            for eid, ets in self._timestamps.items():
                self.set_tsnorm(eid, ets)
        return self._timestamps

    @property
    def clean_timestamps(self):
        return {self.events[eid].TS_norm: eid for eid in self.timestamps}

    @clean_timestamps.setter
    def clean_timestamps(self, tss: Dict[int, int]) -> None:
        for eid, ets in tss.items():
            self.set_tsnorm(eid, ets)  # update event object

    @property
    def reference_timestamp(self) -> int:
        return self.clean_timestamps[self._ref_evtid]

    @property
    def reference_event(self) -> Event:
        if self._ref_evtid is None:
            _ = self.timestamps
        return self.events[self._ref_evtid]

    def set_tsnorm(self, evtid: int, ts: int) -> None:
        event = self.events.get(evtid, None)
        if event is not None:
            event.TS_norm = ts

    @staticmethod
    def reorder_marocs(signal, marocs, reorder_map=[0, 3, 1, -2, -2]) -> np.ndarray:
        reordered_array = np.zeros(signal.shape[0])

        for m, (a, b) in enumerate(marocs):
            reordered_array[a + reorder_map[m] * 64 : b + reorder_map[m] * 64] = signal[
                a:b
            ]
        return reordered_array

    def reorder_signal(self, signal):
        signal_to_correct = np.copy(signal)
        reordered_signal = self.reorder_marocs(signal_to_correct, marocs)
        return reordered_signal

    @staticmethod
    def fix_doubling(signal, common_mode_threshold=400):
        signal_to_correct = np.copy(signal)
        for _, (l, h) in enumerate(marocs):
            mu = np.mean(signal[l:h])
            std = np.std(signal[l:h])
            # over = np.where(signal[l:h]>=mu+3*std)[0]
            under = np.where(signal[l:h] <= mu + 3 * std)[0]
            mean2 = np.mean(signal[l:h][under])
            # print(mu, mean2)
            if mean2 > common_mode_threshold:
                signal_to_correct[l:h] = signal[l:h] / 2.0
        return signal_to_correct

    @staticmethod
    def do_common_mode(signal, noise):
        signal_out = np.copy(signal)
        for i, (l, h) in enumerate(marocs):
            mu = np.mean(signal[l:h])
            std = np.std(signal[l:h])
            # print(i, mu, std)
            over = np.where(signal[l:h] >= mu)[0]
            under = np.where(signal[l:h] < mu)[0]
            if over.shape[0] == 0:
                common_mode = 0
                raise ("No strips available for common mode correction. Skip")
            common_mode = np.mean(signal[l:h][under])
            signal_out[l:h] = signal[l:h] - common_mode
        return signal_out

    @staticmethod
    def check_faulty_ribbon(signal, delta_signal=100, n_oscillations=30):
        counts = np.zeros(5)
        for k, (mi, mj) in enumerate(marocs):
            count_maroc = 0
            diff = []
            for (i, si), (j, sj) in zip(
                enumerate(signal[mi:mj]), enumerate(signal[mi:mj][1:])
            ):
                if np.abs(sj - si) > delta_signal:
                    diff.append(np.abs(sj - si))
                    # print(k, np.abs(sj - si))
                    count_maroc += 1
                    # print(count_maroc)
                    # print(i, j, si, sj)
            if count_maroc > n_oscillations:
                counts[k] = 1
                print("maroc {}, no. oscillations: {}".format(k, count_maroc))
                print(counts, np.mean(diff))
        if np.any(counts == 1):
            # print('counts==1')
            return True
        else:
            # print('false')
            return False

    def musigma_1st(self):
        if self._pedestal_1st is None and self._noise_1st is None:
            self._pedestal_1st = np.mean(
                np.asarray(list(self.signals.values())), axis=0
            )
            self._noise_1st = np.std(np.asarray(list(self.signals.values())), axis=0)
        return self._pedestal_1st, self._noise_1st

    @property
    def pedestal_1st(self):
        self._pedestal_1st, _ = self.musigma_1st()
        return self._pedestal_1st

    @property
    def noise_1st(self):
        _, self._noise_1st = self.musigma_1st()
        return self._noise_1st

    def musigma_2nd(self, sigma=4):
        if self._pedestal_2nd is None and self._noise_2nd is None:
            good = []
            # print(self.bid, len(self.signals.values()))
            for sig in self.signals.values():

                if (
                    np.any((sig - self.pedestal_1st) > self._noise_1st * sigma)
                    or self.check_faulty_ribbon(sig) == True
                ):
                    continue
                good.append(sig)
            # print(len(good))
            self._pedestal_2nd = np.mean(good, axis=0)
            self._noise_2nd = np.std(good, axis=0)
        return self._pedestal_2nd, self._noise_2nd

    @property
    def pedestal_2nd(self):
        self._pedestal_2nd, _ = self.musigma_2nd()
        return self._pedestal_2nd

    @property
    def noise_2nd(self):
        _, self._noise_2nd = self.musigma_2nd()
        return self._noise_2nd

    def musigma_final(self, sigma=4):
        if self._sign_ped_corr is None and self._noise_final is None:
            good = []
            # print(self.bid, len(self.signals.values()))
            for sig in self.signals.values():

                if (
                    np.any((sig - self.pedestal_2nd) > self.noise_2nd * sigma)
                    or self.check_faulty_ribbon(sig) == True
                ):
                    continue
                good.append(sig)
            # print(len(good))
            good = self.do_common_mode(good - self.pedestal_2nd, self.noise_2nd)
            self._sign_ped_corr = np.mean(good, axis=0)
            self._noise_final = np.std(good, axis=0)
        return self._sign_ped_corr, self._noise_final

    @property
    def sign_ped_corr(self):
        self._sign_ped_corr, _ = self.musigma_final()
        return self._sign_ped_corr

    @property
    def noise_final(self):
        _, self._noise_final = self.musigma_final()
        return self._noise_final

    @property
    def noise(self):
        if self._noise is None:
            noise = 0
            noise = (
                np.sum(
                    list(
                        (
                            map(
                                lambda dat: np.sqrt((dat - self._pedestal_1st) ** 2),
                                np.asarray(list(self.signals.values())),
                            )
                        )
                    ),
                    axis=0,
                )
                / (self.tot_events - 1)
            )
            self._noise = noise
        return self._noise


class MarocData:
    def __init__(self, infile):
        self._data = self._read_data(infile)
        self._boards = self._read_events()

    def _read_data(self, fname):
        with open(fname, "rb") as fp:
            data = np.fromfile(fp, dtype=np.uint32)
        return data

    def _read_events(self):
        boards = {}
        tot_events = int(self._data.shape[0] / (STEP_DATA + STEP_HEADER))
        for i in range(tot_events):
            metadata = self._data[
                i * STEP_HEADER + i * STEP_DATA : (i + 1) * STEP_HEADER + STEP_DATA * i
            ]
            ip_dec = metadata[0]
            boardip = BOARDS[ip_dec]
            event_board = boards.setdefault(
                boardip.bid, Board(boardip.bid, boardip.ipv4, boardip.dec)
            )
            data = self._data[
                (i + 1) * STEP_HEADER
                + STEP_DATA * i : (i + 1) * STEP_HEADER
                + (i + 1) * STEP_DATA
            ]
            event = Event(metadata, data)
            event_board.add(event)
            boards[boardip.bid] = event_board
        return boards

    def get_board(self, bid: int) -> Board:
        return self._boards.get(bid, None)

    @property
    def tot_events(self):
        return sum([b.tot_events for b in self._boards.values()])

    @property
    def max_evt(self):
        return np.max([b.tot_events for b in self._boards.values()])

    @property
    def n_events_per_board(self):
        return np.sort([(b.bid, b.tot_events) for b in self._boards.values()], axis=0)

    @property
    def active_boards(self):
        return np.sort([b.bid for b in self._boards.values()])

    def get_event(self, evt_id):
        sorted_ids = sorted(self._boards.keys())
        return [(bid, self._boards[bid].get_event(evt_id)) for bid in sorted_ids]

    def fix_p1(self, debug=True):
        all_ts_stack = list()
        boards_timestamps = dict()
        for bid in self._boards:
            board_ts = self._boards[bid].clean_timestamps
            boards_timestamps[bid] = board_ts
            all_ts_stack.extend(list(board_ts.keys()))

        all_ts_stack = sorted(set(all_ts_stack))
        ts_to_fix = [
            ts for ts, ts_1 in zip(all_ts_stack, all_ts_stack[1:]) if (ts_1 == ts + 1)
        ]

        for ts in ts_to_fix:
            for bid, board_ts in boards_timestamps.items():
                if ts + 1 in board_ts:
                    evt_id = board_ts[ts + 1]
                    if debug:
                        print(f"[LOG]: FIX {evt_id} in {bid}; from {ts+1} to {ts}")
                    self._boards[bid].set_tsnorm(evt_id, ts)
        return

    def noise_tot(self, sigma=5):
        return {bid: (self.get_board(bid).noise * sigma) for bid in self.active_boards}

    @property
    def musigma(self):
        return {bid: norm.fit(self.get_board(bid).noise) for bid in self.active_boards}

    @property
    def pedestals_tot(self):
        return {bid: self.get_board(bid).avg_data for bid in self.active_boards}

    def check_clean_ts(self):
        delta = []
        boards_to_fix = []
        for board in self.active_boards:
            ts = np.asarray(list(self.get_board(board).timestamps.values()))
            delta.append(ts[1])
        for i, val in enumerate(delta):
            if np.abs(val - np.mean(delta)) > np.std(delta):
                boards_to_fix.append(i + 1)
        count = 0
        reference_board = max(
            self._boards.values(), key=lambda b: len(b.clean_timestamps)
        )
        assert (
            reference_board not in boards_to_fix
        ), "Board to correct is reference board. Abort."
        for board_id in boards_to_fix:  # sorted by BID
            # clean_ts = all_clean_ts[board_id]
            if board_id in self.active_boards:
                board = self._boards[board_id]
                ref_timestamps = [
                    ts for ts, _ in reference_board.clean_timestamps.items()
                ]
                ref_ts = ref_timestamps[1]
                original_TS = [event.TS for _, event in board.events.items()]
                original_evt_id = [evt.evt_id for _, evt in board.events.items()]
                if original_evt_id[0] > original_evt_id[1]:
                    timestamps_to_clean = original_TS[1:]
                    evt_id_to_clean = original_evt_id[1:]
                else:
                    timestamps_to_clean = original_TS
                    evt_id_to_clean = original_evt_id
                second_ts = timestamps_to_clean[0]
                start = second_ts - ref_ts
                cleaned_ts = timestamps_to_clean - start
                clean_timestamp_dict = {
                    evt_id: ts for ts, evt_id in zip(cleaned_ts, evt_id_to_clean)
                }
                board.clean_timestamps = clean_timestamp_dict

                print("Timestamps of board {} have been fixed".format(board_id))
                count += 1
        if count == 0:
            print("Clean imestamps were already ok")
        return
