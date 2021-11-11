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
        self._avg_data = None
        self._noise = None
        self._signals = None
        self._ref_evtid = None
        self._second_evt_id = None
        self._timestamps = None

    def add(self, event: Event):
        self.events[event.evt_id] = event
        self._avg_data = None
        self._noise = None
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
                eid: self.correct_signal(evt.signal) for eid, evt in self.events.items()
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

    def correct_signal(self, signal, common_mode_threshold=400):
        signal_to_correct = np.copy(signal)
        maroc_n_strips = 64
        n_marocs = 5
        marocs = [
            (i, j)
            for i, j in zip(
                np.arange(0, maroc_n_strips * (n_marocs + 1), maroc_n_strips),
                np.arange(0, maroc_n_strips * (n_marocs + 1), maroc_n_strips)[1:],
            )
        ]
        for (
            i,
            m,
        ) in enumerate(marocs):
            l, h = m
            maroc_strips = range(l, h)
            if np.mean(signal_to_correct[maroc_strips]) > common_mode_threshold:
                signal_to_correct[maroc_strips] = signal_to_correct[maroc_strips] / 2.0
        reordered_signal = self.reorder_marocs(signal_to_correct, marocs)
        return reordered_signal

    @property
    def avg_data(self):
        if self._avg_data is None:
            self._avg_data = np.mean(np.asarray(list(self.signals.values())), axis=0)
        return self._avg_data

    @property
    def noise(self):
        if self._noise is None:
            noise = 0
            noise = (
                np.sum(
                    list(
                        (
                            map(
                                lambda dat: np.sqrt((dat - self.avg_data) ** 2),
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
            if board in self.active_boards:
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
