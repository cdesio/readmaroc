from datetime import time
import numpy as np
from numpy.lib import add_docstring
from maroc_ipaddress import BOARDS
import matplotlib.pyplot as plt

from dataclasses import dataclass
from dataclasses import InitVar, field
from typing import List
from scipy.stats import norm

STEP_HEADER = 39
STEP_DATA = 320


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
            self._signals = np.asarray([evt.signal for evt in self.events.values()])
        return self._signals

    @property
    def timestamps(self):
        if self._timestamps is None:
            ts_eid_map = {evt.TS: eid for eid, evt in list(self.events.items())[1:]}
            sorted_ts = sorted(ts_eid_map)  # sort ASC by TS value
            self._ref_evtid = ts_eid_map[sorted_ts[0]]
            self._timestamps = {ts_eid_map[ets]: ets for ets in sorted_ts}
            ref_value = self._timestamps[self._ref_evtid]
            for eid, ets in self._timestamps.items():
                tsnorm = ets - ref_value
                self.set_tsnorm(eid, tsnorm)
        return self._timestamps

    @property
    def clean_timestamps(self):
        return {eid: self.events[eid].TS_norm for eid in self.timestamps}

    @clean_timestamps.setter
    def clean_timestamps(self, tss: dict[int, int]) -> None:
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

    @property
    def avg_data(self):
        if self._avg_data is None:
            self._avg_data = np.mean(self.signals, axis=0)
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
                                self.signals,
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

    def get_clean_ts(self):
        reference_board = max(
            self._boards.values(), key=lambda b: len(b.clean_timestamps)
        )
        ref_event = reference_board.reference_event
        # all_clean_ts = {bid: b.clean_timestamps for bid, b in self._boards.items()}
        count = 0
        for i, board_id in enumerate(sorted(self._boards)):  # sorted by BID
            # clean_ts = all_clean_ts[board_id]
            board = self._boards[board_id]
            board_refid = board.reference_event.evt_id
            if board_refid != ref_event.evt_id:
                ref_ts = reference_board.clean_timestamps[board_refid]
                to_fix = board.timestamps[board_refid]
                start = to_fix - ref_ts
                ts_to_clean = {
                    eid: (ets - start) for eid, ets in board.timestamps.items()
                }
                board.clean_timestamps = ts_to_clean
                print("Timestamps of board {} have been fixed".format(board_id))
                count += 1
        if count == 0:
            print("Clean imestamps were already ok")
        return
