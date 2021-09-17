from datetime import time
import numpy as np
from numpy.lib import add_docstring
from maroc_ipaddress import BOARDS
import matplotlib.pyplot as plt

from dataclasses import dataclass
from dataclasses import InitVar, field
from typing import List


STEP_HEADER = 39
STEP_DATA = 320

@dataclass
class Event:
    metadata: InitVar[List[int]]
    signal: List[int]
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

    def __post_init__(self, metadata):
        if (metadata is not None):
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

    def __str__(self) -> str:
        return f"Event({self.evt_id})"

    def __repr__(self) -> str:
        return str(self)


class Board:
    def __init__(self, bid: int, ipv4:int, dec: int):
        self.ip = ipv4
        self.bid = bid
        self.dec = dec
        self.events = {}

    def add(self, event : Event):
        self.events[event.evt_id] = event

    def get_event(self, evt_id):
        return self.events.get(evt_id, None)
    
    @property
    def tot_events(self):
        return len(self.events)

    def __contains__(self, evt_id):
        return evt_id in self.events

    @property
    def event_ids(self):
        return list(self.events.keys())


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
                    i * STEP_HEADER
                    + i * STEP_DATA : (i + 1) * STEP_HEADER
                    + STEP_DATA * i]
            ip_dec = metadata[0]
            boardip = BOARDS[ip_dec]
            event_board = boards.setdefault(boardip.bid, 
                Board(boardip.bid, boardip.ipv4, boardip.dec))
            data = self._data[
                    (i + 1) * STEP_HEADER
                    + STEP_DATA * i : (i + 1) * STEP_HEADER
                    + (i + 1) * STEP_DATA]
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
    def n_events_per_board(self):
        return [(b.board_id, b.tot_events) for b in self._boards.values()]
    @property
    def active_boards(self):
        return [b.board_id for b in self._boards.values()]

    def get_event(self, evt_id):
        sorted_ids = sorted(self._boards.keys())
        return [(bid, self._boards[bid].get_event(evt_id)) for bid in sorted_ids]
        
    




