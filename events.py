from dataclasses import dataclass, InitVar, field
from typing import List
from numpy.typing import ArrayLike
import numpy as np


@dataclass
class Event:
    OFFSET = [(l, h) for l, h in zip(range(0, 384, 64), range(64, 384, 64))]

    signal: ArrayLike = field(init=False)  # 320 (64 block)

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
