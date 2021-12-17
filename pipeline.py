from dataclasses import dataclass
from matplotlib import pyplot as plt
import numpy as np
from typing import Protocol
from typing import Sequence, Dict, List
from numpy.core.numeric import outer
from numpy.lib.ufunclike import fix
from numpy.typing import ArrayLike
from events import Event
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# OFFSET = [(l, h) for l, h in zip(range(0, 384, 64), range(64, 384, 64))]


@dataclass
class Maroc:
    left: int
    right: int

    def __contains__(self, pos: int):
        return pos >= self.left and pos <= self.right

    def __le__(self, a, b):
        return a >= self.left and b <= self.right

    def __lt__(self, a, b):
        return a > self.left and b < self.right

    def __ge__(self, a, b):
        return a <= self.left and b >= self.right

    def __gt__(self, a, b):
        return a < self.left and b > self.right


MAROC = {
    m: Maroc(l, h)
    for m, (l, h) in enumerate(zip(range(0, 384, 64), range(64, 384, 64)))
}


class Operator(Protocol):
    def apply(self, signals: Dict[int, ArrayLike]) -> Dict[int, ArrayLike]:
        pass


def random_noise(signals: Dict[int, ArrayLike], sigma: int = 4) -> ArrayLike:
    all_signals = list()
    for signal in signals.values():
        all_signals.append(signal)
    all_signals = np.vstack(all_signals)
    pedestal = np.mean(all_signals, axis=0)
    noise = np.std(all_signals, axis=0)

    all_signals = list(
        filter(
            lambda s: not (np.any((s - pedestal) > (sigma * noise))),
            all_signals,
        )
    )
    all_signals = np.vstack(all_signals)
    return np.std(all_signals, axis=0)


def check_pedestal(
    signals: Dict[int, ArrayLike], pedestal: ArrayLike, sigma: int = 4
) -> ArrayLike:
    all_signals = list()
    for signal in signals.values():
        all_signals.append(signal)
    all_signals = np.vstack(all_signals)
    return np.mean((all_signals - pedestal), axis=0)


def check_faulty_ribbon(signal, delta_signal=100, n_oscillations=30, output=False):
    counts = np.zeros(5)
    for k, maroc in MAROC.items():

        count_maroc = 0
        diff = []
        for (i, si), (j, sj) in zip(
            enumerate(signal[maroc.left : maroc.right]),
            enumerate(signal[maroc.left : maroc.right][1:]),
        ):
            if np.abs(sj - si) > delta_signal:
                diff.append(np.abs(sj - si))
                # print(k, np.abs(sj - si))
                count_maroc += 1
                # print(count_maroc)
                # print(i, j, si, sj)
        if count_maroc > n_oscillations:
            counts[k] = 1
            if output:
                print("maroc {}, no. oscillations: {}".format(k, count_maroc))
            # print(counts, np.mean(diff))
    return counts

    # class Pedestal:
    def __init__(self, sigma: int = 4):
        self.sigma = sigma
        self.pedestal_ = None
        self.noise_ = None

    def apply(self, signals: Dict[int, ArrayLike]) -> Dict[int, ArrayLike]:
        all_signals = list()
        for i, signal in enumerate(signals.values()):
            all_signals.append(signal)
        all_signals = np.vstack(all_signals)

        for _ in range(2):
            pedestal = np.mean(all_signals, axis=0)
            noise = np.std(all_signals, axis=0)

            all_signals = list(
                filter(
                    lambda signal: not (
                        np.any((signal - pedestal) > (self.sigma * noise))
                    )
                    and np.all(
                        check_faulty_ribbon(signal, delta_signal=20, n_oscillations=30)
                        == 0
                    ),
                    all_signals,
                )
            )
            all_signals = np.vstack(all_signals)
        self.pedestal_ = np.mean(all_signals, axis=0)
        self.noise_ = np.std(all_signals, axis=0)
        fixed_signals = dict()
        for eid, signal in signals.items():
            fixed_signals[eid] = signal - pedestal
        return fixed_signals


class Pedestal:
    def __init__(self, sigma: int = 4):
        self.sigma = sigma
        self.pedestal_ = None
        self.noise_ = None

    def apply(self, signals: Dict[int, ArrayLike]) -> Dict[int, ArrayLike]:
        hits = find_hits(signals)
        all_signals = list()
        for eid, signal in signals.items():
            hit = hits[eid].hit
            if not hit:
                all_signals.append(signal)
        all_signals = np.vstack(all_signals)

        for _ in range(2):
            pedestal = np.mean(all_signals, axis=0)
            noise = np.std(all_signals, axis=0)

            all_signals = list(
                filter(
                    lambda signal: not (
                        np.any((signal - pedestal) > (self.sigma * noise))
                    ),
                    all_signals,
                )
            )
            all_signals = np.vstack(all_signals)
        self.pedestal_ = np.mean(all_signals, axis=0)
        self.noise_ = np.std(all_signals, axis=0)
        fixed_signals = dict()
        for eid, signal in signals.items():
            fixed_signals[eid] = signal - pedestal
        return fixed_signals


class Reordering:

    DEFAULT = {0: 0, 1: 3, 2: 4, 3: 2, 4: 1}
    # DEFAULT = {0: 0, 1: 4, 2: 3, 3: 1, 4: 2}
    def __init__(self, order_map: Dict[int, int] = None):
        self.order_map = order_map if order_map else self.DEFAULT

    def apply(self, signals: Dict[int, ArrayLike]) -> Dict[int, ArrayLike]:
        reordered_signals = dict()
        for eid, signal in signals.items():
            reordered_signal = np.zeros_like(signal)
            for i, maroc in MAROC.items():
                marocmap = MAROC[self.order_map[i]]
                lm = marocmap.left
                hm = marocmap.right
                reordered_signal[maroc.left : maroc.right] = signal[lm:hm]
            reordered_signals[eid] = reordered_signal
        return reordered_signals


class Doubling:
    def __init__(self, threshold: int = 400):
        self.threshold = threshold

    def apply(self, signals: Dict[int, ArrayLike]) -> Dict[int, ArrayLike]:
        fixed_signals = dict()
        for eid, signal in signals.items():
            fixed_signal = np.zeros_like(signal)
            for k, m in MAROC.items():
                maroc_signal = np.copy(signal[m.left : m.right]).astype(np.float32)
                maroc_mu = np.mean(maroc_signal)
                maroc_std = np.std(maroc_signal)
                if (
                    np.mean(maroc_signal[maroc_signal <= maroc_mu + 2 * maroc_std])
                    >= self.threshold
                ):
                    maroc_signal /= 2
                fixed_signal[m.left : m.right] = maroc_signal
            fixed_signals[eid] = fixed_signal
        return fixed_signals


# class Fix_overflow: #no maroc division
#     def __init__(self, threshold: int = 100, plot=False):
#         self.threshold = threshold
#         self.plot = plot

#     def apply(self, signals: Dict[int, ArrayLike]) -> Dict[int, ArrayLike]:
#         fixed_signals = dict()
#         for eid, signal in signals.items():
#             fixed_signal = np.copy(signal)
#             max_signal = np.max(signal)
#             if np.any(signal == 0):
#                 if max_signal >= 600 and not (
#                     check_faulty_ribbon(signal, delta_signal=80, n_oscillations=30)
#                 ):

#                     fixed_signal = np.copy(signal)
#                     zeros = np.where(signal == 0)[0]
#                     l, r = zeros[0], zeros[-1]
#                     for i, strip in enumerate(fixed_signal[l - 5 : r + 5]):
#                         fixed_signal[l - 5 : r + 5][i] = (
#                             strip + max_signal if strip < self.threshold else max_signal
#                         )
#                     if self.plot:
#                         plt.figure()
#                         plt.plot(range(320), fixed_signal)
#                         plt.title("fix overflow evt: {}".format(eid))
#                         for i, maroc in MAROC.items():
#                             plt.axvline(
#                                 maroc.left, linewidth=0.5, linestyle="--", c="grey"
#                             )
#                         plt.axvline(
#                             maroc.right, linewidth=0.5, linestyle="--", c="grey"
#                         )
#                         plt.show()
#             fixed_signals[eid] = fixed_signal

# return fixed_signals


class Fix_overflow:  #  maroc division
    def __init__(self, threshold: int = 100, plot=False):
        self.threshold = threshold
        self.plot = plot

    def apply(self, signals: Dict[int, ArrayLike]) -> Dict[int, ArrayLike]:
        fixed_signals = dict()
        for eid, signal in signals.items():
            fixed = 0
            max_signal = np.max(signal)
            if np.any(signal == 0):  # and max_signal >= 600:
                fixed_signal = np.copy(signal)
                faulty = check_faulty_ribbon(signal, delta_signal=80, n_oscillations=30)
                for i, maroc in MAROC.items():
                    maroc_signal = fixed_signal[maroc.left : maroc.right]

                    if faulty[i] == 0:
                        # print(i, faulty)
                        zeros = np.where(maroc_signal == 0)[0]
                        # print(i, zeros)
                        if len(zeros) > 0:
                            l, r = zeros[0] + maroc.left, zeros[-1] + maroc.left
                            L, R = max(l - 5, maroc.left), min(maroc.right, r + 6)
                            L -= maroc.left
                            R -= maroc.left
                            # print(f"Considering: {L}:{R}")
                            arr = maroc_signal[slice(L, R)]
                            arr[arr + max_signal >= self.threshold] = max_signal

                            fixed = 1
            else:
                fixed_signal = signal
                fixed = 0
            fixed_signals[eid] = fixed_signal

            if fixed == 1 and self.plot:
                plt.figure()
                plt.plot(range(320), fixed_signal)
                plt.title("fix overflow evt: {}".format(eid))
                for i, maroc in MAROC.items():
                    plt.axvline(maroc.left, linewidth=0.5, linestyle="--", c="grey")
                plt.axvline(maroc.right, linewidth=0.5, linestyle="--", c="grey")
                plt.show()

        return fixed_signals


@dataclass
class HitPositions:
    hit: List[int]
    left: List[int]
    right: List[int]


from collections import defaultdict


def find_hits(
    signals: dict[int, ArrayLike],
    noise: ArrayLike = None,
    plot: bool = False,
    output: bool = False,
    ribbon_output=False,
) -> Dict[int, ArrayLike]:

    all_signals = list()
    found_hits = {}
    for i, signal in enumerate(signals.values()):
        all_signals.append(signal)
    all_signals = np.vstack(all_signals)
    if noise is None:
        noise = np.std(all_signals, axis=0)
    for eid, signal in signals.items():
        hits_out = HitPositions([], [], [])
        # hits_out = {}
        # hits_out.setdefault("hit")
        # hits_out.setdefault("left")
        # hits_out.setdefault("right")
        faulty = check_faulty_ribbon(
            signal, delta_signal=50, n_oscillations=30, output=ribbon_output
        )

        # print(faulty)

        if not len(faulty[faulty == 1]):
            peaks, _ = find_peaks(
                signal, height=(4 * noise), distance=30, prominence=30, width=(5, 80)
            )
            # found = []
            # for strip, adc in enumerate(signal):
            #     if strip > 0 and strip < 320 and adc >= 4 * noise[strip]:
            #         found.append(strip)
            if len(peaks) >= 1:

                if len(peaks) > 1:
                    if output:
                        print("evt {} found {} peaks".format(eid, len(peaks)))
                elif len(peaks) == 1:
                    if output:
                        print("evt {} found 1 peak.".format(eid))

                for peak in peaks:
                    max_strip_y = signal[peak]
                    seed = peak
                    if output:
                        print(
                            "evt: {}, seed: {}, signal: {}, searching in [{}, {}]".format(
                                eid,
                                seed,
                                signal[seed],
                                max(seed - 40, 0),
                                min(seed + 40, 320),
                            )
                        )
                    s_l = signal[max(seed - 40, 0) : seed - 3]
                    n_l = noise[max(seed - 40, 0) : seed - 3]
                    mask_l = np.ma.MaskedArray(s_l, s_l >= 2 * n_l)

                    s_u = signal[seed + 3 : min(seed + 40, 320)]
                    n_u = noise[seed + 3 : min(seed + 40, 320)]

                    mask_u = np.ma.MaskedArray(s_u, s_u >= 2 * n_u)
                    if len(mask_l) and len(mask_u):
                        lower_x = np.ma.argmin(mask_l) + max(seed - 40, 0)
                        upper_x = np.ma.argmin(mask_u) + seed + 3
                        if (
                            signal[seed] - signal[lower_x] > 30
                            and signal[seed] - signal[upper_x] > 30
                        ):
                            if output:
                                print(
                                    f"seed: {seed}, lower: {lower_x}, upper: {upper_x}"
                                )
                            hits_out.hit.append(seed)
                            hits_out.left.append(lower_x)
                            hits_out.right.append(upper_x)
                    # hits_out["left"] = lefts.append(lower_x)
                    # hits_out["right"] = rights.append(upper_x)

                    if plot:

                        plt.scatter(seed, max_strip_y, s=3, c="k", marker="s")
                        plt.axhline(
                            noise[seed], linestyle="--", linewidth=0.5, c="grey"
                        )
                        plt.axhline(
                            4 * noise[seed], linestyle="--", linewidth=0.5, c="gold"
                        )
                        plt.axhline(
                            3 * noise[seed], linestyle="--", linewidth=0.5, c="magenta"
                        )
                        plt.axvline(lower_x, linestyle="--", c="green", linewidth=0.75)
                        plt.axvline(upper_x, linestyle="--", c="red", linewidth=0.75)
                if plot:

                    plt.title("find hits: evt {}".format(eid))
                    plt.plot(range(320), signal, linewidth=1)
                    plt.ylim(
                        np.min(signal),
                        np.max(signal) + 10 if np.max(signal) > 50 else 50,
                    )
                    plt.show()
        else:
            # hits_out.hit = None
            # hits_out.left = None
            # hits_out.right = None
            if output:
                print("Skipping event {} because of Faulty ribbon cable\n".format(eid))
        found_hits[eid] = hits_out
    return found_hits


class CommonMode:
    def __init__(self, output=False, plot=False, ribbon_output=False, debug=False):
        self.output = output
        self.plot = plot
        self.ribbon_output = ribbon_output
        self.debug = debug

    def apply(
        self,
        signals: Dict[int, ArrayLike],
    ) -> Dict[int, ArrayLike]:

        fixed_signals = dict()
        hits = find_hits(
            signals,
            plot=self.plot,
            output=self.output,
            ribbon_output=self.ribbon_output,
        )
        for eid, signal in signals.items():

            fixed_signal = np.zeros_like(signal)
            seeds = hits[eid].hit
            lefts = hits[eid].left
            rights = hits[eid].right

            for k, maroc in MAROC.items():
                maroc_signal = signal[maroc.left : maroc.right]
                if seeds:
                    for i, (seed, ls, rs) in enumerate(zip(seeds, lefts, rights)):
                        if ls in maroc and rs in maroc:
                            if self.debug:
                                print(
                                    "hit in maroc {}: left {}, right: {}, width: {}".format(
                                        k, ls, rs, rs - ls
                                    )
                                )
                            if ls <= maroc.left:
                                # print("ls: {} l: {}".format(ls, maroc.left))
                                if len(signal[rs : maroc.right]) > 0:
                                    maroc_mu = np.mean(signal[rs : maroc.right])
                                else:
                                    continue
                            elif rs >= maroc.right:
                                # print("rs: {}, h: {}".format(rs, maroc.right))
                                if len(signal[maroc.left : ls]) > 0:
                                    maroc_mu = np.mean(signal[maroc.left : ls])
                                else:
                                    continue
                            elif ls > maroc.left and rs < maroc.right:
                                # print("ls>l and rs<h".format(eid))
                                maroc_mu = np.mean(
                                    np.hstack(
                                        [
                                            signal[maroc.left : ls],
                                            signal[rs : maroc.right],
                                        ]
                                    )
                                )
                        elif ls in maroc and not rs in maroc:
                            if self.debug:
                                print(
                                    "left in maroc {} but not right, left: {}, right: {}, width: {}".format(
                                        k, ls, rs, rs - ls
                                    )
                                )
                            if len(signal[maroc.left : ls]) > 0:
                                maroc_mu = np.mean(signal[maroc.left : ls])
                            else:
                                continue
                        elif rs in maroc and not ls in maroc:
                            if self.debug:
                                print(
                                    "right in maroc {} but not left, left: {}, right: {}, width: {}".format(
                                        k, ls, rs, rs - ls
                                    )
                                )
                            if len(signal[rs : maroc.right]) > 0:

                                maroc_mu = np.mean(signal[rs : maroc.right])
                            else:
                                continue

                        else:
                            maroc_mu = np.mean(maroc_signal)
                else:
                    # print(
                    #    "Evt {} maroc {} no hits. calculating total mean".format(eid, k)
                    # )
                    maroc_mu = np.mean(maroc_signal)
                    # print(k, maroc_mu)
                    # common_mode = np.mean(maroc_signal[maroc_signal <= maroc_mu])
                maroc_signal -= maroc_mu
                fixed_signal[maroc.left : maroc.right] = maroc_signal

            fixed_signals[eid] = fixed_signal

        return fixed_signals


class Pipeline:
    def __init__(self, operators: Sequence[Operator]):
        self.ops = operators

    def apply(self, signals: Dict[int, ArrayLike]) -> Dict[int, ArrayLike]:
        for op in self.ops:
            signals = op.apply(signals)
        return signals
