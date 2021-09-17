from io import BufferedIOBase
import ipaddress
import itertools
import numpy as np
from collections import namedtuple

lst = list(itertools.product([0, 1], repeat=4))

def ip_last(quad, jumper):
    a, b, c, d = quad
    return (2 ** 0 * a + 2 ** 1 * b + 2 ** 2 * c + 2 ** 3 * d) * 16 + jumper

ipaddr = []
ipdec = []
for quad in lst:
    for i in [0, 1]:
        ipaddr.append("192.168.200.{}".format(ip_last(quad, i)))
        ipdec.append(
            int(ipaddress.IPv4Address("192.168.200.{}".format(ip_last(quad, i))))
        )

idx = np.argsort([int(addr.split(".")[-1]) for addr in ipaddr])

sorted_ipaddr = np.asarray(ipaddr)[idx][2:]
sorted_dec = np.asarray(ipdec)[idx][2:]

BoardIP = namedtuple("BoardIP", ["bid", "ipv4", "dec"])
BOARDS = {dec: BoardIP(bid, ip, dec) for bid, ip, dec in zip(range(1, 31), sorted_ipaddr, sorted_dec)}