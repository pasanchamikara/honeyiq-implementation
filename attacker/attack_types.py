"""
Attack type definitions, kill chain stages, attacker intents,
and per-attack-type network feature simulation distributions.
Based on UNSW-NB15 dataset categories.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Core enumerations
# ---------------------------------------------------------------------------

class AttackType(IntEnum):
    NORMAL         = 0
    RECONNAISSANCE = 1
    ANALYSIS       = 2
    FUZZERS        = 3
    EXPLOITS       = 4
    BACKDOORS      = 5
    SHELLCODE      = 6
    GENERIC        = 7
    DOS            = 8
    WORMS          = 9

    @classmethod
    def names(cls) -> List[str]:
        return [e.name for e in cls]

    @classmethod
    def count(cls) -> int:
        return len(cls)


class KillChainStage(IntEnum):
    RECONNAISSANCE    = 0
    WEAPONIZATION     = 1
    DELIVERY          = 2
    EXPLOITATION      = 3
    INSTALLATION      = 4
    COMMAND_AND_CTRL  = 5
    ACTIONS_ON_OBJ    = 6

    @classmethod
    def names(cls) -> List[str]:
        return [e.name for e in cls]

    @classmethod
    def count(cls) -> int:
        return len(cls)


class AttackerIntent(IntEnum):
    STEALTHY      = 0   # Low and slow, avoids detection
    AGGRESSIVE    = 1   # Fast escalation, high noise
    TARGETED      = 2   # Focused on data exfiltration / specific objectives
    OPPORTUNISTIC = 3   # Exploits whatever is available, scattered

    @classmethod
    def names(cls) -> List[str]:
        return [e.name for e in cls]

    @classmethod
    def count(cls) -> int:
        return len(cls)


# ---------------------------------------------------------------------------
# Severity and threat weights
# ---------------------------------------------------------------------------

# Base attack severity in [0.0, 1.0] — how dangerous this attack type is
ATTACK_SEVERITY: Dict[int, float] = {
    AttackType.NORMAL:         0.00,
    AttackType.RECONNAISSANCE: 0.20,
    AttackType.ANALYSIS:       0.25,
    AttackType.FUZZERS:        0.35,
    AttackType.GENERIC:        0.40,
    AttackType.EXPLOITS:       0.70,
    AttackType.SHELLCODE:      0.75,
    AttackType.BACKDOORS:      0.80,
    AttackType.DOS:            0.85,
    AttackType.WORMS:          0.90,
}

# Kill chain stage threat multiplier — later stages are more dangerous
KILL_CHAIN_WEIGHT: Dict[int, float] = {
    KillChainStage.RECONNAISSANCE:   0.10,
    KillChainStage.WEAPONIZATION:    0.20,
    KillChainStage.DELIVERY:         0.35,
    KillChainStage.EXPLOITATION:     0.55,
    KillChainStage.INSTALLATION:     0.70,
    KillChainStage.COMMAND_AND_CTRL: 0.85,
    KillChainStage.ACTIONS_ON_OBJ:   1.00,
}

# Primary kill chain stage associated with each attack type
ATTACK_PRIMARY_STAGE: Dict[int, int] = {
    AttackType.NORMAL:         KillChainStage.RECONNAISSANCE,
    AttackType.RECONNAISSANCE: KillChainStage.RECONNAISSANCE,
    AttackType.ANALYSIS:       KillChainStage.WEAPONIZATION,
    AttackType.FUZZERS:        KillChainStage.DELIVERY,
    AttackType.EXPLOITS:       KillChainStage.EXPLOITATION,
    AttackType.BACKDOORS:      KillChainStage.INSTALLATION,
    AttackType.SHELLCODE:      KillChainStage.EXPLOITATION,
    AttackType.GENERIC:        KillChainStage.DELIVERY,
    AttackType.DOS:            KillChainStage.ACTIONS_ON_OBJ,
    AttackType.WORMS:          KillChainStage.COMMAND_AND_CTRL,
}


# ---------------------------------------------------------------------------
# Network feature simulation distributions
# ---------------------------------------------------------------------------
# Each entry: feature_name -> (distribution, *params)
# Distributions: "uniform"   (low, high)
#                "lognormal" (mean_log, sigma_log)
#                "poisson"   (lambda)
#                "choice"    (list of values)
#                "constant"  (value)
#
# Features approximate key UNSW-NB15 fields:
#   dur       - flow duration (seconds)
#   sbytes    - source bytes
#   dbytes    - destination bytes
#   sttl      - source TTL
#   dttl      - destination TTL
#   sloss     - source packet loss count
#   dloss     - destination packet loss count
#   sload     - source bits per second
#   dload     - destination bits per second
#   spkts     - source packet count
#   dpkts     - destination packet count
#   swin      - source TCP window size
#   dwin      - destination TCP window size
#   ct_srv_src - connections with same service & source
#   ct_dst_ltm - connections with same destination

FEATURE_DISTRIBUTIONS: Dict[int, Dict[str, Tuple]] = {

    AttackType.NORMAL: {
        "dur":       ("uniform",   0.001, 10.0),
        "sbytes":    ("lognormal", 8.0,   2.0),
        "dbytes":    ("lognormal", 7.5,   2.0),
        "sttl":      ("choice",    [64, 128, 255]),
        "dttl":      ("choice",    [64, 128, 255]),
        "sloss":     ("poisson",   0.5),
        "dloss":     ("poisson",   0.5),
        "sload":     ("uniform",   100.0, 5_000.0),
        "dload":     ("uniform",   100.0, 5_000.0),
        "spkts":     ("poisson",   15),
        "dpkts":     ("poisson",   12),
        "swin":      ("choice",    [8192, 16384, 32768, 65535]),
        "dwin":      ("choice",    [8192, 16384, 32768, 65535]),
        "ct_srv_src":("poisson",   5),
        "ct_dst_ltm":("poisson",   4),
    },

    AttackType.RECONNAISSANCE: {
        "dur":       ("uniform",   0.0001, 0.5),
        "sbytes":    ("lognormal", 4.0,    1.0),
        "dbytes":    ("lognormal", 3.0,    1.0),
        "sttl":      ("choice",    [64, 128]),
        "dttl":      ("choice",    [64, 128]),
        "sloss":     ("poisson",   0.1),
        "dloss":     ("poisson",   0.1),
        "sload":     ("uniform",   50.0,   500.0),
        "dload":     ("uniform",   10.0,   200.0),
        "spkts":     ("poisson",   3),
        "dpkts":     ("poisson",   1),
        "swin":      ("choice",    [1024, 2048]),
        "dwin":      ("choice",    [0, 1024]),
        "ct_srv_src":("poisson",   20),   # many probes to same service
        "ct_dst_ltm":("poisson",   30),   # scanning many destinations
    },

    AttackType.ANALYSIS: {
        "dur":       ("uniform",   0.01,  2.0),
        "sbytes":    ("lognormal", 5.0,   1.2),
        "dbytes":    ("lognormal", 4.5,   1.2),
        "sttl":      ("choice",    [64, 128]),
        "dttl":      ("choice",    [64, 128]),
        "sloss":     ("poisson",   0.3),
        "dloss":     ("poisson",   0.3),
        "sload":     ("uniform",   100.0, 2_000.0),
        "dload":     ("uniform",   50.0,  1_000.0),
        "spkts":     ("poisson",   8),
        "dpkts":     ("poisson",   5),
        "swin":      ("choice",    [2048, 4096, 8192]),
        "dwin":      ("choice",    [1024, 2048]),
        "ct_srv_src":("poisson",   15),
        "ct_dst_ltm":("poisson",   10),
    },

    AttackType.FUZZERS: {
        "dur":       ("uniform",   0.001, 3.0),
        "sbytes":    ("lognormal", 6.0,   1.5),
        "dbytes":    ("lognormal", 3.0,   1.0),
        "sttl":      ("choice",    [64, 128]),
        "dttl":      ("choice",    [64]),
        "sloss":     ("poisson",   2.0),
        "dloss":     ("poisson",   3.0),
        "sload":     ("uniform",   500.0, 20_000.0),
        "dload":     ("uniform",   10.0,  500.0),
        "spkts":     ("poisson",   50),
        "dpkts":     ("poisson",   5),
        "swin":      ("choice",    [512, 1024, 2048]),
        "dwin":      ("choice",    [0, 512]),
        "ct_srv_src":("poisson",   8),
        "ct_dst_ltm":("poisson",   5),
    },

    AttackType.EXPLOITS: {
        "dur":       ("uniform",   0.1,   5.0),
        "sbytes":    ("lognormal", 7.0,   1.5),
        "dbytes":    ("lognormal", 5.0,   1.5),
        "sttl":      ("choice",    [64, 128]),
        "dttl":      ("choice",    [64, 128]),
        "sloss":     ("poisson",   1.0),
        "dloss":     ("poisson",   1.5),
        "sload":     ("uniform",   500.0, 30_000.0),
        "dload":     ("uniform",   200.0, 10_000.0),
        "spkts":     ("poisson",   25),
        "dpkts":     ("poisson",   20),
        "swin":      ("choice",    [4096, 8192, 16384]),
        "dwin":      ("choice",    [4096, 8192]),
        "ct_srv_src":("poisson",   6),
        "ct_dst_ltm":("poisson",   4),
    },

    AttackType.BACKDOORS: {
        "dur":       ("uniform",   10.0,  3_600.0),  # persistent, long sessions
        "sbytes":    ("lognormal", 6.0,   1.0),
        "dbytes":    ("lognormal", 6.5,   1.0),
        "sttl":      ("choice",    [64, 128]),
        "dttl":      ("choice",    [64, 128]),
        "sloss":     ("poisson",   0.2),
        "dloss":     ("poisson",   0.2),
        "sload":     ("uniform",   10.0,  500.0),    # low, stealthy
        "dload":     ("uniform",   10.0,  500.0),
        "spkts":     ("poisson",   8),
        "dpkts":     ("poisson",   8),
        "swin":      ("choice",    [8192, 16384, 32768]),
        "dwin":      ("choice",    [8192, 16384, 32768]),
        "ct_srv_src":("poisson",   2),
        "ct_dst_ltm":("poisson",   2),
    },

    AttackType.SHELLCODE: {
        "dur":       ("uniform",   0.01,  2.0),
        "sbytes":    ("lognormal", 5.5,   0.8),
        "dbytes":    ("lognormal", 4.0,   1.0),
        "sttl":      ("choice",    [64, 128]),
        "dttl":      ("choice",    [64]),
        "sloss":     ("poisson",   0.8),
        "dloss":     ("poisson",   1.0),
        "sload":     ("uniform",   200.0, 8_000.0),
        "dload":     ("uniform",   50.0,  2_000.0),
        "spkts":     ("poisson",   12),
        "dpkts":     ("poisson",   8),
        "swin":      ("choice",    [1024, 2048, 4096]),
        "dwin":      ("choice",    [512, 1024]),
        "ct_srv_src":("poisson",   4),
        "ct_dst_ltm":("poisson",   3),
    },

    AttackType.GENERIC: {
        "dur":       ("uniform",   0.001, 5.0),
        "sbytes":    ("lognormal", 6.5,   1.8),
        "dbytes":    ("lognormal", 5.0,   1.8),
        "sttl":      ("choice",    [64, 128, 255]),
        "dttl":      ("choice",    [64, 128]),
        "sloss":     ("poisson",   1.5),
        "dloss":     ("poisson",   1.5),
        "sload":     ("uniform",   300.0, 15_000.0),
        "dload":     ("uniform",   100.0, 5_000.0),
        "spkts":     ("poisson",   30),
        "dpkts":     ("poisson",   20),
        "swin":      ("choice",    [2048, 4096, 8192]),
        "dwin":      ("choice",    [2048, 4096]),
        "ct_srv_src":("poisson",   10),
        "ct_dst_ltm":("poisson",   8),
    },

    AttackType.DOS: {
        "dur":       ("uniform",   0.0,   0.01),    # flood, very short flows
        "sbytes":    ("lognormal", 5.0,   0.5),
        "dbytes":    ("lognormal", 1.0,   0.5),
        "sttl":      ("choice",    [64]),
        "dttl":      ("choice",    [64]),
        "sloss":     ("poisson",   10.0),
        "dloss":     ("poisson",   8.0),
        "sload":     ("uniform",   50_000.0, 500_000.0),  # very high load
        "dload":     ("uniform",   0.0,      1_000.0),
        "spkts":     ("poisson",   500),
        "dpkts":     ("poisson",   2),
        "swin":      ("choice",    [512, 1024]),
        "dwin":      ("choice",    [0]),
        "ct_srv_src":("poisson",   50),
        "ct_dst_ltm":("poisson",   1),
    },

    AttackType.WORMS: {
        "dur":       ("uniform",   0.001, 1.0),
        "sbytes":    ("lognormal", 5.0,   1.0),
        "dbytes":    ("lognormal", 4.5,   1.0),
        "sttl":      ("choice",    [64, 128]),
        "dttl":      ("choice",    [64, 128]),
        "sloss":     ("poisson",   3.0),
        "dloss":     ("poisson",   2.0),
        "sload":     ("uniform",   1_000.0, 50_000.0),
        "dload":     ("uniform",   500.0,   20_000.0),
        "spkts":     ("poisson",   50),
        "dpkts":     ("poisson",   30),
        "swin":      ("choice",    [2048, 4096]),
        "dwin":      ("choice",    [2048, 4096]),
        "ct_srv_src":("poisson",   25),   # many connections (spreading)
        "ct_dst_ltm":("poisson",   40),   # hitting many hosts
    },
}

# Ordered list of feature names used by the classifier
FEATURE_NAMES: List[str] = [
    "dur", "sbytes", "dbytes", "sttl", "dttl",
    "sloss", "dloss", "sload", "dload", "spkts",
    "dpkts", "swin", "dwin", "ct_srv_src", "ct_dst_ltm",
]
