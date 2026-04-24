"""
Map OpenCanary logtype codes → AttackType and initial KillChainStage.
"""

from __future__ import annotations

from attacker.attack_types import AttackType, KillChainStage
from opencanary_integration.ingest.models import OpenCanaryEvent


_LOGTYPE_TO_ATTACK: dict[int, AttackType] = {
    1000:  AttackType.RECONNAISSANCE,   # HTTP probe
    2000:  AttackType.RECONNAISSANCE,   # Port scan
    3000:  AttackType.ANALYSIS,         # FTP login attempt
    4000:  AttackType.RECONNAISSANCE,   # Telnet banner
    5000:  AttackType.EXPLOITS,         # MySQL exploit
    6000:  AttackType.EXPLOITS,         # MSSQL exploit
    7000:  AttackType.EXPLOITS,         # Telnet login
    8000:  AttackType.ANALYSIS,         # HTTP proxy probe
    9000:  AttackType.BACKDOORS,        # SMB access
    10000: AttackType.RECONNAISSANCE,   # SNMP probe
    11000: AttackType.GENERIC,          # SIP probe
    12000: AttackType.SHELLCODE,        # VNC login
    13000: AttackType.SHELLCODE,        # VNC login (alt)
    14000: AttackType.EXPLOITS,         # Redis exploit
    15000: AttackType.ANALYSIS,         # TFTP probe
    16000: AttackType.ANALYSIS,         # Git probe
    17000: AttackType.RECONNAISSANCE,   # TCP banner grab
    18000: AttackType.RECONNAISSANCE,   # HTTP skin probe
    22000: AttackType.EXPLOITS,         # SSH brute-force
}

_ATTACK_TO_INITIAL_STAGE: dict[AttackType, KillChainStage] = {
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


def map_logtype(event: OpenCanaryEvent) -> AttackType:
    """Return the AttackType corresponding to an OpenCanary logtype."""
    return _LOGTYPE_TO_ATTACK.get(event.logtype, AttackType.GENERIC)


def initial_stage_for(attack_type: AttackType) -> KillChainStage:
    """Return the default kill chain entry stage for an attack type."""
    return _ATTACK_TO_INITIAL_STAGE.get(attack_type, KillChainStage.RECONNAISSANCE)
