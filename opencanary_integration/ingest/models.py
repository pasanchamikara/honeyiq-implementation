"""
Pydantic models for raw OpenCanary log events.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, field_validator


class OpenCanaryEvent(BaseModel):
    """
    Represents a single JSON log line emitted by OpenCanary.

    Field names match the top-level keys in opencanary's JSON output.
    """

    dst_host:   str
    dst_port:   int
    logdata:    dict[str, Any]
    logtype:    int
    node_id:    str
    src_host:   str
    src_port:   int
    utc_time:   str
    local_time: str

    @field_validator("logdata", mode="before")
    @classmethod
    def _ensure_dict(cls, v: Any) -> dict:
        if v is None:
            return {}
        if isinstance(v, str):
            import json
            return json.loads(v)
        return v

    @property
    def service_name(self) -> str:
        """Human-readable service name derived from logtype."""
        _NAMES = {
            1000: "HTTP", 2000: "PORT_SCAN", 3000: "FTP",
            4000: "TELNET_BANNER", 5000: "MYSQL", 6000: "MSSQL",
            7000: "TELNET", 8000: "HTTPPROXY", 9000: "SMB",
            10000: "SNMP", 11000: "SIP", 12000: "VNC",
            13000: "VNC2", 14000: "REDIS", 15000: "TFTP",
            16000: "GIT", 17000: "TCP_BANNER", 18000: "HTTP_SKIN",
            22000: "SSH",
        }
        return _NAMES.get(self.logtype, f"LOGTYPE_{self.logtype}")
