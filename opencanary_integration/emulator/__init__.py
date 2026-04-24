"""
OpenCanary emulator — runs the full HoneyIQ pipeline without a live OpenCanary
instance.  Useful for thesis demonstration, integration testing, and CI.
"""
from .event_generator import OpenCanaryEventGenerator
from .honeypot_emulator import DummyHoneypot
from .scenario import EmulatorScenario

__all__ = ["OpenCanaryEventGenerator", "DummyHoneypot", "EmulatorScenario"]
