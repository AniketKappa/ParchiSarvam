"""Shared Sarvam client. Reads SARVAM_API_KEY from the environment or a local .env file."""
import os
from pathlib import Path

from sarvamai import SarvamAI

env_file = Path(__file__).with_name(".env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

api_key = os.environ.get("SARVAM_API_KEY")
if not api_key:
    raise SystemExit("SARVAM_API_KEY is not set. Add it to .env")

client = SarvamAI(api_subscription_key=api_key)
