import os
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass(frozen=True)
class SRConfig:
    api_key: str
    timeout_s: int = 20

def load_config() -> SRConfig:
    # Load .env from current working directory (repo root)
    load_dotenv()
    key = os.getenv("SPORTRADAR_API_KEY")
    if not key:
        raise RuntimeError("Missing SPORTRADAR_API_KEY (check .env in repo root)")
    return SRConfig(api_key=key)