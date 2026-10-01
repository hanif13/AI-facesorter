from datetime import datetime
from pathlib import Path
import logging
from zoneinfo import ZoneInfo

def setup_logging(name: str) -> Path:
    root = Path("outputs/logs")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}_{datetime.now(ZoneInfo('Asia/Bangkok')).strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(path), logging.StreamHandler()], force=True)
    return path
