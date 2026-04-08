import random
from datetime import datetime, timedelta


def get_random_2025_timestamp() -> datetime:
    """Generates a random timestamp between Jan 1, 2025 and Dec 31, 2025."""
    start = datetime(2025, 1, 1)
    # Total seconds in 2025 (365 days)
    delta_seconds = 365 * 24 * 60 * 60
    random_seconds = random.randint(0, delta_seconds)
    return start + timedelta(seconds=random_seconds)
