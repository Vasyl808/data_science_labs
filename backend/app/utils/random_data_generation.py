import random
from datetime import datetime, timedelta

def get_random_2025_timestamp() -> datetime:
    start = datetime(2025, 1, 1)
    delta_seconds = 365 * 24 * 60 * 60
    random_seconds = random.randint(0, delta_seconds)
    return start + timedelta(seconds=random_seconds)