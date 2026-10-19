"""
Access to NEXRAD data on the NOAA Open Data Dissemination buckets (public, no
credentials needed, plain HTTPS):

* unidata-nexrad-level2         archived Level-II volumes  YYYY/MM/DD/SITE/SITEYYYYMMDD_HHMMSS_V06
* unidata-nexrad-level2-chunks  real-time Level-II chunks  SITE/<vol 1-999>/YYYYMMDD-HHMMSS-NNN-{S,I,E}
* unidata-nexrad-level3         Level-III products (2020+) SSS_PPP_YYYY_MM_DD_HH_MM_SS

In production, ``latest_realtime_volume`` returns bytes of the volume currently being
collected.  Chunks are complete Archive II records, so the bytes can be decoded as
soon as the 0.5/0.9 deg cuts have arrived (~1-2 min into the volume) instead of
waiting for the full volume (~4-6 min).
"""
from __future__ import annotations

import datetime
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Optional, Sequence, Tuple

LEVEL2_BUCKET = "https://unidata-nexrad-level2.s3.amazonaws.com"
LEVEL2_CHUNKS_BUCKET = "https://unidata-nexrad-level2-chunks.s3.amazonaws.com"
LEVEL3_BUCKET = "https://unidata-nexrad-level3.s3.amazonaws.com"
_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def _get(url: str, timeout: float = 60.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def s3_list(bucket: str, prefix: str, delimiter: Optional[str] = None,
            max_keys: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """Lists (keys, common_prefixes) of a public S3 bucket (handles pagination)."""
    keys, prefixes, token = [], [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if delimiter:
            q["delimiter"] = delimiter
        if max_keys:
            q["max-keys"] = str(max_keys)
        if token:
            q["continuation-token"] = token
        root = ET.fromstring(_get(f"{bucket}/?{urllib.parse.urlencode(q)}"))
        keys += [c.find(f"{_NS}Key").text for c in root.findall(f"{_NS}Contents")]
        prefixes += [c.find(f"{_NS}Prefix").text for c in root.findall(f"{_NS}CommonPrefixes")]
        token_el = root.find(f"{_NS}NextContinuationToken")
        if max_keys or token_el is None:
            return keys, prefixes
        token = token_el.text


def download(bucket: str, key: str) -> bytes:
    return _get(f"{bucket}/{urllib.parse.quote(key)}")


# ---------------------------------------------------------------- Level-II archive
_L2_RE = re.compile(r"([A-Z]{4})(\d{8}_\d{6})_V0\d(\.gz)?$")


def level2_key_time(key: str) -> Optional[datetime.datetime]:
    m = _L2_RE.search(key)
    return datetime.datetime.strptime(m.group(2), "%Y%m%d_%H%M%S") if m else None


def list_level2_volumes(site: str, date: datetime.date) -> List[str]:
    keys, _ = s3_list(LEVEL2_BUCKET, f"{date:%Y/%m/%d}/{site.upper()}/")
    return sorted(k for k in keys if level2_key_time(k) is not None)


def find_level2_volume(site: str, time: datetime.datetime,
                       max_diff: datetime.timedelta = datetime.timedelta(minutes=10)) -> str:
    """Key of the archived volume whose start time is closest to ``time``."""
    keys = []
    for d in {(time - datetime.timedelta(days=1)).date(), time.date(), (time + datetime.timedelta(days=1)).date()}:
        keys += list_level2_volumes(site, d)
    if not keys:
        raise FileNotFoundError(f"No Level-II volumes for {site} near {time}")
    best = min(keys, key=lambda k: abs(level2_key_time(k) - time))
    if abs(level2_key_time(best) - time) > max_diff:
        raise FileNotFoundError(f"No Level-II volume for {site} within {max_diff} of {time}")
    return best


# ---------------------------------------------------------------- Level-II real time
_CHUNK_RE = re.compile(r"/(\d{8}-\d{6})-(\d{3})-([SIE])$")


def _volume_chunks(site: str, vol_prefix: str) -> List[str]:
    keys, _ = s3_list(LEVEL2_CHUNKS_BUCKET, vol_prefix)
    keys = [k for k in keys if _CHUNK_RE.search(k)]
    return sorted(keys, key=lambda k: int(_CHUNK_RE.search(k).group(2)))


def _volume_start(vol_prefix: str) -> Optional[datetime.datetime]:
    keys, _ = s3_list(LEVEL2_CHUNKS_BUCKET, vol_prefix, max_keys=1)
    if not keys:
        return None
    m = _CHUNK_RE.search(keys[0])
    return datetime.datetime.strptime(m.group(1), "%Y%m%d-%H%M%S") if m else None


def latest_realtime_volume_prefix(site: str) -> str:
    """
    Prefix (``SITE/<n>/``) of the volume currently being transmitted.
    Volume numbers wrap 999 -> 1, so the sequence of start times is a rotated sorted
    array; its maximum is found with O(log n) listing requests.
    """
    _, prefixes = s3_list(LEVEL2_CHUNKS_BUCKET, f"{site.upper()}/", delimiter="/")
    prefixes = sorted(prefixes, key=lambda p: int(p.rstrip("/").split("/")[-1]))
    if not prefixes:
        raise FileNotFoundError(f"No real-time chunks for {site}")
    times = {}

    def t(i):
        if i not in times:
            times[i] = _volume_start(prefixes[i]) or datetime.datetime.min
        return times[i]

    lo, hi = 0, len(prefixes) - 1
    while lo < hi:  # find the rotation point (index of the minimum start time)
        mid = (lo + hi) // 2
        if t(mid) > t(hi):
            lo = mid + 1
        else:
            hi = mid
    return prefixes[(lo - 1) % len(prefixes)]


def realtime_volume_bytes(site: str, vol_prefix: Optional[str] = None) -> Tuple[bytes, bool]:
    """
    Concatenated chunks of a real-time volume (default: the newest one).
    Returns (archive_bytes, complete) where complete means the 'E' chunk has arrived.
    """
    vol_prefix = vol_prefix or latest_realtime_volume_prefix(site)
    keys = _volume_chunks(site, vol_prefix)
    if not keys or not keys[0].endswith("-S"):
        raise FileNotFoundError(f"Start chunk missing for {vol_prefix}")
    data = b"".join(download(LEVEL2_CHUNKS_BUCKET, k) for k in keys)
    return data, keys[-1].endswith("-E")


# ---------------------------------------------------------------- Level-III
def level3_key_time(key: str) -> datetime.datetime:
    return datetime.datetime.strptime(key[-19:], "%Y_%m_%d_%H_%M_%S")


def find_level3_product(site: str, product: str, time: datetime.datetime,
                        max_diff: datetime.timedelta = datetime.timedelta(seconds=90)) -> Optional[str]:
    """
    Key of the Level-III product (e.g. 'N0K', 'NAK') whose timestamp is closest to
    ``time``.  Level-III timestamps equal the start time of the source cut (the
    volume start time for the first cut), so pass the surveillance sweep time.
    ``site`` may be the 4-letter ICAO (KTLX) or 3-letter id (TLX).
    """
    s3 = site.upper()[-3:]
    keys = []
    for t in {time - datetime.timedelta(hours=1), time, time + datetime.timedelta(hours=1)}:
        k, _ = s3_list(LEVEL3_BUCKET, f"{s3}_{product}_{t:%Y_%m_%d_%H}")
        keys += k
    if not keys:
        return None
    best = min(keys, key=lambda k: abs(level3_key_time(k) - time))
    return best if abs(level3_key_time(best) - time) <= max_diff else None


def fetch_level3_kdp_for_volume(site: str, sweep_times: Sequence[datetime.datetime],
                                products: Sequence[str] = ("N0K", "NAK")) -> List[Optional[bytes]]:
    """Downloads the Level-III KDP product matching each TorNet tilt (None if absent)."""
    out = []
    for prod, t in zip(products, sweep_times):
        key = find_level3_product(site, prod, t)
        out.append(download(LEVEL3_BUCKET, key) if key else None)
    return out


def level3_kdp_fetcher(site: str, product: str, cut_time: datetime.datetime) -> Optional[bytes]:
    """``level3_kdp`` callable for ``level2_to_tornet_volume`` using the AWS Level-III bucket."""
    key = find_level3_product(site, product, cut_time)
    return download(LEVEL3_BUCKET, key) if key else None
