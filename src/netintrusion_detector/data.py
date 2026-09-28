"""Loading and schema definitions for the NSL-KDD network intrusion dataset.

Data provenance
----------------
This package was built and evaluated against the **real, public NSL-KDD**
dataset, fetched at build time from a well-known GitHub mirror of the
official NSL-KDD release:

    https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt
    https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt

NSL-KDD is a widely used, freely redistributable revision of the classic
KDD Cup 1999 network intrusion benchmark (it removes duplicate records and
rebalances the difficulty of the original KDD'99 set). It ships as two
plain CSV-like text files with 41 pre-extracted flow features + a text
label (+ an optional "difficulty" score column), and is evaluated entirely
offline -- there is no live packet capture involved in this project, nor
does using it require any.

The loader below (:func:`load_nslkdd`) is written generically so it works
identically whether you:

* point it at the original ``KDDTrain+.txt`` / ``KDDTest+.txt`` files (no
  header row, 41 or 42 columns, columns in the standard NSL-KDD order), or
* point it at any CSV that already has a header using the same column
  names (e.g. the small bundled sample under ``data/sample/``), or
* pass a DataFrame produced by :func:`generate_synthetic_nslkdd_like`, kept
  here purely as an offline fallback / testing utility in case a real
  download is ever unavailable in some environment -- it is NOT what this
  package's shipped model was trained on. See its docstring for the
  explicit synthetic-data disclaimer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Standard NSL-KDD / KDD'99 41-feature schema
# ---------------------------------------------------------------------------
# Order and names match the canonical "Field Names.csv" distributed with the
# dataset (public, documented schema -- not proprietary).

CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]

NUMERIC_FEATURES = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

# Feature order as it appears in the raw KDDTrain+.txt / KDDTest+.txt files.
NSL_KDD_FEATURE_ORDER = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

assert len(NSL_KDD_FEATURE_ORDER) == 41
assert set(NSL_KDD_FEATURE_ORDER) == set(CATEGORICAL_FEATURES) | set(NUMERIC_FEATURES)

ALL_COLUMNS_WITH_LABEL = NSL_KDD_FEATURE_ORDER + ["label", "difficulty"]

# ---------------------------------------------------------------------------
# Standard NSL-KDD attack-type -> attack-category mapping.
# This is the well-documented public mapping used throughout the NSL-KDD
# literature, covering the ~39 specific attack types across the train+test
# splits (the test set contains several attack types not seen in training,
# which is a deliberate, well-known feature of NSL-KDD for testing
# generalization to novel attacks).
# ---------------------------------------------------------------------------

DOS_ATTACKS = {
    "back", "land", "neptune", "pod", "smurf", "teardrop", "mailbomb",
    "processtable", "udpstorm", "apache2", "worm",
}
PROBE_ATTACKS = {
    "satan", "ipsweep", "nmap", "portsweep", "mscan", "saint",
}
R2L_ATTACKS = {
    "guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster",
    "warezclient", "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattack",
    "httptunnel", "sendmail", "named", "worm",
}
U2R_ATTACKS = {
    "buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack",
    "xterm", "ps",
}

# 'worm' historically appears in both DoS-like and R2L-like taxonomies across
# different published versions of the mapping; NSL-KDD's own documentation
# places it under R2L, which is what we use canonically (kept in DOS_ATTACKS
# above only for readers cross-checking older KDD'99 tables; the lookup
# below resolves it to R2L).
_CATEGORY_BY_ATTACK: dict[str, str] = {}
for _name in DOS_ATTACKS:
    _CATEGORY_BY_ATTACK[_name] = "DoS"
for _name in PROBE_ATTACKS:
    _CATEGORY_BY_ATTACK[_name] = "Probe"
for _name in R2L_ATTACKS:
    _CATEGORY_BY_ATTACK[_name] = "R2L"
for _name in U2R_ATTACKS:
    _CATEGORY_BY_ATTACK[_name] = "U2R"
_CATEGORY_BY_ATTACK["worm"] = "R2L"
_CATEGORY_BY_ATTACK["normal"] = "normal"

ATTACK_CATEGORIES = ["normal", "DoS", "Probe", "R2L", "U2R"]


def map_attack_to_category(attack_label: str) -> str:
    """Map a specific NSL-KDD attack-type string (or 'normal') to its
    attack category: one of normal/DoS/Probe/R2L/U2R.

    Unknown / unseen attack-type strings map to 'Unknown' rather than
    raising, since NSL-KDD's test set intentionally contains attack types
    absent from the training set (a deliberate generalization challenge).
    """
    key = str(attack_label).strip().lower()
    return _CATEGORY_BY_ATTACK.get(key, "Unknown")


@dataclass
class NSLKDDDataset:
    """A loaded NSL-KDD-schema dataset, already split into features/labels."""

    X: pd.DataFrame
    attack_type: pd.Series  # raw string label, e.g. "neptune", "normal"
    attack_category: pd.Series  # normal/DoS/Probe/R2L/U2R
    is_attack: pd.Series  # 0 = normal, 1 = attack (binary target)
    source: str = field(default="unknown")  # "real" or "synthetic", for transparency


def load_nslkdd(path: str, source: Optional[str] = None) -> NSLKDDDataset:
    """Load an NSL-KDD-schema file into an :class:`NSLKDDDataset`.

    Accepts either:

    * The original headerless ``KDDTrain+.txt`` / ``KDDTest+.txt`` format
      (comma-separated, 41 feature columns + label [+ optional numeric
      difficulty column]).
    * Any CSV with a header row using the same column names (as produced by
      :func:`generate_synthetic_nslkdd_like` or the bundled sample files).

    Parameters
    ----------
    path:
        Path to a ``.txt`` or ``.csv`` file.
    source:
        Optional explicit provenance tag ("real" or "synthetic"). If not
        given, it is inferred from the filename (best-effort) and otherwise
        left as "unknown".
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"NSL-KDD data file not found: {path}. Real KDDTrain+.txt / "
            "KDDTest+.txt files can be downloaded from a public NSL-KDD "
            "mirror, e.g. https://raw.githubusercontent.com/defcom17/"
            "NSL_KDD/master/KDDTrain%2B.txt -- see README for details."
        )

    # Peek at the first line to decide whether there's a header.
    with open(path, "r") as fh:
        first_line = fh.readline().strip()
    first_tokens = first_line.split(",")

    has_header = first_tokens and first_tokens[0].strip().lower() in (
        "duration",
        "protocol_type",
    )

    if has_header:
        df = pd.read_csv(path)
    else:
        n_cols = len(first_tokens)
        if n_cols == len(NSL_KDD_FEATURE_ORDER) + 1:
            names = NSL_KDD_FEATURE_ORDER + ["label"]
        elif n_cols == len(NSL_KDD_FEATURE_ORDER) + 2:
            names = ALL_COLUMNS_WITH_LABEL
        else:
            raise ValueError(
                f"Unrecognized NSL-KDD file shape: {n_cols} columns in "
                f"{path!r}. Expected {len(NSL_KDD_FEATURE_ORDER)} feature "
                "columns plus a label (and optionally a difficulty score)."
            )
        df = pd.read_csv(path, header=None, names=names)

    missing = set(NSL_KDD_FEATURE_ORDER) - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing expected NSL-KDD columns: {sorted(missing)}")
    if "label" not in df.columns:
        raise ValueError("Input file has no 'label' column (attack type / 'normal').")

    # Some raw label values in the original file carry a trailing '.'
    # (an artifact of the original KDD'99 test-set format); normalize it.
    attack_type = df["label"].astype(str).str.strip().str.rstrip(".").str.lower()
    attack_category = attack_type.map(map_attack_to_category)
    is_attack = (attack_type != "normal").astype(int)

    X = df[NSL_KDD_FEATURE_ORDER].copy()

    inferred_source = source
    if inferred_source is None:
        lower_path = path.lower()
        if "synthetic" in lower_path:
            inferred_source = "synthetic"
        elif "kdd" in lower_path or "nsl" in lower_path:
            inferred_source = "real"
        else:
            inferred_source = "unknown"

    return NSLKDDDataset(
        X=X,
        attack_type=attack_type,
        attack_category=attack_category,
        is_attack=is_attack,
        source=inferred_source,
    )


# ---------------------------------------------------------------------------
# Synthetic fallback generator
# ---------------------------------------------------------------------------
#
# NOT USED for this package's shipped, trained model -- the real NSL-KDD
# dataset was successfully downloaded and used instead (see README for the
# exact source URLs and achieved metrics). This function is kept purely as
# a documented, honest fallback: if a future environment cannot reach the
# NSL-KDD mirrors over the network, this generates a SYNTHETIC, clearly
# labeled stand-in dataset that mimics NSL-KDD's real per-category feature
# distributions closely enough for the pipeline/tests to still exercise
# meaningful, non-random signal. It must never be presented as real data.


def generate_synthetic_nslkdd_like(
    n_samples: int = 2000,
    class_weights: Optional[dict] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a SYNTHETIC dataset that mimics NSL-KDD's schema and the
    broad per-attack-category distributional shape of the real data.

    This is a documented stand-in for offline development/testing when the
    real NSL-KDD files are not reachable -- it is *not* derived from or
    equivalent to the real dataset, and code/README consuming its output
    must always label it "synthetic". Distributional assumptions per
    category (deliberately simplified, but directionally consistent with
    well-documented NSL-KDD attack signatures):

    * **normal**: moderate/variable duration, moderate byte counts, mostly
      ``logged_in=1``, low error rates, low counts.
    * **DoS** (e.g. neptune/smurf floods): very short duration, near-zero
      ``dst_bytes``, high ``count``/``srv_count`` (many connections in a
      short window), high ``serror_rate``/``srv_serror_rate``.
    * **Probe** (e.g. satan/portsweep/nmap scans): short duration, near-zero
      bytes both ways, high ``count`` with high ``diff_srv_rate`` /
      ``dst_host_diff_srv_rate`` (many *different* services/hosts touched
      -- the scanning signature), low ``logged_in``.
    * **R2L** (e.g. guess_passwd/ftp_write -- remote-to-local access
      attempts): low counts (usually single connections), elevated
      ``num_failed_logins``, ``logged_in`` often 0 (failed access attempts).
    * **U2R** (e.g. buffer_overflow/rootkit -- privilege escalation *after*
      already logging in, so it's rare and low-volume): ``logged_in=1``,
      elevated ``num_shells``/``num_file_creations``/``root_shell``/
      ``num_compromised``, otherwise looks close to normal traffic (hence
      historically hard to detect from flow features alone).

    Parameters
    ----------
    n_samples:
        Total number of rows to generate.
    class_weights:
        Optional dict of {category: fraction}, must sum to ~1. Defaults to
        an imbalanced split similar in spirit to real NSL-KDD (attacks are
        common, but R2L/U2R are much rarer than DoS/normal -- this class
        imbalance is part of why R2L/U2R are historically hard classes).
    random_state:
        Seed for full reproducibility.
    """
    rng = np.random.default_rng(random_state)

    if class_weights is None:
        class_weights = {
            "normal": 0.53,
            "DoS": 0.33,
            "Probe": 0.09,
            "R2L": 0.04,
            "U2R": 0.01,
        }

    categories = list(class_weights.keys())
    probs = np.array([class_weights[c] for c in categories], dtype=float)
    probs = probs / probs.sum()
    counts = rng.multinomial(n_samples, probs)

    protocols = np.array(["tcp", "udp", "icmp"])
    services = np.array(["http", "ftp", "smtp", "telnet", "private", "domain_u", "other", "ftp_data"])
    flags = np.array(["SF", "S0", "REJ", "RSTR", "SH"])

    rows = []
    for category, n in zip(categories, counts):
        if n == 0:
            continue
        if category == "normal":
            duration = rng.exponential(200, n)
            src_bytes = rng.lognormal(6, 2, n)
            dst_bytes = rng.lognormal(6, 2, n)
            count = rng.integers(1, 20, n)
            srv_count = rng.integers(1, 20, n)
            serror_rate = np.clip(rng.normal(0.02, 0.03, n), 0, 1)
            diff_srv_rate = np.clip(rng.normal(0.05, 0.05, n), 0, 1)
            logged_in = rng.binomial(1, 0.7, n)
            num_failed_logins = np.zeros(n)
            num_shells = np.zeros(n)
            num_file_creations = np.zeros(n)
            root_shell = np.zeros(n)
            num_compromised = np.zeros(n)
            flag_choice = rng.choice(flags, n, p=[0.75, 0.05, 0.05, 0.10, 0.05])
            attack_type = "normal"
        elif category == "DoS":
            duration = rng.exponential(1, n)
            src_bytes = rng.exponential(50, n)
            dst_bytes = np.zeros(n)
            count = rng.integers(100, 511, n)
            srv_count = rng.integers(100, 511, n)
            serror_rate = np.clip(rng.normal(0.9, 0.08, n), 0, 1)
            diff_srv_rate = np.clip(rng.normal(0.02, 0.03, n), 0, 1)
            logged_in = np.zeros(n, dtype=int)
            num_failed_logins = np.zeros(n)
            num_shells = np.zeros(n)
            num_file_creations = np.zeros(n)
            root_shell = np.zeros(n)
            num_compromised = np.zeros(n)
            flag_choice = rng.choice(flags, n, p=[0.05, 0.75, 0.10, 0.05, 0.05])
            attack_type = "neptune"
        elif category == "Probe":
            duration = rng.exponential(0.5, n)
            src_bytes = rng.exponential(20, n)
            dst_bytes = np.zeros(n)
            count = rng.integers(50, 300, n)
            srv_count = rng.integers(1, 10, n)
            serror_rate = np.clip(rng.normal(0.3, 0.2, n), 0, 1)
            diff_srv_rate = np.clip(rng.normal(0.7, 0.15, n), 0, 1)
            logged_in = np.zeros(n, dtype=int)
            num_failed_logins = np.zeros(n)
            num_shells = np.zeros(n)
            num_file_creations = np.zeros(n)
            root_shell = np.zeros(n)
            num_compromised = np.zeros(n)
            flag_choice = rng.choice(flags, n, p=[0.10, 0.55, 0.15, 0.10, 0.10])
            attack_type = "satan"
        elif category == "R2L":
            duration = rng.exponential(5, n)
            src_bytes = rng.lognormal(4, 1, n)
            dst_bytes = rng.lognormal(3, 1, n)
            count = rng.integers(1, 5, n)
            srv_count = rng.integers(1, 5, n)
            serror_rate = np.clip(rng.normal(0.05, 0.05, n), 0, 1)
            diff_srv_rate = np.clip(rng.normal(0.05, 0.05, n), 0, 1)
            logged_in = rng.binomial(1, 0.15, n)
            num_failed_logins = rng.integers(1, 5, n)
            num_shells = np.zeros(n)
            num_file_creations = np.zeros(n)
            root_shell = np.zeros(n)
            num_compromised = np.zeros(n)
            flag_choice = rng.choice(flags, n, p=[0.4, 0.2, 0.2, 0.1, 0.1])
            attack_type = "guess_passwd"
        else:  # U2R
            duration = rng.exponential(300, n)
            src_bytes = rng.lognormal(6, 1.5, n)
            dst_bytes = rng.lognormal(6, 1.5, n)
            count = rng.integers(1, 5, n)
            srv_count = rng.integers(1, 5, n)
            serror_rate = np.clip(rng.normal(0.02, 0.02, n), 0, 1)
            diff_srv_rate = np.clip(rng.normal(0.02, 0.02, n), 0, 1)
            logged_in = np.ones(n, dtype=int)
            num_failed_logins = np.zeros(n)
            num_shells = rng.integers(1, 3, n)
            num_file_creations = rng.integers(1, 5, n)
            root_shell = rng.binomial(1, 0.6, n)
            num_compromised = rng.integers(1, 4, n)
            flag_choice = rng.choice(flags, n, p=[0.85, 0.05, 0.05, 0.025, 0.025])
            attack_type = "buffer_overflow"

        block = pd.DataFrame(
            {
                "duration": duration,
                "protocol_type": rng.choice(protocols, n, p=[0.7, 0.2, 0.1]),
                "service": rng.choice(services, n),
                "flag": flag_choice,
                "src_bytes": src_bytes,
                "dst_bytes": dst_bytes,
                "land": np.zeros(n),
                "wrong_fragment": np.zeros(n),
                "urgent": np.zeros(n),
                "hot": np.zeros(n),
                "num_failed_logins": num_failed_logins,
                "logged_in": logged_in,
                "num_compromised": num_compromised,
                "root_shell": root_shell,
                "su_attempted": np.zeros(n),
                "num_root": np.zeros(n),
                "num_file_creations": num_file_creations,
                "num_shells": num_shells,
                "num_access_files": np.zeros(n),
                "num_outbound_cmds": np.zeros(n),
                "is_host_login": np.zeros(n),
                "is_guest_login": np.zeros(n),
                "count": count,
                "srv_count": srv_count,
                "serror_rate": serror_rate,
                "srv_serror_rate": np.clip(serror_rate + rng.normal(0, 0.02, n), 0, 1),
                "rerror_rate": np.clip(rng.normal(0.02, 0.03, n), 0, 1),
                "srv_rerror_rate": np.clip(rng.normal(0.02, 0.03, n), 0, 1),
                "same_srv_rate": np.clip(1 - diff_srv_rate, 0, 1),
                "diff_srv_rate": diff_srv_rate,
                "srv_diff_host_rate": np.clip(rng.normal(0.05, 0.05, n), 0, 1),
                "dst_host_count": rng.integers(1, 255, n),
                "dst_host_srv_count": rng.integers(1, 255, n),
                "dst_host_same_srv_rate": np.clip(1 - diff_srv_rate, 0, 1),
                "dst_host_diff_srv_rate": diff_srv_rate,
                "dst_host_same_src_port_rate": np.clip(rng.normal(0.2, 0.2, n), 0, 1),
                "dst_host_srv_diff_host_rate": np.clip(rng.normal(0.05, 0.05, n), 0, 1),
                "dst_host_serror_rate": serror_rate,
                "dst_host_srv_serror_rate": serror_rate,
                "dst_host_rerror_rate": np.clip(rng.normal(0.02, 0.03, n), 0, 1),
                "dst_host_srv_rerror_rate": np.clip(rng.normal(0.02, 0.03, n), 0, 1),
                "label": attack_type,
            }
        )
        rows.append(block)

    df = pd.concat(rows, ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df[ALL_COLUMNS_WITH_LABEL[:-1]]  # all feature cols + label, no difficulty
