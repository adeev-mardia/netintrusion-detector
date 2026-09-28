# netintrusion-detector

A machine-learning network intrusion / anomaly detector trained and evaluated
on **NSL-KDD**, the standard public benchmark for offline, flow-feature-based
network intrusion detection research.

## Data provenance (read this first)

This project uses the **real, public NSL-KDD dataset**, downloaded directly
from a well-known public GitHub mirror of the official release:

- Train: <https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt> (125,973 rows)
- Test: <https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt> (22,544 rows)

These are plain, freely redistributable text files — NSL-KDD is an improved,
de-duplicated revision of the classic 1999 DARPA/KDD Cup intrusion dataset,
built specifically to be a fair, reusable ML benchmark. **The model shipped
and reported on in this README was trained on this real data, not synthetic
data.**

`src/netintrusion_detector/data.py` also ships a clearly-named
`generate_synthetic_nslkdd_like(...)` function. **It was not needed and was
not used for this project's results** — the real dataset downloaded
successfully. It exists purely as a documented, honest fallback: if some
future environment can't reach the mirror above (e.g. a stricter network
sandbox), `data.py`'s loader (`load_nslkdd`) is written generically enough to
load either the real `KDDTrain+.txt` / `KDDTest+.txt` files *or* a CSV
produced by the synthetic generator, using the same column schema, so nothing
else in the package needs to change. Anything that function produces is
always labeled `synthetic` and must never be described as real NSL-KDD data.

To reproduce the real download yourself:

```bash
mkdir -p data/raw
curl -o data/raw/KDDTrain+.txt "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
curl -o data/raw/KDDTest+.txt  "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
```

Because the full files are ~19MB/~3MB, they are **not** committed to git
(`data/raw/*.txt` is gitignored). Instead, `data/sample/` bundles small,
deterministic, stratified CSV samples of the *real* data (built with
`scripts/make_sample_data.py`, one row per attack type up to a cap, fixed
seed) so the package, its tests, and the CLI are usable instantly with no
download and no network access required.

## Why an offline, pre-extracted-feature benchmark is legitimate

NSL-KDD (like the original KDD Cup '99 set it improves on) does **not**
require live packet capture. Each row is already a fully-extracted flow
record: 41 features summarizing one network connection (duration, byte
counts, protocol/service/flag, counts of failed logins, rates of SYN/REJ
errors over recent connections to the same host/service, etc.), plus a label
of `normal` or one of ~39 specific attack types. This is exactly the standard
methodology used throughout the published network-intrusion-detection ML
literature: it isolates the *feature engineering already agreed to be
security-relevant* from the *detection model*, so different papers/models
can be compared apples-to-apples on identical, offline, fully-reproducible
data. That's what this package does.

## Problem setup

- **41 features**: 3 categorical (`protocol_type`, `service`, `flag`) + 38
  numeric (durations, byte counts, connection/error-rate statistics, etc.)
  — the standard, public NSL-KDD schema.
- **Attack categories**: each of the ~39 specific attack types is mapped to
  one of four standard categories:
  - **DoS** — denial of service floods (`neptune`, `smurf`, `back`, `pod`, `teardrop`, `land`, ...)
  - **Probe** — surveillance/scanning (`satan`, `portsweep`, `ipsweep`, `nmap`, ...)
  - **R2L** — remote-to-local unauthorized access attempts (`guess_passwd`, `ftp_write`, `warezmaster`, ...)
  - **U2R** — user-to-root privilege escalation (`buffer_overflow`, `rootkit`, `loadmodule`, ...)
- **Two supervised tasks**:
  - *Binary*: `normal` (0) vs `attack` (1) — the practical "should this flow raise an alert?" question.
  - *Multiclass*: `normal` / `DoS` / `Probe` / `R2L` / `U2R` — what *kind* of attack, for triage.
- NSL-KDD's official test set deliberately contains attack types **not
  present in the training set**, to test generalization to novel attacks —
  this is a well-known, intentional property of the benchmark (not a bug in
  this project's split), and it's part of why test accuracy on NSL-KDD is
  famously lower than train accuracy across the entire published literature,
  not just for this implementation.

## Model

`RandomForestClassifier` (scikit-learn), `n_estimators=200`,
`class_weight="balanced_subsample"`, wrapped in a single
`ColumnTransformer` → `Pipeline`:

- categorical features (`protocol_type`, `service`, `flag`) → `OneHotEncoder(handle_unknown="ignore")`
- numeric features (38 columns) → `StandardScaler`

A random forest is a defensible, realistic choice for this kind of tabular,
mixed categorical/numeric security data: it captures nonlinear interactions
between features (e.g. high `count` + high `serror_rate` together, not
either alone, is what signals a DoS flood) without needing careful manual
feature-scale tuning, and it's used throughout the published NSL-KDD
literature as a strong baseline — a meaningfully more capable choice than a
single decision stump.

## Actual results (real run, real NSL-KDD test set)

Trained on the full real `KDDTrain+.txt` (125,973 rows), evaluated on the
full real, official `KDDTest+.txt` (22,544 rows) — **not** train/test-split
from the same pool, but NSL-KDD's own canonical, harder test file (this is
why the numbers below are lower than the ~99%+ accuracy that's easy to get
by evaluating on a random split of the training file alone — that would be a
misleadingly easy benchmark, and this project deliberately avoids it).

```
Binary (normal vs attack) evaluation
----------------------------------------
Accuracy:  0.7775
Precision: 0.9684
Recall:    0.6298
F1:        0.7632
FPR:       0.0272
Confusion matrix: TN=9447 FP=264 FN=4751 TP=8082

FPR / recall at operating thresholds:
 threshold      FPR   recall
       0.3   0.0316   0.7127
       0.5   0.0272   0.6314
       0.7   0.0255   0.5651

Multiclass (normal/DoS/Probe/R2L/U2R) evaluation
--------------------------------------------------
Accuracy:      0.7385
Macro precision: 0.7448
Macro recall:    0.4742
Macro F1:        0.4779

     class  precision     recall         f1
       DoS     0.9615     0.7700     0.8552
     Probe     0.8579     0.5936     0.7017
       R2L     0.7692     0.0035     0.0069
       U2R     0.5000     0.0299     0.0563
    normal     0.6356     0.9738     0.7692

Confusion matrix (rows=true, cols=pred), labels=['DoS', 'Probe', 'R2L', 'U2R', 'normal']:
  5743     48      0      0   1667
   163   1437      0      0    821
     0      4     10      1   2872
     0      0      3      2     62
    67    186      0      1   9457
```

(Full JSON: reproduce with `netintrusion-detector evaluate --output-json report.json`.)

### Reading these numbers honestly

- **High precision (0.97), low-ish recall (0.63) on the binary task, with a
  low FPR (2.7%)**: the model is conservative — when it flags something as
  an attack it's almost always right, and it rarely cries wolf on real
  normal traffic, but it misses over a third of attacks in the official
  test file. That miss rate is inflated by novel attack types in the test
  set that never appeared in training (a known, deliberate NSL-KDD design
  choice) — a random forest trained only on `KDDTrain+` has no way to have
  learned their signature. This is exactly the accuracy range (upper 70s%)
  widely reported for standard classifiers evaluated on the official
  `KDDTest+` file in the published literature, as opposed to the
  unrealistically high (>99%) numbers that show up when a paper instead
  evaluates on a random split of the training file alone.
- **R2L and U2R are, as expected, the hardest classes** (recall 0.003 and
  0.03 respectively) — this reproduces a well-documented, famous property
  of NSL-KDD, for two compounding reasons visible directly in this run's
  own confusion matrix: (1) severe class imbalance — R2L/U2R are a tiny
  fraction of training rows compared to DoS/normal, so the classifier has
  very little signal to learn their pattern from, and (2) both categories
  are attacks that, at the flow-feature level used here, look deceptively
  close to normal traffic (R2L is often just one failed/successful login
  attempt; U2R happens *after* a legitimate login). Both categories are
  historically the hardest on this exact benchmark across the literature,
  not just in this implementation.
- **FPR at different thresholds** (0.3 / 0.5 / 0.7): lowering the
  probability threshold to flag "attack" trades a higher false-positive
  rate (more normal traffic incorrectly alerted on) for higher recall
  (catching more real attacks) — the operational trade-off a security
  analyst has to tune based on how much alert volume their team can absorb.

## Design notes: why these evaluation choices matter operationally

- **FPR is reported explicitly, not just accuracy/F1**, because in a real
  deployment every flagged flow becomes a ticket for a human analyst to
  triage. A detector with excellent recall but a high FPR causes "alert
  fatigue" — analysts start ignoring or disabling it — so FPR at the actual
  deployed threshold is often the single most important operational number,
  arguably more than raw accuracy.
- **Evaluation uses the official, harder `KDDTest+` file** (with novel
  attack types absent from training) rather than a random split of the
  training file, because the latter substantially overstates real-world
  generalization — it's a well-known pitfall in intrusion-detection ML
  papers that this project deliberately avoids.
- **Multiclass macro metrics are reported per-class**, not just as one
  averaged number, specifically because macro-averaging can hide that a
  model has effectively failed on rare-but-important classes (R2L/U2R
  here) while still looking "good" overall.

## Repository layout

```
netintrusion-detector/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/netintrusion_detector/
│   ├── data.py            # schema, attack-category mapping, real-data loader, synthetic fallback generator
│   ├── preprocessing.py   # ColumnTransformer/Pipeline: one-hot + scaling
│   ├── model.py           # RandomForest train/predict/save/load (binary + multiclass)
│   ├── evaluate.py        # accuracy/precision/recall/F1, FPR @ thresholds, confusion matrices
│   └── cli.py             # `netintrusion-detector train|evaluate|predict`
├── scripts/make_sample_data.py   # deterministic stratified sample of the real data
├── data/
│   ├── raw/               # full real KDDTrain+.txt / KDDTest+.txt (gitignored; see above to fetch)
│   └── sample/             # small bundled real-data CSV samples (committed)
└── tests/
```

## Installation

```bash
python3 -m pip install -e ".[dev]"
```

## Usage

```bash
# Train both binary and multiclass models (uses data/raw/ if present,
# otherwise automatically falls back to the bundled data/sample/ CSVs).
netintrusion-detector train --task both

# Evaluate on the test set and print a report (+ optional JSON export).
netintrusion-detector evaluate --task both --output-json report.json

# Predict on any CSV with the 41 NSL-KDD feature columns.
netintrusion-detector predict path/to/flows.csv --task binary --output-csv preds.csv
```

All three subcommands accept `--train-path` / `--test-path` / `--model-dir`
to point at custom files/locations.

## Tests

```bash
python3 -m pytest -v
```

26 tests, fully offline and deterministic (fixed random seeds throughout,
run against the bundled real-data sample). Covers: dataset schema and
loader correctness, the standard attack-type → category mapping, the
preprocessing pipeline's shape/round-trip behavior (including graceful
handling of a categorical value never seen during training), model training
and prediction shape/label correctness for both tasks, model save/load
round-tripping, evaluation metrics checked against hand-computed values on
a tiny fixture (not just trusted from sklearn), and an end-to-end CLI
train → evaluate → predict run on the bundled sample data.
