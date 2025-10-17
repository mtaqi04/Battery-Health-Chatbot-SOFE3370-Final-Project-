# 📂 data/

**Purpose:** Store the dataset and any cleaned/derived artifacts.

## Expected contents
- `PulseBat Dataset.xlsx` — raw dataset (U1–U21 + SOH).
- `PulseBat Data Description.md` — feature & collection notes.
- `cleaned_pulsebat.csv` — **OUTPUT** after Sprint 1 Task 2 (U1–U21 + SOH only).

## When you implement
1. Document any cleaning (rows removed, imputation, scaling).
2. Note schema (types, ranges; SOH should be in [0, 1]).
3. Keep raw vs processed clearly separated (consider `/raw` and `/processed`).
