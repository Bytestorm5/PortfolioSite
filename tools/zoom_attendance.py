
from __future__ import annotations
import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np
import warnings
from typing import IO, Iterable, Sequence, Any

def _get_session_total(path: Path, sheet_name: str | None = None) -> float | None:
    """
    Extract meeting total duration (minutes) from metadata block if present.
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            df0 = pd.read_csv(path)
            cols = [str(c).strip().lower() for c in df0.columns]
            if "duration (minutes)" in cols:
                col = cols.index("duration (minutes)")
                series = pd.to_numeric(df0.iloc[:, col], errors="coerce")
                val = series.dropna()
                if not val.empty:
                    return float(val.iloc[0])
        elif suffix in (".xls", ".xlsx"):
            df0 = pd.read_excel(path, nrows=5, sheet_name=sheet_name)
            cols = [str(c).strip().lower() for c in df0.columns]
            if "duration (minutes)" in cols:
                col = cols.index("duration (minutes)")
                series = pd.to_numeric(df0.iloc[:, col], errors="coerce")
                val = series.dropna()
                if not val.empty:
                    return float(val.iloc[0])
    except Exception:
        return None
    return None


# -------------------------------
# Helper utilities
# -------------------------------

def _read_sheet(path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    """
    Read a participants_* file (.csv or .xlsx). Return a DataFrame with at least:
      - 'email' (lowercased, stripped)
      - 'duration' (numeric, in minutes)
    The function tries a variety of common column names.
    """
    suffix = path.suffix.lower()

    if suffix == ".csv":
        # First try a straightforward read
        df = pd.read_csv(path)
        # If we don't find an email/duration later, we'll attempt a header-scan pass.
    elif suffix in (".xls", ".xlsx"):
        # Read the first sheet by default
        try:
            df = pd.read_excel(path, sheet_name=sheet_name)
        except ValueError as exc:  # sheet not found or similar
            raise ValueError(f"Failed to read sheet '{sheet_name}' in {path.name}: {exc}") from exc
    else:
        raise ValueError(f"Unsupported file type: {suffix} ({path.name})")

    # Normalize column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Common email column variants
    email_cols = [
        "email", "user email", "user_email", "participant email", "participant_email",
        "mail", "e-mail"
    ]
    email_col = next((c for c in email_cols if c in df.columns), None)

    if email_col is None:
        # try to guess: any column containing 'email'
        email_col = next((c for c in df.columns if "email" in c), None)

    # If still not found, try scanning for a header row inside the CSV (Zoom export style)
    if email_col is None and suffix == ".csv":
        raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
        header_row_idx = None
        for i in range(min(len(raw), 50)):  # search first 50 rows for a header containing 'email'
            row_vals = [str(x).strip().lower() for x in raw.iloc[i].tolist()]
            if any(v == "email" for v in row_vals):
                header_row_idx = i
                break
        if header_row_idx is not None:
            # Reconstruct dataframe from this header row onward
            header = [str(x).strip() if str(x).strip() != "" else f"col_{j}" for j, x in enumerate(raw.iloc[header_row_idx])]
            df = raw.iloc[header_row_idx+1:].copy()
            df.columns = [str(c).strip().lower() for c in header]
            # Update normalized df for subsequent logic
            # (Note: we already normalized to lowercase below after this block)
        else:
            # leave df as initially read; we'll error out below if columns still not found
            pass

        # Re-run normalized columns
        df.columns = [str(c).strip().lower() for c in df.columns]
        email_col = next((c for c in df.columns if c == "email"), None)
        if email_col is None:
            email_col = next((c for c in df.columns if "email" in c), None)

    if email_col is None:
        raise ValueError(f"Could not find an email column in {path.name}. Columns: {list(df.columns)}")

    # Common duration column variants
    dur_candidates = [
        "total duration (minutes)", "duration (minutes)", "duration_minutes",
        "duration", "total duration", "attendance duration", "time in meeting (minutes)",
        "minutes"
    ]
    dur_col = next((c for c in dur_candidates if c in df.columns), None)

    if dur_col is None:
        # fallback: any column containing 'duration' or 'minutes'
        dur_col = next((c for c in df.columns if ("duration" in c or "minutes" in c)), None)

    # If still not found (Zoom export often uses 'time in meeting (minutes)')
    if dur_col is None and suffix == ".csv":
        # Look explicitly for 'time in meeting (minutes)'
        for c in df.columns:
            if "time in meeting" in c and "minute" in c:
                dur_col = c
                break

    if dur_col is None:
        raise ValueError(f"Could not find a duration column in {path.name}. Columns: {list(df.columns)}")

    # Clean email and duration
    out = pd.DataFrame({
        "email": df[email_col].astype(str).str.strip().str.lower(),
        "duration": pd.to_numeric(df[dur_col], errors="coerce")
    })
    out = out[out["email"].notna() & out["email"].str.contains("@")]
    out["duration"] = out["duration"].fillna(0)

    # In case the input has multiple rows per person in a day file, aggregate (sum) per email
    out = out.groupby("email", as_index=False)["duration"].sum()

    return out


def _colname_from_filename(path: Path) -> str:
    """
    Construct a friendly column name from the file name.
    Tries to extract a YYYY_MM_DD (or YYYY-MM-DD) date. Falls back to the base name.
    """
    name = path.stem  # no suffix
    # Look for date patterns
    m = re.search(r"(20\d{2})[_-](\d{1,2})[_-](\d{1,2})", name)
    if m:
        y, mo, d = m.groups()
        try:
            mo = int(mo)
            d = int(d)
            return f"{int(y):04d}-{mo:02d}-{d:02d}"
        except Exception:
            pass
    # As a secondary try, if the file is like participants_<id>_YYYY_MM_DD
    return name


def _derive_day_label(path: Path, sheet_name: str | None, existing: set[str]) -> str:
    """
    Produce a display/column name for a given file and optional sheet, while
    keeping names unique inside a run.
    """
    base = _colname_from_filename(path)
    candidate = base

    if sheet_name:
        sheet_candidate = _colname_from_filename(Path(sheet_name))
        normalized = sheet_name.strip()
        default_sheet_names = {"sheet", "sheet1", "sheet 1"}

        if sheet_candidate.lower() not in default_sheet_names:
            if sheet_candidate == base:
                candidate = base
            else:
                candidate = sheet_candidate if re.search(r"\d", sheet_candidate) else f"{base} - {sheet_candidate}"
        elif normalized:
            candidate = f"{base} - {normalized}"

    unique = candidate
    i = 2
    while unique in existing:
        unique = f"{candidate} ({i})"
        i += 1

    existing.add(unique)
    return unique


def _build_matrix_from_day_frames(day_frames: Iterable[tuple[str, pd.DataFrame, float | None, Any]]) -> tuple[pd.DataFrame, list[str], dict[str, float], dict[str, Any]]:
    """
    Shared helper to merge day-wise attendance dataframes into a single matrix.
    """
    day_frames = list(day_frames)
    if not day_frames:
        raise SystemExit("No attendance data found.")

    all_emails: set[str] = set()
    day_names: list[str] = []
    session_totals: dict[str, float] = {}
    day_sources: dict[str, Any] = {}

    normalized_frames: list[tuple[str, pd.DataFrame]] = []
    for day_name, df_day, session_total, source in day_frames:
        day_names.append(day_name)
        normalized_frames.append((day_name, df_day.rename(columns={"duration": day_name})))
        all_emails.update(df_day["email"].unique())
        session_totals[day_name] = float(session_total) if session_total is not None else float('nan')
        day_sources[day_name] = source

    matrix = pd.DataFrame({"Email": sorted(all_emails)})
    for day_name, df_day in normalized_frames:
        matrix = matrix.merge(df_day, how="left", left_on="Email", right_on="email")
        matrix = matrix.drop(columns=["email"])

    for c in day_names:
        if c in matrix.columns:
            matrix[c] = pd.to_numeric(matrix[c], errors="coerce").fillna(0)

    return matrix[["Email"] + day_names], day_names, session_totals, day_sources


def apply_day_label_overrides(
    df: pd.DataFrame,
    day_cols: list[str],
    session_totals: dict[str, float],
    label_map: dict[str, str] | None = None,
    day_sources: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, float], dict[str, str], dict[str, Any]]:
    """
    Rename day columns according to label_map (original -> desired).
    Ensures uniqueness by suffixing (n) if needed.
    Returns (df, new_day_cols, new_session_totals, applied_map).
    """
    if not label_map:
        return df, day_cols, session_totals, {}, day_sources or {}

    applied: dict[str, str] = {}
    used: set[str] = set()
    rename_map: dict[str, str] = {}
    new_day_cols: list[str] = []
    new_session_totals: dict[str, float] = {}
    new_day_sources: dict[str, Any] = {}

    for col in day_cols:
        target = str(label_map.get(col, col)).strip()
        if not target:
            target = col
        candidate = target
        i = 2
        while candidate in used:
            candidate = f"{target} ({i})"
            i += 1
        used.add(candidate)
        applied[col] = candidate
        rename_map[col] = candidate
        new_day_cols.append(candidate)
        new_session_totals[candidate] = session_totals.get(col, float("nan"))
        if day_sources is not None:
            new_day_sources[candidate] = day_sources.get(col)

    df = df.rename(columns=rename_map)
    return df, new_day_cols, new_session_totals, applied, new_day_sources


def build_attendance_matrix(input_dir: Path) -> tuple[pd.DataFrame, list[str], dict[str, float], dict[str, Any]]:
    """
    Scan input_dir for files matching participants_*.* csv|xlsx (or any .csv/.xls/.xlsx if none match),
    build a matrix:
      - rows: unique emails
      - columns: one per file/sheet/day with attendance duration in minutes
    Returns (df, day_cols, session_totals) where df has 'Email' as first col, followed by day columns.
    """
    files = sorted([
        p for p in input_dir.glob("participants_*.*")
        if p.suffix.lower() in (".csv", ".xlsx", ".xls") and not p.name.startswith("~$")
    ])
    if not files:
        files = sorted([
            p for p in input_dir.glob("*.*")
            if p.suffix.lower() in (".csv", ".xlsx", ".xls") and not p.name.startswith("~$")
        ])
    if not files:
        raise SystemExit(f"No attendance files (.csv/.xls/.xlsx) found in {input_dir}")

    day_frames = []
    used_names: set[str] = set()

    for f in files:
        suffix = f.suffix.lower()
        if suffix in (".xls", ".xlsx"):
            try:
                with pd.ExcelFile(f) as workbook:
                    sheet_names = list(workbook.sheet_names)
            except Exception as e:
                warnings.warn(f"Skipping {f.name}: unable to read workbook ({e})")
                continue

            if not sheet_names:
                warnings.warn(f"Skipping {f.name}: no sheets detected")
                continue

            for sheet in sheet_names:
                try:
                    df_day = _read_sheet(f, sheet_name=sheet)
                except Exception as e:
                    warnings.warn(f"Skipping {f.name} [{sheet}]: {e}")
                    continue

                colname = _derive_day_label(f, sheet, used_names)
                st = _get_session_total(f, sheet_name=sheet)
                source = {"file": f.name, "sheet": sheet, "source": f"{f.name} [{sheet}]"}
                day_frames.append((colname, df_day, st, source))
        else:
            try:
                df_day = _read_sheet(f)
            except Exception as e:
                warnings.warn(f"Skipping {f.name}: {e}")
                continue

            colname = _derive_day_label(f, None, used_names)
            st = _get_session_total(f)
            source = {"file": f.name, "sheet": None, "source": f.name}
            day_frames.append((colname, df_day, st, source))

    if not day_frames:
        raise SystemExit("No valid input files after parsing.")

    return _build_matrix_from_day_frames(day_frames)


def apply_thresholds(df: pd.DataFrame, day_cols: Sequence[str], thresholds: dict[str, float]) -> tuple[pd.Series, dict[str, float]]:
    """
    Apply per-day minute thresholds to the attendance matrix.
    Returns (flags, sanitized_thresholds) where flags is a boolean Series indicating
    whether each attendee met or exceeded the threshold on every day (0 counts as below threshold).
    """
    sanitized: dict[str, float] = {}
    for c in day_cols:
        raw_val = thresholds.get(c, 0)
        try:
            sanitized[c] = float(raw_val)
        except Exception:
            sanitized[c] = 0.0

    def row_ok(row):
        for c in day_cols:
            val = float(row[c])
            thr = sanitized.get(c, 0.0)
            if val < thr:
                return False
        return True

    flags = df.apply(row_ok, axis=1)
    return flags, sanitized


def compute_percentile_flags(df: pd.DataFrame, day_cols: list[str], N: float) -> pd.Series:
    """
    For each day column, compute the Nth percentile of durations across *that day*.
    Then return (flags, thresholds) where flags is True only if the attendee is
    at/above the Nth percentile for all days they have non-zero attendance.

    (If an attendee didn't attend a day (0 minutes), that day is not required for 'True'.)
    """
    thresholds = {}
    for c in day_cols:
        # Compute threshold for the day based on that column's distribution
        col = pd.to_numeric(df[c], errors="coerce").fillna(0)
        # If all zeros, threshold is 0
        if (col > 0).any():
            thresholds[c] = float(np.percentile(col[col >= 0], N * 100.0))
        else:
            thresholds[c] = 0.0

    flags, sanitized = apply_thresholds(df, day_cols, thresholds)
    return flags, sanitized




def save_to_excel(df: pd.DataFrame, day_cols: list[str], thresholds: dict[str, float], out_path: Path | IO, session_totals: dict[str, float] | None = None, percentile: float | None = None):
    """
    Save:
      - Sheet 'Thresholds': Day | Session_Total_Minutes | Threshold_Minutes (direct values, editable)
      - Sheet 'Attendance': Email + day columns + All_Above_Threshold formula referencing the thresholds table.
    """
    from xlsxwriter.utility import xl_rowcol_to_cell

    out_df = df.copy()
    thresholds_map = session_totals or {}
    threshold_values = [float(thresholds.get(d, 0.0)) for d in day_cols]
    session_values = [float(thresholds_map.get(d, float("nan"))) for d in day_cols]

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        # Thresholds sheet
        th_name = "Thresholds"
        out_th = pd.DataFrame({
            "Day": day_cols,
            "Session_Total_Minutes": session_values,
            "Threshold_Minutes": threshold_values,
        })
        out_th.to_excel(writer, index=False, sheet_name=th_name, startrow=1)
        th_ws = writer.sheets[th_name]
        th_ws.write(0, 0, "Percentile_Base")
        if percentile is not None:
            th_ws.write_number(0, 1, float(percentile))
        else:
            th_ws.write(0, 1, "")

        th_header_row = 1
        th_first_data_row = th_header_row + 1
        th_last_data_row = th_first_data_row + len(day_cols) - 1

        th_ws.add_table(th_header_row, 0, th_last_data_row, 2, {
            "columns": [
                {"header": "Day"},
                {"header": "Session_Total_Minutes"},
                {"header": "Threshold_Minutes"},
            ],
            "style": "Table Style Medium 9",
        })

        # Attendance sheet
        at_name = "Attendance"
        out_df.to_excel(writer, index=False, sheet_name=at_name, startrow=1)
        at_ws = writer.sheets[at_name]

        nrows, ncols = out_df.shape
        header_row = 1
        first_data_row = header_row + 1
        last_data_row = header_row + nrows

        bool_col_index = ncols
        at_ws.write(header_row, bool_col_index, "All_Above_Threshold")

        # Build AND of comparisons against thresholds from Thresholds sheet
        for r in range(first_data_row, last_data_row + 1):
            conditions = []
            for j, day in enumerate(day_cols):
                day_col = 1 + j
                cell = xl_rowcol_to_cell(r, day_col)
                day_header_cell = xl_rowcol_to_cell(header_row, day_col, row_abs=True, col_abs=True)
                thr = (
                    f"INDEX({th_name}!$C$3:$C${th_last_data_row+1},"
                    f"MATCH({at_name}!{day_header_cell},{th_name}!$A$3:$A${th_last_data_row+1},0))"
                )
                conditions.append(f"IF({cell}=0,FALSE,{cell}>={thr})")
            and_expr = ",".join(conditions) if conditions else "FALSE"
            at_ws.write_formula(r, bool_col_index, f"=AND({and_expr})")

        table_end_row = last_data_row
        table_end_col = bool_col_index
        at_ws.add_table(header_row, 0, table_end_row, table_end_col, {
            "columns": [{"header": col} for col in list(out_df.columns)] + [{"header": "All_Above_Threshold"}],
            "style": "Table Style Medium 9",
        })

        # Autofit
        out_cols = list(out_df.columns) + ["All_Above_Threshold"]
        for i, col in enumerate(out_cols):
            series = out_df[col] if col in out_df.columns else []
            max_len = max([len(str(col))] + [len(str(x)) for x in series])
            at_ws.set_column(i, i, min(50, max(12, max_len + 2)))


def main():
    parser = argparse.ArgumentParser(description="Aggregate attendance durations across participants_* files and flag Nth-percentile attendees per day.")
    parser.add_argument("--input", type=str, default="input", help="Input folder containing participants_* files (.csv/.xlsx). Default: ./input")
    parser.add_argument("--output", type=str, default="attendance_summary.xlsx", help="Output Excel file path. Default: attendance_summary.xlsx")
    parser.add_argument("--percentile", type=float, default=0.9, help="N as a fraction (e.g., 0.9 for 90th percentile). Default: 0.9")
    args = parser.parse_args()

    input_dir = Path(args.input)
    out_path = Path(args.output)
    N = float(args.percentile)
    if not (0 < N < 1):
        raise SystemExit("--percentile must be between 0 and 1 (e.g., 0.9 for 90th)")

    df, day_cols, session_totals, _ = build_attendance_matrix(input_dir)
    _, thresholds = compute_percentile_flags(df, day_cols, N)
    save_to_excel(df, day_cols, thresholds, out_path, session_totals=session_totals, percentile=N)
    print(f"Wrote {out_path} with {len(df)} rows and {len(day_cols)} day columns.")

if __name__ == "__main__":
    main()
