import io
import json
import math
import platform
from pathlib import Path
from tempfile import TemporaryDirectory

from flask import Flask, render_template, send_file, request, jsonify, url_for
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta

from tools.zoom_attendance import (
    apply_thresholds,
    build_attendance_matrix,
    compute_percentile_flags,
    save_to_excel,
    apply_day_label_overrides,
)

load_dotenv()

app = Flask(__name__)
DEFAULT_PERCENTILE = 0.9


def _clean_number(value):
    try:
        num = float(value)
    except Exception:
        return None
    return None if math.isnan(num) else num


def _build_matrix_from_uploaded_files(file_storages):
    uploads = [f for f in file_storages if f and f.filename]
    if not uploads:
        raise ValueError("Please upload at least one CSV/XLSX file.")

    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for upload in uploads:
            filename = secure_filename(upload.filename)
            if not filename:
                continue
            dest = tmp_path / filename
            upload.save(dest)

        df, day_cols, session_totals, day_sources = build_attendance_matrix(tmp_path)

    return df, day_cols, session_totals, day_sources


def _parse_day_labels(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception as exc:
        raise ValueError(f"Invalid day_labels payload: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("day_labels must be a JSON object.")
    return {str(k): str(v) for k, v in data.items()}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tools')
def tools_index():
    tools = [{
        "name": "Zoom Attendance Aggregator",
        "description": "Upload Zoom participant exports and get an aggregated sheet showing student attendance, with configurable attendance thresholds.",
        "href": url_for('zoom_attendance_page'),
        "pill": "New" if ((datetime.now() - datetime(2025, 11, 29)).days <= 14) else None
    }]
    return render_template('tools/index.html', tools=tools)


@app.route("/tools/zoom_attendance")
def zoom_attendance_page():
    return render_template("tools/zoom_attendance.html")


@app.post("/tools/zoom_attendance/parse")
def zoom_attendance_parse():
    try:
        label_overrides = _parse_day_labels(request.form.get("day_labels"))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        df, day_cols, session_totals, day_sources = _build_matrix_from_uploaded_files(request.files.getlist("files"))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    raw_day_cols = list(day_cols)
    df, day_cols, session_totals, applied_map, day_sources = apply_day_label_overrides(
        df, day_cols, session_totals, label_overrides, day_sources=day_sources
    )
    flags, thresholds = compute_percentile_flags(df, day_cols, DEFAULT_PERCENTILE)

    rows = []
    for idx, row in df.iterrows():
        record = {"Email": row["Email"]}
        for day in day_cols:
            record[day] = float(row[day])
        record["metThreshold"] = bool(flags.iloc[idx])
        rows.append(record)

    day_meta = []
    for idx, day in enumerate(day_cols):
        info = day_sources.get(day, {}) if isinstance(day_sources, dict) else {}
        day_meta.append({
            "label": day,
            "raw_label": raw_day_cols[idx] if idx < len(raw_day_cols) else day,
            "session_total": _clean_number(session_totals.get(day)),
            "threshold": float(thresholds.get(day, 0.0)),
            "source": info.get("source"),
            "file": info.get("file"),
            "sheet": info.get("sheet"),
        })

    return jsonify({
        "day_columns": day_cols,
        "raw_day_columns": raw_day_cols,
        "applied_day_labels": applied_map,
        "rows": rows,
        "thresholds": {day: float(thresholds.get(day, 0.0)) for day in day_cols},
        "session_totals": {day: _clean_number(val) for day, val in session_totals.items()},
        "percentile": DEFAULT_PERCENTILE,
        "day_meta": day_meta,
    })


@app.post("/tools/zoom_attendance/export")
def zoom_attendance_export():
    try:
        label_overrides = _parse_day_labels(request.form.get("day_labels"))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    thresholds_raw = request.form.get("thresholds", "{}")
    try:
        threshold_map = json.loads(thresholds_raw) if thresholds_raw else {}
        if not isinstance(threshold_map, dict):
            raise ValueError("Thresholds payload must be an object.")
    except Exception as exc:
        return jsonify({"error": f"Invalid thresholds payload: {exc}"}), 400

    try:
        df, day_cols, session_totals, day_sources = _build_matrix_from_uploaded_files(request.files.getlist("files"))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    df, day_cols, session_totals, _, _ = apply_day_label_overrides(
        df, day_cols, session_totals, label_overrides
    )
    _, sanitized = apply_thresholds(df, day_cols, threshold_map)

    output = io.BytesIO()
    save_to_excel(df, day_cols, sanitized, output, session_totals=session_totals)
    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="attendance_summary.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == '__main__':
    if platform.system() == 'Windows':
        app.config["TEMPLATES_AUTO_RELOAD"] = True
        app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
        app.run(debug=True, port=5000)
    else:
        app.run(port=80, host='0.0.0.0')
