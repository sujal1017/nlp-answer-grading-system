import re
import sqlite3
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, root_mean_squared_error


def is_target_student(exam_id: str, student_id: str) -> bool:
    if exam_id != "IA1_RM_2026":
        return False
    sid = str(student_id).strip()
    return bool(re.fullmatch(r"\d{3}", sid)) and (1 <= int(sid) <= 37)


def compute_ia_total(scores: dict[str, float]) -> float:
    q1 = scores.get("Q1A", 0.0) + scores.get("Q1B", 0.0)
    q2 = scores.get("Q2A", 0.0) + scores.get("Q2B", 0.0)
    q3 = scores.get("Q3A", 0.0) + scores.get("Q3B", 0.0)
    return max(q1, q2) + q3


def compute_metrics(y_true: list[float], y_pred: list[float], max_marks: float, tolerance: float) -> dict:
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    diffs = yp - yt  # AI - Actual

    mae = mean_absolute_error(yt, yp)
    rmse = root_mean_squared_error(yt, yp)
    norm_acc = (1.0 - (mae / max_marks)) * 100.0
    exact_acc = np.mean(np.isclose(yt, yp, atol=1e-5)) * 100.0
    tol_acc = np.mean(np.abs(diffs) <= tolerance) * 100.0
    mean_bias = float(np.mean(diffs))

    # Discretize half-marks (x2) for Quadratic Weighted Kappa
    yt_int = np.round(yt * 2).astype(int)
    yp_int = np.round(yp * 2).astype(int)
    qwk = cohen_kappa_score(yt_int, yp_int, weights="quadratic") if len(set(yt_int)) > 1 else None

    r_val = float(pearsonr(yt, yp)[0]) if len(set(yt)) > 1 and len(set(yp)) > 1 else None
    rho_val = float(spearmanr(yt, yp)[0]) if len(set(yt)) > 1 and len(set(yp)) > 1 else None

    return {
        "N": len(yt),
        "MAE": round(float(mae), 3),
        "RMSE": round(float(rmse), 3),
        "Normalized_Accuracy_%": round(float(norm_acc), 2),
        "Exact_Match_%": round(float(exact_acc), 2),
        f"Within_±{tolerance}_%": round(float(tol_acc), 2),
        "Mean_Diff_(AI-Actual)": round(mean_bias, 3),
        "Pearson_r": round(r_val, 3) if r_val is not None else "N/A",
        "Spearman_rho": round(rho_val, 3) if rho_val is not None else "N/A",
        "QWK_Kappa": round(float(qwk), 3) if qwk is not None else "N/A",
    }


def run_evaluation(db_path: str = "grading_system.db"):
    # Read-only connection: guarantees zero modifications to grading_system.db
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()

    cur.execute("""
        SELECT exam_id, student_id, question_id, ai_score, teacher_score, teacher_verified
        FROM submissions
    """)
    all_rows = cur.fetchall()
    conn.close()

    # Filter strictly to IA1_RM_2026 and students 001..037
    valid_rows = [r for r in all_rows if is_target_student(r[0], r[1])]
    excluded_ids = sorted({r[1] for r in all_rows if not is_target_student(r[0], r[1])})
    included_ids = sorted({r[1] for r in valid_rows})

    print(f"Included Students ({len(included_ids)}): {included_ids}")
    print(f"Excluded Test/Other IDs ({len(excluded_ids)}): {excluded_ids}\n")

    # 1. Per-Question Accuracy (from submissions where teacher_verified = 1)
    q_true, q_pred = [], []
    by_qid: dict[str, dict[str, list[float]]] = {}

    # Track per-student scores for IA total calculation
    student_scores: dict[str, dict[str, dict[str, float]]] = {
        sid: {"ai": {}, "actual": {}, "verified_count": 0} for sid in included_ids
    }

    for _, sid, qid, ai_s, t_s, verified in valid_rows:
        qid_clean = qid.strip().upper()
        ai_val = float(ai_s) if ai_s is not None else 0.0
        actual_val = float(t_s) if (verified == 1 and t_s is not None) else ai_val

        student_scores[sid]["ai"][qid_clean] = ai_val
        student_scores[sid]["actual"][qid_clean] = actual_val

        if verified == 1 and t_s is not None and ai_s is not None:
            student_scores[sid]["verified_count"] += 1
            q_true.append(float(t_s))
            q_pred.append(float(ai_s))
            by_qid.setdefault(qid_clean, {"true": [], "pred": []})
            by_qid[qid_clean]["true"].append(float(t_s))
            by_qid[qid_clean]["pred"].append(float(ai_s))

    print("=== 1. OVERALL PER-QUESTION ACCURACY (Max: 5.0 Marks, Tolerance: ±0.5) ===")
    if q_true:
        for k, v in compute_metrics(q_true, q_pred, max_marks=5.0, tolerance=0.5).items():
            print(f"  {k:<24}: {v}")
    else:
        print("  No verified question rows found.")

    print("\n=== 2. BREAKDOWN BY QUESTION ID (Max: 5.0 Marks) ===")
    print(f"  {'QID':<6} {'N':<5} {'MAE':<8} {'Norm Acc %':<12} {'Within ±0.5 %':<15} {'Mean Diff (AI-Act)'}")
    for qid in sorted(by_qid.keys()):
        m = compute_metrics(by_qid[qid]["true"], by_qid[qid]["pred"], max_marks=5.0, tolerance=0.5)
        print(
            f"  {qid:<6} {m['N']:<5} {m['MAE']:<8} {m['Normalized_Accuracy_%']:<12} "
            f"{m['Within_±0.5_%']:<15} {m['Mean_Diff_(AI-Actual)']:+}"
        )

    # 3. Per-Student Total IA Marks & Difference (AI - Actual)
    print("\n=== 3. PER-STUDENT IA TOTAL DIFFERENCE (001 to 037, Max: 20.0 Marks) ===")
    print(f"  {'Student':<9} {'AI Total':<10} {'Actual Total':<14} {'Diff (AI - Actual)':<20} {'Abs Error':<10} {'Status'}")
    print("  " + "-" * 78)

    b_true, b_pred = [], []
    for sid in included_ids:
        data = student_scores[sid]
        ai_tot = compute_ia_total(data["ai"])
        act_tot = compute_ia_total(data["actual"])
        diff = ai_tot - act_tot
        abs_err = abs(diff)
        status = f"Verified ({data['verified_count']} Qs)" if data["verified_count"] > 0 else "Unverified (AI only)"

        print(f"  {sid:<9} {ai_tot:<10.2f} {act_tot:<14.2f} {diff:+<20.2f} {abs_err:<10.2f} {status}")

        # Include in aggregate booklet benchmark if student has verified ground truth
        if data["verified_count"] > 0:
            b_true.append(act_tot)
            b_pred.append(ai_tot)

    print("\n=== 4. OVERALL WHOLE-BOOKLET ACCURACY (Verified Students, Max: 20.0 Marks) ===")
    if b_true:
        for k, v in compute_metrics(b_true, b_pred, max_marks=20.0, tolerance=1.0).items():
            print(f"  {k:<24}: {v}")
    else:
        print("  No verified student booklets found.")


if __name__ == "__main__":
    run_evaluation()