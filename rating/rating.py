import numpy as np

CRITERIA = {
    "sharpness": True,
    "noise": False,
    "dynamic_range": True,
    "exposure": True,
    "contrast": True,
    "colorfulness": True,
    "chroma_noise": False,
    "highlight_rec": True,
    "shadow_detail": True,
}

DEFAULT_WEIGHTS = {
    "sharpness": 0.20,
    "noise": 0.15,
    "dynamic_range": 0.15,
    "exposure": 0.10,
    "contrast": 0.10,
    "colorfulness": 0.10,
    "chroma_noise": 0.08,
    "highlight_rec": 0.06,
    "shadow_detail": 0.06,
}

REFERENCE = {
    "sharpness": 8.0,
    "noise": 2.0,
    "dynamic_range": 6.5,
    "exposure": 0.90,
    "contrast": 0.20,
    "colorfulness": 40.0,
    "chroma_noise": 6.0,
    "highlight_rec": 0.60,
    "shadow_detail": 0.50,
}


def normalize_minmax(values, invert=False):
    arr = np.array(values, dtype=float)
    if arr.max() - arr.min() < 1e-9:
        return [50.0] * len(arr)
    norm = (arr - arr.min()) / (arr.max() - arr.min()) * 100
    return (100 - norm if invert else norm).tolist()


def normalize_reference(values, ref, higher_better=True):
    out = []
    for v in values:
        if ref <= 0:
            out.append(50.0)
            continue
        ratio = v / ref if higher_better else ref / max(v, 1e-6)
        out.append(float(np.clip(ratio * 100, 0, 150)))
    return out


def score_weighted(results, weights=None, use_reference=False):
    weights = weights or DEFAULT_WEIGHTS
    if not results:
        return []

    norm = {}
    for key, higher_better in CRITERIA.items():
        vals = [r[key] for r in results]
        if use_reference:
            norm[key] = normalize_reference(vals, REFERENCE[key], higher_better)
        else:
            norm[key] = normalize_minmax(vals, invert=not higher_better)

    out = []
    for i, r in enumerate(results):
        score = sum(weights[k] * norm[k][i] for k in CRITERIA)
        out.append({**r, "score": round(score, 2)})
    return out


def score_topsis(results, weights=None):
    weights = weights or DEFAULT_WEIGHTS
    if not results:
        return []

    keys = list(CRITERIA.keys())
    X = np.array([[r[k] for k in keys] for r in results], dtype=float)

    norm = np.sqrt((X ** 2).sum(axis=0))
    norm[norm == 0] = 1e-9
    R = X / norm

    w = np.array([weights[k] for k in keys])
    V = R * w

    ideal, anti = [], []
    for j, k in enumerate(keys):
        if CRITERIA[k]:
            ideal.append(V[:, j].max())
            anti.append(V[:, j].min())
        else:
            ideal.append(V[:, j].min())
            anti.append(V[:, j].max())
    ideal, anti = np.array(ideal), np.array(anti)

    d_pos = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((V - anti) ** 2).sum(axis=1))

    closeness = d_neg / (d_pos + d_neg + 1e-9)
    scores = (closeness * 100).round(2)

    return [{**r, "score": float(s)} for r, s in zip(results, scores)]


def build_rating(results, weights=None, method="weighted", use_reference=False):
    if method == "topsis":
        out = score_topsis(results, weights)
    else:
        out = score_weighted(results, weights, use_reference)

    out.sort(key=lambda x: x["score"], reverse=True)
    for rank, item in enumerate(out, 1):
        item["rank"] = rank
    return out


def pairwise_diff(rated, key="score"):
    rows = []
    for i in range(len(rated)):
        for j in range(i + 1, len(rated)):
            a, b = rated[i], rated[j]
            diff = {
                "A": a["camera"],
                "B": b["camera"],
                "Δ score": round(a[key] - b[key], 2),
            }
            for k in CRITERIA:
                diff[f"Δ {k}"] = round(a[k] - b[k], 4)
            rows.append(diff)
    return rows