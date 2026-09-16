import numpy as np


DEFAULT_WEIGHTS = {
    "sharpness": 0.35,
    "noise": 0.25,
    "dynamic_range": 0.25,
    "exposure": 0.15,
}


def normalize(values: list[float], invert: bool = False) -> list[float]:
    arr = np.array(values, dtype=float)
    if arr.max() - arr.min() < 1e-9:
        return [50.0] * len(arr)
    norm = (arr - arr.min()) / (arr.max() - arr.min()) * 100
    if invert:
        norm = 100 - norm
    return norm.tolist()


def build_rating(results: list[dict], weights: dict = None) -> list[dict]:
    weights = weights or DEFAULT_WEIGHTS
    if not results:
        return []

    sharp = normalize([r["sharpness"] for r in results])
    noise = normalize([r["noise"] for r in results], invert=True)
    dr = normalize([r["dynamic_range"] for r in results])
    exp = normalize([r["exposure"] for r in results])

    out = []
    for i, r in enumerate(results):
        score = (
            weights["sharpness"] * sharp[i]
            + weights["noise"] * noise[i]
            + weights["dynamic_range"] * dr[i]
            + weights["exposure"] * exp[i]
        )
        out.append({**r, "score": round(score, 2)})

    out.sort(key=lambda x: x["score"], reverse=True)
    for rank, item in enumerate(out, 1):
        item["rank"] = rank
    return out