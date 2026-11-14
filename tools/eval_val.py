#!/usr/bin/env python3
"""Оценка качества ONNX-артефактов на val.jsonl: AUC/AP по категориям + накладные от квантования.

    python -m tools.eval_val --artifacts artifacts/int8 --dataset ../contentmoderration/dataset --limit 300
    python -m tools.eval_val --artifacts artifacts/int8 --ref artifacts/fp32 ...   # сравнить с fp32-ONNX
Используйте с реальным checkpoint.pt (экспорт с --checkpoint), иначе голова случайная и метрики бессмысленны.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "service"))
from app.engine import Engine, EngineConfig

def run(art, recs, root, max_images):
    from tqdm import tqdm

    e = Engine(EngineConfig(Path(art), max_images=max_images, max_text_tokens=512))
    P, T = [], []
    bar = tqdm(recs, desc=str(art), unit="rec")
    for r in bar:
        imgs = []
        for p in r["images"][:max_images]:
            im = Image.open(root / p); im.thumbnail((1024, 1024)); imgs.append(im.convert("RGB"))
        t0 = time.perf_counter(); res = e.predict(r["text"] or "", imgs); dt = (time.perf_counter() - t0) * 1000
        T.append(dt)
        P.append([res.probs[c] for c in e.categories])
        bar.set_postfix(n_img=len(imgs), ms=f"{dt:.0f}", mean_ms=f"{np.mean(T):.0f}")
    return e.categories, np.array(P), np.array(T)

def main():
    from sklearn.metrics import roc_auc_score
    ap = argparse.ArgumentParser(); ap.add_argument("--artifacts", required=True); ap.add_argument("--ref")
    ap.add_argument("--dataset", default=str(ROOT.parent / "contentmoderration/dataset")); ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--max-images", type=int, default=4); a = ap.parse_args()
    root = Path(a.dataset)
    recs = [json.loads(l) for l in open(root / "val.jsonl", encoding="utf-8")]
    rng = np.random.default_rng(0); rng.shuffle(recs); recs = recs[: a.limit]
    cats, P, T = run(a.artifacts, recs, root, a.max_images)
    Y = np.array([[r["categories"][c] for c in cats] for r in recs])
    print(f"{a.artifacts}: n={len(recs)} latency mean={T.mean():.0f}ms p95={np.percentile(T,95):.0f}ms max={T.max():.0f}ms")
    for i, c in enumerate(cats):
        print(f"  {c:10s} AUC={roc_auc_score(Y[:, i], P[:, i]):.4f}" if len(set(Y[:, i])) > 1 else f"  {c}: n/a")
    print(f"  overall    AUC={roc_auc_score(Y.max(1), P.max(1)):.4f}")
    if a.ref:
        _, R, _ = run(a.ref, recs, root, a.max_images)
        print(f"  vs ref {a.ref}: max|dprob|={np.abs(P-R).max():.4f} mean|dprob|={np.abs(P-R).mean():.4f} "
              f"decision flips@0.5={(((P>.5)!=(R>.5)).any(1)).mean()*100:.1f}%")
if __name__ == "__main__": main()
