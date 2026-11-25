"""Косинус/ошибка hidden-вектора между эталонными fp32-артефактами и квантованными."""
import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "service"))
from app.engine import Engine, EngineConfig
from tools.verify_smolvlm import make_cases

def main(ref_dir, *dirs):
    cases = make_cases(np.random.default_rng(0))
    ref = Engine(EngineConfig(Path(ref_dir)))
    refs = [ref.predict(t, i) for _, t, i in cases]
    del ref
    for d in dirs:
        e = Engine(EngineConfig(Path(d)))
        row = []
        for (n, t, i), r in zip(cases, refs):
            g = e.predict(t, i)
            cos = float(np.dot(g.hidden, r.hidden) / (np.linalg.norm(g.hidden) * np.linalg.norm(r.hidden)))
            row.append(f"{n}: cos={cos:.5f} rel_err={np.linalg.norm(g.hidden-r.hidden)/np.linalg.norm(r.hidden):.3f} {g.timings_ms['total']:.0f}ms")
        print(d, "\n  " + "\n  ".join(row))

if __name__ == "__main__":
    main(*sys.argv[1:])
