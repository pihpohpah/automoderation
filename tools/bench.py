"""Бенчмарк худшего случая против запущенного сервиса + пиковый RSS процесса.

    python -m tools.bench --url http://127.0.0.1:8000 --pid <pid uvicorn>
"""
import argparse, io, json, subprocess, time
import numpy as np, httpx
from PIL import Image

def jpg(h, w, rng):
    # шум + градиент: максимально «тяжёлый» для декодера/кодировщика
    # «фото»: низкочастотный шум, растянутый до нужного размера + лёгкий мелкий шум (реалистичный размер JPEG)
    small = (rng.random((max(2, h // 64), max(2, w // 64), 3)) * 255).astype("uint8")
    img = Image.fromarray(small).resize((w, h), Image.BICUBIC)
    a = np.asarray(img, dtype=np.int16) + rng.integers(-6, 7, (h, w, 3))
    b = io.BytesIO(); Image.fromarray(np.clip(a, 0, 255).astype("uint8")).save(b, "JPEG", quality=90); return b.getvalue()

def rss_mb(pid):
    try: return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)]).split()[0]) / 1024
    except Exception: return float("nan")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8000");     ap.add_argument("--n", type=int, default=3); a = ap.parse_args()
    rng = np.random.default_rng(0)
    long_text = ("Отличное место для прогулок, рекомендую всем. " * 80)[:4000]
    cases = {
        "text_only_short": ("Концерт в субботу", []),
        "text_only_4000ch": (long_text, []),
        "1img_1024": ("Кафе", [jpg(768, 1024, rng)]),
        "4img_1024_longtext": (long_text, [jpg(768, 1024, rng) for _ in range(4)]),
        "4img_big_4000px": ("Парк", [jpg(3000, 4000, rng) for _ in range(4)]),
    }
    with httpx.Client(timeout=30) as c:
        for name, (t, imgs) in cases.items():
            for i in range(a.n):
                files = [("images", (f"{k}.jpg", d, "image/jpeg")) for k, d in enumerate(imgs)]
                t0 = time.perf_counter()
                r = c.post(a.url + "/moderate", data={"text": t}, files=files or None)
                dt = (time.perf_counter() - t0) * 1000
                j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                print(f"{name:22s} #{i} http={r.status_code} wall={dt:7.0f}ms tokens={j.get('n_tokens')} "
                      f"timings={j.get('timings_ms')}")
            h = c.get(a.url + "/health").json(); print(f"   rss={h['rss_mb']}MB peak={h['peak_rss_mb']}MB")

if __name__ == "__main__": main()
