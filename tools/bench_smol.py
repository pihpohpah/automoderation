"""Оценка скорости/памяти готовых ONNX SmolVLM2-256M (int8 и fp32) на 1 CPU. Prefill без генерации."""
import sys, time, resource, numpy as np, onnxruntime as ort
d, var = sys.argv[1], sys.argv[2]            # каталог onnx, суффикс: "_int8" или ""
def rss():
    import os
    with open("/proc/self/statm") as f: return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
def sess(n):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.enable_cpu_mem_arena = False; so.enable_mem_pattern = False
    return ort.InferenceSession(f"{d}/{n}{var}.onnx", so, providers=["CPUExecutionProvider"])
print("start", round(rss()))
vis, dec, emb = sess("vision_encoder"), sess("decoder_model_merged"), sess("embed_tokens")
print("loaded", round(rss()))
rng = np.random.default_rng(0)
def run_vis(n):
    pv = rng.standard_normal((1, n, 3, 512, 512)).astype("float32"); m = np.ones((1, n, 512, 512), bool)
    t = time.perf_counter(); o = vis.run(None, {"pixel_values": pv, "pixel_attention_mask": m})[0]; return o, (time.perf_counter() - t) * 1000
def run_dec(T):
    x = rng.standard_normal((1, T, 576)).astype("float32") * 0.1
    feeds = {"inputs_embeds": x, "attention_mask": np.ones((1, T), np.int64), "position_ids": np.arange(T, dtype=np.int64)[None]}
    for i in range(30):
        feeds[f"past_key_values.{i}.key"] = np.zeros((1, 3, 0, 64), np.float32); feeds[f"past_key_values.{i}.value"] = np.zeros((1, 3, 0, 64), np.float32)
    t = time.perf_counter(); dec.run(["logits"], feeds); return (time.perf_counter() - t) * 1000
run_vis(1); run_dec(32)
for n in (1, 4):
    o, ms = run_vis(n); o, ms = run_vis(n); print(f"vision {n} img -> {o.shape} {ms:.0f} ms rss={rss():.0f}")
for T in (64, 300, 600, 1000):
    run_dec(T); ms = run_dec(T); print(f"decoder prefill T={T}: {ms:.0f} ms rss={rss():.0f}")
print("peak", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
