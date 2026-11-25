"""Тесты: препроцессинг (юнит) и API (нужны артефакты: ARTIFACTS_DIR или artifacts/smolvlm_int8)."""
import io, os, sys
from pathlib import Path
import numpy as np, pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "service"))
from app import preprocess as P  # noqa: E402

ART = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts/smolvlm_int8"))


def jpg(h=300, w=400, seed=0):
    a = (np.random.default_rng(seed).random((h, w, 3)) * 255).astype("uint8")
    b = io.BytesIO(); Image.fromarray(a).save(b, "JPEG"); return b.getvalue()


def test_resize_to_square_shape():
    img = P.resize_to_square(Image.new("RGB", (400, 300)), image_size=512)
    assert img.size == (512, 512)


def test_image_to_pixel_values_shape():
    v = P.image_to_pixel_values(Image.new("RGB", (400, 300)), image_size=512)
    assert v.shape == (3, 512, 512) and v.dtype == np.float32


def test_build_prompt_string_text_only():
    s = P.build_prompt_string("привет", n_images=0, tokens_per_image=64)
    assert s == "<|im_start|>User: привет<end_of_utterance>\n"
    assert "<image>" not in s


def test_build_prompt_string_with_images_has_right_image_token_count():
    s = P.build_prompt_string("текст", n_images=2, tokens_per_image=64)
    assert s.count("<image>") == 2 * 64
    assert s.startswith("<|im_start|>User:")
    assert s.endswith("текст<end_of_utterance>\n")


def test_text_rope_shape():
    inv_freq = np.array([1.0, 0.5], dtype=np.float32)
    cos, sin = P.text_rope(5, inv_freq)
    assert cos.shape == (5, 4) and sin.shape == (5, 4)


def test_cv_features_empty_and_shape():
    assert P.compute_cv_features([]).shape == (9,)
    f = P.compute_cv_features([Image.new("RGB", (64, 64), (0, 0, 0))])
    assert f.shape == (9,) and f[-1] == 1.0


needs_art = pytest.mark.skipif(not (ART / "lm.onnx").exists(), reason="нет ONNX-артефактов")


@pytest.fixture(scope="module")
def client():
    os.environ["ARTIFACTS_DIR"] = str(ART); os.environ["WARMUP"] = "0"
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


@needs_art
def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


@needs_art
def test_text_only(client):
    r = client.post("/moderate", data={"text": "Концерт в субботу"})
    assert r.status_code == 200
    j = r.json()
    assert set(j["categories"]) == {"dangerous", "spam", "junk", "mismatch"} and j["label"] in (0, 1) and j["n_images"] == 0


@needs_art
def test_with_images(client):
    files = [("images", (f"{i}.jpg", jpg(seed=i), "image/jpeg")) for i in range(3)]
    r = client.post("/moderate", data={"text": "Кафе"}, files=files)
    assert r.status_code == 200 and r.json()["n_images"] == 3


@needs_art
def test_too_many_images(client):
    files = [("images", (f"{i}.jpg", jpg(64, 64, i), "image/jpeg")) for i in range(5)]
    assert client.post("/moderate", data={"text": "x"}, files=files).status_code == 422


@needs_art
def test_bad_image(client):
    r = client.post("/moderate", data={"text": "x"}, files=[("images", ("a.jpg", b"not an image", "image/jpeg"))])
    assert r.status_code == 422


@needs_art
def test_empty_request_ok(client):
    assert client.post("/moderate", data={"text": ""}).status_code == 200
