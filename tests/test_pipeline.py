"""Unit tests for the AI Voice Recovery pipeline.

The heavy ML models (Whisper, FLAN-T5) are mocked so the suite runs fast,
offline and without a GPU.
"""

import os

import pytest

import pipeline


@pytest.fixture(autouse=True)
def clear_caches():
    """Ensure lazy-loaded model caches don't leak between tests."""
    pipeline.get_whisper_model.cache_clear()
    pipeline.get_correction_model.cache_clear()
    yield
    pipeline.get_whisper_model.cache_clear()
    pipeline.get_correction_model.cache_clear()


# ---------- speech_to_text ----------

def test_speech_to_text_empty_input_returns_empty():
    assert pipeline.speech_to_text(None) == ""
    assert pipeline.speech_to_text("") == ""


def test_speech_to_text_forces_english_translation(monkeypatch):
    captured = {}

    class FakeWhisper:
        def transcribe(self, audio, **kwargs):
            captured.update(kwargs)
            captured["audio"] = audio
            return {"text": "  hello world  "}

    monkeypatch.setattr(pipeline, "get_whisper_model", lambda: FakeWhisper())

    assert pipeline.speech_to_text("clip.wav") == "hello world"
    # English output is guaranteed by Whisper's translate task.
    assert captured["task"] == "translate"
    assert captured["audio"] == "clip.wav"


def test_speech_to_text_handles_errors(monkeypatch):
    class Boom:
        def transcribe(self, *a, **k):
            raise RuntimeError("decode failure")

    monkeypatch.setattr(pipeline, "get_whisper_model", lambda: Boom())
    result = pipeline.speech_to_text("clip.wav")
    assert "Speech recognition error" in result


# ---------- fix_text ----------

def test_fix_text_empty_returns_empty():
    assert pipeline.fix_text("") == ""


def test_fix_text_passthrough_when_model_unavailable(monkeypatch):
    monkeypatch.setattr(pipeline, "get_correction_model", lambda: (None, None))
    assert pipeline.fix_text("some text") == "some text"


def test_fix_text_uses_model_output(monkeypatch):
    class FakeTokenizer:
        def __call__(self, *a, **k):
            return {"input_ids": [[0]]}

        def decode(self, *a, **k):
            return "  Corrected sentence.  "

    class FakeModel:
        def generate(self, **kwargs):
            return [[1, 2, 3]]

    monkeypatch.setattr(
        pipeline, "get_correction_model", lambda: (FakeTokenizer(), FakeModel())
    )
    assert pipeline.fix_text("corectd sentance") == "Corrected sentence."


def test_fix_text_falls_back_on_error(monkeypatch):
    class BadTokenizer:
        def __call__(self, *a, **k):
            raise ValueError("tokenize failure")

    monkeypatch.setattr(
        pipeline, "get_correction_model", lambda: (BadTokenizer(), object())
    )
    assert pipeline.fix_text("keep me") == "keep me"


# ---------- process_voice ----------

def test_process_voice_runs_full_pipeline(monkeypatch):
    monkeypatch.setattr(pipeline, "speech_to_text", lambda audio: "raw text")
    monkeypatch.setattr(pipeline, "fix_text", lambda text: text.upper())
    raw, corrected = pipeline.process_voice("clip.wav")
    assert raw == "raw text"
    assert corrected == "RAW TEXT"


# ---------- text_to_voice ----------

def test_text_to_voice_empty_returns_none():
    assert pipeline.text_to_voice("") is None
    assert pipeline.text_to_voice("   ") is None


def test_text_to_voice_creates_mp3(monkeypatch):
    saved = {}

    class FakeGTTS:
        def __init__(self, text, lang):
            saved["text"] = text
            saved["lang"] = lang

        def save(self, path):
            with open(path, "wb") as fh:
                fh.write(b"ID3fake-mp3-bytes")

    monkeypatch.setattr(pipeline, "gTTS", FakeGTTS)

    path = pipeline.text_to_voice("hello there")
    try:
        assert path is not None
        assert path.endswith(".mp3")
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
        assert saved["lang"] == "en"  # output voice is English
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


def test_text_to_voice_handles_failure(monkeypatch):
    class BadGTTS:
        def __init__(self, *a, **k):
            raise RuntimeError("network down")

    monkeypatch.setattr(pipeline, "gTTS", BadGTTS)
    assert pipeline.text_to_voice("hello") is None


# ---------- Gradio UI (optional) ----------

def test_build_demo_constructs_blocks():
    gr = pytest.importorskip("gradio")
    import app

    assert isinstance(app.build_demo(), gr.Blocks)
