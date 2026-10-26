"""Распознаёт wav в voices/<голос>/ через whisper и пишет voices/<голос>/transcripts.json.
Запуск: python transcribe.py myvoice [--model medium]
Текст в transcripts.json можно поправить руками (поле "text") до запуска clone_voice.py."""
import argparse
import json
import os
import re

import torch
import whisper

ROOT = os.path.dirname(os.path.abspath(__file__))


def numbers_to_words(text):
    try:
        from num2words import num2words
    except ImportError:
        return text
    return re.sub(r"\d+", lambda m: num2words(int(m.group()), lang="ru"), text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("voice")
    ap.add_argument("--model", default="medium")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    folder = os.path.join(ROOT, "voices", args.voice)
    meta_path = os.path.join(folder, "transcripts.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    model = whisper.load_model(args.model, device=dev)
    for fname in sorted(f for f in os.listdir(folder) if f.lower().endswith(".wav")):
        if fname in meta:
            continue
        # fp16 на старых GTX (16xx) даёт NaN -> мусор вида "!!!!", поэтому fp32
        res = model.transcribe(os.path.join(folder, fname), fp16=False, temperature=0.0, condition_on_previous_text=False)
        lang = res.get("language", "ru")
        meta[fname] = {"lang": lang, "text": numbers_to_words(res["text"].strip()) if lang == "ru" else res["text"].strip()}
        print(f"[{args.voice}] {lang} {fname[:40]}: {meta[fname]['text'][:160]}", flush=True)
        with open(meta_path, "w", encoding="utf-8") as f:  # после каждого файла — прогресс не теряется
            json.dump(meta, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
