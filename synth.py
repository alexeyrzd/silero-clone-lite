"""Синтез речи клонированным голосом (без модифицированной модели — через собственный граф).

  python synth.py --voice voices/myvoice/myvoice_clone_v5_ru.pt --text "Привет, мир." --out out.wav
  python synth.py --voice voices/myvoice/myvoice_extended_v1.pt --text "Привет, мир." --out out.wav
--voice: вектор (2,128) из clone_voice.py ИЛИ файл из extended_voice.py (вектор + сдвиги акустики)
"""
import argparse
import os
import sys

import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clone_voice as cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--model", default=os.path.join(cv.ROOT, "v5_ru.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    S = cv.load_silero(args.model, args.device)
    params = dict(S.m.named_parameters())

    ck = torch.load(args.voice, map_location="cpu")
    E = ck["voice"] if isinstance(ck, dict) else ck
    for path in [args.voice]:
        if path is None:
            continue
        c = torch.load(path, map_location="cpu")
        if isinstance(c, dict):
            for n, off in c["offsets"].items():
                params[n].add_(off.to(args.device))
    audio = cv.render(S, cv.normalize_text(args.text), E.to(args.device))
    sf.write(args.out, audio / max(1e-6, abs(audio).max()) * 0.9, 48000)
    print("сохранено:", args.out)


if __name__ == "__main__":
    main()
