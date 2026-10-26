"""Расширенный голос Silero v5: вектор (2,128) + аккуратное дообучение акустики (декодер мела). Вокодер не трогается.
  scope=decoder (по умолчанию) — веса декодера и мел-выхода (~3 млн), с сильным штрафом за уход от исходных
  scope=bias — только bias нормировок декодера и мел-выхода (~1.9 тыс. чисел), слабее
Отклонения штрафуются; лучшая точка выбирается по отложенным фразам.

Обучение:
  python extended_voice.py train myvoice --start voices/myvoice/myvoice_clone_v5_ru.pt --steps 3000
"""
import argparse
import os
import random
import sys
import time
import warnings

import soundfile as sf
import torch
import torchaudio

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clone_voice as cv


def bias_params(m, scope="decoder"):
    """scope='decoder' — все веса декодера акустики и мел-выхода; scope='bias' — только сдвиги."""
    out = {}
    for n, p in m.named_parameters():
        if scope == "decoder":
            if n.startswith("tacotron.decoder.") or n.startswith("tacotron.lin."):
                out[n] = p
            continue
        if not n.endswith("bias"):
            continue
        if (n.startswith("tacotron.decoder.") and "norm" in n) or n == "tacotron.lin.bias":
            out[n] = p
    return out


def apply_extended(S, path, strength=1.0):
    """Прописывает сдвиги в модель (SileroV5 из clone_voice.load_silero). Возвращает вектор голоса."""
    ck = torch.load(path, map_location="cpu")
    params = dict(S.m.named_parameters())
    with torch.no_grad():
        for n, off in ck["offsets"].items():
            params[n].add_(off.to(params[n].device) * strength)
    return ck["voice"]


def load_clips(S, feat, folder, max_clips, holdout, seed):
    meta = cv.load_texts(folder)
    names = [f for f, v in meta.items() if v["lang"] == "ru" and os.path.exists(os.path.join(folder, f))]
    random.Random(seed).shuffle(names)
    names = names[: max_clips + holdout]
    clips = []
    for f in names:
        text = cv.normalize_text(cv.mark_text(meta[f]["text"]))
        x, sr = sf.read(os.path.join(folder, f), dtype="float32", always_2d=True)
        w = torch.from_numpy(x.mean(1)).to(S.device)
        if sr != 48000:
            w = torchaudio.functional.resample(w, sr, 48000)
        clips.append(dict(name=f, text=text, seq=S.sequence(text), wav=w, lm=feat.logmel(w)))
    return clips[holdout:], clips[:holdout]


def train(args):
    dev = args.device
    folder = os.path.join(cv.VOICES, args.voice)
    log_f = open(os.path.join(folder, f"extended_log_{args.tag}.txt"), "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()

    torch.set_grad_enabled(False)
    S = cv.load_silero(args.model, dev)
    S.auto_yo = not args.no_auto_yo
    feat = cv.Features(dev, args.fmax)
    fit, hold = load_clips(S, feat, folder, args.max_clips, args.holdout, args.seed)
    E = torch.load(args.start, map_location=dev)
    log(f"фраз в обучении {len(fit)}, на проверке {len(hold)} | старт {os.path.basename(args.start)}")

    # длительности фиксируем по выравниванию со стартовым голосом (это не голос, а ритм конкретных записей)
    t0 = time.time()
    cl = cv.Cloner(S, feat, fit + hold, lambda *_: None)
    cl.realign(E)
    log(f"выравнивание {time.time()-t0:.0f}s")

    params = bias_params(S.m, args.scope)
    orig = {n: p.detach().clone() for n, p in params.items()}
    n_off = sum(p.numel() for p in params.values())
    log(f"обучаемых параметров ({args.scope}): {n_off} чисел в {len(params)} тензорах + вектор голоса")

    torch.set_grad_enabled(True)
    for p in S.m.parameters():
        p.requires_grad_(False)
    for p in params.values():
        p.requires_grad_(True)
    e_t = E[0].clone().requires_grad_(True)
    e_p = E[1, 64:].clone().requires_grad_(True)
    gains = torch.zeros(len(fit) + len(hold), device=dev, requires_grad=True)
    opt = torch.optim.Adam([{"params": [e_t, e_p], "lr": args.lr_voice},
                            {"params": list(params.values()), "lr": args.lr_offsets},
                            {"params": [gains], "lr": 0.01}])
    zero = torch.zeros(len(fit) + len(hold), device=dev)

    def reg():
        return sum(((p - orig[n]) ** 2).sum() for n, p in params.items())

    def eval_hold():
        with torch.no_grad():
            random.seed(123)
            return sum(cl.clip_loss(len(fit) + i, e_t, e_p, zero, gains).item()
                       for i in range(len(hold)) for _ in range(3)) / (3 * len(hold))

    base_hold = eval_hold()
    best = (base_hold, 0, None)
    log(f"проверка до обучения: {base_hold:.4f}")
    t0, run = time.time(), None
    for step in range(1, args.steps + 1):
        idx = random.sample(range(len(fit)), min(args.batch, len(fit)))
        loss = sum(cl.clip_loss(k, e_t, e_p, zero, gains) for k in idx) / len(idx)
        total = loss + args.reg * reg()
        opt.zero_grad(); total.backward(); opt.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % args.eval_every == 0 or step == args.steps:
            h = eval_hold()
            off_norm = sum(((p - orig[n]) ** 2).sum().item() for n, p in params.items()) ** 0.5
            mark = ""
            if h < best[0]:
                best = (h, step, {"voice": torch.stack([e_t.detach(), torch.cat([E[1, :64], e_p.detach()])]).cpu(),
                                  "offsets": {n: (p.detach() - orig[n]).cpu() for n, p in params.items()}})
                mark = " <- лучшая"
            log(f"  step {step:5d} | train {run:.4f} | проверка {h:.4f}{mark} | норма сдвигов {off_norm:.2f} | {time.time()-t0:.0f}s")
            if step - best[1] >= args.patience:
                log(f"  проверка не улучшается {args.patience} шагов — стоп")
                break

    torch.set_grad_enabled(False)
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(orig[n])
    if best[2] is None:
        log("сдвиги не улучшили проверку — ничего не сохраняю")
        return
    out = os.path.join(folder, f"{args.voice}_extended_{args.tag}.pt")
    torch.save({**best[2], "meta": {"start": args.start, "step": best[1], "hold_loss": best[0], "hold_before": base_hold}}, out)
    log(f"лучшая точка: шаг {best[1]}, проверка {base_hold:.4f} -> {best[0]:.4f} | сохранено {out}")

    # демо: базовый голос vs расширенный
    demo_dir = os.path.join(folder, f"extended_listen_{args.tag}")
    os.makedirs(demo_dir, exist_ok=True)
    text = cv.normalize_text(cv.DEMO_TEXT)
    a0 = cv.render(S, text, E)
    sf.write(os.path.join(demo_dir, "demo_1_base.wav"), a0 / max(1e-6, abs(a0).max()) * 0.9, 48000)
    apply_extended(S, out)
    a1 = cv.render(S, text, best[2]["voice"].to(dev))
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(orig[n])
    sf.write(os.path.join(demo_dir, "demo_2_extended.wav"), a1 / max(1e-6, abs(a1).max()) * 0.9, 48000)
    log(f"демо: {demo_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train"])
    ap.add_argument("voice")
    ap.add_argument("--start", required=True, help="стартовый вектор голоса (2,128) из clone_voice.py")
    ap.add_argument("--model", default=os.path.join(cv.ROOT, "v5_ru.pt"))
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr_voice", type=float, default=0.01)
    ap.add_argument("--scope", choices=["decoder", "bias"], default="decoder")
    ap.add_argument("--lr_offsets", type=float, default=None, help="по умолчанию 1e-5 (decoder) / 3e-3 (bias)")
    ap.add_argument("--reg", type=float, default=None, help="штраф за уход от исходных весов; по умолчанию 0.1 (decoder) / 1e-3 (bias)")
    ap.add_argument("--max_clips", type=int, default=100)
    ap.add_argument("--holdout", type=int, default=5)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=600)
    ap.add_argument("--no_auto_yo", action="store_true", help="«ё» только явно написанные в тексте")
    ap.add_argument("--fmax", type=float, default=16500.0, help="верхняя частота лосса (по полосе записей)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.lr_offsets is None:
        args.lr_offsets = 1e-5 if args.scope == "decoder" else 3e-3
    if args.reg is None:
        args.reg = 0.1 if args.scope == "decoder" else 1e-3
    random.seed(args.seed); torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
