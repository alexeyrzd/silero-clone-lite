"""Клонирование голоса Silero v5 по аудио+тексту без энкодера: подбор (2,128) градиентом через сам граф Silero.

Запуск (python 3.10+, CUDA желательна):
  python clone_voice.py myvoice
  python clone_voice.py myvoice --max_clips 12 --holdout 1
Нужен voices/<голос>/transcripts.json (делает transcribe.py).
На выходе в папке голоса: <голос>_clone_<tag>.pt, demo_clone_<tag>.wav, holdout_*.wav, clone_log_<tag>.txt
"""
import argparse
import copy
import json
import os
import random
import re
import sys
import time
import warnings

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
VOICES = os.path.join(ROOT, "voices")
HOP = 600
DEMO_TEXT = ("Уважаемые пассажиры! Скорый поезд номер двести девяносто четыре, сообщением Кисловодск - Москва, "
             "прибывает на второй путь. Нумерация вагонов начинается с головы поезда.")


# ─────────────────────────────────────────────────────────────────────────────
# Дифференцируемая разборка Silero v5
# ─────────────────────────────────────────────────────────────────────────────

def istft_same(spec, n_fft, hop, win, window, padding="same"):
    """Копия ISTFT из tts_package.package_utils (Vocos, padding='same'), дифференцируемая."""
    if padding == "center":
        return torch.istft(spec, n_fft, hop, win, window, center=True)
    pad = (win - hop) // 2
    T = spec.shape[-1]
    ifft = torch.fft.irfft(spec, n_fft, dim=1, norm="backward") * window[None, :, None]
    out_size = (T - 1) * hop + win
    y = F.fold(ifft, output_size=(1, out_size), kernel_size=(1, win), stride=(1, hop))[:, 0, 0, pad:-pad]
    env = F.fold(window.square().expand(1, T, -1).transpose(1, 2), output_size=(1, out_size),
                 kernel_size=(1, win), stride=(1, hop)).squeeze()[pad:-pad]
    return y / env


class SileroV5:
    family = "v5"
    slot = 0                 # голос пишется в слот aidar (так делает модифицированная v5_ru.pt для random)
    demo_speaker = "aidar"
    auto_yo = True           # False — «ё» только там, где она явно написана в тексте

    def __init__(self, path, device, M=None):
        M = M or torch.package.PackageImporter(path).load_pickle("tts_models", "model")
        M.to(device)
        self.M, self.pk, self.m = M, M.packages[0], M.packages[0].models[0]
        self.pk.load_random_voice = lambda voice_path=None: None  # не затирать слот рандомом
        self.device = device
        # v5_ru: вокодер отдаёт (mag, cos, sin), ISTFT в python-обёртке; v5_2/v5_4/cis/turkic: ISTFT внутри JIT, сразу аудио
        wj = getattr(self.pk, "wrapped_jit_v", None)
        self.istft = copy.deepcopy(wj.istft).to(device) if wj is not None else None
        if self.istft is None:
            # автоград TorchScript падает на complex внутри встроенного ISTFT — берём параметры и окно, считаем сами
            ist = self.m.vocoder.head.istft
            self.jit_istft = dict(n_fft=ist.n_fft, hop=ist.hop_length, win=ist.win_length,
                                  window=ist.window.detach().to(device), padding=ist.padding)
        for p in self.m.parameters():
            p.requires_grad_(False)
        if not getattr(self.pk, "q_model_unpacked", True):
            self.pk.unpack_q_model(); self.pk.q_model_unpacked = True
        self.punkt = torch.tensor(self.m.punkt_symbols, device=device)
        self.pitch_shift_coef = float(self.m.mean_std_coef[0])  # слот 0 = random/aidar

    def sequence(self, text):
        ssml = text.lstrip().startswith("<speak")
        with torch.no_grad():
            try:
                prep = self.pk.prepare_tts_model_input(text, ssml=ssml, speaker_ids=[0])
            except TypeError:  # v5_2+ / cis: добавлен аргумент lang
                prep = self.pk.prepare_tts_model_input(text, ssml=ssml, speaker_ids=[0], lang=None)
            sents, _, breaks, rates, pitches, _ = prep
            # auto_yo=False: акцентор не превращает «е» в «ё» сам (иначе ломает «пчЕлка», если так и сказано)
            seq = self.pk.merge_batch_model(sents, breaks, rates, pitches,
                                            put_yo=self.auto_yo, put_yo_homo=self.auto_yo)[0].to(self.device)
            if hasattr(self.pk, "build_type_ids_inference"):
                # v5_4: тип фразы (утверждение/вопрос/восклицание) прибавляется на входе питч-предиктора
                g = self.pk.build_type_ids_inference.__func__.__globals__
                plain = re.sub(r"<[^>]+>", "", text)
                seq.type_ids = self.pk.build_type_ids_inference(plain, g["classify_text"](plain), seq.shape[1]).to(self.device)
        return seq

    @staticmethod
    def tf(module, x, mask):
        """Трансформеры v5/v4 принимают (x, mask, style), у v3 — только (x, mask)."""
        try:
            return module(x, mask, None)
        except RuntimeError as e:
            if "expected at most" not in str(e):
                raise
            return module(x, mask)

    def log_dur(self, seq, e_dur):
        dp = self.m.dur_predictor.dur_pred
        return dp.lin(self.tf(dp.transformer, dp.embedding(seq) + e_dur.view(1, 1, -1), seq == 0)).squeeze(-1)

    @staticmethod
    def dur_from_log(logd, rate=1.0):
        d = ((logd.exp() - 1).clamp(min=0).round() / rate).round()
        d[0, 0] = d[0, 0].clamp(max=5)
        d[0, -1] = d[0, -1].clamp(max=7)
        d[0, -2] = 13
        d[0, -3] = d[0, -3].clamp(max=13)
        return d

    def pitch(self, seq, e_pitch, coef=None):
        pp = self.m.pitch_predictor.pitch_pred
        x = pp.embedding(seq) + e_pitch.view(1, 1, -1)
        type_ids = getattr(seq, "type_ids", None)
        if type_ids is not None and getattr(pp, "utt_type_emb", False):
            x = x + pp.type_embedding(type_ids)
        p = pp.lin(self.tf(pp.transformer, x, seq == 0)).transpose(1, 2)
        if coef is not None:  # как update_pitch_coef в MultiTTSModel (SSML <prosody pitch>)
            nz = (p.abs() >= 1e-3).float()
            p = p * nz * coef + (coef - 1) * self.pitch_shift_coef * nz
        return p

    def mel(self, seq, e_timbre, pitch, dur):
        tac = self.m.tacotron
        h = self.tf(tac.encoder, tac.embedding(seq), seq == 0)
        h = h + e_timbre.view(1, 1, -1) + tac.pitch_proj(pitch).transpose(1, 2) * tac.pitch_strength
        h = tac.len_reg(h, dur.clone())
        return tac.lin(self.tf(tac.decoder, h, None)).transpose(1, 2)

    def audio(self, mel):
        if self.istft is None:  # ISTFT внутри JIT-вокодера: бэкбон и out-слой из JIT, хвост считаем сами
            voc = self.m.vocoder
            x = voc.head.out(voc.backbone(mel)).transpose(1, 2)
            mag, p = x.chunk(2, dim=1)
            mag = mag.exp().clamp(max=100.0)
            return istft_same(mag * torch.complex(p.cos(), p.sin()), **self.jit_istft)[0]
        mag, c, s = self.m.vocoder(mel, 48000, 0.0, True)
        return self.istft(mag * torch.complex(c, s))[0]

    def vocoder_call(self, mel):
        try:
            return self.m.vocoder(mel, 48000, 0.0, True)
        except RuntimeError as e:  # v3: forward(x, sample_rate)
            if "expected at most" not in str(e):
                raise
            return self.m.vocoder(mel, 48000)

    def set_voice(self, E):
        self.m.tacotron.speaker_embedding.weight.data[0].copy_(E[0])
        self.m.dur_predictor.dur_pred.speaker_embedding.weight.data[0].copy_(E[1, :64])
        self.m.pitch_predictor.pitch_pred.speaker_embedding.weight.data[0].copy_(E[1, 64:])

    def builtin_voices(self):
        t = self.m.tacotron.speaker_embedding.weight.data
        d = self.m.dur_predictor.dur_pred.speaker_embedding.weight.data
        p = self.m.pitch_predictor.pitch_pred.speaker_embedding.weight.data
        return [torch.stack([t[i], torch.cat([d[i], p[i]])]).clone() for i in range(t.shape[0])]

    @property
    def symbol_to_id(self):
        return self.pk.symbol_to_id


class SileroV4(SileroV5):
    """v4_ru / v3_1_ru: та же FastPitch-акустика, но HiFiGAN-multiband вокодер (128 мел), акцентор внутри TorchScript,
    длительности без жёстких ограничений на крайние символы. Голос пишется в слот random (5)."""
    family = "v4"
    demo_speaker = "random"

    @property
    def symbol_to_id(self):
        return dict(self.sysm.symbol_to_id)

    def __init__(self, path, device, M=None):
        M = M or torch.package.PackageImporter(path).load_pickle("tts_models", "model")
        torch.set_grad_enabled(False)
        M.to(device)
        if not getattr(M, "q_model_unpacked", True) and hasattr(M, "unpack_q_model"):
            M.unpack_q_model(); M.q_model_unpacked = True
        self.M = M
        self.sysm = M.models[0] if hasattr(M, "models") else M.model
        self.m = self.sysm.tts_model
        M.load_random_voice = lambda voice_path=None: None
        self.device = device
        ids = getattr(M, "speaker_to_ids", None)
        self.slot = (ids[0] if isinstance(ids, list) else getattr(M, "speaker_to_id", {})).get("random", 5)
        for p in self.m.parameters():
            p.requires_grad_(False)
        sym = self.sysm.symbol_to_id
        self.punkt = torch.tensor([sym[c] for c in ".,!?…:;–-" if c in sym], device=device)
        self.pitch_shift_coef = float(self.m.mean_std_coef[self.slot])

    def sequence(self, text):
        s = self.sysm
        with torch.no_grad():
            sents, clean, breaks, rates, pitches, _ = self.M.prepare_tts_model_input(
                text, ssml=text.lstrip().startswith("<speak"), speaker_ids=[self.slot])
            if not hasattr(s, "_tokenize_clean"):  # v3_en и др. без акцентора: forward сразу склеивает предложения
                return s.merge_batch_model(sents, breaks, rates, pitches)[0].to(self.device)
            acc = []
            for sent, cl in zip(sents, clean):
                raw, ct, pm = s._tokenize_clean(sent, cl)
                sp, spred, yp, ypred = s._get_model_preds(ct)
                try:
                    words = s.postprocess_accentor(raw, ct, pm, sp, spred, yp, ypred, self.auto_yo)
                except RuntimeError:  # v3_1: без флага put_yo
                    words = s.postprocess_accentor(raw, ct, pm, sp, spred, yp, ypred)
                acc.append(s._fuse_words_to_sentence(words))
            seq = s.merge_batch_model([a.lower() for a in acc], breaks, rates, pitches)[0]
        return seq.to(self.device)

    @staticmethod
    def dur_from_log(logd, rate=1.0):
        return ((logd.exp() - 1).clamp(min=0).round() / rate).round()

    def audio(self, mel):
        return self.vocoder_call(mel).reshape(-1)

    def set_voice(self, E):
        k = self.slot
        self.m.tacotron.speaker_embedding.weight.data[k].copy_(E[0])
        self.m.dur_predictor.dur_pred.speaker_embedding.weight.data[k].copy_(E[1, :64])
        self.m.pitch_predictor.pitch_pred.speaker_embedding.weight.data[k].copy_(E[1, 64:])

    def builtin_voices(self):
        return [v for i, v in enumerate(super().builtin_voices()) if i != self.slot]


def load_silero(path, device):
    # профилирующий JIT-исполнитель на CUDA склеивает ops и падает на complex внутри ISTFT (v5_2+/cis/turkic):
    # обёртка Silero сама его выключает перед синтезом, делаем так же
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    M = torch.package.PackageImporter(path).load_pickle("tts_models", "model")
    if hasattr(M, "packages"):
        return SileroV5(path, device, M)
    return SileroV4(path, device, M)


# ─────────────────────────────────────────────────────────────────────────────
# Признаки, лосс, выравнивание
# ─────────────────────────────────────────────────────────────────────────────

class Features:
    def __init__(self, device, fmax):
        self.fmax = fmax
        self.mel_fn = torchaudio.transforms.MelSpectrogram(48000, n_fft=2400, hop_length=HOP, n_mels=100,
                                                           f_max=fmax).to(device)
        self.windows = {n: torch.hann_window(n, device=device) for n in (512, 1024, 2048)}

    def logmel(self, a):
        return (self.mel_fn(a) + 1e-5).log()

    def loss(self, a, b):
        T = min(a.numel(), b.numel()); a, b = a[:T], b[:T]
        total = 0
        for n, w in self.windows.items():
            kmax = int(n * self.fmax / 48000) + 1
            A = torch.stft(a, n, n // 4, window=w, return_complex=True).abs()[:kmax]
            B = torch.stft(b, n, n // 4, window=w, return_complex=True).abs()[:kmax]
            total = total + (torch.log(A + 1e-5) - torch.log(B + 1e-5)).abs().mean() + (A - B).norm() / (B.norm() + 1e-7)
        return total / 3 + (self.logmel(a) - self.logmel(b)).abs().mean()


def target_durations(feat, syn_audio, dur_syn, tgt_lm):
    a = feat.logmel(syn_audio)
    Ts = min(a.shape[-1], int(dur_syn.sum()))
    X = (a[:, :Ts] - a[:, :Ts].mean(1, keepdim=True)).cpu().double().numpy()
    Y = (tgt_lm - tgt_lm.mean(1, keepdim=True)).cpu().double().numpy()
    _, wp = librosa.sequence.dtw(X=X, Y=Y, metric="cosine")
    frame_sym = torch.repeat_interleave(torch.arange(dur_syn.shape[1]), dur_syn[0].long().cpu())[:Ts].numpy()
    tgt_sym = np.full(Y.shape[1], -1)
    for i, j in wp[::-1]:
        if tgt_sym[j] < 0:
            tgt_sym[j] = frame_sym[i]
    tgt_sym[tgt_sym < 0] = frame_sym[-1]
    return torch.tensor(np.bincount(tgt_sym, minlength=dur_syn.shape[1]), dtype=torch.float32).unsqueeze(0)


def render(S, text, E):
    """Синтез через свой граф: работает на любой модели, даже если в ней нет спикеров random/aidar."""
    with torch.no_grad():
        seq = S.sequence(text)
        d = S.dur_from_log(S.log_dur(seq, E[1, :64]))
        return S.audio(S.mel(seq, E[0], S.pitch(seq, E[1, 64:]), d)).reshape(-1).cpu().numpy()


# акцентор Silero ошибается в некоторых словах; '+' перед ударной гласной. Слова, где '+' уже стоит, не трогаются.
# Пример: STRESS_FIX = {"второй": "втор+ой"}
STRESS_FIX = {}


def apply_stress_fix(text):
    def repl(m):
        w = m.group(0)
        fixed = STRESS_FIX.get(w.lower())
        if fixed is None:
            return w
        return fixed.capitalize() if w[0].isupper() else fixed
    return re.sub(r"(?<![+\w])[А-Яа-яЁё]+(?![+\w])", repl, text)


def normalize_text(text):
    """Приводит текст whisper к тому, что Silero реально произносит: он молча выкидывает всё, чего нет в алфавите."""
    text = text.replace("№", " номер ").replace("%", " процентов ")
    text = re.sub(r"(\d+)[.:](\d+)", r"\1 \2", text)  # 11.30 / 11:30 -> 11 30
    try:
        from num2words import num2words
        text = re.sub(r"\d+", lambda m: " " + num2words(int(m.group()), lang="ru") + " ", text)
    except ImportError:
        pass
    text = re.sub(r"[«»\"“”„()\[\]]", "", text)
    text = apply_stress_fix(text)
    return re.sub(r"\s+", " ", text).strip()


END_PUNCT = ".!?…"


def mark_text(raw):
    """Если в конце нет знака конца предложения, ставит точку."""
    out = raw.strip()
    if out and out[-1] not in END_PUNCT:
        out += "."
    return out


def detect_cutoff(x, sr):
    L = min(len(x), sr * 20)
    S = np.abs(np.fft.rfft(x[:L] * np.hanning(L))) + 1e-12
    fr = np.fft.rfftfreq(L, 1 / sr)
    db = 20 * np.log10(S / S.max())
    idx = np.where(np.convolve(db > -70, np.ones(200), "same") > 20)[0]
    return float(fr[idx.max()]) if len(idx) else sr / 2


def load_texts(folder):
    """transcripts.json: {файл.wav: {"lang": "ru", "text": "..."}}. Текст можно поправить руками."""
    return json.load(open(os.path.join(folder, "transcripts.json"), encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# Подбор
# ─────────────────────────────────────────────────────────────────────────────

class Cloner:
    def __init__(self, S, feat, clips, log, window_frames=400, margin=32, batch=3, norm_reg=0.0, fit_pauses=False,
                 timbre_norm=0.0):
        self.S, self.feat, self.clips, self.log = S, feat, clips, log
        # timbre_norm>0: тембр держится ровно на этой длине, подбирается только направление
        self.timbre_norm = timbre_norm
        # fit_pauses: учить и длительности на знаках препинания (паузы) — для живого голоса
        self.fit_pauses = fit_pauses
        # мягкий штраф за выход норм тембра/pitch за масштаб встроенных голосов (иначе тембр улетает и звучит искажённо)
        self.norm_reg = norm_reg
        bv = torch.stack(S.builtin_voices()) if norm_reg > 0 else None
        self.norm_caps = None if bv is None else (bv[:, 0].norm(dim=1).max().item() * 1.15,
                                                  bv[:, 1, 64:].norm(dim=1).max().item() * 1.15)
        self.W, self.margin, self.batch = window_frames, margin, batch
        dev = S.device
        n = len(clips)
        self.log_rate = torch.zeros(n, device=dev)
        self.log_pcoef = torch.zeros(n, device=dev)
        self.log_gain = torch.zeros(n, device=dev)

    @staticmethod
    def centered(x):
        return x - x.mean() if x.numel() > 2 else x * 0

    def realign(self, E):
        with torch.no_grad():
            for k, c in enumerate(self.clips):
                pc = self.centered(self.log_pcoef)[k].exp()
                d = self.S.dur_from_log(self.S.log_dur(c["seq"], E[1, :64]), self.centered(self.log_rate)[k].exp().item())
                a = self.S.audio(self.S.mel(c["seq"], E[0], self.S.pitch(c["seq"], E[1, 64:], pc), d))
                c["dur"] = target_durations(self.feat, a, d.cpu(), c["lm"]).to(self.S.device)

    def clip_loss(self, k, e_t, e_p, log_pcoef, log_gain):
        c = self.clips[k]
        mel = self.S.mel(c["seq"], e_t, self.S.pitch(c["seq"], e_p, log_pcoef[k].exp()), c["dur"])
        T = mel.shape[-1]
        if T > self.W + 2 * self.margin:
            s = random.randint(0, T - self.W - 2 * self.margin)
            a = self.S.audio(mel[..., s: s + self.W + 2 * self.margin])
            a = a[self.margin * HOP: (self.margin + self.W) * HOP]
            t0 = (s + self.margin) * HOP
            tgt = c["wav"][t0: t0 + self.W * HOP]
        else:
            a, tgt = self.S.audio(mel), c["wav"]
        return self.feat.loss(a * log_gain[k].exp(), tgt)

    def fit_dur(self, E, steps=300):
        e_d = E[1, :64].clone().requires_grad_(True)
        lr_ = self.log_rate.clone().requires_grad_(True)
        opt = torch.optim.Adam([e_d, lr_], lr=0.05)
        for _ in range(steps):
            rates = self.centered(lr_) if len(self.clips) >= 3 else lr_ * 0
            loss = 0
            for k, c in enumerate(self.clips):
                ld = self.S.log_dur(c["seq"], e_d)
                pred = ((ld.exp() - 1).clamp(min=0) / rates[k].exp())
                mask = torch.ones_like(ld, dtype=torch.bool)
                mask[0, 0] = False; mask[0, -3:] = False
                if not self.fit_pauses:
                    mask &= ~torch.isin(c["seq"], self.S.punkt)
                loss = loss + F.mse_loss(torch.log1p(pred[mask]), torch.log1p(c["dur"][mask]))
            opt.zero_grad(); loss.backward(); opt.step()
        self.log_rate = lr_.detach()
        E = E.clone(); E[1, :64] = e_d.detach()
        return E, loss.item() / len(self.clips)

    def timbre(self, v):
        if self.timbre_norm > 0:
            return F.normalize(v, dim=0) * self.timbre_norm
        return v

    def fit_acoustic(self, E, steps, lr=0.1):
        init_t = E[0].clone()
        if self.timbre_norm > 0 and init_t.norm() < 1e-3:
            init_t = torch.randn_like(init_t) * 0.01
        e_t = init_t.requires_grad_(True)
        e_p = E[1, 64:].clone().requires_grad_(True)
        lpc = self.log_pcoef.clone().requires_grad_(True)
        lg = self.log_gain.clone().requires_grad_(True)
        opt = torch.optim.Adam([{"params": [e_t, e_p], "lr": lr}, {"params": [lpc, lg], "lr": lr * 0.2}])
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, [lr, lr * 0.2], total_steps=steps, pct_start=0.1)
        n = len(self.clips)
        t0, run = time.time(), 0.0
        for step in range(steps):
            idx = random.sample(range(n), min(self.batch, n))
            pcoef = self.centered(lpc) if n >= 3 else lpc * 0
            loss = sum(self.clip_loss(k, self.timbre(e_t), e_p, pcoef, lg) for k in idx) / len(idx)
            if self.norm_reg > 0:
                cap_t, cap_p = self.norm_caps
                loss = loss + self.norm_reg * (F.relu(e_t.norm() - cap_t) ** 2 + F.relu(e_p.norm() - cap_p) ** 2)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            run = 0.95 * run + 0.05 * loss.item() if step else loss.item()
            if step % 100 == 0 or step == steps - 1:
                self.log(f"    acoustic step {step:4d}/{steps} loss {run:.4f} | {time.time()-t0:.0f}s")
        self.log_pcoef, self.log_gain = lpc.detach(), lg.detach()
        return torch.stack([self.timbre(e_t.detach()), torch.cat([E[1, :64], e_p.detach()])])

    def pick_init(self, extra=()):
        dev = self.S.device
        # у cis/turkic по 30-60 голосов — перебирать все долго, берём первые 8
        cands = [("zeros", torch.zeros(2, 128, device=dev))] + \
                [(f"builtin{i}", e) for i, e in enumerate(self.S.builtin_voices()[:8])] + list(extra)
        best = None
        for name, E in cands:
            self.realign(E)
            with torch.no_grad():
                l = sum(self.clip_loss(k, E[0], E[1, 64:], self.log_pcoef, self.log_gain).item()
                        for k in range(len(self.clips))) / len(self.clips)
            self.log(f"  init {name:9s} loss {l:.4f}")
            if best is None or l < best[0]:
                best = (l, name, E.clone())
        self.log(f"  -> старт с {best[1]}")
        return best[2]

    def run(self, E, rounds=3, short=300, long=1200):
        for r in range(rounds):
            self.realign(E)
            E, dl = self.fit_dur(E)
            self.log(f"  round {r}: dur mse {dl:.4f} | rate {self.centered(self.log_rate).exp().cpu().numpy().round(3)}")
            E = self.fit_acoustic(E, long if r == rounds - 1 else short)
            self.log(f"  round {r}: pitch coef {self.centered(self.log_pcoef).exp().cpu().numpy().round(3)} "
                     f"gain {self.log_gain.exp().cpu().numpy().round(3)}")
        return E


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("voice", help="папка voices/<голос>")
    ap.add_argument("--model", default=os.path.join(ROOT, "v5_ru.pt"))
    ap.add_argument("--max_clips", type=int, default=12)
    ap.add_argument("--holdout", type=int, default=1, help="сколько файлов отложить на проверку (0 = все в фит)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--short", type=int, default=300)
    ap.add_argument("--long", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default=None, help="суффикс выходных файлов (по умолчанию имя модели: v5_ru...)")
    ap.add_argument("--no_auto_yo", action="store_true",
                    help="не расставлять «ё» автоматически — только явно написанные в тексте")
    ap.add_argument("--timbre_norm", type=float, default=0.0,
                    help="держать норму тембра ровно на этом значении при подборе (например 13); 0 — свободно")
    ap.add_argument("--fit_pauses", action="store_true",
                    help="учить длительности пауз на знаках препинания (живой голос)")
    ap.add_argument("--norm_reg", type=float, default=0.05,
                    help="штраф за нормы тембра/pitch выше встроенных голосов x1.15; 0 — выключено")
    args = ap.parse_args()
    tag = args.tag or os.path.splitext(os.path.basename(args.model))[0]

    random.seed(args.seed); torch.manual_seed(args.seed)
    folder = os.path.join(VOICES, args.voice)
    log_f = open(os.path.join(folder, f"clone_log_{tag}.txt"), "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()

    dev = args.device
    meta = load_texts(folder)
    items = [(f, v["text"]) for f, v in meta.items()
             if v["lang"] == "ru" and os.path.exists(os.path.join(folder, f)) and v["text"].strip()]
    skipped = [f for f, v in meta.items() if v["lang"] != "ru"]
    if skipped:
        log(f"пропущены не-ru файлы: {skipped}")
    random.shuffle(items)
    items = items[: args.max_clips + args.holdout]

    S = load_silero(args.model, dev)
    S.auto_yo = not args.no_auto_yo
    log(f"модель: {os.path.basename(args.model)} (семейство {S.family}, слот голоса {S.slot}, авто-ё {'вкл' if S.auto_yo else 'выкл'})")
    wavs, cutoffs = [], []
    for f, _ in items:
        x, sr = sf.read(os.path.join(folder, f), dtype="float32", always_2d=True)
        x = x.mean(1)
        cutoffs.append(detect_cutoff(x, sr))
        w = torch.from_numpy(x).to(dev)
        wavs.append(torchaudio.functional.resample(w, sr, 48000) if sr != 48000 else w)
    # 10-й перцентиль, а не минимум: один файл с обрезанным верхом не должен срезать полосу всему подбору
    cut = float(np.percentile(cutoffs, 10)) if len(cutoffs) >= 10 else min(cutoffs)
    fmax = min(22000.0, cut - 500)
    feat = Features(dev, fmax)
    n_low = sum(c < cut for c in cutoffs)
    log(f"голос {args.voice}: {len(items)} файлов, срез спектра ~{cut/1000:.1f} кГц (min {min(cutoffs)/1000:.1f}, ниже порога {n_low}) -> лосс до {fmax/1000:.1f} кГц")

    clips = []
    for (f, raw), w in zip(items, wavs):
        text = normalize_text(mark_text(raw))
        seq = S.sequence(text)
        clips.append(dict(name=f, text=text, raw_text=raw, seq=seq, wav=w, lm=feat.logmel(w)))
        log(f"  {f[:50]:50s} {w.numel()/48000:5.1f}s  символов {seq.shape[1]}")
    hold, fit = clips[: args.holdout], clips[args.holdout:]

    torch.set_grad_enabled(True)
    t0 = time.time()
    kw = dict(norm_reg=args.norm_reg, fit_pauses=args.fit_pauses, timbre_norm=args.timbre_norm)
    cloner = Cloner(S, feat, fit, log, **kw)
    E0 = cloner.pick_init()
    E = cloner.run(E0, rounds=args.rounds, short=args.short, long=args.long)
    log(f"подбор занял {time.time()-t0:.0f}s")
    out_pt = os.path.join(folder, f"{args.voice}_clone_{tag}.pt")
    torch.save(E.cpu(), out_pt)
    log(f"голос сохранён: {out_pt}  нормы тембр {E[0].norm():.2f} dur {E[1,:64].norm():.2f} pitch {E[1,64:].norm():.2f}")

    # ---- проверка на отложенных файлах ----
    def eval_voice(name, Ev):
        res = []
        for c in hold:
            ev = Cloner(S, feat, [c], lambda *_: None)
            ev.realign(Ev)
            with torch.no_grad():
                a = S.audio(S.mel(c["seq"], Ev[0], S.pitch(c["seq"], Ev[1, 64:]), c["dur"]))
                T = min(a.numel(), c["wav"].numel())
                a = a[:T] * (c["wav"][:T].norm() / (a[:T].norm() + 1e-7))
                ml = (feat.logmel(a) - feat.logmel(c["wav"][:T])).abs().mean().item()
            res.append(ml)
            log(f"  [{name}] {c['name'][:40]} melL1(выровн.) {ml:.3f}")
            sf.write(os.path.join(folder, f"holdout_{name}_{tag}_{len(res)}.wav"), a.cpu().numpy(), 48000)
        return float(np.mean(res)) if res else float("nan")

    if hold:
        for i, c in enumerate(hold):
            sf.write(os.path.join(folder, f"holdout_target_{i+1}.wav"), c["wav"].cpu().numpy(), 48000)
        eval_voice("clone", E)
        eval_voice("zeros", torch.zeros(2, 128, device=dev))

    torch.set_grad_enabled(False)
    sf.write(os.path.join(folder, f"demo_clone_{tag}.wav"), render(S, normalize_text(DEMO_TEXT), E), 48000)
    log(f"демо: demo_clone_{tag}.wav")
    log_f.close()


if __name__ == "__main__":
    main()
