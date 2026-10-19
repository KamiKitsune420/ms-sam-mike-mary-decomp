"""Build a Sam-format voice (.spd) out of Anna's recordings.

Sam's voice is a pitch-synchronous LPC vocoder: per unit a list of frames, each with a period length
(sign = voiced), 20 LSFs as 6 split-VQ bytes, a gain, and for voiced frames the residual spectrum as
11 (or 13 for a delta frame) VQ bytes.  This builds such a file from Anna's corpus.

Her units are demisyllables ("w+aa+n"), so the phones have to be cut out first: stage `align` runs a
small forced aligner (diagonal-Gaussian models over MFCCs, bootstrapped from her 60 single-phone
consonant types, then re-estimated), stage `analyze` does the pitch-synchronous LPC analysis on the
resulting phone segments, and stage `build` trains the codebooks and writes the file.

  python spdbuild.py align   [--corpus DIR] [--per-type N] [--iters N]
  python spdbuild.py analyze [--corpus DIR] [--per-unit N]
  python spdbuild.py build   [--corpus DIR] --sam Sam.spd --out Anna.spd
"""
import argparse
import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np

REC = np.dtype([("type", "<i4"), ("start", "<i4"), ("len", "<i4"), ("pcm_off", "<i8"),
                ("nep", "<i4"), ("ep_off", "<i8"), ("feat", "u1", 9), ("pad", "u1", 3)])
RATE = 16000
HOP = 80            # 5 ms
WIN = 400           # 25 ms
NMEL = 20
NCEP = 13
MINFR = 2           # shortest a phone may be, in frames


# ---------------------------------------------------------------- corpus access
class Corpus:
    def __init__(self, d):
        self.dir = d
        with open(os.path.join(d, "units.bin"), "rb") as f:
            hdr = f.read(12)
            assert hdr[:4] == b"ANCP", "not a corpus dump"
            n = int(np.frombuffer(hdr[8:12], "<u4")[0])
            self.u = np.frombuffer(f.read(n * REC.itemsize), dtype=REC, count=n)
        self.pcm = np.memmap(os.path.join(d, "audio.pcm"), dtype="<i2", mode="r")
        self.ep = np.memmap(os.path.join(d, "epochs.i16"), dtype="<i2", mode="r")
        self.name = {}
        for line in open(os.path.join(d, "types.txt")):
            i, _, nm = line.strip().partition(" ")
            self.name[int(i)] = nm
        self.phones = {}                      # type -> list of phone names ([] = letter unit / unusable)
        for t, nm in self.name.items():
            if not nm or nm == "?" or nm.startswith("_"):
                self.phones[t] = []
            else:
                self.phones[t] = nm.split("+")

    def audio(self, i):
        r = self.u[i]
        return np.asarray(self.pcm[r["pcm_off"]:r["pcm_off"] + r["len"]], dtype=np.float64)

    def epochs(self, i):
        r = self.u[i]
        return np.asarray(self.ep[r["ep_off"]:r["ep_off"] + r["nep"]], dtype=np.int32)


# ---------------------------------------------------------------- features
def mel_filters(nmel=NMEL, nfft=512, rate=RATE):
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    edges = mel2hz(np.linspace(hz2mel(50), hz2mel(rate / 2 - 100), nmel + 2))
    bins = np.floor((nfft + 1) * edges / rate).astype(int)
    fb = np.zeros((nmel, nfft // 2 + 1))
    for m in range(nmel):
        a, b, c = bins[m], bins[m + 1], bins[m + 2]
        if b == a:
            b = a + 1
        if c == b:
            c = b + 1
        fb[m, a:b] = (np.arange(a, b) - a) / max(b - a, 1)
        fb[m, b:c] = (c - np.arange(b, c)) / max(c - b, 1)
    return fb


_FB = mel_filters()
_DCT = np.array([[np.cos(np.pi * k * (2 * m + 1) / (2 * NMEL)) for m in range(NMEL)] for k in range(NCEP)])
_WIN = np.hanning(WIN + 2)[1:-1]


def mfcc(x):
    """x -> [nframes, NCEP] (frame t covers samples [t*HOP, t*HOP+WIN))"""
    n = max(1 + (len(x) - WIN) // HOP, 0)
    if n <= 0:
        return np.zeros((0, NCEP))
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    seg = x[idx] * _WIN
    S = np.abs(np.fft.rfft(seg, 512, axis=1)) ** 2
    M = np.log(S @ _FB.T + 1e-6)
    return M @ _DCT.T


# ---------------------------------------------------------------- forced alignment
class Models:
    """one diagonal Gaussian per phone"""

    def __init__(self):
        self.sum = {}
        self.sq = {}
        self.n = {}
        self.mu = {}
        self.var = {}

    def add(self, ph, X):
        if len(X) == 0:
            return
        self.sum[ph] = self.sum.get(ph, 0) + X.sum(0)
        self.sq[ph] = self.sq.get(ph, 0) + (X ** 2).sum(0)
        self.n[ph] = self.n.get(ph, 0) + len(X)

    def estimate(self):
        self.mu, self.var = {}, {}
        for ph, n in self.n.items():
            if n < 5:
                continue
            mu = self.sum[ph] / n
            var = np.maximum(self.sq[ph] / n - mu ** 2, 1.0)
            self.mu[ph] = mu
            self.var[ph] = var
        self.sum, self.sq, self.n = {}, {}, {}

    def cost(self, ph, X):
        """-log likelihood per frame, or None when the phone has no model yet"""
        if ph not in self.mu:
            return None
        mu, var = self.mu[ph], self.var[ph]
        return 0.5 * (np.log(2 * np.pi * var).sum() + (((X - mu) ** 2) / var).sum(1))


def viterbi(costs, minfr=MINFR):
    """costs[state][frame] -> boundaries: a left-to-right pass with a minimum duration per state"""
    ns, nf = len(costs), len(costs[0])
    if ns == 1:
        return [(0, nf)]
    big = 1e18
    D = np.full((ns, nf + 1), big)
    B = np.zeros((ns, nf + 1), dtype=int)
    cum = [np.concatenate([[0.0], np.cumsum(c)]) for c in costs]   # cum[s][b] - cum[s][a] = cost of [a,b)
    D[0, 0] = 0.0
    for s in range(ns):
        lo_prev = s * minfr
        for e in range(lo_prev + minfr, nf + 1):
            if s == 0:
                D[0, e] = cum[0][e]
                continue
            best, arg = big, -1
            for b in range(lo_prev, e - minfr + 1):
                if D[s - 1, b] >= big:
                    continue
                v = D[s - 1, b] + cum[s][e] - cum[s][b]
                if v < best:
                    best, arg = v, b
            D[s, e] = best
            B[s, e] = arg
    if D[ns - 1, nf] >= big:
        return None
    bounds, e = [], nf
    for s in range(ns - 1, -1, -1):
        b = B[s, e] if s else 0
        bounds.append((b, e))
        e = b
    return bounds[::-1]


def stage_align(args):
    c = Corpus(args.corpus)
    # which corpus units to use: every single-phone unit, and up to --per-type of the others
    by_type = {}
    for i, t in enumerate(c.u["type"]):
        by_type.setdefault(int(t), []).append(i)
    picks = []
    for t, idx in by_type.items():
        ph = c.phones.get(t, [])
        if not ph:
            continue
        picks += idx if len(ph) == 1 else idx[: args.per_type]
    picks.sort()
    print("aligning %d of %d corpus units (%d types)" % (len(picks), len(c.u), len(by_type)))

    feats, phs = {}, {}
    t0 = time.time()
    for k, i in enumerate(picks):
        x = c.audio(i)
        f = mfcc(x)
        if len(f) < MINFR:
            continue
        feats[i] = f.astype(np.float32)
        phs[i] = c.phones[int(c.u[i]["type"])]
        if k % 5000 == 0:
            print("  features %d/%d (%.0f s)" % (k, len(picks), time.time() - t0))
    print("  features done: %d units, %.1f MB" % (len(feats), sum(f.nbytes for f in feats.values()) / 1e6))

    m = Models()
    for i, f in feats.items():                      # bootstrap
        p = phs[i]
        if len(p) == 1:
            m.add(p[0], f)
        else:                                       # split proportionally, drop the transitions
            cuts = np.linspace(0, len(f), len(p) + 1).astype(int)
            for j, ph in enumerate(p):
                a, b = cuts[j], cuts[j + 1]
                if b - a > 2:
                    m.add(ph, f[a + 1:b - 1])
    m.estimate()
    print("  bootstrapped %d phone models" % len(m.mu))

    seg = {}
    for it in range(args.iters):
        nm = Models()
        ok = bad = 0
        t0 = time.time()
        for i, f in feats.items():
            p = phs[i]
            costs = []
            for ph in p:
                cst = m.cost(ph, f)
                if cst is None:
                    costs = None
                    break
                costs.append(cst)
            if costs is None or len(f) < MINFR * len(p):
                bad += 1
                continue
            b = viterbi(costs)
            if b is None:
                bad += 1
                continue
            seg[i] = b
            ok += 1
            for j, ph in enumerate(p):
                a, e = b[j]
                nm.add(ph, f[a:e])
        nm.estimate()
        m = nm
        print("  pass %d: aligned %d, skipped %d (%.0f s)" % (it + 1, ok, bad, time.time() - t0))

    np.savez_compressed(os.path.join(args.corpus, "segments.npz"),
                        unit=np.array(sorted(seg)),
                        bounds=np.array([np.array(seg[i], dtype=np.int32).ravel() for i in sorted(seg)], dtype=object),
                        allow_pickle=True)
    print("wrote %s/segments.npz (%d units)" % (args.corpus, len(seg)))


# ---------------------------------------------------------------- stage 2: choose and analyse instances
def stage_analyze(args):
    c = Corpus(args.corpus)
    z = np.load(os.path.join(args.corpus, "segments.npz"), allow_pickle=True)
    seg = {int(u): np.asarray(b).reshape(-1, 2) for u, b in zip(z["unit"], z["bounds"])}
    tab, names, nunits, ubase = load_leaf_table(os.path.join(args.corpus, "samleaf.bin"))
    pid = {n: i for i, n in enumerate(names)}
    print("%d aligned units, Sam has %d phones / %d units" % (len(seg), len(names), nunits))

    # recording order, so a unit's neighbours give the phones on either side
    order = np.argsort(c.u["start"], kind="stable")
    pos_in_rec = {int(i): k for k, i in enumerate(order)}
    ends = c.u["start"] + c.u["len"]

    cand = {}                                    # sam unit -> list of (corpus unit, s0, s1, nper)
    ninst = 0
    for i, bounds in seg.items():
        ph = c.phones[int(c.u[i]["type"])]
        if len(ph) != len(bounds):
            continue
        k = pos_in_rec[i]
        prev_i = int(order[k - 1]) if k > 0 else -1
        next_i = int(order[k + 1]) if k + 1 < len(order) else -1
        lctx = "sil"
        if prev_i >= 0 and ends[prev_i] == c.u[i]["start"]:
            pp = c.phones[int(c.u[prev_i]["type"])]
            lctx = pp[-1] if pp else "sil"
        rctx = "sil"
        if next_i >= 0 and ends[i] == c.u[next_i]["start"]:
            np_ = c.phones[int(c.u[next_i]["type"])]
            rctx = np_[0] if np_ else "sil"
        for j, p in enumerate(ph):
            left = ph[j - 1] if j > 0 else lctx
            right = ph[j + 1] if j + 1 < len(ph) else rctx
            tgts = sam_targets(p, pid, int(c.u[i]["feat"][7]))
            if not tgts:
                continue
            li, ri = pid.get(sam_context(left, pid), pid["SIL"]), pid.get(sam_context(right, pid), pid["SIL"])
            a, b = bounds[j]
            s0 = int(a) * HOP + WIN // 2
            s1 = int(b) * HOP + WIN // 2
            s1 = min(s1, int(c.u[i]["len"]))
            if s1 - s0 < 160:
                continue
            ninst += 1
            for t in tgts:
                for q in range(4):
                    uid = int(tab[pid[t], li, ri, q])
                    if uid >= 0:
                        cand.setdefault(uid, []).append((i, s0, s1))
    print("%d phone instances -> candidates for %d of %d Sam units" % (ninst, len(cand), nunits))

    # a fallback per phone, for the contexts her corpus never had
    by_phone = {}
    for i, bounds in seg.items():
        ph = c.phones[int(c.u[i]["type"])]
        if len(ph) != len(bounds):
            continue
        for j, p in enumerate(ph):
            a, b = bounds[j]
            s0, s1 = int(a) * HOP + WIN // 2, min(int(b) * HOP + WIN // 2, int(c.u[i]["len"]))
            if s1 - s0 >= 160:
                by_phone.setdefault(p, []).append((i, s0, s1))
    fallback = {}
    for t in names:
        srcs = [p for p in list(CONS) + list(VOW) + list(EXTRA) if t in sam_targets(p, pid)]
        pool = [x for p in srcs for x in by_phone.get(p, [])]
        if pool:
            d = np.array([s1 - s0 for _, s0, s1 in pool])
            rank = np.argsort(np.abs(d - np.median(d)))
            fallback[t] = [pool[int(k)] for k in rank[:12]]         # the most typical ones
    print("fallbacks for %d phones" % len(fallback))

    lag = np.exp(-0.5 * (2 * np.pi * 60.0 / RATE * np.arange(21)) ** 2)
    frames = {}
    durms = {}                                    # her own duration for each unit, in ms
    t0 = time.time()
    done = 0
    for uid in range(nunits):
        pool = cand.get(uid, [])
        if not pool:
            continue                                  # filled from the phone's fallback below
        d = np.array([s1 - s0 for _, s0, s1 in pool])
        rank = np.argsort(np.abs(d - np.median(d)))   # the instances closest to this unit's typical length
        got = []
        for k in rank[: args.per_unit * 3]:
            i, s0, s1 = pool[int(k)]
            fr = analyse_instance(c, i, s0, s1, lag, args.max_frames)
            if fr and len(fr) >= 3:
                got.append(fr)
            if len(got) >= args.per_unit:
                break
        if got:
            frames[uid] = average_instances(got) if len(got) > 1 else got[0]
            durms[uid] = float(np.median([pool[int(k)][2] - pool[int(k)][1] for k in rank[:len(got)]])) / 16.0
        done += 1
        if done % 500 == 0:
            print("  %d units analysed (%.0f s)" % (done, time.time() - t0))
    # units with no candidate get their phone's fallback instance
    miss = 0
    for uid in range(nunits):
        if uid in frames:
            continue
        p = unit_phone(uid, tab, names)
        pool = fallback.get(p) or []
        got = []
        for (i, s0, s1) in pool:
            fr = analyse_instance(c, i, s0, s1, lag, args.max_frames)
            if fr and len(fr) >= 3:
                got.append(fr)
            if len(got) >= args.per_unit:
                break
        if got:
            frames[uid] = average_instances(got) if len(got) > 1 else got[0]
            miss += 1
    # anything still empty (a phone with no usable recording at all, e.g. SIL or +NOISE+) borrows the
    # nearest filled unit of the same phone, so no unit renders as silence
    empty = [u for u in range(nunits) if u not in frames]
    borrowed = 0
    for u in empty:
        p = unit_phone(u, tab, names)
        same = [k for k in frames if unit_phone(k, tab, names) == p]
        src = min(same, key=lambda k: abs(k - u)) if same else (min(frames, key=lambda k: abs(k - u)) if frames else None)
        if src is not None:
            frames[u] = frames[src]
            borrowed += 1
    print("filled %d units (%d from a fallback, %d borrowed), %d still empty"
          % (len(frames), miss, borrowed, nunits - len(frames)))
    np.savez_compressed(os.path.join(args.corpus, "frames.npz"),
                        uid=np.array(sorted(frames)),
                        data=np.array([frames[u] for u in sorted(frames)], dtype=object),
                        dur=np.array([durms.get(u, 0.0) for u in sorted(frames)]), allow_pickle=True)
    print("wrote %s/frames.npz" % args.corpus)


def load_wav(path):
    """Read mono or multichannel audio and resample it to the analysis rate."""
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    rate, audio = wavfile.read(path)
    if rate <= 0 or len(audio) == 0:
        raise ValueError("WAV must have a positive sample rate and contain audio")
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        if info.min == 0:
            midpoint = float(info.max + 1) / 2.0
            audio = (audio.astype(np.float64) - midpoint) / midpoint
        else:
            scale = float(max(abs(info.min), abs(info.max)))
            audio = audio.astype(np.float64) / scale
    else:
        audio = audio.astype(np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    elif audio.ndim != 1:
        raise ValueError("WAV must contain mono or multichannel PCM audio")
    if not np.all(np.isfinite(audio)):
        raise ValueError("WAV contains non-finite samples")
    if rate != RATE:
        divisor = math.gcd(int(rate), RATE)
        audio = resample_poly(audio, RATE // divisor, int(rate) // divisor)
    return np.asarray(audio, dtype=np.float64)


def parse_htk_lab(path, nsamples, rate=RATE):
    """Read phone-level HTK labels; timestamps use HTK 100 ns units."""
    labels = []
    previous_end = 0
    silence = {"SIL", "SP", "PAU", "H#", "BOUNDARY"}
    with open(path, encoding="utf-8-sig") as lab:
        lines = enumerate(lab, 1)
        for line_no, line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 3:
                raise ValueError("lab line %d must have start, end, and phone label" % line_no)
            try:
                start, end = int(fields[0]), int(fields[1])
            except ValueError:
                raise ValueError("lab line %d has non-integer HTK timestamps" % line_no)
            if start < 0 or end <= start or start < previous_end:
                raise ValueError("lab line %d has invalid or overlapping timestamps" % line_no)
            a = int(round(start * rate / 10_000_000))
            b = min(nsamples, int(round(end * rate / 10_000_000)))
            if a >= nsamples:
                raise ValueError("lab line %d starts after the end of the WAV" % line_no)
            label = fields[2].upper()
            lower_label = label.lower()
            if label in silence:
                targets = ["SIL"]
            else:
                targets = [label]
                if lower_label[-1:] in ("0", "1", "2") and lower_label[:-1] in VOW:
                    targets = [VOW[lower_label[:-1]] + ("1" if lower_label[-1] in ("1", "2") else "0")]
                elif lower_label in VOW:
                    targets = [VOW[lower_label] + "0", VOW[lower_label] + "1"]
            labels.append((a, b, label, targets))
            previous_end = end
    if not labels:
        raise ValueError("lab file contains no phone labels")
    return labels


def estimate_period(audio, center, min_period=40, max_period=320):
    """Estimate a voiced pitch period by the strongest normalized autocorrelation peak."""
    half = 300
    start, end = max(0, center - half), min(len(audio), center + half)
    frame = audio[start:end]
    if len(frame) < 2 * min_period or np.dot(frame, frame) < 1e-7:
        return None
    frame = frame - np.mean(frame)
    nfft = 1 << (2 * len(frame) - 1).bit_length()
    ac = np.fft.irfft(np.abs(np.fft.rfft(frame, nfft)) ** 2, nfft)[:max_period + 1]
    lags = np.arange(min_period, min(max_period + 1, len(ac)))
    normalized = ac[lags] / np.maximum(len(frame) - lags, 1)
    normalized /= max(ac[0] / len(frame), 1e-12)
    peaks = np.flatnonzero((normalized[1:-1] >= normalized[:-2]) &
                           (normalized[1:-1] > normalized[2:])) + 1
    if not len(peaks):
        return None
    peaks = peaks[normalized[peaks] >= 0.3]
    return int(lags[peaks[0]]) if len(peaks) else None


def make_pitch_epochs(audio):
    """Create signed pitch-period lengths from a recording (negative marks unvoiced frames)."""
    periods = []
    position = 0
    while position < len(audio):
        center = position + 200
        period = estimate_period(audio, center)
        if period is None:
            period = min(80, len(audio) - position)
            signed = -period
        else:
            period = min(period, len(audio) - position)
            signed = period
        if period < 16:
            break
        periods.append(signed)
        position += period
    return np.asarray(periods, dtype=np.int16)


class WaveRecording:
    def __init__(self, audio, epochs):
        self._audio = audio
        self._epochs = epochs
        self.marks = np.concatenate([[0], np.cumsum(np.abs(epochs))])

    def audio(self, _index):
        return self._audio

    def epochs(self, _index):
        return self._epochs


def stage_wav(args):
    """Build a complete Sam-format voice from a WAV and phone-level HTK lab."""
    if args.per_unit < 1 or args.max_frames < 3:
        raise ValueError("--per-unit must be positive and --max-frames must be at least 3")
    audio = load_wav(args.wav)
    labels = parse_htk_lab(args.lab, len(audio))
    tab, names, nunits, _ = load_leaf_table(args.leaf_table)
    pid = {name: i for i, name in enumerate(names)}
    if "SIL" not in pid:
        raise ValueError("the template Sam voice has no SIL phone")
    for _, _, label, targets in labels:
        missing = [target for target in targets if target not in pid]
        if missing:
            raise ValueError("lab phone %r maps to phone(s) missing from the Sam voice: %s" %
                             (label, ", ".join(missing)))

    epochs = make_pitch_epochs(audio)
    recording = WaveRecording(audio, epochs)
    lag = np.exp(-0.5 * (2 * np.pi * 60.0 / RATE * np.arange(21)) ** 2)
    candidates, fallback = {}, {}
    analysis_cache = {}

    def analyze(instance, start, end):
        key = (instance, start, end)
        if key not in analysis_cache:
            analysis_cache[key] = analyse_instance(recording, instance, start, end, lag, args.max_frames)
        return analysis_cache[key]

    for i, (_, _, _, targets) in enumerate(labels):
        left_label = labels[i - 1][3][0] if i else "SIL"
        right_label = labels[i + 1][3][0] if i + 1 < len(labels) else "SIL"
        left = pid[left_label] if left_label in pid else pid["SIL"]
        right = pid[right_label] if right_label in pid else pid["SIL"]
        a, b, _, phones = labels[i]
        if b - a < 160:
            continue
        for phone in phones:
            phone_id = pid[phone]
            fallback.setdefault(phone, []).append((i, a, b))
            for position in range(4):
                uid = int(tab[phone_id, left, right, position])
                if uid >= 0:
                    candidates.setdefault(uid, []).append((i, a, b))

    frames, durations = {}, {}
    for uid, pool in candidates.items():
        lengths = np.array([b - a for _, a, b in pool])
        rank = np.argsort(np.abs(lengths - np.median(lengths)))
        got, selected_lengths = [], []
        for index in rank[:args.per_unit * 3]:
            instance, start, end = pool[int(index)]
            result = analyze(instance, start, end)
            if result and len(result) >= 3:
                got.append(result)
                selected_lengths.append(end - start)
            if len(got) >= args.per_unit:
                break
        if got:
            frames[uid] = average_instances(got) if len(got) > 1 else got[0]
            durations[uid] = float(np.median(selected_lengths)) / 16.0

    fallback_frames = {}
    for phone, pool in fallback.items():
        lengths = np.array([end - start for _, start, end in pool])
        rank = np.argsort(np.abs(lengths - np.median(lengths)))
        got = []
        for index in rank:
            instance, start, end = pool[int(index)]
            result = analyze(instance, start, end)
            if result and len(result) >= 3:
                got.append(result)
            if len(got) >= args.per_unit:
                break
        if got:
            fallback_frames[phone] = average_instances(got) if len(got) > 1 else got[0]

    for uid in range(nunits):
        if uid not in frames:
            phone = unit_phone(uid, tab, names)
            if phone in fallback_frames:
                frames[uid] = fallback_frames[phone]

    empty = [uid for uid in range(nunits) if uid not in frames]
    for uid in empty:
        phone = unit_phone(uid, tab, names)
        matching = [source for source in frames if unit_phone(source, tab, names) == phone]
        source = min(matching, key=lambda source: abs(source - uid)) if matching else (
            min(frames, key=lambda source: abs(source - uid)) if frames else None)
        if source is not None:
            frames[uid] = frames[source]
    if not frames:
        raise ValueError("no usable phone instances were found in the WAV and lab file")
    if not any(frame[4] is not None for sequence in frames.values() for frame in sequence):
        raise ValueError("no voiced frames were found; the SPD builder needs voiced speech for excitation codebooks")
    print("analysed %d of %d units; %d units use a borrowed frame sequence" %
          (nunits - len(empty), nunits, len(empty)))

    with tempfile.TemporaryDirectory(prefix="wav2spd-") as corpus:
        shutil.copyfile(args.leaf_table, os.path.join(corpus, "samleaf.bin"))
        uids = sorted(frames)
        np.savez_compressed(os.path.join(corpus, "frames.npz"),
                            uid=np.array(uids),
                            data=np.array([frames[uid] for uid in uids], dtype=object),
                            dur=np.array([durations.get(uid, 0.0) for uid in uids]),
                            allow_pickle=True)
        build_args = argparse.Namespace(
            corpus=corpus, sam=args.sam, out=args.out, rate=args.rate, books=args.books,
            keep_durations=args.keep_durations)
        stage_build(build_args)


def unit_phone(uid, tab, names):
    """which phone a Sam unit index belongs to (the table says which phone produced it)"""
    if not hasattr(unit_phone, "map"):
        m = {}
        for p in range(tab.shape[0]):
            for u in np.unique(tab[p]):
                if u >= 0:
                    m.setdefault(int(u), names[p])
        unit_phone.map = m
    return unit_phone.map.get(uid, "SIL")


def resample_frames(fr, T, order=20, nfft=512):
    """one instance's frames -> exactly T frames on a common time axis (LSF and log gain interpolated,
    the pulse spectrum taken from the nearest source frame)"""
    m = len(fr)
    src = np.linspace(0.0, m - 1.0, T)
    lsf = np.array([f[2] for f in fr], dtype=np.float64)
    gain = np.log(np.maximum([f[3] for f in fr], 1e-9))
    per = np.array([f[0] for f in fr], dtype=np.float64)
    vo = np.array([f[1] for f in fr], dtype=np.float64)
    out = []
    for t in range(T):
        p = src[t]
        i0, w = int(np.floor(p)), p - np.floor(p)
        i1 = min(i0 + 1, m - 1)
        k = int(round(p))
        out.append(((1 - w) * lsf[i0] + w * lsf[i1],
                    (1 - w) * gain[i0] + w * gain[i1],
                    (1 - w) * per[i0] + w * per[i1],
                    (1 - w) * vo[i0] + w * vo[i1],
                    fr[k][4]))
    return out


def to_complex(packed, nfft=512):
    X = np.zeros(nfft // 2 + 1, dtype=complex)
    X.real = packed[:nfft // 2 + 1]
    X.imag[1:nfft // 2] = packed[nfft // 2 + 1:][::-1]
    return X


def from_complex(X, nfft=512):
    packed = np.zeros(nfft)
    packed[:nfft // 2 + 1] = X.real
    packed[nfft // 2 + 1:] = X.imag[1:nfft // 2][::-1]
    return packed


def lsf_to_lpc(lsf):
    """LSF (0..0.5) -> a[0..p]; the inverse of lpc_to_lsf, same convention as sam.c"""
    p = len(lsf)
    w = 2 * np.pi * np.sort(np.asarray(lsf, dtype=np.float64))
    P = np.array([1.0])
    Q = np.array([1.0])
    for i in range(0, p, 2):
        P = np.convolve(P, [1.0, -2.0 * np.cos(w[i]), 1.0])
    for i in range(1, p, 2):
        Q = np.convolve(Q, [1.0, -2.0 * np.cos(w[i]), 1.0])
    a = 0.5 * (np.convolve(P, [1.0, 1.0])[:p + 1] + np.convolve(Q, [1.0, -1.0])[:p + 1])
    a[0] = 1.0
    return a


def average_envelopes(lsfs, order=20, nfft=512):
    """average several filters as SPECTRA, not as LSF vectors.

    The mean of LSF vectors sits between the instances' formants and comes out flatter than any of them,
    which is what made the averaged voice duller than the single-instance one.  Averaging the log power
    spectra and re-fitting keeps the peaks where the recordings actually put them."""
    acc = None
    for lsf in lsfs:
        H = np.fft.rfft(lsf_to_lpc(lsf), nfft)
        lp = -2.0 * np.log(np.abs(H) + 1e-12)
        acc = lp if acc is None else acc + lp
    pw = np.exp(acc / len(lsfs))
    r = np.fft.irfft(pw)[:order + 1]
    r[0] *= 1.0001
    a, _ = levinson(r, order)
    out = lpc_to_lsf(a)
    return out if out is not None and len(out) == order else np.sort(np.mean(lsfs, axis=0))


def average_spectra(specs, nfft=512):
    """average pulse spectra in the MAGNITUDE domain, keeping one instance's phase.

    Averaging the complex spectra cancels the high band: the pulses agree on the main peak (that is what
    the rotation aligns) but not on the fine phase structure above a couple of kHz."""
    X = [to_complex(np.asarray(s, dtype=np.float64), nfft) for s in specs]
    mag = np.mean([np.abs(x) for x in X], axis=0)
    k = int(np.argmin([float(np.sum((np.abs(x) - mag) ** 2)) for x in X]))
    return from_complex(mag * np.exp(1j * np.angle(X[k])), nfft)


def average_instances(insts, order=20, nfft=512, smooth=True):
    """several instances of the same unit -> one averaged frame sequence (this is what kills the roughness:
    a single recording carries its own noise and its own pitch jitter into every use of the unit)"""
    lens = sorted(len(f) for f in insts)
    T = int(np.clip(lens[len(lens) // 2], 3, 40))
    R = [resample_frames(f, T) for f in insts]
    out = []
    for t in range(T):
        lsf = average_envelopes([r[t][0] for r in R]) if len(R) > 1 else R[0][t][0]
        gain = float(np.exp(np.mean([r[t][1] for r in R])))
        per = int(round(np.mean([r[t][2] for r in R])))
        voiced = np.mean([r[t][3] for r in R]) >= 0.5
        spec = None
        if voiced:
            specs = [r[t][4] for r in R if r[t][4] is not None]
            if specs:
                s = average_spectra(specs) if len(specs) > 1 else np.asarray(specs[0], dtype=np.float64)
                n = float(np.sqrt(np.mean(s ** 2)))
                if n > 1e-9:
                    spec = (s / n).astype(np.float32)
            if spec is None:
                voiced = False
        out.append((max(per, 16), 1 if voiced else 0, lsf.astype(np.float32), gain, spec))
    if smooth and T >= 3:                      # light smoothing along the unit, as the averaging leaves steps
        L = np.array([f[2] for f in out], dtype=np.float64)
        S = L.copy()
        S[1:-1] = 0.15 * L[:-2] + 0.70 * L[1:-1] + 0.15 * L[2:]
        G = np.log(np.maximum([f[3] for f in out], 1e-9))
        H = G.copy()
        H[1:-1] = 0.25 * G[:-2] + 0.5 * G[1:-1] + 0.25 * G[2:]   # gain steps click, so smooth those too
        out = [(p, v, np.sort(S[i]).astype(np.float32), float(np.exp(H[i])), s)
               for i, (p, v, _l, _g, s) in enumerate(out)]
    return out


def analyse_instance(c, i, s0, s1, lag, max_frames, order=20):
    """one phone instance -> Sam frames: (period, voiced, lsf[20], gain, spectrum[512] or None)"""
    x = c.audio(i)
    ep = c.epochs(i)
    if len(ep) == 0 or len(x) == 0:
        return None
    marks = getattr(c, "marks", None)
    if marks is None:
        marks = np.concatenate([[0], np.cumsum(np.abs(ep))])
    out = []
    first = max(0, int(np.searchsorted(marks, s0, side="right")) - 1)
    for k in range(first, len(ep)):
        s, n = int(marks[k]), int(abs(ep[k]))
        if n < 16 or n > 800 or s < s0 or s + n > min(s1, len(x)):
            continue
        r = analyse_period(x, s, n, order, lag)
        if r is None:
            continue
        lsf, res = r
        if ep[k] > 0:
            spec = pack_spectrum(res)
            g = float(np.sqrt(np.mean(spec ** 2)))
            if g <= 1e-9:
                continue
            out.append((n, 1, lsf.astype(np.float32), g, (spec / g).astype(np.float32)))
        else:
            g = float(np.sqrt(np.mean(res ** 2)) / 0.02)
            out.append((n, 0, lsf.astype(np.float32), g, None))
        if len(out) >= max_frames:
            break
    return out or None


# ---------------------------------------------------------------- Anna's phones -> Sam's phone set
# Sam splits vowels into stressed (AA1) and unstressed (AA0); Anna's unit names carry no stress, so both
# get the same recordings.  Her extra phones: ix (reduced high vowel) -> IH0, dx (flap) -> D.
CONS = {"b": "B", "ch": "CH", "d": "D", "dh": "DH", "f": "F", "g": "G", "h": "HH", "jh": "JH", "k": "K",
        "l": "L", "m": "M", "n": "N", "ng": "NG", "p": "P", "r": "R", "s": "S", "sh": "SH", "t": "T",
        "th": "TH", "v": "V", "w": "W", "y": "Y", "z": "Z", "zh": "ZH", "sil": "SIL", "dx": "D"}
VOW = {"aa": "AA", "ae": "AE", "ah": "AH", "ao": "AO", "aw": "AW", "ay": "AY", "eh": "EH", "er": "ER",
       "ey": "EY", "ih": "IH", "iy": "IY", "ow": "OW", "oy": "OY", "uh": "UH", "uw": "UW"}
EXTRA = {"ax": "AH0", "ix": "IH0"}


def sam_targets(ph, names, stress=None):
    """the Sam phones an Anna phone can fill.

    Sam keeps stressed and unstressed vowels apart (AA1 / AA0); her recorded units carry their stress in
    UNT feature 7 (0 none, 1 primary, 2 secondary), so a stressed recording fills the stressed phone."""
    if ph in CONS:
        return [CONS[ph]] if CONS[ph] in names else []
    if ph in EXTRA:
        return [EXTRA[ph]] if EXTRA[ph] in names else []
    if ph in VOW:
        both = [v for v in (VOW[ph] + "1", VOW[ph] + "0") if v in names]
        if stress is None or len(both) < 2:
            return both
        want = VOW[ph] + ("1" if stress >= 1 else "0")
        return [want] if want in names else both
    return []


def sam_context(ph, names):
    """one Sam phone id to use as left/right context (the tree asks about phone classes)"""
    t = sam_targets(ph, names)
    return t[0] if t else "SIL"


def load_leaf_table(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"SLF1"
        nph, nunits = np.frombuffer(f.read(8), "<i4")
        tab = np.frombuffer(f.read(nph * nph * nph * 4 * 4), "<i4").reshape(nph, nph, nph, 4)
        base = np.frombuffer(f.read(nph * 4), "<i4")          # first unit index of each phone
        names = f.read().split(b"\0")[:nph]
    return tab, [n.decode() for n in names], int(nunits), np.asarray(base)


# ---------------------------------------------------------------- LPC / LSF
def levinson(r, p):
    a = np.zeros(p + 1)
    a[0] = 1.0
    e = r[0]
    if e <= 0:
        return a, 1.0
    for i in range(1, p + 1):
        acc = r[i] + (np.dot(a[1:i], r[i - 1:0:-1]) if i > 1 else 0.0)
        k = -acc / e
        a[1:i + 1] = a[1:i + 1] + k * a[i - 1::-1][:i]
        e *= 1.0 - k * k
        if e <= 0:
            return a, 1e-9
    return a, e


def lpc_to_lsf(a):
    """-> the 'order' line spectral frequencies, sorted, normalized 0..0.5 (Sam: cos(2*pi*lsf))"""
    p = len(a) - 1
    ar = a[::-1]
    P = np.polydiv(np.concatenate([a, [0.0]]) + np.concatenate([[0.0], ar]), np.array([1.0, 1.0]))[0]
    Q = np.polydiv(np.concatenate([a, [0.0]]) - np.concatenate([[0.0], ar]), np.array([1.0, -1.0]))[0]
    f = np.concatenate([np.angle(np.roots(P)), np.angle(np.roots(Q))])
    f = np.sort(f[f > 1e-9]) / (2 * np.pi)
    return f if len(f) == p else None


def analyse_period(x, s, n, order, lag_win, nfft=512):
    """LPC + residual pulse of one pitch period starting at sample s of x"""
    w = max(2 * n, 200)
    c = s + n // 2
    a0, b0 = max(c - w // 2, 0), min(c + w // 2, len(x))
    seg = x[a0:b0]
    if len(seg) < order * 2:
        return None
    seg = seg * np.hanning(len(seg) + 2)[1:-1]
    r = np.correlate(seg, seg, "full")[len(seg) - 1:len(seg) - 1 + order + 1]
    if r[0] < 1e-6:
        return None
    r = r * lag_win
    r[0] *= 1.0001
    a, _ = levinson(r, order)
    a = a * (0.997 ** np.arange(order + 1))            # mild bandwidth expansion, for filter stability
    lsf = lpc_to_lsf(a)
    if lsf is None:
        return None
    # residual over the period (needs the previous `order` samples for the FIR memory)
    lo = max(s - order, 0)
    piece = x[lo:s + n]
    # convolve output index t is the residual of piece[t]; only the first len(piece) samples are the
    # signal (the rest is the FIR tail), and the period we want is the last n of those.
    res = np.convolve(piece, a)[:len(piece)][-n:]
    if len(res) < n:
        res = np.concatenate([np.zeros(n - len(res)), res])
    return lsf, res


def pack_spectrum(res, nfft=512, align=True):
    """one residual period -> the half-complex spectrum Sam stores (re[0..n/2], im[k] at n-k).

    The pulse is first rotated so its peak sits at a fixed place in the period.  Without that the glottal
    pulse lands at a different offset in every period, the phases disagree, and averaging or quantizing the
    spectra turns the pulse train into buzz."""
    n = len(res)
    if align and n > 4:
        k = int(np.argmax(np.abs(res)))
        res = np.roll(res, -k + n // 8)          # peak at 1/8 into the period, like a glottal closure
    m = min(n, nfft // 2)
    buf = np.zeros(nfft)
    buf[:m] = res[:m]
    if n > m:
        buf[nfft - (n - m):] = res[m:]
    X = np.fft.rfft(buf)
    packed = np.zeros(nfft)
    packed[:nfft // 2 + 1] = X.real
    packed[nfft // 2 + 1:] = X.imag[1:nfft // 2][::-1]
    return packed


# ---------------------------------------------------------------- stage 3: codebooks and the file
def kmeans(X, k, iters=12, seed=1):
    """plain Lloyd with k-means++-ish seeding; X[n, d] -> centroids[k, d]"""
    rng = np.random.default_rng(seed)
    n = len(X)
    if n <= k:
        C = np.zeros((k, X.shape[1]), dtype=np.float64)
        C[:n] = X
        C[n:] = X[rng.integers(0, max(n, 1), k - n)] if n else 0.0
        return C
    C = X[rng.choice(n, k, replace=False)].astype(np.float64).copy()
    for _ in range(iters):
        lab = nearest(X, C)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(0)
            else:
                C[j] = X[rng.integers(0, n)]
    return C


def nearest(X, C, block=4096):
    """index of the closest centroid for every row of X"""
    out = np.empty(len(X), dtype=np.int32)
    cn = (C ** 2).sum(1)
    for a in range(0, len(X), block):
        b = X[a:a + block]
        d = cn[None, :] - 2.0 * (b @ C.T)
        out[a:a + block] = np.argmin(d, axis=1)
    return out


def escape_periods(vals):
    """Sam's period encoding: signed bytes, 127 / -128 continue into the next byte"""
    out = bytearray()
    for v in vals:
        v = int(v)
        if -127 < v < 127:
            out.append(v & 0xFF)
            continue
        if v >= 127:
            k, r = divmod(v, 127)
            if r == 127:
                k, r = k + 1, 0
            out += bytes([127]) * k
            out.append(r & 0xFF)
        else:
            k, r = divmod(-v, 128)
            if r == 128:
                k, r = k + 1, 0
            out += bytes([0x80]) * k
            out.append((-r) & 0xFF)
    return bytes(out)


def stage_build(args):
    with np.load(os.path.join(args.corpus, "frames.npz"), allow_pickle=True) as z:
        uids = list(z["uid"])
        frame_values = list(z["data"])
        duration_values = np.asarray(z["dur"]).copy()
    data = {int(u): d for u, d in zip(uids, frame_values)}
    tab, names, nunits, ubase = load_leaf_table(os.path.join(args.corpus, "samleaf.bin"))
    with open(args.sam, "rb") as f:
        sam = f.read()
    import struct
    off3, sz3 = struct.unpack("<II", sam[0x18 + 24:0x18 + 32])
    old = sam[off3:off3 + sz3]

    def u32(o, s=old):
        return struct.unpack("<I", s[o:o + 4])[0]

    order, nfft, log2 = u32(0x498), u32(0x49C), u32(0x4A0)
    if args.books == "sam":                      # exactly Sam's split
        lsf_dims = [u32(0x14 + 12 * i) for i in range(u32(4))]
        exc_dims = [u32(0x194 + 12 * i) for i in range(u32(8))]
    else:                                        # finer: 2 LSFs per book, and 15 excitation bands
        lsf_dims = [2] * (order // 2)
        exc_dims = [2 * h for h in (4, 4, 6, 8, 10, 12, 14, 16, 20, 20, 25, 25, 30, 30, 31)]
    print("target: order %d, FFT %d, LSF books %s, excitation books %s" % (order, nfft, lsf_dims, exc_dims))

    # ---- gather the training vectors
    all_lsf, all_spec = [], []
    for u in uids:
        for (n, v, lsf, g, spec) in data[u]:
            all_lsf.append(lsf)
            if spec is not None:
                all_spec.append(spec)
    L = np.array(all_lsf, dtype=np.float64)
    S = np.array(all_spec, dtype=np.float64)
    print("%d frames (%d voiced) from %d units" % (len(L), len(S), len(uids)))

    # ---- LSF codebooks (256 entries each, split by the original dims)
    lsf_cb, o = [], 0
    for d in lsf_dims:
        C = kmeans(L[:, o:o + d], 256, iters=25, seed=o + 1)
        lsf_cb.append(C)
        o += d
    # ---- excitation codebooks (128 entries): book b covers bins [bin, bin+half),
    #      vector = [Re(bin..bin+half), Im(bin+half-1..bin)]  (the second half is reversed)
    exc_cb, bins = [], []
    bin0 = 1
    for d in exc_dims:
        half = d // 2
        re = S[:, bin0:bin0 + half]
        im = S[:, nfft - bin0 - half + 1:nfft - bin0 + 1]
        V = np.concatenate([re, im], axis=1)
        exc_cb.append(kmeans(V, 256, iters=20, seed=bin0 + 7))  # a byte indexes 256, Sam only used 128
        bins.append((bin0, half))
        bin0 += half
    print("codebooks trained (%d LSF, %d excitation), spectrum bins 1..%d" % (len(lsf_cb), len(exc_cb), bin0 - 1))

    # ---- quantize every frame at once (one nearest-centroid pass per codebook, not per frame)
    units = []
    for uid in range(nunits):
        fr = data.get(uid)
        if fr is None or len(fr) == 0:                     # a unit her corpus never covered: near-silence
            fr = [(120, 0, np.linspace(0.02, 0.48, order).astype(np.float32), 1.0, None)]
        units.append(fr)
    flat = [f for fr in units for f in fr]
    FL = np.array([f[2] for f in flat], dtype=np.float64)
    voiced = [f for f in flat if f[4] is not None]
    FS = np.array([f[4] for f in voiced], dtype=np.float64) if voiced else np.zeros((0, nfft))
    lsf_idx, o = [], 0
    for bi, d in enumerate(lsf_dims):
        lsf_idx.append(nearest(FL[:, o:o + d], lsf_cb[bi]))
        o += d
    exc_idx = []
    for bi, (b0, half) in enumerate(bins):
        V = np.concatenate([FS[:, b0:b0 + half], FS[:, nfft - b0 - half + 1:nfft - b0 + 1]], axis=1)
        exc_idx.append(nearest(V, exc_cb[bi]) if len(FS) else np.zeros(0, dtype=np.int32))
    print("quantized %d frames (%d voiced)" % (len(FL), len(FS)))

    unit_blobs = []
    fi = vi = 0
    for fr in units:
        blob = bytearray()
        blob += struct.pack("<i", len(fr))
        blob += escape_periods([(n if v else -n) for (n, v, _l, _g, _s) in fr])
        for k in range(len(fr)):                             # LSF bytes
            for bi in range(len(lsf_dims)):
                blob.append(int(lsf_idx[bi][fi + k]))
        for f in fr:                                         # gains
            blob += struct.pack("<f", float(f[3]))
        for k, f in enumerate(fr):                           # excitation bytes, voiced frames only
            if f[4] is None:
                continue
            for bi in range(len(bins)):
                blob.append(int(exc_idx[bi][vi]))
            vi += 1
        fi += len(fr)
        unit_blobs.append(bytes(blob))

    # ---- lay out the new synth section
    hdr = bytearray(0x4B0)
    body = bytearray()

    def put_floats(a):
        off = 0x4B0 + len(body)
        body.extend(np.asarray(a, dtype="<f4").tobytes())
        return off

    def setu32(o, v):
        hdr[o:o + 4] = struct.pack("<I", v & 0xFFFFFFFF)

    setu32(0, args.rate)
    setu32(4, len(lsf_cb))
    setu32(8, len(exc_cb))
    setu32(0xC, 0)                       # no delta-excitation books: every voiced frame stands alone
    for i, C in enumerate(lsf_cb):
        off = put_floats(C.ravel())
        setu32(0x14 + 12 * i, C.shape[1])
        setu32(0x18 + 12 * i, off)
        setu32(0x1C + 12 * i, C.shape[0])
    for i, C in enumerate(exc_cb):
        off = put_floats(C.ravel())
        setu32(0x194 + 12 * i, C.shape[1])
        setu32(0x198 + 12 * i, off)
        setu32(0x19C + 12 * i, C.shape[0])
    sine = np.sin(2 * np.pi * np.arange(2 * nfft) / (2.0 * nfft))
    setu32(0x4A4, put_floats(sine))
    win = 0.5 * (1.0 + np.cos(np.pi * np.arange(nfft) / nfft))        # 1 -> 0 taper for fit_period
    setu32(0x4A8, put_floats(win))
    rng = np.random.default_rng(7)
    setu32(0x4AC, put_floats(rng.standard_normal(args.rate)))         # unit-variance noise source
    setu32(0x498, order)
    setu32(0x49C, nfft)
    setu32(0x4A0, log2)
    setu32(0x490, nunits)
    # unit table then the unit blobs
    tab_off = 0x4B0 + len(body)
    setu32(0x494, tab_off)
    body.extend(b"\0" * 4 * nunits)
    offs = []
    for blob in unit_blobs:
        offs.append(0x4B0 + len(body))
        body.extend(blob)
    for i, o in enumerate(offs):
        p = tab_off - 0x4B0 + 4 * i
        body[p:p + 4] = struct.pack("<I", o)
    new = bytes(hdr) + bytes(body)
    print("new synth section: %d bytes (%d units)" % (len(new), nunits))

    # ---- her own durations into section 1 (the front end's only per-unit timing input:
    #      sam_front.c reads base = float at leaf*16 in milliseconds)
    off1, sz1 = struct.unpack("<II", sam[0x18 + 8:0x18 + 16])
    sec1 = bytearray(sam[off1:off1 + sz1])
    dur = {int(u): float(d) for u, d in zip(uids, duration_values) if float(d) > 0}
    nph = len(names)
    changed = 0
    if not args.keep_durations:
        for p in range(nph):
            toff = struct.unpack("<I", bytes(sec1[p * 4:p * 4 + 4]))[0]
            if toff == 0 or toff >= sz1:
                continue
            for uid, ms in dur.items():
                leaf = uid - int(ubase[p])
                if leaf < 0:
                    continue
                pos = toff + leaf * 16
                if pos + 16 > sz1 or unit_phone(uid, tab, names) != names[p]:
                    continue
                ms = float(np.clip(ms, 15.0, 400.0))
                sec1[pos:pos + 4] = struct.pack("<f", ms)
                changed += 1
        print("durations: %d leaves set from her recordings" % changed)

    # ---- rebuild the container: sections 0-2 and 4 keep Sam's content, section 3 is ours
    out = bytearray(sam[:off3])
    out[off1:off1 + sz1] = sec1                     # section 1 with her timings
    out += new
    tail_off = off3 + sz3
    shift = len(new) - sz3
    out += sam[tail_off:]
    out[0x18 + 24:0x18 + 32] = struct.pack("<II", off3, len(new))
    for i in range(6):                                   # sections after ours move
        o, s = struct.unpack("<II", out[0x18 + 8 * i:0x20 + 8 * i])
        if i != 3 and o > off3:
            out[0x18 + 8 * i:0x1C + 8 * i] = struct.pack("<I", o + shift)
    with open(args.out, "wb") as f:
        f.write(bytes(out))
    print("wrote %s (%.1f MB)" % (args.out, len(out) / 1e6))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("align")
    a.add_argument("--corpus", default="corpus")
    a.add_argument("--per-type", type=int, default=150)
    a.add_argument("--iters", type=int, default=3)
    a.set_defaults(func=stage_align)
    b = sub.add_parser("analyze")
    b.add_argument("--corpus", default="corpus")
    b.add_argument("--per-unit", type=int, default=6)
    b.add_argument("--max-frames", type=int, default=48)
    b.set_defaults(func=stage_analyze)
    c = sub.add_parser("build")
    c.add_argument("--corpus", default="corpus")
    c.add_argument("--sam", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--rate", type=int, default=RATE)
    c.add_argument("--books", choices=["sam", "fine"], default="sam")
    c.add_argument("--keep-durations", action="store_true", help="keep Sam's timings")
    c.set_defaults(func=stage_build)
    w = sub.add_parser("wav", help="build a voice from a WAV and phone-level HTK lab")
    w.add_argument("--wav", required=True)
    w.add_argument("--lab", required=True, help="phone labels with HTK 100 ns timestamps")
    w.add_argument("--leaf-table", required=True, help="samleaf.bin generated from the template SPD")
    w.add_argument("--sam", required=True, help="template Sam-format SPD voice")
    w.add_argument("--out", required=True)
    w.add_argument("--rate", type=int, default=RATE)
    w.add_argument("--books", choices=["sam", "fine"], default="sam")
    w.add_argument("--per-unit", type=int, default=6)
    w.add_argument("--max-frames", type=int, default=48)
    w.add_argument("--keep-durations", action="store_true", help="keep template voice timings")
    w.set_defaults(func=stage_wav)
    args = ap.parse_args()
    try:
        args.func(args)
    except (OSError, ValueError) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
