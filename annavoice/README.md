# Microsoft Anna as a Sam voice

Microsoft Anna (Windows Vista and 7) is a **concatenative** voice: 5.5 hours of a real person's speech,
cut into demisyllables and glued back together. Microsoft Sam is a **vocoder**: a 1.5 MB file of LPC
filters and pulse shapes. This builds a Sam-format voice file out of Anna's recordings, so Sam's engine
speaks with her vowels and her consonants — her timbre, his buzz, at a five-hundredth of the size.

Everything Sam's engine can do then applies to her: his pitch contours, `--sing`, the SAPI 4 reverb and
RoboSoft effects, any rate, any pitch.

**No voice data is included here, and the file you build should stay on your machine.** The builder reads
your own installed Anna and your own `Sam.spd`; the result contains her recorded speech in another form.

## What you need

- **Microsoft Anna installed** — `C:\Program Files (x86)\Common Files\SpeechEngines\Microsoft\TTS20`
- **The Microsoft Anna port**, for its corpus decoder: <https://github.com/KamiKitsune420/ms-ana-decomp>
- **`Sam.spd`** from your own Windows / SAPI 5 install (the phone set, the trees and the unit layout)
- Python with numpy and scipy, and MSVC for the two C tools

## Building the voice

```
annavoice\build.bat ..\..\ms-ana-decomp\src          (builds annacorpus, annaunits, samleaf, samunit)

build\x64\annacorpus.exe --out corpus                 decode her corpus  (~5 min, 600 MB)
build\x64\samleaf.exe Sam.spd corpus\samleaf.bin      dump Sam's unit tree
python spdbuild.py align   --corpus corpus            forced alignment   (~2 min)
python spdbuild.py analyze --corpus corpus            pitch-synchronous analysis (~10 min)
python spdbuild.py build   --corpus corpus --sam Sam.spd --out Anna.spd
```

Then put `Anna.spd` next to `LTTS1033.LXA` and `r1033tts.LXA`, give it an `.sdf` for its base pitch
(Mary's, at 189 Hz, suits her), and speak:

```
build\x64\sam_say.exe --data voice --voice Anna --rate 16000 "Hello, my name is Anna." out.wav
```

## Building from a WAV and HTK labels

For a custom recording, supply a phone-level HTK `.lab` file alongside a WAV. Labels should use
the template voice's ARPAbet phone names (for example `AA1`, `T`, `SIL`); unstressed vowel labels
such as `AA` populate both stress variants, and `sil`, `sp`, `pau`, or `h#` are treated as `SIL`.
HTK timestamps are integer 100 ns units, as in `0 5000000 HH`. Use a Sam-format template SPD to
provide the phone set, trees, and unit layout:

```
build\x64\samleaf.exe Sam.spd samleaf.bin
python annavoice\spdbuild.py wav --wav recording.wav --lab recording.lab ^
  --leaf-table samleaf.bin --sam Sam.spd --out Custom.spd
```

The input audio is mixed to mono and resampled to 16 kHz for pitch-synchronous analysis. The builder
estimates pitch periods from the waveform, trains fresh codebooks, and uses the HTK phone boundaries
to select context-matched units and set their durations. Units not represented by the recording borrow
another instance of the same phone when possible; limited phone coverage may reduce voice quality.

## How it works

Sam's voice file is a pitch-synchronous LPC vocoder. Each of its 3365 units is a list of frames, and each
frame holds a pitch-period length (negative = unvoiced), 20 line spectral frequencies as 6 split-VQ bytes,
a gain, and — for voiced frames — the residual pulse's spectrum as 11 more bytes. A CART picks one unit
per phone from the phone itself, its neighbours and its position in the word.

The problem is that Anna's units are **demisyllables** (`w+aa+n`), so her vowels have no boundaries: only
her 60 single-phone types (all consonants) are directly usable.

1. **`annacorpus.c`** decodes all 159043 corpus units with their pitch epochs from her `.WIH`.
2. **`spdbuild.py align`** is a small forced aligner: MFCCs, one diagonal Gaussian per phone, bootstrapped
   from the single-phone consonants, Viterbi-aligned and re-estimated. That cuts ~142000 phones out.
3. **`samleaf.c`** asks Sam's own CART for all 562432 (phone, left, right, position) combinations, so her
   phones land in the same buckets his front end would use. Contexts her corpus never had fall back to the
   phone's most typical instances.
4. **`spdbuild.py analyze`** does the pitch-synchronous analysis per period: LPC order 20 → LSFs, residual
   pulse, gain. Six instances per unit are averaged (see below).
5. **`spdbuild.py build`** trains fresh codebooks on her data with k-means (nothing is reused from
   Microsoft's codebooks), quantizes every frame, and writes the file. Her measured phone durations go into
   section 1, so the voice keeps her rhythm rather than Sam's.

## Things that turned out to matter

- **The residual must line up with its filter.** `np.convolve(x, a)` returns the residual of sample *t* at
  index *t*; taking `[order:]` shifts every pulse 20 samples away from the filter state that produced it.
  That one slice sprayed broadband noise at every pitch period: +18 dB above 2.5 kHz, and it was audible as
  a rough, hissy voice. `roundtrip.py` and `rtdebug.py` exist to catch exactly this class of bug — they
  encode and decode one instance and compare it band by band with the original.
- **Pulses must be phase-aligned before they are averaged or quantized.** Each period's residual is rotated
  so the glottal pulse sits 1/8 into the period; otherwise the stored phases disagree and the pulse train
  turns into buzz.
- **Average pulse spectra as magnitudes, not complex.** Complex averaging cancels the high band, because the
  pulses agree on the main peak but not on the fine phase structure above a couple of kHz: it cost 4-5 dB
  between 1.5 and 8 kHz.
- **Average filters as spectra, not as LSF vectors.** The mean of several LSF vectors sits between their
  formants and is flatter than any of them.
- **Quantization was not the bottleneck.** Going from Sam's 6 LSF and 11 excitation books to 10 and 15
  (the format allows up to 16 of each, and every index is a full byte) changed the result by 0.04 dB.

## Measuring it

`voicecheck.py` compares any render with a reference: the distance between their long-term average spectra,
and the harmonic-to-noise ratio. `bandcmp.py` prints the same spectra band by band, which is how the +18 dB
and the missing high band were both found. Against real Anna saying the same sentence:

| build | spectrum distance | harmonic-to-noise |
|---|---|---|
| first working version | 16.49 | 6.6 dB |
| pulse alignment + averaging | 12.29 | 6.9 dB |
| residual alignment fixed | 8.30 | 10.7 dB |
| magnitude averaging, envelope averaging, stress-aware vowels | 5.28 | 9.8 dB |
| *real Anna* | — | 6.6 dB |

`samunit.exe` renders a single unit at a chosen pitch and length, which is how you tell whether a problem
is in one unit's encoding or in the joins between units.

## Known gaps

- She is still about 4 dB darker than the original between 1.5 and 8 kHz.
- Her `ix` and `dx` (the flap in "butter") have no home in Sam's phone set and are mapped to `IH0` and `D`.
- 32 of the 3365 units have no matching recording at all and borrow the nearest unit of the same phone.
- Sam's per-leaf amplitudes are kept; only the durations are hers.
