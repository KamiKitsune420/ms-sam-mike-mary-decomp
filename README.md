# Microsoft Sam, Mike and Mary in portable C

A reconstruction of the SAPI 5 "Microsoft TTS engine" (`spttseng.dll`, Windows XP) that speaks as
**Microsoft Sam, Microsoft Mike and Microsoft Mary**, in plain C99, plus the **SAPI 4 voice effects**
("Mike in Hall", "in Stadium", "in Space", the RoboSoft robots and Whisper) rebuilt as a reusable
audio effect. It reads the **original voice data files** at runtime and produces the same
audio as the real engine: identical length and every sample within ±1 LSB of the 16-bit output.
About 99.7% of samples match exactly; the rest differ by float rounding.

No Microsoft code is included. The C was written from a disassembly and decompilation of the engine
and checked function by function against the running DLL.

The three voices are the same engine and the same code; only the voice file (`Sam.spd`, `Mike.spd`,
`Mary.spd`) and each voice's base pitch (100, 110 and 189 Hz, read from its `.sdf`) differ. Mike and Mary
match the real voices as closely as Sam does (identical length, every sample within ±1 LSB).

**The voice data is not included.** You need `Sam.spd` (and/or `Mike.spd` / `Mary.spd` with their
`.sdf` files), `LTTS1033.LXA` and `r1033tts.LXA` from your own install (Windows XP, or the SAPI 5.1 runtime; look in
`C:\Program Files\Common Files\Microsoft Shared\Speech\` and its `1033` subfolder).

## Status

| stage | source | verified against the real engine |
|---|---|---|
| waveform synthesis (LSF/LPC vocoder, pitch marks, excitation) | `src/sam.c` | 99.7% of samples bit-exact, the rest ±1 LSB |
| compressed lexicon `LTTS1033.LXA` | `src/sam_lex.c` | 28,980 / 28,980 words |
| letter-to-sound `r1033tts.LXA` | `src/sam_lex.c` | 5,000 / 5,000 words |
| morphology (plural / past / -ing ... stems) | `src/sam_morph.c` | 2,000 / 2,000 words |
| text normalizer (sentences, abbreviations, numbers, dates, times, currency, phone numbers, ...) | `src/sam_norm.c` | token lists match on all test texts |
| part-of-speech tagger + homographs (read, live, wind, "the" before vowels ...) | `src/sam_pos.c` | 100 / 100 homograph sentences |
| phrasing, accents, durations, F0, unit selection | `src/sam_front.c` | exact |
| **end to end, text → WAV** | `sam_say` | **21/21 corpus sentences, 104/104 number/date lines, 100/100 homograph sentences, 180/180 paragraphs of *Alice in Wonderland* (Gutenberg edition, including its license text)** |

Not done yet:
- SAPI XML markup (`<rate>`, `<pitch>`, `<emph>`, `<spell>` ...) and user lexicons. Plain text only.
- The SAPI 4 voices themselves (`msttsl`, a different engine) are not ported; only their effects are.

## Building

Needs a C99 compiler. There are no platform dependencies in the library.

```
src\build.bat x64          (MSVC; finds Visual Studio / Build Tools itself; output in build\x64\)
src\build_clang.bat        (clang -std=c99 -pedantic, output in build\clang\)
make                       (Linux / macOS / MinGW, output in build/)
```

Windows builds produce **`sam.dll`** with its import library, and `build\<arch>\dist\` collects the DLL,
the `.lib`, the headers (`sam_tts.h` is the one to include) and the CLI — everything another program needs.
On other platforms `make dist` does the same with `libsam.so` / `.dylib`.

### Standalone exe (voice data compiled in)

Put the three data files in `data/voice/`, then:

```
python tools/embed_data.py            (writes src/sam_data.c, ~10 MB)
src\build.bat x64                     (also builds build\x64\sam_standalone.exe)
make standalone
```

The result needs no data files, but it contains Microsoft's data, so keep it to yourself.
`src/sam_data.c` is in `.gitignore`.

## Running

Point `--data` at the folder holding the voice files and the two `.LXA` files; `--voice` picks the voice.

```
build\x64\sam_say.exe --data data\voice "Hello, my name is Microsoft Sam." out.wav
build\x64\sam_say.exe --data data\voice --voice Mike "Hello, I am Mike." out.wav      (Mike.spd + Mike.sdf)
build\x64\sam_say.exe --data data\voice --voice Mary @story.txt out.wav              (read a UTF-8 file)
build\x64\sam_say.exe --data data\voice --voice Mike --effect hall "Mike in Hall." out.wav
```

### SAPI 4 voice effects

`--effect NAME` gives any voice the voice modes of the 1999 SAPI 4 engine (`msttssyn.dll`):

| name | original voice | what it is |
|---|---|---|
| `hall` | Mike / Mary in Hall | 4 allpass filters in series, 30.6 / 20.8 / 14.9 / 11.0 ms, feedback about 0.75 |
| `stadium` | ... in Stadium | the same with delays 5.3x longer (162 / 111 / 71 / 44 ms): separate slaps |
| `space` | ... in Space | one 400 ms allpass: a long fading repeat |
| `room` | (never used by Microsoft) | 81 / 55 / 36 / 22 ms |
| `robosoft1` ... `robosoft6` | RoboSoft One ... Six | a 10 ms allpass at 94% feedback rings at 100 Hz and its harmonics (the metallic buzz); One/Six shorten the loop by a random 0-40% on every chunk (the warble); One, Two, Five and Six are also monotone |
| `robot` | = `robosoft1` | |
| `whisper` | Male / Female / Sam Whisper | every frame gets noise excitation, then the FIR `[0.25, -0.5, 0.25]` |
| `monotone` | | flat pitch only |

Each effect is `out = clip(dry*x + allpasses(send*x))`; each allpass is `v = x + g*d, y = d - g*v` with the
preset's delays and dB gains (details and the original table addresses are in `src/sam4fx.c`). The
RoboSoft presets add up to +9 dB, which clips on the louder SAPI 5 voices, so `sam_say` trims their input
by 5-10 dB; `--no-trim` restores the original level. `src/sam4fx.h` works on any 16-bit audio stream.

Experiment knobs (not in the original engine):

```
--pitch 1.5          scale every pitch value
--speed 0.7          slower (<1) or faster (>1)
--gain 2             louder / clipped
--vibrato 3 6        the engine's hidden vibrato: depth, rate in Hz
--no-reverse         turn off time-reversal of repeated unvoiced frames
--reverse-units      play every sound unit backwards ("inside out")
--rate 48000         run the vocoder at another sample rate (real 48 kHz synthesis, not resampling)
```

### BigVGAN-inspired vocoder

`--vocoder bigvgan-inspired` selects an optional native-C harmonic source/filter renderer.
It builds a band-limited, phase-continuous harmonic excitation for voiced frames. The recorded
residual sets excitation level only; its waveform is not mixed into the output. The decoded LPC
envelope and unvoiced noise are retained. This mode needs no model files or runtime dependencies
and is intended as an experimental BigVGAN-inspired alternative; it is **not** the neural BigVGAN
model and cannot reproduce its
learned sound without trained weights. Voiced excitation uses a softer harmonic slope and lower
level to reduce harshness and leave headroom for vowel formants; a low-harmonic trim reduces bass,
unvoiced consonant frames get a gentle 5.5 kHz low-pass to tame hiss, and a smooth soft-knee
clipper rounds loud peaks without frame-to-frame limiter gain changes;
the default `--vocoder sam` keeps the original renderer unchanged.
The same option is available in the `sam` CLI, and library hosts can select it with
`sam_tts_set_vocoder(t, "bigvgan-inspired")`.

### Neural BigVGAN (offline)

`tools/bigvgan_say.py` runs an actual NVIDIA BigVGAN checkpoint. It feeds the SAM reference waveform
through NVIDIA's own waveform-to-mel frontend, with the volume normalization used during BigVGAN
training, then runs one whole-utterance PyTorch inference. This keeps the input features consistent
with standard checkpoints rather than approximating mels from LPC coefficients, which could cause
unstable output. The wrapper also checks that the exported timing and reference audio agree and
preserves exact silence spans. This requires Python, NumPy, librosa, PyTorch, a local checkout of
NVIDIA's BigVGAN repository with its dependencies installed, and a compatible checkpoint directory
containing `config.json` and `bigvgan_generator.pt`. No model weights or voice data are included.

```
python tools\bigvgan_say.py `
  --sam-say build\x64\sam_say.exe `
  --data data\voice `
  --bigvgan-repo C:\models\BigVGAN `
  --checkpoint C:\models\bigvgan-v2 `
  "Hello, my name is Microsoft Sam." out.wav
```

Use `--checkpoint` with either that model directory or its `bigvgan_generator.pt` file. The
`--device` option accepts `auto` (default), `cpu`, or `cuda`.

### Singing

`--sing` treats the text as a score, one word per line: the word with syllables split by `-`, then a
note and a length in beats per syllable. `- 2` is a rest, `tempo 120` sets the beats per minute.

```
tempo 100
twin-kle   C4 1  C4 1
twin-kle   G4 1  G4 1
lit-tle    A4 1  A4 1
star       G4 2
```

```
build\x64\sam_say.exe --data data\voice --sing @demo\twinkle.txt twinkle.wav
```

Consonants keep their natural length and the vowel stretches to fill the note. A trailing r, l, m,
n or ng shares the note with the vowel. Long vowels hold their steadiest frame instead of looping
the unit, so they don't stutter. Singing options:

```
--sing-vibrato 30 5.5   vibrato depth in cents and rate in Hz (fades in on long notes)
--transpose -12         shift every note (semitones), e.g. an octave down
--no-smooth             turn off frame interpolation (sounds choppier, more like stock Sam)
--smooth                turn frame interpolation on when speaking too
```

The library API is in `src/sam.h`. Call `sam_tts_new` with the three data files and a PCM callback,
then `sam_tts_speak`. At a lower level, `sam_synth_segment` renders single acoustic units, and
`sam_unit_decode` exposes the LPC frames of any of the 3,365 units.

## How the engine works (short version)

1. **Normalizer**: text is split at whitespace into tokens. Punctuation is peeled off the ends.
   Periods are resolved as sentence end, abbreviation (177-entry table), or initials. Each token is
   classified: word, number, ordinal, decimal, fraction, year, decade, percent, degrees, currency,
   clock time, duration, date (numeric or with month names), phone number, state + ZIP, range,
   hyphenated, or "spell it out". Anything unrecognized is spelled piece by piece. Sentences end at
   `. ! ?` or after 50 tokens.
2. **Lookup**: vendor lexicon, then morphology, then letter-to-sound, giving pronunciations and parts
   of speech.
3. **Tagger**: 63 Brill-style contextual rules pick the part of speech, and with it the pronunciation of
   homographs. Special-word rules handle *a*, *the*, *read*, *US*, *Dr.*/*St.*, units, and so on.
4. **Word prosody**: emphasis for `!`, phrase breaks, accents. Accent strength comes from the C
   runtime's `rand()` (seed 1). Numbers, times, phone numbers and currency get their own accent
   patterns and short pauses.
5. **Units**: decision trees in `Sam.spd` choose one of 3,365 diphone-like units per phone. Durations
   come from per-unit statistics with phrase-final lengthening. The F0 contour is built from
   top/mid/bottom lines plus accents and boundary tones.
6. **Synthesis**: a pitch-synchronous LPC vocoder. Vector-quantized LSF frames give the filters, and
   irfft pulses or noise give the excitation.

## Microsoft Anna as a Sam voice

`annavoice/` builds a Sam-format voice file out of **Microsoft Anna's** recordings, so this engine speaks
with her timbre: her 5.5 hours of recorded speech become a 1.7 MB vocoder voice, and everything here
applies to it — pitch, rate, singing, the SAPI 4 effects.

It needs Anna installed, the [Microsoft Anna port](https://github.com/KamiKitsune420/ms-ana-decomp) for her
corpus decoder, and your own `Sam.spd` for the phone set and trees. Her units are demisyllables, so the
builder forced-aligns her corpus to cut phones out, analyses them pitch-synchronously, trains fresh
codebooks on her data and writes the file. See `annavoice/README.md`.

**The voice file is not included and should not be redistributed** — it is Microsoft's recorded speech in
another format. The builder produces it locally from your own installation.

## Verification tools

- `harness/samtap.exe`: loads the real `spttseng.dll` and hot-patches internal functions. It speaks
  text through SAPI and logs internal records (units, pitch marks, filters, word records, tagger input
  and output, token nodes). Build it with `harness/build.bat` (32-bit MSVC).
- `tools/compare_corpus.py`: the 21 reference sentences in `ref/corpus`, C versus real WAVs (render the
  references yourself with `tools/render_ref.ps1`; they are not included).
- `tools/bulk_compare.py file.txt`: renders every line with both engines and compares the audio.
- `tools/norm_compare.py file.txt`: compares token classification, words, pronunciations and parts of
  speech line by line.
- Python models of each stage are in `tools/` (`samvoc.py`, `prosody_model.py`, `word_prosody.py`,
  `lxa.py`, `lts.py`, `morph.py`, ...).
- `tools/pe_read.py` reads data tables straight from the DLL. (The Ghidra output used during the work
  is not included, since it is Microsoft's code.)

## Fun facts

- **"soy" and the roflcopter.** The unit for the *oy* glide in "soy" is made entirely of unvoiced
  frames. When the engine stretches it, it repeats frames and plays every other copy backwards; the
  repeating noise pattern turns into a buzz at about 93 Hz, the famous helicopter sound. This port
  does it too (`--no-reverse` changes the texture).
- **Accents are random.** Word accent strength uses the C runtime's `rand()` seeded with 1, so Sam says
  the same text the same way every time, but changing an earlier sentence changes later accents.
- **Unpronounceable words kill the sentence.** In the real engine an unknown symbol makes SAPI reject
  the whole Speak call, and the last ~5000 samples of audio buffered so far are lost. The port
  reproduces this.

## License

MIT for the code in this repo (see `LICENSE`). Microsoft Sam's voice data and the original engine
are Microsoft's and are not included. The small tables in `src/sam_abbrev_tab.h`,
`src/sam_pos_rules.h` and `src/sam_words_tab.h` (abbreviations, tagger rules, special words) were
transcribed from the engine because the program needs them to behave the same way.
