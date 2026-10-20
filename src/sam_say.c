/* sam_say: Microsoft Sam text to speech from the original voice data files.
 *
 * usage: sam_say [options] "text" out.wav      ("@file.txt" speaks a UTF-8 text file)
 *   --data DIR      folder containing Sam.spd, LTTS1033.LXA and r1033tts.LXA
 *   --pitch x --speed x --gain x --vibrato depth rate --no-reverse   (see sam.h)
 *   --vocoder residual|pulse|bigvgan|spectral|bigvgan-concat   select the excitation engine
 *   --reverse-units   play every sound unit backwards      --rate HZ   run the vocoder at another rate
 *   --sing            the text is a score: "twin-kle C4 1 C4 1" per line, "- 1" rest, "tempo 100"
 *     --no-smooth  --sing-vibrato CENTS HZ  --transpose SEMITONES
 */
#include "sam.h"
#include "sam4fx.h"
#ifdef SAM_EMBEDDED
#include "sam_data.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    FILE *f;
    unsigned samples;
    sam4fx *fx; /* SAPI 4 voice effect, or NULL */
} sink;

static void put(FILE *f, unsigned v, int n)
{
    int i;
    for (i = 0; i < n; i++) fputc((int)((v >> (8 * i)) & 0xFF), f);
}

static unsigned g_rate = 22050;

static void header(FILE *f, unsigned bytes)
{
    fwrite("RIFF", 1, 4, f);
    put(f, 36 + bytes, 4);
    fwrite("WAVEfmt ", 1, 8, f);
    put(f, 16, 4);
    put(f, 1, 2);
    put(f, 1, 2);
    put(f, g_rate, 4);
    put(f, g_rate * 2, 4);
    put(f, 2, 2);
    put(f, 16, 2);
    fwrite("data", 1, 4, f);
    put(f, bytes, 4);
}

static void on_pcm(const int16_t *pcm, size_t n, void *user)
{
    sink *s = user;
    size_t i;
    if (s->fx) { /* the effect works in place on a copy of the chunk */
        int16_t *b = malloc(n * sizeof *b);
        if (!b) return;
        memcpy(b, pcm, n * sizeof *b);
        sam4fx_process_s16(s->fx, b, (int)n);
        for (i = 0; i < n; i++) put(s->f, (uint16_t)b[i], 2);
        free(b);
        s->samples += (unsigned)n;
        return;
    }
    for (i = 0; i < n; i++) put(s->f, (uint16_t)pcm[i], 2);
    s->samples += (unsigned)n;
}

int main(int argc, char **argv)
{
    const char *dir = NULL, *text = NULL, *out = NULL, *voice = "Sam", *effect = NULL;
    char spd[1024], lex[1024], lts[1024], err[256];
    sam_params p;
    sam_tts *t;
    sink s;
    int i, sing = 0, no_smooth = 0, fx_preset = 0, fx_on = 0, no_trim = 0;
    sam_params_default(&p);
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--data") && i + 1 < argc) dir = argv[++i];
        else if (!strcmp(argv[i], "--voice") && i + 1 < argc) voice = argv[++i];
        else if (!strcmp(argv[i], "--effect") && i + 1 < argc) effect = argv[++i];
        else if (!strcmp(argv[i], "--vocoder") && i + 1 < argc) {
            const char *name = argv[++i];
            if (!strcmp(name, "residual")) p.vocoder = SAM_VOCODER_RESIDUAL;
            else if (!strcmp(name, "pulse")) p.vocoder = SAM_VOCODER_PULSE;
            else if (!strcmp(name, "bigvgan")) p.vocoder = SAM_VOCODER_BIGVGAN;
            else if (!strcmp(name, "spectral")) p.vocoder = SAM_VOCODER_SPECTRAL;
            else if (!strcmp(name, "bigvgan-concat")) p.vocoder = SAM_VOCODER_BIGVGAN_CONCAT;
            else {
                fprintf(stderr, "sam_say: unknown vocoder %s (residual, pulse, bigvgan, spectral or bigvgan-concat)\n", name);
                return 1;
            }
        }
        else if (!strcmp(argv[i], "--no-trim")) no_trim = 1; /* original RoboSoft loudness (clips) */
        else if (!strcmp(argv[i], "--pitch") && i + 1 < argc) p.pitch_scale = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--speed") && i + 1 < argc) p.speed = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--gain") && i + 1 < argc) p.gain = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--vibrato") && i + 2 < argc) {
            p.vibrato = (float)atof(argv[++i]);
            p.vibrato_rate = (float)atof(argv[++i]);
        } else if (!strcmp(argv[i], "--no-reverse")) p.no_reverse = 1;
        else if (!strcmp(argv[i], "--reverse-units")) p.reverse_units = 1;
        else if (!strcmp(argv[i], "--sing")) sing = 1;
        else if (!strcmp(argv[i], "--smooth")) p.smooth = 1;
        else if (!strcmp(argv[i], "--no-smooth")) no_smooth = 1;
        else if (!strcmp(argv[i], "--sing-vibrato") && i + 2 < argc) {
            p.sing_vibrato = (float)atof(argv[++i]);
            p.sing_vibrato_rate = (float)atof(argv[++i]);
        } else if (!strcmp(argv[i], "--transpose") && i + 1 < argc) p.transpose = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--rate") && i + 1 < argc) g_rate = (unsigned)(p.out_rate = atoi(argv[++i]));
        else if (!text) text = argv[i];
        else if (!out) out = argv[i];
    }
    if (sing && !no_smooth) p.smooth = 1; /* singing blends frames unless told not to */
    if (effect) { /* SAPI 4 voice modes: hall stadium space room whisper robosoft1..6 robot monotone */
        if (sam4fx_lookup(effect, &fx_preset, &p.monotone, &p.whisper)) {
            fprintf(stderr, "sam_say: unknown effect %s (hall stadium space room whisper robosoft1..robosoft6 monotone)\n", effect);
            return 1;
        }
        fx_on = fx_preset != 0 || p.whisper;
    }
    if (text && text[0] == '@') { /* read the text from a file */
        FILE *f = fopen(text + 1, "rb");
        char *buf;
        long n;
        if (!f) {
            fprintf(stderr, "cannot read %s\n", text + 1);
            return 1;
        }
        fseek(f, 0, SEEK_END);
        n = ftell(f);
        fseek(f, 0, SEEK_SET);
        buf = malloc((size_t)n + 1);
        if (!buf) return 1;
        n = (long)fread(buf, 1, (size_t)n, f);
        buf[n] = 0;
        fclose(f);
        text = buf;
        if ((unsigned char)text[0] == 0xef && (unsigned char)text[1] == 0xbb && (unsigned char)text[2] == 0xbf) text += 3;
    }
    if (!text || !out) {
        fprintf(stderr, "usage: sam_say [--data DIR] [--vocoder residual|pulse|bigvgan|spectral|bigvgan-concat] [--pitch x] [--speed x] [--gain x] [--vibrato d r] \"text\" out.wav\n");
        return 1;
    }
#ifdef SAM_EMBEDDED
    if (!dir) /* voice data compiled into the program */
        t = sam_tts_new_mem(sam_data_spd, sam_data_spd_size, sam_data_lex, sam_data_lex_size, sam_data_lts,
                            sam_data_lts_size, &p, err, sizeof err);
    else
#endif
    {
        if (!dir) dir = ".";
        snprintf(spd, sizeof spd, "%s/%s.spd", dir, voice);
        { /* the voice's base pitch lives in its .sdf ("Vois" header, int at 0x318) */
            char sdf[1024];
            unsigned char b[0x31c];
            FILE *f;
            snprintf(sdf, sizeof sdf, "%s/%s.sdf", dir, voice);
            f = fopen(sdf, "rb");
            if (f) {
                if (fread(b, 1, sizeof b, f) == sizeof b && !memcmp(b, "Vois", 4))
                    p.base_pitch = (float)(b[0x318] | b[0x319] << 8 | b[0x31a] << 16 | (unsigned)b[0x31b] << 24);
                fclose(f);
            }
        }
        snprintf(lex, sizeof lex, "%s/LTTS1033.LXA", dir);
        snprintf(lts, sizeof lts, "%s/r1033tts.LXA", dir);
        t = sam_tts_new(spd, lex, lts, &p, err, sizeof err);
    }
    if (!t) {
        fprintf(stderr, "sam_say: %s\n", err);
        return 1;
    }
    s.fx = NULL;
    if (fx_on) {
        s.fx = sam4fx_new(fx_preset, (int)g_rate);
        if (s.fx && !no_trim) sam4fx_set_input_gain(s.fx, (float)pow(10.0, sam4fx_default_trim_db(fx_preset) / 20.0));
        if (s.fx && p.whisper) {
            static const float taps[3] = {0.25f, -0.5f, 0.25f};
            sam4fx_set_fir(s.fx, taps, 3);
        }
    }
    s.f = fopen(out, "wb");
    s.samples = 0;
    if (!s.f) {
        fprintf(stderr, "cannot write %s\n", out);
        return 1;
    }
    header(s.f, 0);
    if ((sing ? sam_tts_sing(t, text, on_pcm, &s) : sam_tts_speak_pcm(t, text, on_pcm, &s)) != 0) fprintf(stderr, "sam_say: synthesis error\n");
    if (s.fx) { /* let the echoes die away: run the effect over silence, like the engine does between sentences */
        static int16_t z[22050];
        int k;
        for (k = 0; k < 2; k++) {
            memset(z, 0, sizeof z);
            on_pcm(z, (size_t)(g_rate / 2 < 22050 ? g_rate / 2 : 22050), &s);
        }
        sam4fx_free(s.fx);
    }
    fseek(s.f, 0, SEEK_SET);
    header(s.f, s.samples * 2);
    fclose(s.f);
    sam_tts_free(t);
    printf("%u samples (%.2f s)\n", s.samples, s.samples / (double)g_rate);
    return 0;
}
