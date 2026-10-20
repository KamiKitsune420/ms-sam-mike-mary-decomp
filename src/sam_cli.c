/* sam: speak text with the Sam library (sam_tts.h) into a WAV file.
 *
 * usage: sam [options] "text"|@file.txt out.wav
 *   --data DIR     folder with <Voice>.spd, <Voice>.sdf, LTTS1033.LXA and r1033tts.LXA (or SAM_DATA)
 *   --voice NAME   Sam (default), Mike or Mary
 *   --rate N       -10..18   --volume N  0..100   --pitch N  -10..10
 *   --effect NAME  hall stadium space room whisper robosoft1..robosoft6 monotone
 *   --vocoder NAME sam (default) or bigvgan-inspired
 *   --xml          the text is SAPI XML        --events   print sentence/word/bookmark events
 */
#include "sam_tts.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    FILE *f;
    const char *text;
    unsigned samples;
    int events;
} sink;

static void put(FILE *f, unsigned v, int n)
{
    int i;
    for (i = 0; i < n; i++) fputc((int)((v >> (8 * i)) & 0xFF), f);
}

static void header(FILE *f, unsigned bytes, unsigned rate)
{
    fwrite("RIFF", 1, 4, f);
    put(f, 36 + bytes, 4);
    fwrite("WAVEfmt ", 1, 8, f);
    put(f, 16, 4);
    put(f, 1, 2);
    put(f, 1, 2);
    put(f, rate, 4);
    put(f, rate * 2, 4);
    put(f, 2, 2);
    put(f, 16, 2);
    fwrite("data", 1, 4, f);
    put(f, bytes, 4);
}

static int on_audio(const int16_t *pcm, size_t n, void *user)
{
    sink *s = user;
    size_t i;
    for (i = 0; i < n; i++) put(s->f, (unsigned)(uint16_t)pcm[i], 2);
    s->samples += (unsigned)n;
    return 0;
}

static void on_event(const sam_event *e, void *user)
{
    sink *s = user;
    static const char *const names[] = {"?", "sentence", "word", "bookmark", "end"};
    if (!s->events) return;
    printf("%8.3f s  %-8s", e->audio_pos / (double)SAM_TTS_SAMPLE_RATE, names[e->type > 0 && e->type <= 4 ? e->type : 0]);
    if (e->type == SAM_EV_BOOKMARK) printf(" \"%s\"", e->name ? e->name : "");
    else if (e->type != SAM_EV_END) printf(" \"%.*s\"", (int)e->text_len, s->text + e->text_pos);
    printf("\n");
}

static char *read_file(const char *fn)
{
    FILE *f = fopen(fn, "rb");
    char *b;
    long n;
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    b = malloc((size_t)n + 1);
    if (b) {
        n = (long)fread(b, 1, (size_t)n, f);
        b[n] = 0;
    }
    fclose(f);
    return b;
}

int main(int argc, char **argv)
{
    const char *dir = getenv("SAM_DATA") ? getenv("SAM_DATA") : ".";
    const char *voice = "Sam", *effect = NULL, *vocoder = NULL, *text = NULL, *out = NULL;
    int i, rate = 0, volume = 100, pitch = 0, flags = 0, rc;
    char err[256];
    sam_speech *t;
    sam_callbacks cb;
    sink s;
    memset(&s, 0, sizeof s);
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--data") && i + 1 < argc) dir = argv[++i];
        else if (!strcmp(argv[i], "--voice") && i + 1 < argc) voice = argv[++i];
        else if (!strcmp(argv[i], "--effect") && i + 1 < argc) effect = argv[++i];
        else if (!strcmp(argv[i], "--vocoder") && i + 1 < argc) vocoder = argv[++i];
        else if (!strcmp(argv[i], "--rate") && i + 1 < argc) rate = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--volume") && i + 1 < argc) volume = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pitch") && i + 1 < argc) pitch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--xml")) flags |= SAM_SPEAK_XML;
        else if (!strcmp(argv[i], "--events")) s.events = 1;
        else if (!text) text = argv[i];
        else if (!out) out = argv[i];
    }
    if (!text || !out) {
        fprintf(stderr, "usage: sam [--data DIR] [--voice NAME] [--rate N] [--volume N] [--pitch N]"
                        " [--effect NAME] [--vocoder sam|bigvgan-inspired] [--xml] [--events]"
                        " \"text\"|@file.txt out.wav\n");
        return 1;
    }
    if (text[0] == '@') {
        char *b = read_file(text + 1);
        if (!b) {
            fprintf(stderr, "cannot read %s\n", text + 1);
            return 1;
        }
        text = b;
        if ((unsigned char)b[0] == 0xef && (unsigned char)b[1] == 0xbb && (unsigned char)b[2] == 0xbf) text += 3;
    }
    t = sam_tts_open(dir, voice, err, sizeof err);
    if (!t) {
        fprintf(stderr, "sam: %s\n", err);
        return 1;
    }
    if (effect && sam_tts_set_effect(t, effect)) {
        fprintf(stderr, "sam: unknown effect %s (hall stadium space room whisper robosoft1..robosoft6 monotone)\n", effect);
        return 1;
    }
    if (vocoder && sam_tts_set_vocoder(t, vocoder)) {
        fprintf(stderr, "sam: unknown vocoder %s (sam or bigvgan-inspired)\n", vocoder);
        return 1;
    }
    sam_tts_set_rate(t, rate);
    sam_tts_set_volume(t, volume);
    sam_tts_set_pitch(t, pitch);
    s.f = fopen(out, "wb");
    if (!s.f) {
        fprintf(stderr, "cannot write %s\n", out);
        return 1;
    }
    s.text = text;
    header(s.f, 0, (unsigned)sam_tts_sample_rate(t));
    cb.audio = on_audio;
    cb.event = on_event;
    cb.user = &s;
    rc = sam_tts_speak(t, text, flags, &cb);
    fseek(s.f, 0, SEEK_SET);
    header(s.f, s.samples * 2, (unsigned)sam_tts_sample_rate(t));
    fclose(s.f);
    printf("%u samples (%.2f s)\n", s.samples, s.samples / (double)sam_tts_sample_rate(t));
    sam_tts_close(t);
    if (rc < 0) fprintf(stderr, "sam: synthesis error\n");
    return rc < 0;
}
