/* tests/lib: behaviour of the sam_tts library that the CLI cannot show.
 *
 *   lib_test [data_dir]        (default: ..\..\data\voices relative to the repo root, or SAM_DATA)
 *
 * 1. cancel from another thread stops within one audio chunk and sam_tts_speak returns 1
 * 2. returning nonzero from the audio callback stops it too, and also returns 1
 * 3. word / sentence / bookmark events are in order, inside the text, and their audio positions
 *    never run past the audio that has been delivered
 * 4. two handles opened the same way render the same text identically
 *
 * Note: one handle does not repeat itself exactly. The engine's accent prominences come from the CRT
 * rand(), whose state runs on from one Speak() to the next, as it does in the original spttseng.dll;
 * so does the vocoder's position in the voice's noise buffer.
 */
#include "../../src/sam_tts.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
typedef HANDLE thread_t;
typedef DWORD(WINAPI *thread_fn)(void *);
#define THREAD_RET DWORD WINAPI
static thread_t thread_start(thread_fn fn, void *arg)
{
    return CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)fn, arg, 0, NULL);
}
static void thread_join(thread_t t) { WaitForSingleObject(t, INFINITE); CloseHandle(t); }
static void ms_sleep(int ms) { Sleep((DWORD)ms); }
#else
#include <pthread.h>
#include <unistd.h>
typedef pthread_t thread_t;
typedef void *(*thread_fn)(void *);
#define THREAD_RET void *
static thread_t thread_start(thread_fn fn, void *arg)
{
    pthread_t t;
    pthread_create(&t, NULL, fn, arg);
    return t;
}
static void thread_join(thread_t t) { pthread_join(t, NULL); }
static void ms_sleep(int ms) { usleep(ms * 1000); }
#endif

static int failures = 0;

static void check(int ok, const char *what)
{
    printf("%-6s %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) failures++;
}

/* ---- the text used everywhere below ---- */
static const char *LONG_TEXT =
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. The five boxing wizards jump quickly at dawn.";

typedef struct {
    sam_speech *t;
    volatile long long samples;
    long long stop_after; /* 0 = never */
    int stopped;
    /* event checking */
    const char *text;
    int nev, order_ok, span_ok, pos_ok, nword, nsent, nbm;
    long long last_ev;
    volatile long long at_cancel;
    int16_t *pcm;
    long long ncap;
    int max_sample;
} ctx;

static int audio(const int16_t *pcm, size_t n, void *user)
{
    ctx *c = user;
    size_t i;
    for (i = 0; i < n; i++) {
        int magnitude = pcm[i] < 0 ? -(int)pcm[i] : pcm[i];
        if (magnitude > c->max_sample) c->max_sample = magnitude;
    }
    if (c->pcm && c->samples + (long long)n <= c->ncap) memcpy(c->pcm + c->samples, pcm, n * sizeof *pcm);
    c->samples += (long long)n;
    if (c->stop_after && c->samples >= c->stop_after) {
        c->stopped = 1;
        return 1;
    }
    return 0;
}

static void event(const sam_event *e, void *user)
{
    ctx *c = user;
    long len = (long)strlen(c->text);
    c->nev++;
    if ((long long)e->audio_pos < c->last_ev) c->order_ok = 0;
    c->last_ev = (long long)e->audio_pos;
    if ((long long)e->audio_pos > c->samples + 2 * SAM_TTS_SAMPLE_RATE) c->pos_ok = 0; /* never far ahead of the audio */
    if (e->type == SAM_EV_WORD || e->type == SAM_EV_SENTENCE) {
        if (e->text_pos < 0 || e->text_len < 0 || e->text_pos + e->text_len > len) c->span_ok = 0;
    }
    if (e->type == SAM_EV_WORD) c->nword++;
    if (e->type == SAM_EV_SENTENCE) c->nsent++;
    if (e->type == SAM_EV_BOOKMARK) c->nbm++;
}

static THREAD_RET canceller(void *arg)
{
    ctx *c = (ctx *)arg;
    while (c->samples < SAM_TTS_SAMPLE_RATE / 4) ms_sleep(1); /* let a quarter of a second out first */
    c->at_cancel = c->samples;
    sam_tts_cancel(c->t);
    return 0;
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : getenv("SAM_DATA") ? getenv("SAM_DATA") : "../data/voices";
    char err[256];
    sam_speech *t = sam_tts_open(dir, "Sam", err, sizeof err);
    sam_callbacks cb;
    ctx c;
    int rc;
    long long full;
    if (!t) {
        fprintf(stderr, "lib_test: %s (pass the data folder as argv[1])\n", err);
        return 2;
    }
    cb.audio = audio;
    cb.event = event;

    /* 1. full render, and event sanity */
    memset(&c, 0, sizeof c);
    c.t = t;
    c.text = LONG_TEXT;
    c.order_ok = c.span_ok = c.pos_ok = 1;
    c.ncap = 60LL * SAM_TTS_SAMPLE_RATE;
    c.pcm = malloc((size_t)c.ncap * sizeof *c.pcm);
    cb.user = &c;
    rc = sam_tts_speak(t, LONG_TEXT, 0, &cb);
    full = c.samples;
    check(rc == 0, "speak returns 0 when it finishes");
    check(full > SAM_TTS_SAMPLE_RATE, "it produced more than a second of audio");
    check(c.nsent == 4, "one sentence event per sentence");
    check(c.nword == 31, "one word event per word (31)");
    check(c.order_ok, "event audio positions are monotonic");
    check(c.span_ok, "word / sentence spans lie inside the text");
    check(c.pos_ok, "event audio positions do not run ahead of the audio");

    /* 2. a second handle renders the same text identically */
    {
        sam_speech *t2 = sam_tts_open(dir, "Sam", err, sizeof err);
        ctx d;
        int vocoder_rc;
        memset(&d, 0, sizeof d);
        d.t = t2;
        d.text = LONG_TEXT;
        d.order_ok = d.span_ok = d.pos_ok = 1;
        d.ncap = c.ncap;
        d.pcm = malloc((size_t)d.ncap * sizeof *d.pcm);
        cb.user = &d;
        rc = t2 ? sam_tts_speak(t2, LONG_TEXT, 0, &cb) : -1;
        check(rc == 0 && d.samples == full && !memcmp(c.pcm, d.pcm, (size_t)full * sizeof *d.pcm),
              "a freshly opened handle renders the same text identically");
        check(d.nword == c.nword && d.nsent == c.nsent, "and reports the same events");
        vocoder_rc = t2 ? sam_tts_set_vocoder(t2, "bigvgan-inspired") : -1;
        check(vocoder_rc == 0, "the BigVGAN-inspired vocoder can be selected");
        if (vocoder_rc == 0) {
            size_t i;
            int different = 0;
            memset(d.pcm, 0, (size_t)d.ncap * sizeof *d.pcm);
            d.samples = 0;
            d.nword = d.nsent = d.nbm = d.nev = 0;
            d.last_ev = 0;
            d.max_sample = 0;
            d.order_ok = d.span_ok = d.pos_ok = 1;
            rc = sam_tts_speak(t2, LONG_TEXT, 0, &cb);
            for (i = 0; i < (size_t)(d.samples < full ? d.samples : full); i++)
                if (d.pcm[i] != c.pcm[i]) { different = 1; break; }
            check(rc == 0 && d.samples == full && different,
                  "the alternative vocoder changes voiced audio without changing duration");
            check(d.max_sample <= 30000, "the alternative vocoder keeps peaks below the clipping ceiling");
        }
        free(d.pcm);
        sam_tts_close(t2);
    }

    /* 3. cancel from another thread */
    {
        thread_t th;
        memset(&c, 0, sizeof c);
        c.t = t;
        c.text = LONG_TEXT;
        c.order_ok = c.span_ok = c.pos_ok = 1;
        cb.user = &c;
        th = thread_start(canceller, &c);
        rc = sam_tts_speak(t, LONG_TEXT, 0, &cb);
        thread_join(th);
        check(rc == 1, "cancel from another thread returns 1");
        check(c.samples < full, "it stopped before the end");
        /* the engine hands audio over in chunks of about 5000 samples; at most one more may come out
           after the cancel, plus the item being rendered when it arrived */
        check(c.samples - c.at_cancel <= 2 * 5000 + SAM_TTS_SAMPLE_RATE / 4,
              "it stopped within about one chunk of the cancel");
        printf("       (%lld samples came out after the cancel; %lld of %lld in total)\n",
               c.samples - c.at_cancel, c.samples, full);
    }

    /* 4. stopping from the audio callback */
    {
        memset(&c, 0, sizeof c);
        c.t = t;
        c.text = LONG_TEXT;
        c.order_ok = c.span_ok = c.pos_ok = 1;
        c.stop_after = SAM_TTS_SAMPLE_RATE / 2;
        cb.user = &c;
        rc = sam_tts_speak(t, LONG_TEXT, 0, &cb);
        check(rc == 1, "a nonzero audio callback return stops the speak and returns 1");
        check(c.samples < full, "it stopped before the end");
    }

    /* 5. it still works after being cancelled */
    {
        memset(&c, 0, sizeof c);
        c.t = t;
        c.text = LONG_TEXT;
        c.order_ok = c.span_ok = c.pos_ok = 1;
        c.ncap = 60LL * SAM_TTS_SAMPLE_RATE;
        c.pcm = malloc((size_t)c.ncap * sizeof *c.pcm);
        cb.user = &c;
        rc = sam_tts_speak(t, LONG_TEXT, 0, &cb);
        check(rc == 0 && c.samples > full / 2 && c.nword == 31,
              "a new speak after a cancel renders the whole text again");
        free(c.pcm);
    }

    /* 6. bookmarks and XML */
    {
        static const char *xml = "<bookmark mark=\"a\"/>Hello there. How <bookmark mark=\"b\"/>are you?";
        memset(&c, 0, sizeof c);
        c.t = t;
        c.text = xml;
        c.order_ok = c.span_ok = c.pos_ok = 1;
        cb.user = &c;
        rc = sam_tts_speak(t, xml, SAM_SPEAK_XML, &cb);
        check(rc == 0 && c.nbm == 2, "both bookmarks are reported");
        check(c.nsent == 2 && c.nword == 5, "the tags do not become words");
        check(c.order_ok && c.span_ok, "XML event positions are sane");
    }

    sam_tts_close(t);
    printf(failures ? "%d FAILURES\n" : "all tests passed (%d failures)\n", failures);
    return failures != 0;
}
