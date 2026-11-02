# Linux / macOS / MinGW build.   make            -> build/sam_say, build/sam (library CLI), libsam
#                                 make standalone -> build/sam_standalone (needs src/sam_data.c from tools/embed_data.py)
#                                 make test       -> build/lib_test, the library's own tests
CC ?= cc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -Wno-unused-parameter
LIB = src/sam.c src/sam_lex.c src/sam_morph.c src/sam_pos.c src/sam_norm.c src/sam_front.c
TTSLIB = $(LIB) src/sam4fx.c src/sam_tts.c
SOEXT ?= so

all: build/sam_say build/sam build/libsam.$(SOEXT) build/samsynth build/lextest

build:
	mkdir -p build

build/sam_say: $(LIB) src/sam4fx.c src/sam_say.c | build
	$(CC) $(CFLAGS) $^ -lm -o $@

# the screen-reader library (src/sam_tts.h) and its command line front end
build/sam: $(TTSLIB) src/sam_cli.c | build
	$(CC) $(CFLAGS) $^ -lm -o $@

build/libsam.$(SOEXT): $(TTSLIB) | build
	$(CC) $(CFLAGS) -shared -fPIC -fvisibility=hidden -DSAM_BUILD_DLL $^ -lm -o $@

build/samsynth: src/sam.c src/samsynth.c | build
	$(CC) $(CFLAGS) $^ -lm -o $@

build/lextest: src/sam_lex.c src/sam_morph.c src/lextest.c | build
	$(CC) $(CFLAGS) $^ -lm -o $@

standalone: build/sam_standalone

build/sam_standalone: $(LIB) src/sam4fx.c src/sam_say.c src/sam_data.c | build
	$(CC) $(CFLAGS) -DSAM_EMBEDDED $^ -lm -o $@

test: build/lib_test

build/lib_test: $(TTSLIB) tests/lib/lib_test.c | build
	$(CC) $(CFLAGS) -D_DEFAULT_SOURCE $^ -lm -lpthread -o $@

# the library drop for other programs: the shared library, its headers and the CLI
dist: build/libsam.$(SOEXT) build/sam
	mkdir -p build/dist
	cp build/libsam.$(SOEXT) build/sam build/dist/
	cp src/sam_tts.h src/sam.h src/sam4fx.h build/dist/
	printf '%s\n' \
	  'Microsoft Sam, Mike and Mary - portable C port' '' \
	  '  libsam.$(SOEXT)   the library: include sam_tts.h and link against it' \
	  '  sam               the command line front end' '' \
	  'No voice data is included. Point --data at a folder holding Sam.spd,' \
	  'LTTS1033.LXA and r1033tts.LXA from your own installation.' > build/dist/README.txt
	@echo "dist: build/dist"

clean:
	rm -rf build

.PHONY: all standalone test dist clean
