# The ISIS-II emulator scripts/intel_oracle.py runs Intel's PL/M-80 on.
#
#   make -C tools/isis            # builds tools/isis/isis
#   make -C tools/isis qkz80-source
#
# It needs the qkz80 Z80/8080 core from cpmemu, found the way romwbw_emu's
# src/Makefile finds it: pkg-config, else a cpmemu checkout next to this
# repository (../cpmemu/src, with libqkz80.a built there by `make -C
# ../cpmemu/src`), else /usr/local.  QKZ80_CFLAGS and QKZ80_LIBS on the
# command line choose another.

THIS_DIR := $(dir $(lastword $(MAKEFILE_LIST)))

all: $(THIS_DIR)isis

SISTER_QKZ80_PATH = $(THIS_DIR)../../../cpmemu/src
SISTER_QKZ80 = $(realpath $(SISTER_QKZ80_PATH))

ifeq ($(origin QKZ80_CFLAGS),undefined)
  ifneq ($(shell pkg-config --exists qkz80 2>/dev/null && echo yes),)
    QKZ80_SOURCE = pkg-config ($(shell pkg-config --variable=prefix qkz80 2>/dev/null))
  else ifneq ($(wildcard $(SISTER_QKZ80_PATH)/qkz80.h),)
    QKZ80_SOURCE = the sister checkout, $(SISTER_QKZ80)
  else
    QKZ80_SOURCE = /usr/local - nothing else was found
    $(warning qkz80: no pkg-config entry and no $(SISTER_QKZ80_PATH)/qkz80.h;)
    $(warning qkz80: trying -I/usr/local/include/qkz80 -L/usr/local/lib.)
  endif
else
  QKZ80_SOURCE = QKZ80_CFLAGS, set by the caller ($(origin QKZ80_CFLAGS))
endif

QKZ80_CFLAGS ?= $(shell pkg-config --cflags qkz80 2>/dev/null || \
    ([ -f "$(SISTER_QKZ80_PATH)/qkz80.h" ] && echo "-I$(SISTER_QKZ80)") || \
    echo "-I/usr/local/include/qkz80")
# The archive by name: -lqkz80 would pick an uninstalled libqkz80.dylib in the
# same directory, which then fails to load (see romwbw_emu's src/Makefile).
QKZ80_LIBS ?= $(shell pkg-config --libs qkz80 2>/dev/null || \
    ([ -f "$(SISTER_QKZ80_PATH)/libqkz80.a" ] && echo "$(SISTER_QKZ80)/libqkz80.a") || \
    echo "-L/usr/local/lib -lqkz80")

CXX ?= c++
CXXFLAGS ?= -std=c++11 -O2 -Wall

$(THIS_DIR)isis: $(THIS_DIR)isis.cc
	$(CXX) $(CXXFLAGS) $(QKZ80_CFLAGS) $< $(QKZ80_LIBS) -o $(THIS_DIR)isis

qkz80-source:
	@echo "qkz80: $(QKZ80_SOURCE)"
	@echo "  QKZ80_CFLAGS = $(QKZ80_CFLAGS)"
	@echo "  QKZ80_LIBS   = $(QKZ80_LIBS)"

clean:
	rm -f $(THIS_DIR)isis

.PHONY: all clean qkz80-source
