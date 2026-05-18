# Makefile for building a PL/M-80 source through the uplm80 toolchain.
#
# Contributed by Martin Homuth-Rosemann (@Ho-Ro), GitHub issue #5.
# https://github.com/avwohl/uplm80/issues/5
#
# Usage:
#   make TARGET=hellocpm          # build hellocpm.com
#   make TARGET=hellocpm disasm   # also disassemble to 8080 and Z80 mnemonics
#   make TARGET=hellocpm clean    # remove intermediates
#   make TARGET=hellocpm distclean  # remove .com as well
#
# Drop this Makefile into a directory next to a single <TARGET>.plm source
# and the .com file will be produced via uplm80 -> um80 -> ul80.

TARGET := hellocpm

PLM := $(TARGET).plm
MAC := $(TARGET).mac
REL := $(TARGET).rel
COM := $(TARGET).com

DIS8080 := $(TARGET)_8080.mac
DISZ80  := $(TARGET)_z80.mac


all: $(COM)


$(MAC): $(PLM)
	uplm80 -o $@ $<

$(REL): $(MAC)
	um80 -o $@ $<

$(COM): $(REL)
	ul80 -o $@ $<
	ls -l $@


.PHONY: disasm
disasm: $(COM)
	ud80 -o $(DIS8080) $<
	ux80 -o $(DISZ80) $(DIS8080)


.PHONY: clean
clean:
	rm -f $(MAC) $(REL) $(DIS8080) $(DISZ80) *~

.PHONY: distclean
distclean: clean
	rm -f $(COM)
