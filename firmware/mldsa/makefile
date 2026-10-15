TARGET = simpleserial-dilithium2-xshuffle

# The connected target is the CW308T STM32F3/STM32F303 board.  A command-line
# PLATFORM value may still override this default for reproducibility checks.
PLATFORM ?= CW308_STM32F3

SRC += simpleserial-dilithium2-xshuffle.c \
       fips202.c ntt.c packing.c poly.c polyvec.c reduce.c \
       rounding.c sign.c symmetric-shake.c xshuffle.c cw_randombytes.c \
       key_storage.c

# The C implementation in ntt.c is complete for this target.  Do not add the
# lowercase .s file here: the local ChipWhisperer make rules pass it through
# as a raw linker input instead of assembling it.

CFLAGS += -DDILITHIUM_XSHUFFLE
CFLAGS += -fstack-usage

# Optional high-resolution acquisition mode.  When set to 0..L-1, the
# trigger surrounds only that polynomial's 256-coefficient traversal, which
# allows clkgen_x4 sampling within the CW Lite trace-memory limit.
ifneq ($(CAPTURE_POLY_ONLY),)
  CFLAGS += -DDILITHIUM_CAPTURE_POLY_ONLY=$(CAPTURE_POLY_ONLY)
endif

# Accepted-after replay mode: suppress triggers around ordinary candidate
# traversals and trigger only around the repeated c*s1 traversal after all
# rejection checks have passed.
ifneq ($(CAPTURE_ACCEPTED_CS1),)
  CFLAGS += -DDILITHIUM_CAPTURE_ACCEPTED_CS1
endif

ifeq ($(CRYPTO_TARGET),)
  CRYPTO_TARGET = NONE
endif

# The target is kept in the research workspace, while the ChipWhisperer
# firmware support is taken from the locally installed SDK.  Override
# FIRMWAREPATH on the make command line when using another SDK checkout.
ifndef FIRMWAREPATH
$(error Set FIRMWAREPATH to the ChipWhisperer firmware/mcu directory)
endif
include $(FIRMWAREPATH)/simpleserial/Makefile.simpleserial
include $(FIRMWAREPATH)/Makefile.inc

# The checked-out framework contains an assembler-listing flag whose automatic
# filename expansion is incompatible with this Windows GNU Make invocation.
# It can overwrite a C object with the listing text, which then breaks the
# linker.  Listings are not required for the firmware, so remove that flag.
CFLAGS := $(filter-out -Wa% -adhlns=%,$(CFLAGS))
