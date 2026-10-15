TARGET = simpleserial-kyber512-ushuffle-scalar
PLATFORM ?= CW308_STM32F3

SRC += cbd.c fips202.c indcpa.c kem.c matacc.c ntt.c poly.c polyvec.c \
       randombytes.c red.c simpleserial-kyber.c symmetric-fips202.c \
       verify.c keccakf1600.c ushuffle.c
ASRC += fastaddsub.S fastbasemul.S fastinvntt.S fastntt.S matacc_asm.S \
        poly_asm.S reduce.S

# The x-domain subtraction experiment uses the ARMv7E-M USUB16 primitive.
CFLAGS += -DKYBER_CAPTURE_USHUF
CFLAGS += -fstack-usage
ASFLAGS += -mfpu=fpv4-sp-d16

# Optional post-trigger export used only to validate offline labels.  The
# formal attack firmware is built without this instrumentation.
ifneq ($(KYBER_DEBUG_X),)
  CFLAGS += -DKYBER_DEBUG_X
endif
CRYPTO_TARGET ?= NONE
ifndef FIRMWAREPATH
$(error Set FIRMWAREPATH to the ChipWhisperer firmware/mcu directory)
endif

include $(FIRMWAREPATH)/simpleserial/Makefile.simpleserial
include $(FIRMWAREPATH)/Makefile.inc

# Makefile.simpleserial appends assembler listing flags that are not accepted
# by the assembler used by this standalone copy.
CFLAGS := $(filter-out -Wa% -adhlns=%,$(CFLAGS))
