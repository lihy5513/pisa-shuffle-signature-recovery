#include <stdint.h>
#include <stddef.h>
#include "randombytes.h"

/*
 * Experimental target RNG.
 *
 * This is deliberately separated from the deterministic Dilithium test key.
 * It supplies fresh per-call bytes for the NTT and x-domain permutations.
 * It is not intended as a production DRBG; replace it with the board's TRNG
 * before making implementation-security claims.
 */
static uint32_t rng_state = 0x6D2B79F5u;

static uint32_t xorshift32(void)
{
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x ? x : 0xA341316Cu;
    return rng_state;
}

int randombytes(uint8_t *output, size_t n)
{
    size_t i;
    for (i = 0; i < n; ++i) {
        if ((i & 3u) == 0u) {
            (void)xorshift32();
        }
        output[i] = (uint8_t)(rng_state >> ((i & 3u) * 8u));
    }
    return 0;
}

/* Host-only experiment helpers.  They are harmless in the firmware build and
 * allow the offline message selector to test a candidate without consuming
 * RNG state when that candidate is rejected. */
uint32_t cw_randombytes_get_state(void)
{
    return rng_state;
}

void cw_randombytes_set_state(uint32_t state)
{
    rng_state = state ? state : 0xA341316Cu;
}
