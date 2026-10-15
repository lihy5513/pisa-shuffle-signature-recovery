#ifndef CONFIG_H
#define CONFIG_H

#define DILITHIUM_MODE 2

/* Use the coarse NTT/INTT schedule from the public M4 implementation.
 * The x-domain schedule is added separately in xshuffle.c. */
#define NTT_PROTECTION_MODE COARSE_SHUFFLING

#endif
