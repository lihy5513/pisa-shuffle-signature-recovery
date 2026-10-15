#include "fips202.h"
#include "packing.h"
#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include "randombytes.h"
#include "sign.h"
#include "symmetric.h"
#include "xshuffle.h"
#ifdef DILITHIUM_XSHUFFLE
#include "hal.h"
#endif
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int PQCLEAN_DILITHIUM2_CLEAN_crypto_sign_keypair_from_seed(uint8_t *pk,
        uint8_t *sk, const uint8_t seed[SEEDBYTES]);
void trigger_high(void);
void trigger_low(void);

/*
 * The CW308T-STM32F3 target has 40 KB of SRAM.  The reference clean signer
 * places its complete matrix and all intermediate vectors on the call stack,
 * which requires about 51 KB for crypto_sign_signature().  The following
 * single-call workspace keeps the long-lived vectors in .bss and streams one
 * matrix row at a time.  The union members are deliberately reused only after
 * the preceding value has reached its last use.
 */
typedef union {
    polyvecl l;
    polyveck k;
} dilithium_polyvec_workspace;

static polyvecl ws_s1;
static polyveck ws_s2;
static dilithium_polyvec_workspace ws_y_h;
static dilithium_polyvec_workspace ws_z_w0;
static polyveck ws_w1;
static dilithium_polyvec_workspace ws_mat_t0;
static polyvecl ws_z;
static poly ws_cp;
static uint16_t ws_x_order[L][N];

/* Host-side validation metadata; it does not affect the signature. */
unsigned int xshuffle_attempts;

static void unpack_t0_only(polyveck *t0,
                           const uint8_t sk[PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_SECRETKEYBYTES]) {
    unsigned int i;
    const uint8_t *packed = sk + 2 * SEEDBYTES + TRBYTES
        + (L + K) * POLYETA_PACKEDBYTES;

    for (i = 0; i < K; ++i) {
        PQCLEAN_DILITHIUM2_CLEAN_polyt0_unpack(
            &t0->vec[i], packed + i * POLYT0_PACKEDBYTES);
    }
}

static void matrix_pointwise_stream(polyveck *t,
                                    polyvecl *mat_row,
                                    const uint8_t rho[SEEDBYTES],
                                    const polyvecl *v) {
    unsigned int i, j;

    for (i = 0; i < K; ++i) {
        for (j = 0; j < L; ++j) {
            PQCLEAN_DILITHIUM2_CLEAN_poly_uniform(
                &mat_row->vec[j], rho, (uint16_t)((i << 8) + j));
        }
        PQCLEAN_DILITHIUM2_CLEAN_polyvecl_pointwise_acc_montgomery(
            &t->vec[i], mat_row, v);
    }
}
/*************************************************
* Name:        PQCLEAN_DILITHIUM2_CLEAN_crypto_sign_keypair
*
* Description: Generates public and private key.
*
* Arguments:   - uint8_t *pk: pointer to output public key (allocated
*                             array of PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_PUBLICKEYBYTES bytes)
*              - uint8_t *sk: pointer to output private key (allocated
*                             array of PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_SECRETKEYBYTES bytes)
*
* Returns 0 (success)
**************************************************/
int PQCLEAN_DILITHIUM2_CLEAN_crypto_sign_keypair(uint8_t *pk, uint8_t *sk) {
    uint8_t seed[SEEDBYTES] = {0};
    return PQCLEAN_DILITHIUM2_CLEAN_crypto_sign_keypair_from_seed(pk, sk, seed);
}

int PQCLEAN_DILITHIUM2_CLEAN_crypto_sign_keypair_from_seed(uint8_t *pk,
        uint8_t *sk, const uint8_t seed[SEEDBYTES]) {
    uint8_t seedbuf[2 * SEEDBYTES + CRHBYTES];
    uint8_t tr[TRBYTES];
    const uint8_t *rho, *rhoprime, *key;
    polyvecl mat[K];
    polyvecl s1, s1hat;
    polyveck s2, t1, t0;

    /* Get randomness for rho, rhoprime and key */
    memcpy(seedbuf, seed, SEEDBYTES);
    shake256(seedbuf, 2 * SEEDBYTES + CRHBYTES, seedbuf, SEEDBYTES);
    rho = seedbuf;
    rhoprime = rho + SEEDBYTES;
    key = rhoprime + CRHBYTES;

    /* Expand matrix */
    PQCLEAN_DILITHIUM2_CLEAN_polyvec_matrix_expand(mat, rho);

    /* Sample short vectors s1 and s2 */
    PQCLEAN_DILITHIUM2_CLEAN_polyvecl_uniform_eta(&s1, rhoprime, 0);
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_uniform_eta(&s2, rhoprime, L);

    /* Matrix-vector multiplication */
    s1hat = s1;
    PQCLEAN_DILITHIUM2_CLEAN_polyvecl_ntt(&s1hat);
    // for ( int i = 0 ;i < L; i++){
    //     for ( int j = 0; j < 256; j++)
    //     {
    //         printf("0x%x, ", s1hat.vec[i].coeffs[j]);
    //     }
    //     printf("\n\n\n");
    // }
    PQCLEAN_DILITHIUM2_CLEAN_polyvec_matrix_pointwise_montgomery(&t1, mat, &s1hat);
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_reduce(&t1);
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_invntt_tomont(&t1);

    /* Add error vector s2 */
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_add(&t1, &t1, &s2);

    /* Extract t1 and write public key */
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_caddq(&t1);
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_power2round(&t1, &t0, &t1);
    PQCLEAN_DILITHIUM2_CLEAN_pack_pk(pk, rho, &t1);
    // for ( int i = 0 ;i < L; i++){
    //     for ( int j = 0; j < 256; j++)
    //     {
    //         printf("%x, ", t1.vec[i].coeffs[j]);
    //     }
    //     printf("\n\n\n");
    // }
    
    // print byte array pk
    // printf("pk: \n");
    // for ( int i = 0 ;i < PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_PUBLICKEYBYTES; i++)
    // {
    //     printf("0x%x, ", pk[i]);
    // }
    // printf("\n\n\n");

    /* Compute H(rho, t1) and write secret key */
    shake256(tr, TRBYTES, pk, PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_PUBLICKEYBYTES);
    PQCLEAN_DILITHIUM2_CLEAN_pack_sk(sk, rho, tr, key, &t0, &s1, &s2);

    return 0;
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_CLEAN_crypto_sign_signature
*
* Description: Computes signature.
*
* Arguments:   - uint8_t *sig:   pointer to output signature (of length PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_BYTES)
*              - size_t *siglen: pointer to output length of signature
*              - uint8_t *m:     pointer to message to be signed
*              - size_t mlen:    length of message
*              - uint8_t *sk:    pointer to bit-packed secret key
*
* Returns 0 (success)
**************************************************/
int PQCLEAN_DILITHIUM2_CLEAN_crypto_sign_signature(uint8_t *sig,
        size_t *siglen,
        const uint8_t *m,
        size_t mlen,
        const uint8_t *sk) {
    unsigned int n;
    int x_reject;
    uint8_t seedbuf[2 * SEEDBYTES + TRBYTES + RNDBYTES + 2 * CRHBYTES];
    uint8_t *rho, *tr, *key, *mu, *rhoprime, *rnd;
    uint16_t nonce = 0;
    shake256incctx state;

    rho = seedbuf;
    tr = rho + SEEDBYTES;
    key = tr + TRBYTES;
    rnd = key + SEEDBYTES;
    mu = rnd + RNDBYTES;
    rhoprime = mu + CRHBYTES;
    PQCLEAN_DILITHIUM2_CLEAN_unpack_sk(
        rho, tr, key, &ws_mat_t0.k, &ws_s1, &ws_s2, sk);

    /* Compute mu = CRH(tr, msg) */
    shake256_inc_init(&state);
    shake256_inc_absorb(&state, tr, TRBYTES);
    shake256_inc_absorb(&state, m, mlen);
    shake256_inc_finalize(&state);
    shake256_inc_squeeze(mu, CRHBYTES, &state);
    shake256_inc_ctx_release(&state);

    for (n = 0; n < RNDBYTES; n++) {
        rnd[n] = 0;
    }
    shake256(rhoprime, CRHBYTES, key, SEEDBYTES + RNDBYTES + CRHBYTES);

    /* Transform the secret vectors.  The matrix is streamed row by row below
     * because the STM32F3 target cannot hold the complete K-by-L matrix. */
    PQCLEAN_DILITHIUM2_CLEAN_polyvecl_ntt(&ws_s1);
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_ntt(&ws_s2);

    xshuffle_attempts = 0;
rej:
    ++xshuffle_attempts;
#ifdef XSHUFFLE_LOG
    /* Keep only the accepted attempt in the host-side oracle log.  The
     * firmware build does not define XSHUFFLE_LOG, so this has no target
     * behavior or timing impact. */
    xshuffle_log_call = 0;
#endif
    /* Sample intermediate vector y */
    PQCLEAN_DILITHIUM2_CLEAN_polyvecl_uniform_gamma1(
        &ws_y_h.l, rhoprime, nonce++);

    /* Matrix-vector multiplication */
    ws_z_w0.l = ws_y_h.l;
    PQCLEAN_DILITHIUM2_CLEAN_polyvecl_ntt(&ws_z_w0.l);
    matrix_pointwise_stream(&ws_w1, &ws_mat_t0.l, rho, &ws_z_w0.l);
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_reduce(&ws_w1);
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_invntt_tomont(&ws_w1);

    /* Decompose w and call the random oracle */
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_caddq(&ws_w1);
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_decompose(
        &ws_w1, &ws_z_w0.k, &ws_w1);
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_pack_w1(sig, &ws_w1);

    shake256_inc_init(&state);
    shake256_inc_absorb(&state, mu, CRHBYTES);
    shake256_inc_absorb(&state, sig, K * POLYW1_PACKEDBYTES);
    shake256_inc_finalize(&state);
    shake256_inc_squeeze(sig, CTILDEBYTES, &state);
    shake256_inc_ctx_release(&state);
    PQCLEAN_DILITHIUM2_CLEAN_poly_challenge(
        &ws_cp, sig); /* uses only the first SEEDBYTES bytes of sig */
    PQCLEAN_DILITHIUM2_CLEAN_poly_ntt(&ws_cp);

    /* Compute z, reject if it reveals secret */
    PQCLEAN_DILITHIUM2_CLEAN_polyvecl_pointwise_poly_montgomery(
        &ws_z, &ws_cp, &ws_s1);
    PQCLEAN_DILITHIUM2_CLEAN_polyvecl_invntt_tomont(&ws_z);
    /* Prepare all coefficient permutations before the single acquisition
     * trigger.  No per-coefficient trigger is used: the rising edge marks
     * the beginning of the complete x-domain traversal. */
    for (n = 0; n < L; ++n) {
        xshuffle_prepare(ws_x_order[n]);
    }
    x_reject = 0;
#ifdef DILITHIUM_XSHUFFLE
#ifndef DILITHIUM_CAPTURE_ACCEPTED_CS1
#ifndef DILITHIUM_CAPTURE_POLY_ONLY
    trigger_high();
#endif
#endif
#endif
    for (n = 0; n < L; ++n) {
#ifdef DILITHIUM_CAPTURE_POLY_ONLY
        if (n == (unsigned int)DILITHIUM_CAPTURE_POLY_ONLY) {
            trigger_high();
        }
#endif
        if (xshuffle_add_freeze_chk_norm(
                &ws_z.vec[n], &ws_z.vec[n], &ws_y_h.l.vec[n],
                GAMMA1 - BETA, ws_x_order[n])) {
            x_reject = 1;
            break;
        }
#ifdef DILITHIUM_CAPTURE_POLY_ONLY
        if (n == (unsigned int)DILITHIUM_CAPTURE_POLY_ONLY) {
            trigger_low();
        }
#endif
    }
#ifdef DILITHIUM_XSHUFFLE
#ifndef DILITHIUM_CAPTURE_ACCEPTED_CS1
    trigger_low();
#endif
#endif
    if (x_reject) {
        goto rej;
    }

    /* Check that subtracting cs2 does not change high bits of w and low bits
     * do not reveal secret information */
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_pointwise_poly_montgomery(
        &ws_y_h.k, &ws_cp, &ws_s2);
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_invntt_tomont(&ws_y_h.k);
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_sub(
        &ws_z_w0.k, &ws_z_w0.k, &ws_y_h.k);
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_reduce(&ws_z_w0.k);
    if (PQCLEAN_DILITHIUM2_CLEAN_polyveck_chknorm(
            &ws_z_w0.k, GAMMA2 - BETA)) {
        goto rej;
    }

    /* Compute hints for w1 */
    unpack_t0_only(&ws_mat_t0.k, sk);
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_ntt(&ws_mat_t0.k);
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_pointwise_poly_montgomery(
        &ws_y_h.k, &ws_cp, &ws_mat_t0.k);
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_invntt_tomont(&ws_y_h.k);
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_reduce(&ws_y_h.k);
    if (PQCLEAN_DILITHIUM2_CLEAN_polyveck_chknorm(&ws_y_h.k, GAMMA2)) {
        goto rej;
    }

    PQCLEAN_DILITHIUM2_CLEAN_polyveck_add(
        &ws_z_w0.k, &ws_z_w0.k, &ws_y_h.k);
    n = PQCLEAN_DILITHIUM2_CLEAN_polyveck_make_hint(
        &ws_y_h.k, &ws_z_w0.k, &ws_w1);
    if (n > OMEGA) {
        goto rej;
    }

#ifdef DILITHIUM_CAPTURE_ACCEPTED_CS1
    /* Finalize the real signature before reusing ws_z as the replay output
     * buffer.  The replay branch must not change any signature bytes. */
    PQCLEAN_DILITHIUM2_CLEAN_pack_sig(sig, sig, &ws_z, &ws_y_h.k);
    *siglen = PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_BYTES;

    /*
     * Accepted-after replay acquisition mode.
     *
     * The normal signing attempt above has already passed every rejection
     * condition. Reconstruct the same y and c*s1 for the accepted nonce and
     * replay the original shuffled z = y + c*s1 traversal.  The signature
     * has already been packed, so this branch does not alter any output
     * bytes. Reusing ws_x_order keeps the replay within the same
     * per-signature unknown permutation and does not consume extra RNG.
     */
    PQCLEAN_DILITHIUM2_CLEAN_polyvecl_uniform_gamma1(
        &ws_y_h.l, rhoprime, (uint16_t)(nonce - 1u));
    PQCLEAN_DILITHIUM2_CLEAN_polyvecl_pointwise_poly_montgomery(
        &ws_z, &ws_cp, &ws_s1);
    PQCLEAN_DILITHIUM2_CLEAN_polyvecl_invntt_tomont(&ws_z);
    for (n = 0; n < L; ++n) {
        xshuffle_prepare(ws_x_order[n]);
    }
#ifdef XSHUFFLE_LOG
    /* In host validation mode, retain labels for the replay rather than the
     * ordinary pre-acceptance traversal. */
    xshuffle_log_call = 0;
#endif
    trigger_high();
    for (n = 0; n < L; ++n) {
        (void)xshuffle_add_freeze_chk_norm(
            &ws_z.vec[n], &ws_z.vec[n], &ws_y_h.l.vec[n],
            GAMMA1 - BETA, ws_x_order[n]);
    }
    trigger_low();
#else
    /* Write signature */
    PQCLEAN_DILITHIUM2_CLEAN_pack_sig(sig, sig, &ws_z, &ws_y_h.k);
    *siglen = PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_BYTES;
#endif
    return 0;
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_CLEAN_crypto_sign
*
* Description: Compute signed message.
*
* Arguments:   - uint8_t *sm: pointer to output signed message (allocated
*                             array with PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_BYTES + mlen bytes),
*                             can be equal to m
*              - size_t *smlen: pointer to output length of signed
*                               message
*              - const uint8_t *m: pointer to message to be signed
*              - size_t mlen: length of message
*              - const uint8_t *sk: pointer to bit-packed secret key
*
* Returns 0 (success)
**************************************************/
int PQCLEAN_DILITHIUM2_CLEAN_crypto_sign(uint8_t *sm,
        size_t *smlen,
        const uint8_t *m,
        size_t mlen,
        const uint8_t *sk) {
    size_t i;

    for (i = 0; i < mlen; ++i) {
        sm[PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_BYTES + mlen - 1 - i] = m[mlen - 1 - i];
    }
    PQCLEAN_DILITHIUM2_CLEAN_crypto_sign_signature(sm, smlen, sm + PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_BYTES, mlen, sk);
    *smlen += mlen;
    return 0;
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_CLEAN_crypto_sign_verify
*
* Description: Verifies signature.
*
* Arguments:   - uint8_t *m: pointer to input signature
*              - size_t siglen: length of signature
*              - const uint8_t *m: pointer to message
*              - size_t mlen: length of message
*              - const uint8_t *pk: pointer to bit-packed public key
*
* Returns 0 if signature could be verified correctly and -1 otherwise
**************************************************/
int PQCLEAN_DILITHIUM2_CLEAN_crypto_sign_verify(const uint8_t *sig,
        size_t siglen,
        const uint8_t *m,
        size_t mlen,
        const uint8_t *pk) {
    unsigned int i;
    uint8_t buf[K * POLYW1_PACKEDBYTES];
    uint8_t rho[SEEDBYTES];
    uint8_t mu[CRHBYTES];
    uint8_t c[CTILDEBYTES];
    uint8_t c2[CTILDEBYTES];
    poly cp;
    polyvecl mat[K], z;
    polyveck t1, w1, h;
    shake256incctx state;

    if (siglen != PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_BYTES) {
        return -1;
    }

    PQCLEAN_DILITHIUM2_CLEAN_unpack_pk(rho, &t1, pk);
    if (PQCLEAN_DILITHIUM2_CLEAN_unpack_sig(c, &z, &h, sig)) {
        return -1;
    }
    if (PQCLEAN_DILITHIUM2_CLEAN_polyvecl_chknorm(&z, GAMMA1 - BETA)) {
        return -1;
    }

    /* Compute CRH(H(rho, t1), msg) */
    shake256(mu, CRHBYTES, pk, PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_PUBLICKEYBYTES);
    shake256_inc_init(&state);
    shake256_inc_absorb(&state, mu, CRHBYTES);
    shake256_inc_absorb(&state, m, mlen);
    shake256_inc_finalize(&state);
    shake256_inc_squeeze(mu, CRHBYTES, &state);
    shake256_inc_ctx_release(&state);

    /* Matrix-vector multiplication; compute Az - c2^dt1 */
    PQCLEAN_DILITHIUM2_CLEAN_poly_challenge(&cp, c); /* uses only the first SEEDBYTES bytes of c */
    PQCLEAN_DILITHIUM2_CLEAN_polyvec_matrix_expand(mat, rho);

    PQCLEAN_DILITHIUM2_CLEAN_polyvecl_ntt(&z);
    PQCLEAN_DILITHIUM2_CLEAN_polyvec_matrix_pointwise_montgomery(&w1, mat, &z);

    PQCLEAN_DILITHIUM2_CLEAN_poly_ntt(&cp);
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_shiftl(&t1);
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_ntt(&t1);
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_pointwise_poly_montgomery(&t1, &cp, &t1);

    PQCLEAN_DILITHIUM2_CLEAN_polyveck_sub(&w1, &w1, &t1);
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_reduce(&w1);
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_invntt_tomont(&w1);

    /* Reconstruct w1 */
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_caddq(&w1);
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_use_hint(&w1, &w1, &h);
    PQCLEAN_DILITHIUM2_CLEAN_polyveck_pack_w1(buf, &w1);

    /* Call random oracle and verify PQCLEAN_DILITHIUM2_CLEAN_challenge */
    shake256_inc_init(&state);
    shake256_inc_absorb(&state, mu, CRHBYTES);
    shake256_inc_absorb(&state, buf, K * POLYW1_PACKEDBYTES);
    shake256_inc_finalize(&state);
    shake256_inc_squeeze(c2, CTILDEBYTES, &state);
    shake256_inc_ctx_release(&state);
    for (i = 0; i < CTILDEBYTES; ++i) {
        if (c[i] != c2[i]) {
            return -1;
        }
    }
    return 0;
}

/*************************************************
* Name:        PQCLEAN_DILITHIUM2_CLEAN_crypto_sign_open
*
* Description: Verify signed message.
*
* Arguments:   - uint8_t *m: pointer to output message (allocated
*                            array with smlen bytes), can be equal to sm
*              - size_t *mlen: pointer to output length of message
*              - const uint8_t *sm: pointer to signed message
*              - size_t smlen: length of signed message
*              - const uint8_t *pk: pointer to bit-packed public key
*
* Returns 0 if signed message could be verified correctly and -1 otherwise
**************************************************/
int PQCLEAN_DILITHIUM2_CLEAN_crypto_sign_open(uint8_t *m,
        size_t *mlen,
        const uint8_t *sm,
        size_t smlen,
        const uint8_t *pk) {
    size_t i;

    if (smlen < PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_BYTES) {
        goto badsig;
    }

    *mlen = smlen - PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_BYTES;
    if (PQCLEAN_DILITHIUM2_CLEAN_crypto_sign_verify(sm, PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_BYTES, sm + PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_BYTES, *mlen, pk)) {
        goto badsig;
    } else {
        /* All good, copy msg, return 0 */
        for (i = 0; i < *mlen; ++i) {
            m[i] = sm[PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_BYTES + i];
        }
        return 0;
    }

badsig:
    /* Signature verification failed */
    *mlen = -1;
    for (i = 0; i < smlen; ++i) {
        m[i] = 0;
    }

    return -1;
}
