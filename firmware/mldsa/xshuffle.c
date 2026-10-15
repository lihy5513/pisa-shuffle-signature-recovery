#include "xshuffle.h"
#include "randombytes.h"
#include "reduce.h"

#ifdef XSHUFFLE_LOG
uint16_t xshuffle_log_order[L][N];
int32_t xshuffle_log_x[L][N];
int32_t xshuffle_log_y[L][N];
int32_t xshuffle_log_raw[L][N];
int32_t xshuffle_log_reduced[L][N];
unsigned int xshuffle_log_call;
#endif

static void swap_u16(uint16_t *a, uint16_t *b)
{
    uint16_t t = *a;
    *a = *b;
    *b = t;
}

void xshuffle_prepare(uint16_t order[N])
{
    xshuffle_prepare_n(order, N);
}

void xshuffle_prepare_n(uint16_t *order, unsigned int n)
{
    unsigned int i;
    uint8_t b[2];

    for (i = 0; i < n; ++i) {
        order[i] = (uint16_t)i;
    }

    /* Fisher--Yates with rejection-free bounded reduction. */
    for (i = n - 1; i > 0; --i) {
        uint16_t j;
        randombytes(b, sizeof(b));
        j = (uint16_t)(((uint16_t)b[0] << 8 | b[1]) % (i + 1u));
        swap_u16(&order[i], &order[j]);
    }
}

int xshuffle_add_freeze_chk_norm(poly *out,
                                 const poly *x,
                                 const poly *y,
                                 uint32_t bound,
                                 uint16_t order[N])
{
    unsigned int k;
    int32_t t;
    int32_t a;
#ifdef XSHUFFLE_LOG
    unsigned int call = xshuffle_log_call++;
#endif

    for (k = 0; k < N; ++k) {
        unsigned int i = order[k];
        /* The signing path intentionally aliases out and x.  Preserve the
         * pre-add operand before writing out->coeffs[i], otherwise the
         * optional offline logger would record x+y as both x and raw. */
        int32_t x_value = x->coeffs[i];
        int32_t y_value = y->coeffs[i];
        /* This is the same reduce-and-norm computation as the reference
         * helper, with only the coefficient traversal order changed. */
        out->coeffs[i] = x_value + y_value;
#ifdef XSHUFFLE_LOG
        if (call < L) {
            xshuffle_log_order[call][k] = (uint16_t)i;
            xshuffle_log_x[call][k] = x_value;
            xshuffle_log_y[call][k] = y_value;
            xshuffle_log_raw[call][k] = out->coeffs[i];
        }
#endif
        /* Match poly_reduce() exactly: it applies reduce32, but not caddq.
         * Keeping the reference helper here avoids changing the arithmetic
         * while only changing the coefficient traversal order. */
        a = out->coeffs[i];
        t = (a + (1 << 22)) >> 23;
        out->coeffs[i] = a - t * Q;
#ifdef XSHUFFLE_LOG
        if (call < L) {
            xshuffle_log_reduced[call][k] = out->coeffs[i];
        }
#endif

        /* Match poly_chknorm(): compute the absolute value of the centered
         * representative before applying the strict bound. */
        t = out->coeffs[i] >> 31;
        t = out->coeffs[i] - (t & (2 * out->coeffs[i]));
        if (t >= (int32_t)bound) {
            return 1;
        }
    }
    return 0;
}
