#ifndef XSHUFFLE_H
#define XSHUFFLE_H

#include <stdint.h>
#include "params.h"
#include "poly.h"

/* Randomize coefficient processing without changing coefficient storage. */
void xshuffle_prepare(uint16_t order[N]);
void xshuffle_prepare_n(uint16_t *order, unsigned int n);
int xshuffle_add_freeze_chk_norm(poly *out,
                                 const poly *x,
                                 const poly *y,
                                 uint32_t bound,
                                 uint16_t order[N]);

#ifdef XSHUFFLE_LOG
extern uint16_t xshuffle_log_order[L][N];
extern int32_t xshuffle_log_x[L][N];
extern int32_t xshuffle_log_y[L][N];
extern int32_t xshuffle_log_raw[L][N];
extern int32_t xshuffle_log_reduced[L][N];
extern unsigned int xshuffle_log_call;
#endif

#endif
