#ifndef XSHUFFLE_HAL_H
#define XSHUFFLE_HAL_H

/* The original M4 benchmark source uses hal_get_time() only for optional
 * timing accounting.  Timing is disabled in this ChipWhisperer target. */
static inline unsigned long long hal_get_time(void) { return 0; }

#endif
