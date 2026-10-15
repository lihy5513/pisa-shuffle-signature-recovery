#include <stdint.h>
#include <stddef.h>

#include "api.h"
#include "hal.h"
#include "simpleserial.h"

/* The key buffer starts empty and is filled through the SimpleSerial k command. */
extern uint8_t sk[PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_SECRETKEYBYTES];
static uint8_t sig[PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_BYTES];
static size_t siglen;

uint8_t reset_target(uint8_t *x, uint8_t len)
{
    (void)x;
    (void)len;
    return 0;
}

/* Write one 32-byte key block and echo it for host-side verification. */
static uint8_t set_key_block(uint8_t *data, uint8_t len)
{
    unsigned int index = data[0];
    if (len != 33 || index >= sizeof(sk) / 32) {
        return 1;
    }
    for (unsigned int i = 0; i < 32; ++i) {
        sk[index * 32 + i] = data[i + 1];
    }
    simpleserial_put('k', 32, sk + index * 32);
    return 0;
}

uint8_t sign_message(uint8_t *msg, uint8_t len)
{
    int ret;
    ret = PQCLEAN_DILITHIUM2_CLEAN_crypto_sign_signature(
        sig, &siglen, msg, len, sk);
    simpleserial_put('r', 32, sig);
    return (uint8_t)ret;
}

int main(void)
{
    platform_init();
    init_uart();
    trigger_setup();

    simpleserial_init();
    simpleserial_addcmd('p', 32, sign_message);
    simpleserial_addcmd('k', 33, set_key_block);
    simpleserial_addcmd('x', 0, reset_target);
    while (1) {
        simpleserial_get();
    }
}
