/*
 * Compute incremental SHA-256 and explicit-width identity-field encodings.
 *
 * Byte length is checked before mutation and finalization is one-shot. Callers decide whether a
 * computed digest is identity or trust evidence.
 */
#include <yvex/internal/core.h>

#include <ctype.h>
#include <limits.h>
#include <string.h>

static uint32_t sha256_rotate_right(uint32_t value, unsigned int bits)
{
    return (value >> bits) | (value << (32u - bits));
}

static void sha256_transform(yvex_sha256 *context,
                             const unsigned char block[64])
{
    static const uint32_t constants[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
    };
    uint32_t words[64];
    uint32_t a, b, c, d, e, f, g, h;
    unsigned int index;

    for (index = 0u; index < 16u; ++index) {
        words[index] = ((uint32_t)block[index * 4u] << 24u) |
                       ((uint32_t)block[index * 4u + 1u] << 16u) |
                       ((uint32_t)block[index * 4u + 2u] << 8u) |
                       (uint32_t)block[index * 4u + 3u];
    }
    for (index = 16u; index < 64u; ++index) {
        uint32_t s0 = sha256_rotate_right(words[index - 15u], 7u) ^
                      sha256_rotate_right(words[index - 15u], 18u) ^
                      (words[index - 15u] >> 3u);
        uint32_t s1 = sha256_rotate_right(words[index - 2u], 17u) ^
                      sha256_rotate_right(words[index - 2u], 19u) ^
                      (words[index - 2u] >> 10u);
        words[index] = words[index - 16u] + s0 + words[index - 7u] + s1;
    }
    a = context->state[0]; b = context->state[1];
    c = context->state[2]; d = context->state[3];
    e = context->state[4]; f = context->state[5];
    g = context->state[6]; h = context->state[7];
    for (index = 0u; index < 64u; ++index) {
        uint32_t s1 = sha256_rotate_right(e, 6u) ^
                      sha256_rotate_right(e, 11u) ^
                      sha256_rotate_right(e, 25u);
        uint32_t choose = (e & f) ^ ((~e) & g);
        uint32_t first = h + s1 + choose + constants[index] + words[index];
        uint32_t s0 = sha256_rotate_right(a, 2u) ^
                      sha256_rotate_right(a, 13u) ^
                      sha256_rotate_right(a, 22u);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t second = s0 + majority;
        h = g; g = f; f = e; e = d + first;
        d = c; c = b; b = a; a = first + second;
    }
    context->state[0] += a; context->state[1] += b;
    context->state[2] += c; context->state[3] += d;
    context->state[4] += e; context->state[5] += f;
    context->state[6] += g; context->state[7] += h;
}

void yvex_sha256_init(yvex_sha256 *context)
{
    if (!context) return;
    memset(context, 0, sizeof(*context));
    context->state[0] = 0x6a09e667u;
    context->state[1] = 0xbb67ae85u;
    context->state[2] = 0x3c6ef372u;
    context->state[3] = 0xa54ff53au;
    context->state[4] = 0x510e527fu;
    context->state[5] = 0x9b05688cu;
    context->state[6] = 0x1f83d9abu;
    context->state[7] = 0x5be0cd19u;
}

int yvex_sha256_update(yvex_sha256 *context, const void *data, size_t length)
{
    const unsigned char *bytes = (const unsigned char *)data;
    size_t offset = 0u;

    if (!context || (!data && length != 0u) || context->finalized ||
        UINT64_MAX / 8u - context->length < (uint64_t)length) return 0;
    context->length += (uint64_t)length;
    while (offset < length) {
        size_t available = sizeof(context->block) - context->used;
        size_t take = length - offset < available ? length - offset : available;
        memcpy(context->block + context->used, bytes + offset, take);
        context->used += take;
        offset += take;
        if (context->used == sizeof(context->block)) {
            sha256_transform(context, context->block);
            context->used = 0u;
        }
    }
    return 1;
}

int yvex_sha256_final(yvex_sha256 *context,
                      unsigned char digest[YVEX_SHA256_DIGEST_BYTES])
{
    uint64_t bit_length;
    unsigned int index;

    if (!context || !digest || context->finalized ||
        context->length > UINT64_MAX / 8u) return 0;
    bit_length = context->length * 8u;
    context->block[context->used++] = 0x80u;
    if (context->used > 56u) {
        memset(context->block + context->used, 0,
               sizeof(context->block) - context->used);
        sha256_transform(context, context->block);
        context->used = 0u;
    }
    memset(context->block + context->used, 0, 56u - context->used);
    for (index = 0u; index < 8u; ++index)
        context->block[63u - index] =
            (unsigned char)((bit_length >> (index * 8u)) & 0xffu);
    sha256_transform(context, context->block);
    for (index = 0u; index < 8u; ++index) {
        digest[index * 4u] = (unsigned char)(context->state[index] >> 24u);
        digest[index * 4u + 1u] =
            (unsigned char)(context->state[index] >> 16u);
        digest[index * 4u + 2u] =
            (unsigned char)(context->state[index] >> 8u);
        digest[index * 4u + 3u] = (unsigned char)context->state[index];
    }
    memset(context->block, 0, sizeof(context->block));
    context->used = 0u;
    context->finalized = 1;
    return 1;
}

/* Converts a borrowed binary digest to lowercase hexadecimal without allocation. */

void yvex_sha256_hex(const unsigned char digest[YVEX_SHA256_DIGEST_BYTES],
                     char output[YVEX_SHA256_HEX_BYTES])
{
    static const char alphabet[] = "0123456789abcdef";
    unsigned int index;

    if (!digest || !output) return;
    for (index = 0u; index < YVEX_SHA256_DIGEST_BYTES; ++index) {
        output[index * 2u] = alphabet[digest[index] >> 4u];
        output[index * 2u + 1u] = alphabet[digest[index] & 0x0fu];
    }
    output[64] = '\0';
}

int yvex_sha256_hex_valid(const char *text)
{
    size_t index;

    if (!text) return 0;
    for (index = 0u; index < 64u; ++index) {
        if (!text[index] || !isxdigit((unsigned char)text[index])) return 0;
    }
    return text[64] == '\0';
}

/*
 * Append one unsigned scalar through the canonical little-endian identity encoding.
 *
 * Never hashes native object representation or padding.
 */
int yvex_sha256_update_u64(yvex_sha256 *context, unsigned long long value)
{
    unsigned char bytes[8];
    unsigned int index;

    for (index = 0u; index < 8u; ++index) {
        bytes[index] = (unsigned char)((value >> (index * 8u)) & 0xffu);
    }
    return yvex_sha256_update(context, bytes, sizeof(bytes));
}

/* Append nullable text through the canonical length-delimited identity encoding. */
int yvex_sha256_update_text(yvex_sha256 *context, const char *text)
{
    size_t length = text ? strlen(text) : 0u;

    return yvex_sha256_update_u64(context, (unsigned long long)length) &&
           yvex_sha256_update(context, text, length);
}
