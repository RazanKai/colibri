/* ink_kv.h -- the Inkling host KV cache's element type: f32, f16, q8 or q4.
 *
 * INK2_KV selects it once, at the first kv_alloc (or model load):
 *
 *   INK2_KV=f32 | f16 | q8 | q4        (default f32)
 *
 * A cache row is `hd` values of one KV head at one position. The layout of a row:
 *
 *   f32   4 bytes a value, the original cache.
 *   f16   2 bytes a value, kv16.h's rounding (IEEE half, ties to even).
 *   q8    blocks of 32 values: one float32 scale, then 32 signed bytes.
 *         value = int8 * scale, scale = amax/127.
 *   q4    blocks of 32 values: one float32 scale, then 16 bytes, two 4-bit
 *         two's-complement nibbles a byte (even index in the low nibble).
 *         value = nibble * scale, scale = amax/7.
 *
 * The scale is f32, not f16: a K row can hold values large enough (q/k clips,
 * the log-length tau) that amax/7 passes f16's 65504 and would saturate the grid.
 *
 * K and V use the same type. The scale lives inside the block, so a row is
 * self-describing and a byte copy (prefix store, park, a chain mirror) needs no
 * knowledge of the type -- only the row's size, ink_kv_row_bytes().
 *
 * The attention widens a row on the fly: ink_kv_dot for the score (an f16 row
 * straight from kv16_dot_lanes, a q4/q8 row through the block scale) and
 * ink_kv_axpy for V. f32 rows keep the exact dot_f32_lanes path, so an f32 run
 * is bit-for-bit what it was.
 */
#ifndef COLI_INK_KV_H
#define COLI_INK_KV_H

#include <stdint.h>
#include <string.h>
#include <math.h>
#include "kv16.h"   /* kv16_from/kv16_to, kv16_dot_lanes, kv16_axpy */

#define INK_KV_F32 0
#define INK_KV_F16 1
#define INK_KV_Q8  2
#define INK_KV_Q4  3

static inline const char *ink_kv_name(int t) {
    return t == INK_KV_F16 ? "f16" : t == INK_KV_Q8 ? "q8" : t == INK_KV_Q4 ? "q4" : "f32";
}
/* INK2_KV's value; -1 when it is not one of the four (the caller refuses the run). */
static inline int ink_kv_parse(const char *s) {
    if (!s || !*s) return INK_KV_F32;
    if (!strcmp(s, "f32") || !strcmp(s, "fp32")) return INK_KV_F32;
    if (!strcmp(s, "f16") || !strcmp(s, "fp16") || !strcmp(s, "half")) return INK_KV_F16;
    if (!strcmp(s, "q8") || !strcmp(s, "k8")) return INK_KV_Q8;
    if (!strcmp(s, "q4") || !strcmp(s, "k4")) return INK_KV_Q4;
    return -1;
}
/* bytes of one row of `hd` values */
static inline size_t ink_kv_row_bytes(int type, int hd) {
    switch (type) {
        case INK_KV_F16: return (size_t)hd * 2;
        case INK_KV_Q8:  return (size_t)((hd + 31) / 32) * 36;
        case INK_KV_Q4:  return (size_t)((hd + 31) / 32) * 20;
        default:         return (size_t)hd * 4;
    }
}
/* one row's worth of floats in bytes, for a plan or a log line */
static inline double ink_kv_mib(int type, int hd, int values) {
    return (double)values * (double)ink_kv_row_bytes(type, hd) / (double)hd / 1048576.0;
}

/* ---- q8: 32 signed bytes + an f16 scale ---- */
static inline void ink_q8_pack_block(uint8_t *d, const float *x, int m) {
    float amax = 0.f;
    for (int i = 0; i < m; i++) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    float s = amax > 1e-30f ? amax / 127.f : 1.f;
    float inv = amax > 1e-30f ? 127.f / amax : 0.f;
    memcpy(d, &s, 4);
    for (int i = 0; i < 32; i++) {
        if (i < m) {
            float v = rintf(x[i] * inv);
            if (v > 127.f) v = 127.f;
            if (v < -127.f) v = -127.f;
            d[4 + i] = (uint8_t)(int8_t)v;
        } else d[4 + i] = 0;
    }
}
/* ---- q4: 16 packed nibbles + an f16 scale ---- */
static inline void ink_q4_pack_block(uint8_t *d, const float *x, int m) {
    float amax = 0.f;
    for (int i = 0; i < m; i++) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    float s = amax > 1e-30f ? amax / 7.f : 1.f;
    float inv = amax > 1e-30f ? 7.f / amax : 0.f;
    memcpy(d, &s, 4);
    for (int i = 0; i < 16; i++) {
        int lo = 0, hi = 0;
        if (2 * i < m) {
            float v = rintf(x[2 * i] * inv);
            if (v > 7.f) v = 7.f; if (v < -8.f) v = -8.f;
            lo = ((int)v) & 0xF;
        }
        if (2 * i + 1 < m) {
            float v = rintf(x[2 * i + 1] * inv);
            if (v > 7.f) v = 7.f; if (v < -8.f) v = -8.f;
            hi = ((int)v) & 0xF;
        }
        d[4 + i] = (uint8_t)(lo | (hi << 4));
    }
}

/* src[0, n) -> a row of `type` */
static inline void ink_kv_put(int type, void *row, const float *src, int n) {
    if (type == INK_KV_F32) { memcpy(row, src, (size_t)n * sizeof(float)); return; }
    if (type == INK_KV_F16) { kv16_pack((uint16_t *)row, src, n); return; }
    uint8_t *p = (uint8_t *)row;
    for (int b = 0; b < n; b += 32) {
        int m = n - b < 32 ? n - b : 32;
        if (type == INK_KV_Q8) { ink_q8_pack_block(p, src + b, m); p += 36; }
        else                   { ink_q4_pack_block(p, src + b, m); p += 20; }
    }
}
/* a row of `type` -> dst[0, n) */
static inline void ink_kv_get(int type, float *dst, const void *row, int n) {
    if (type == INK_KV_F32) { memcpy(dst, row, (size_t)n * sizeof(float)); return; }
    if (type == INK_KV_F16) { kv16_unpack(dst, (const uint16_t *)row, n); return; }
    const uint8_t *p = (const uint8_t *)row;
    for (int b = 0; b < n; b += 32) {
        int m = n - b < 32 ? n - b : 32;
        const float s = *(const float *)p;
        if (type == INK_KV_Q8) {
            for (int i = 0; i < m; i++) dst[b + i] = (float)(int8_t)p[4 + i] * s;
            p += 36;
        } else {
            for (int i = 0; i < m; i++) {
                int v = (p[4 + (i >> 1)] >> ((i & 1) * 4)) & 0xF;
                dst[b + i] = (float)((v ^ 8) - 8) * s;
            }
            p += 20;
        }
    }
}

/* q: n floats dot a row. f32 and f16 are the lanes' own kernels (bit-identical to the
 * f32 cache where the values are representable); q4/q8 scale each block's integer sum. */
static inline float ink_kv_dot(int type, const float *q, const void *row, int n) {
    if (type == INK_KV_F32) return dot_f32_lanes(q, (const float *)row, n);
    if (type == INK_KV_F16) return kv16_dot_lanes(q, (const uint16_t *)row, n);
    const uint8_t *p = (const uint8_t *)row;
    float acc = 0.f;
    for (int b = 0; b < n; b += 32) {
        int m = n - b < 32 ? n - b : 32;
        const float s = *(const float *)p;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        if (type == INK_KV_Q8) {
            const int8_t *v = (const int8_t *)(p + 4);
            int i = 0;
            for (; i + 4 <= m; i += 4) {
                a0 += q[b + i]     * (float)v[i];
                a1 += q[b + i + 1] * (float)v[i + 1];
                a2 += q[b + i + 2] * (float)v[i + 2];
                a3 += q[b + i + 3] * (float)v[i + 3];
            }
            for (; i < m; i++) a0 += q[b + i] * (float)v[i];
            p += 36;
        } else {
            int i = 0;
            for (; i + 4 <= m; i += 4) {
                for (int j = 0; j < 4; j++) {
                    int t = (p[4 + ((i + j) >> 1)] >> (((i + j) & 1) * 4)) & 0xF;
                    float f = (float)((t ^ 8) - 8);
                    if (j == 0) a0 += q[b + i] * f;
                    else if (j == 1) a1 += q[b + i + 1] * f;
                    else if (j == 2) a2 += q[b + i + 2] * f;
                    else a3 += q[b + i + 3] * f;
                }
            }
            for (; i < m; i++) {
                int t = (p[4 + (i >> 1)] >> ((i & 1) * 4)) & 0xF;
                a0 += q[b + i] * (float)((t ^ 8) - 8);
            }
            p += 20;
        }
        acc += s * ((a0 + a1) + (a2 + a3));
    }
    return acc;
}
/* y += a * row, the row widened (a block of 32 at a time for q4/q8: no scratch row
 * of the whole head dim, and the block's scale folded into a first) */
static inline void ink_kv_axpy(int type, float *y, float a, const void *row, int n) {
    if (type == INK_KV_F32) {
        const float *x = (const float *)row;
        for (int i = 0; i < n; i++) y[i] = MATMUL_F32_MADD(y[i], a, x[i]);
        return;
    }
    if (type == INK_KV_F16) { kv16_axpy(y, a, (const uint16_t *)row, n); return; }
    const uint8_t *p = (const uint8_t *)row;
    for (int b = 0; b < n; b += 32) {
        int m = n - b < 32 ? n - b : 32;
        const float s = *(const float *)p * a;
        if (type == INK_KV_Q8) {
            for (int i = 0; i < m; i++) y[b + i] = MATMUL_F32_MADD(y[b + i], s, (float)(int8_t)p[4 + i]);
            p += 36;
        } else {
            for (int i = 0; i < m; i++) {
                int v = (p[4 + (i >> 1)] >> ((i & 1) * 4)) & 0xF;
                y[b + i] = MATMUL_F32_MADD(y[b + i], s, (float)((v ^ 8) - 8));
            }
            p += 20;
        }
    }
}
#endif /* COLI_INK_KV_H */
