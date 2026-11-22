/* tests/test_ink_kv.c -- the host KV codecs (c/ink_kv.h): round trip, error bounds and
 * the dot's agreement with the widened row. Build and run:
 *
 *   cc -O2 -march=native -I c c/tests/test_ink_kv.c -lm -o /tmp/test_ink_kv && /tmp/test_ink_kv
 *
 * f16 is exact where its grid holds the value (and the reference for "near" values); q8
 * and q4 must stay inside their block's half-step; every dot must equal the dot of the
 * widened row to f32's own lane order. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ink_kv.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static float frnd(void) {   /* a spread of magnitudes, signed, with a few exact halves */
    static unsigned s = 12345;
    s = s * 1103515245u + 12345u;
    double u = ((s >> 8) & 0xffffff) / (double)0x1000000;   /* [0,1) */
    double e = pow(10.0, -8.0 + 16.0 * u);                  /* 1e-8 .. 1e7 */
    return (float)((u < 0.5 ? -1 : 1) * e * (0.3 + 0.7 * u));
}

int main(void) {
    const int dims[] = {128, 64, 96};
    for (size_t di = 0; di < sizeof dims / sizeof dims[0]; di++) {
        const int n = dims[di];
        const size_t rb16 = ink_kv_row_bytes(INK_KV_F16, n);
        const size_t rb8 = ink_kv_row_bytes(INK_KV_Q8, n);
        const size_t rb4 = ink_kv_row_bytes(INK_KV_Q4, n);
        CHECK(rb16 == (size_t)n * 2, "f16 row bytes %zu", rb16);
        CHECK(rb8 >= (size_t)n + 2 * ((n + 31) / 32), "q8 row bytes %zu", rb8);
        CHECK(rb4 >= (size_t)n / 2 + 2 * ((n + 31) / 32), "q4 row bytes %zu", rb4);

        unsigned char *r16 = malloc(rb16), *r8 = malloc(rb8), *r4 = malloc(rb4);
        float *x = malloc((size_t)n * sizeof(float)), *y = malloc((size_t)n * sizeof(float));
        float *q = malloc((size_t)n * sizeof(float));

        for (int it = 0; it < 200; it++) {
            for (int i = 0; i < n; i++) { x[i] = frnd(); q[i] = frnd(); }
            /* f32: exact */
            float *r32 = malloc((size_t)n * sizeof(float));
            ink_kv_put(INK_KV_F32, r32, x, n);
            ink_kv_get(INK_KV_F32, y, r32, n);
            for (int i = 0; i < n; i++) CHECK(y[i] == x[i], "f32 round trip %g != %g", y[i], x[i]);
            free(r32);

            /* f16: the value widened is the float it was rounded to; and a value already on
             * the grid comes back bit-exactly */
            ink_kv_put(INK_KV_F16, r16, x, n);
            ink_kv_get(INK_KV_F16, y, r16, n);
            for (int i = 0; i < n; i++) {
                uint16_t b; memcpy(&b, r16 + 2 * i, 2);
                CHECK(y[i] == kv16_to(b), "f16 get %g", y[i]);
                if (fabsf(x[i]) < 65504.f) CHECK(fabsf(y[i] - x[i]) <= fmaxf(fabsf(x[i]) * 0.001f, 6e-8f),
                                                  "f16 err %g vs %g", y[i], x[i]);
            }
            /* a dot is the dot of the widened row (lane order of dot_f32_lanes) */
            float want = dot_f32_lanes(q, y, n), got = ink_kv_dot(INK_KV_F16, q, r16, n);
            CHECK(got == want, "f16 dot %g != %g", got, want);

            /* q8: error inside half a quantization step (amax/254), and the dot matches the
             * widened row's dot */
            ink_kv_put(INK_KV_Q8, r8, x, n);
            ink_kv_get(INK_KV_Q8, y, r8, n);
            for (int b = 0; b < n; b += 32) {
                float amax = 0.f;
                for (int i = b; i < n && i < b + 32; i++) amax = fmaxf(amax, fabsf(x[i]));
                const float step = amax > 1e-30f ? amax / 127.f : 0.f;
                for (int i = b; i < n && i < b + 32; i++) CHECK(fabsf(y[i] - x[i]) <= step * 0.5001f + 1e-30f,
                                                                 "q8 err %g step %g", y[i] - x[i], step);
            }
            want = dot_f32_lanes(q, y, n); got = ink_kv_dot(INK_KV_Q8, q, r8, n);
            CHECK(fabsf(got - want) <= fmaxf(1e-5f * fabsf(want), 1e-5f), "q8 dot %g != %g", got, want);

            /* q4: half a step (amax/14) */
            ink_kv_put(INK_KV_Q4, r4, x, n);
            ink_kv_get(INK_KV_Q4, y, r4, n);
            for (int b = 0; b < n; b += 32) {
                float amax = 0.f;
                for (int i = b; i < n && i < b + 32; i++) amax = fmaxf(amax, fabsf(x[i]));
                const float step = amax > 1e-30f ? amax / 7.f : 0.f;
                for (int i = b; i < n && i < b + 32; i++) CHECK(fabsf(y[i] - x[i]) <= step * 0.5001f + 1e-30f,
                                                                 "q4 err %g step %g", y[i] - x[i], step);
            }
            want = dot_f32_lanes(q, y, n); got = ink_kv_dot(INK_KV_Q4, q, r4, n);
            CHECK(fabsf(got - want) <= fmaxf(2e-5f * fabsf(want), 2e-5f), "q4 dot %g != %g", got, want);

            /* V's accumulation widens the same row */
            float *acc = calloc((size_t)n, sizeof(float)), *acc2 = calloc((size_t)n, sizeof(float));
            for (int i = 0; i < n; i++) acc2[i] = 0.7f * y[i] + 0.1f;
            for (int i = 0; i < n; i++) acc[i] = 0.1f;
            ink_kv_axpy(INK_KV_Q4, acc, 0.7f, r4, n);
            for (int i = 0; i < n; i++) CHECK(fabsf(acc[i] - acc2[i]) <= 1e-5f * (1.f + fabsf(acc2[i])),
                                               "q4 axpy %g vs %g", acc[i], acc2[i]);
            free(acc); free(acc2);
        }
        /* a zero row decodes to zero; a huge row does not overflow */
        for (int i = 0; i < n; i++) x[i] = 0.f;
        ink_kv_put(INK_KV_Q4, r4, x, n); ink_kv_get(INK_KV_Q4, y, r4, n);
        for (int i = 0; i < n; i++) CHECK(y[i] == 0.f, "q4 zero row");
        for (int i = 0; i < n; i++) x[i] = (i & 1 ? -1.f : 1.f) * 3e30f;
        ink_kv_put(INK_KV_Q8, r8, x, n); ink_kv_get(INK_KV_Q8, y, r8, n);
        for (int i = 0; i < n; i++) CHECK(isfinite(y[i]), "q8 huge row");
        free(r16); free(r8); free(r4); free(x); free(y); free(q);
    }
    if (fails) { printf("%d failure(s)\n", fails); return 1; }
    printf("ink_kv: ok\n");
    return 0;
}
