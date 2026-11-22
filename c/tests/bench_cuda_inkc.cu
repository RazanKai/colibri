/* Kernel-level bench for Inkling-2's attention on the CUDA chain (cuda_inkc.cuh): the decode
 * step (S = 1, the split walk) and the prompt path (S rows, the tiled kernel) against a synthetic
 * q4 cache at a served context, with no engine around it. P0a of PLAN-colibri.md's speed work:
 * ms, achieved GB/s and effective TFLOPS per kernel, so a kernel change is measured in seconds
 * rather than in a 60-minute 250K prefill.
 *
 *   make -C c cuda-inkc-bench CUDA=1 CUDA_HOME=/opt/cuda
 *   ./c/cuda_inkc_bench [ctx] [rows] [reps] [kvtype] [heads]
 *
 * ctx the positions in the cache (262144), rows the prompt chunk (1024, a decode step when 0),
 * kvtype 0 f32, 1 f16, 2 q8, 3 q4 (the 250K/1M setting), heads the query heads (24: Inkling-2-
 * Mini's, 3 of them a KV head). The bytes read are the mirror's, the FLOPs are 2 * pairs * H * hd
 * * 2 (QK^T and V, 2 FLOPs a MAC, causal pairs inside the chunk). */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>
#include "../backend_cuda.h"
#include "../cuda_chain.h"
#include "../ink_kv.h"

static const ColiCudaChainOps *T;
static unsigned g_rng = 1234567;
static unsigned rnd() { g_rng = g_rng * 1103515245u + 12345u; return (g_rng >> 8) & 0xffffff; }
static float frnd() { return ((int)(rnd() % 2001) - 1000) / 1000.0f; }

static CcBuf *upf(const std::vector<float> &v) {
    CcBuf *b = T->buf(v.empty() ? 4 : v.size() * 4, CC_DEV);
    if (!b || !T->begin() || !T->write(b, 0, v.data(), v.size() * 4) || !T->submit(1)) { printf("  FAIL upload\n"); exit(1); }
    return b;
}
static CcBuf *upb(const void *p, size_t bytes) {
    CcBuf *b = T->buf(bytes ? bytes : 4, CC_DEV);
    if (!b || !T->begin() || !T->write(b, 0, p, bytes) || !T->submit(1)) { printf("  FAIL upload\n"); exit(1); }
    return b;
}

int main(int argc, char **argv) {
    const int ctx = argc > 1 ? atoi(argv[1]) : 262144;
    const int S = argc > 2 ? atoi(argv[2]) : 1024;
    const int reps = argc > 3 ? atoi(argv[3]) : 3;
    const int kvtype = argc > 4 ? atoi(argv[4]) : 3;
    const int H = argc > 5 ? atoi(argv[5]) : 24, KVH = 8, hd = 128, dr = 16, ext = 1024;
    const int nrows = S > 0 ? S : 1;
    if (H % KVH) { printf("bench: %d heads over %d KV heads\n", H, KVH); return 1; }
    const int pos_base = S > 0 ? ctx - S : ctx;
    T = coli_cuda_chain_ops();
    if (!T || !T->init(0) || !T->ready()) { printf("bench: no CUDA chain\n"); return 1; }

    const size_t rb = ink_kv_row_bytes(kvtype, hd);
    const size_t kvm = (size_t)KVH * (size_t)ctx * rb;
    printf("bench: ctx %d, %s, S %d, %d KV heads, %d heads, hd %d, mirror %.1f MiB a side\n",
           ctx, ink_kv_name(kvtype), S, KVH, H, hd, kvm / 1048576.0);
    std::vector<float> kf((size_t)hd), vf((size_t)hd);
    std::vector<unsigned char> kt(kvm, 0), vt(kvm, 0);
    for (int h = 0; h < KVH; h++)
        for (int t = 0; t < ctx; t++) {
            for (int d = 0; d < hd; d++) { kf[d] = frnd(); vf[d] = frnd(); }
            ink_kv_put(kvtype, &kt[((size_t)h * ctx + t) * rb], kf.data(), hd);
            ink_kv_put(kvtype, &vt[((size_t)h * ctx + t) * rb], vf.data(), hd);
        }
    CcBuf *bk = upb(kt.data(), kt.size()), *bv = upb(vt.data(), vt.size());
    std::vector<unsigned char>().swap(kt); std::vector<unsigned char>().swap(vt);

    std::vector<float> q((size_t)nrows * H * hd), r((size_t)nrows * H * dr), relp((size_t)dr * ext), tau(16 + (size_t)nrows * H * 2);
    for (auto &x : q) x = frnd();
    for (auto &x : r) x = frnd() * 0.5f;
    for (auto &x : relp) x = frnd() * 0.5f;
    for (int s = 0; s < nrows; s++) tau[16 + s] = 1.f + (rnd() % 1000) / 2000.f;
    const int toff = 8, soff = 16 + nrows + 2;   /* tau at toff, the sink logits at soff */
    for (int s = 0; s < nrows * H; s++) tau[soff + s] = frnd() * 3.f;
    std::vector<float> scr((size_t)nrows * KVH * hd * 2), obuf((size_t)nrows * H * hd);
    CcBuf *bq = upf(q), *brr = upf(r), *bp = upf(relp), *bt = upf(tau), *bs = upf(scr);
    CcBuf *bo = T->buf((size_t)nrows * H * hd * 4, CC_DEV);
    CcRelAttn p = {S > 0 ? S : 1, H, KVH, hd, pos_base, ctx, 0, ext, dr, 0, H * hd, 0, H * hd, 0, 0,
                   (int)((size_t)nrows * KVH * hd), (int)((size_t)nrows * KVH * hd * 2), KVH * hd,
                   0, H * dr, 0, toff, (float)(1.0 / sqrt((double)hd)), 1, soff};
    p.k_off = p.v_off = 0;
    p.k_rowb = (int)rb;
    p.kv_type = kvtype;

    /* the causal pairs the chunk attends to: the whole prefix, plus its own rows up to s */
    double pairs = 0;
    for (int s = 0; s < nrows; s++) pairs += (double)pos_base + s + 1;
    const double flop = 2.0 * pairs * H * hd * 2.0;
    /* the mirror rows the tiled prompt kernel reads: a block a (KV head, 8 rows) */
    const double reads = S > 0 ? (double)KVH * ((S + 7) / 8) * pos_base * 2.0 * rb : 0;

    double best = 1e30, sum = 0; int ok = 1;
    for (int i = 0; i < reps; i++) {
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        ok = T->begin() && T->relattn(bq, bk, bv, bo, bs, brr, bp, bt, &p) && T->submit(1);
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        cudaEventDestroy(a); cudaEventDestroy(b);
        if (!ok) { printf("  FAIL relattn\n"); return 1; }
        if (ms < best) best = ms;
        sum += ms;
    }
    printf("  %-28s %8.2f ms (mean %8.2f)  %7.1f GFLOP/s  ", S > 0 ? "prefill chunk" : "decode step", best, sum / reps,
           flop / (best * 1e-3) / 1e9);
    if (reads > 0) printf("%6.1f GB/s mirror  %.2f MiB read", reads / (best * 1e-3) / 1e9, reads / 1048576.0);
    else printf("%7.1f MiB x %d heads walked", (double)ctx * KVH * 2.0 * rb / 1048576.0, H);
    printf("\n");
    (void)obuf;
    return 0;
}
