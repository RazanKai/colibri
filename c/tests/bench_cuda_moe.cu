/* The one-layer prompt-MoE microbench of PLAN-colibri.md §15.10: what one real prompt layer-step
 * costs on the GPU (the routed experts' block tiles + the cold set's upload), against the host's
 * measured 56-70 ms a layer-step (127.0 s of expert-mm over 63 chunks x 36 steps at 64K; 159.1 s
 * of chain-reported routed experts).
 *
 *   make -C c cuda-moe-bench CUDA=1 CUDA_HOME=/opt/cuda
 *   ./c/cuda_moe_bench [residency %] [steps] [overlap] [contend]
 *
 * residency 75 (2,296 of 3,072 experts, the 64K tier) or 56 (1,718, the 250K tier); steps 1 or 36
 * (24 layer-steps + the 12 loop layers twice more, each layer its own routing and format); overlap
 * 1 prefetches the next layer-step's cold set on the chain's copy stream while this one's tiles run;
 * contend 1 runs a ~1 GiB KV-mirror upload on a second stream across the same window.
 *
 * The table is synthetic, in the snapshot's real per-layer formats (gate/up IQ2_S 122 or IQ3_XXS 118,
 * down NVFP4 140 or IQ4_NL 120 - parsed from Inkling-2-Mini-UD-IQ3_XXS.gguf's tensor header, see
 * LAYER_GU/LAYER_DN), real dimensions (D 2304, I 1152, E 128, top-8), rows of the snapshot's
 * .coli_usage histogram per layer (ranked, not uniform 64 an expert). Only the resident experts
 * are held on the device; each layer-step uploads the layer's cold experts through the chain's
 * staging path (cc_fetch: the tier's RAM cache to a DEV slot on the copy stream) and points their
 * table entries into that staging, which is what a GPU prompt path has to do. Every buffer's bytes,
 * the device's free VRAM and the equivalent tier budget are printed.
 *
 * Provenance: every ms, GiB and TFLOP/s it prints is measured; the projection to 64K/250K seconds is
 * arithmetic on the measured host expert phases (PLAN §15.4/§14.9) and is printed as derived.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <cuda_runtime.h>
#include "../backend_cuda.h"
#include "../cuda_chain.h"
#include "../blockdec_host.h"

static const ColiCudaChainOps *T;
static const int D = 2304, I = 1152, Sdef = 1024, K = 8, NC = 24, E = 128;
/* The snapshot's per-layer expert formats (fmt = 100 + the GGUF block type), parsed from
 * Inkling-2-Mini-UD-IQ3_XXS.gguf's tensor header: ffn_gate_up_exps.weight and ffn_down_exps.weight.
 * 116 IQ2_XXS (L0), 117 IQ2_XS (L4), 122 IQ2_S, 118 IQ3_XXS (the loop layers), 140 NVFP4 (down,
 * L0-19), 120 IQ4_NL (down, L20-23). */
static const int LAYER_GU[NC] = {116, 122, 122, 122, 117, 122, 122, 122, 122, 118, 118, 118,
                                 122, 118, 118, 118, 118, 118, 122, 118, 118, 118, 118, 118};
static const int LAYER_DN[NC] = {140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140,
                                 140, 140, 140, 140, 140, 140, 140, 140, 120, 120, 120, 120};

static unsigned g_rng = 0x9e3779b9u;
static unsigned rnd(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5; return g_rng; }
static float frnd(void) { return ((int)(rnd() % 20001) - 10000) / 10000.f; }
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double free_gib(void) { size_t f = 0, tot = 0; cudaMemGetInfo(&f, &tot); return f / 1073741824.0; }
/* a row of a GGUF block format of n weights, padded to whole words (bd_tensor_upload's pitch) */
static size_t rowb(int fmt, int n) {
    const int t = fmt - 100;
    return ((size_t)n / BD_BLK[t] * BD_BYTES[t] + 3) / 4 * 4;
}
static size_t expert_bytes(int l) {   /* gate's I rows and up's I rows (of D inputs), down's D rows (of I inputs) */
    return 2 * (size_t)I * rowb(LAYER_GU[l], D) + (size_t)D * rowb(LAYER_DN[l], I);
}
/* valid-looking random blocks: clear bits 13-14 of every 16-bit word so no fp16 scale/offset is inf
 * or NaN (tests/test_expert_group_rows_cuda.cu's trick); the 4-bit blocks stay arbitrary */
static void fill_blocks(unsigned char *p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = (unsigned char)rnd();
    for (size_t i = 0; i + 1 < n; i += 2) p[i + 1] &= (unsigned char)~0x60u;
}
__global__ void bench_spin_k(float *p, int n, int it) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = p[i];
    for (int k = 0; k < it; k++) x = sinf(x * 1.0001f) + 1e-6f;
    p[i] = x;
}
static CcBuf *dev(size_t bytes) {
    CcBuf *b = T->buf(bytes ? bytes : 4, CC_DEV);
    if (!b) { printf("moe-bench: FAIL %zu bytes on the device\n", bytes); exit(1); }
    return b;
}

static double g_res_share = 0.75, g_cold_mib = 0;

/* ---- the routing history ---- */
static std::vector<double> usage_[NC];          /* per layer, per expert count */
static int load_usage(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { printf("moe-bench: no history at %s\n", path); return 0; }
    for (int l = 0; l < NC; l++) usage_[l].assign(E, 0.0);
    char line[256];
    int nl = 0, nrec = 0, ne = 0;
    while (fgets(line, sizeof line, f)) {
        int l, e; unsigned long long c;
        if (sscanf(line, "%d %d %llu", &l, &e, &c) != 3) continue;
        if (l < 0) { if (l == -1) { nl = e; } continue; }
        if (l >= NC || e < 0 || e >= E) continue;
        usage_[l][e] = (double)c; nrec++;
    }
    fclose(f);
    if (nl && nl != NC) printf("moe-bench: the history has %d layers, the model %d\n", nl, NC);
    printf("moe-bench: history %s: %d records, %d layers x %d experts\n", path, nrec, NC, E);
    (void)ne;
    return 1;
}

int main(int argc, char **argv) {
    const int pct = argc > 1 ? atoi(argv[1]) : 75;
    const int nsteps_in = argc > 2 ? atoi(argv[2]) : 36;
    const int overlap = argc > 3 ? atoi(argv[3]) : 1;
    const int contend = argc > 4 ? atoi(argv[4]) : 0;
    const char *upath = getenv("MOEB_USAGE");
    if (!upath || !*upath) upath = "/run/media/nazar/Gaaaymes/models/Inkling-2-Mini-GGUF/iq3xxs.colibri/.coli_usage";
    const int mirror_mib = getenv("MOEB_MIRROR") ? atoi(getenv("MOEB_MIRROR")) : 1024;
    const int reps = getenv("MOEB_REPS") ? atoi(getenv("MOEB_REPS")) : 3;

    T = coli_cuda_chain_ops();
    int devs[1] = {0};
    if (!coli_cuda_init(devs, 1)) { printf("moe-bench: no CUDA device\n"); return 1; }
    if (!T || !T->init(0) || !T->ready()) { printf("moe-bench: no CUDA chain\n"); return 1; }
    const double free0 = free_gib();
    if (!load_usage(upath)) return 1;

    /* ---- residency: the global top G experts of the history, as the tier's warm start fills ---- */
    const int G = getenv("MOEB_N") ? atoi(getenv("MOEB_N")) : (int)((double)3072 * pct / 100.0 + 0.5);
    std::vector<std::pair<double, int> > rank;
    rank.reserve((size_t)NC * E);
    for (int l = 0; l < NC; l++) for (int e = 0; e < E; e++) rank.push_back(std::make_pair(-usage_[l][e], l * E + e));
    std::sort(rank.begin(), rank.end());
    std::vector<char> resid((size_t)NC * E, 0);
    for (int i = 0; i < G && i < (int)rank.size(); i++) resid[rank[i].second] = 1;
    std::vector<int> nres(NC, 0), ncold(NC, 0);
    size_t res_bytes = 0, cold_host_bytes = 0, max_cold_bytes = 0;
    for (int l = 0; l < NC; l++) {
        for (int e = 0; e < E; e++) { if (resid[l * E + e]) nres[l]++; else ncold[l]++; }
        res_bytes += (size_t)nres[l] * expert_bytes(l);
        const size_t cb = (size_t)ncold[l] * expert_bytes(l);
        cold_host_bytes += cb;
        if (cb > max_cold_bytes) max_cold_bytes = cb;
    }
    printf("moe-bench: residency %d%%: %d of 3,072 experts resident (%.1f a layer), %.2f GiB of weights "
           "(tier budget %.2f GiB = %d experts), free VRAM %.2f GiB at the start\n",
           pct, G, (double)G / NC, res_bytes / 1073741824.0, res_bytes / 1073741824.0, G, free0);

    /* ---- device: the resident table, the staging banks, the KV mirror, the step's buffers ---- */
    CcBuf *tabres = dev(res_bytes);
    const size_t nt = (size_t)NC * E;
    const size_t esc_off = (nt * sizeof(CcExpert) + 255) & ~(size_t)255;
    CcBuf *tab[3];   /* two staging banks' tables and one of the residents only: each written once, so a
                      * step carries no pageable host-to-device copy on the compute stream */
    for (int b = 0; b < 3; b++) tab[b] = dev(esc_off + nt * 2 * sizeof(float));
    CcBuf *stg[2] = {dev(max_cold_bytes), dev(max_cold_bytes)};
    CcBuf *kvm = dev((size_t)mirror_mib << 20);
    const size_t hidf = cc_moe_hid_floats(Sdef, K, D, I, E);
    CcBuf *bx = dev((size_t)Sdef * D * 4), *bids = dev((size_t)Sdef * K * 4), *bw = dev((size_t)Sdef * K * 4),
          *bhid = dev(hidf * 4), *by = dev((size_t)Sdef * D * 4), *blg = dev((size_t)Sdef * E * 4);
    printf("moe-bench: buffers: x+y %.1f MiB, ids+w %.1f MiB, hid %.1f MiB (%llu floats), logits %.1f MiB; "
           "staging %.1f MiB x2; table %.1f KiB; KV mirror %d MiB; free VRAM %.2f GiB after the table\n",
           2.0 * (size_t)Sdef * D * 4 / 1048576.0, 2.0 * (size_t)Sdef * K * 4 / 1048576.0, hidf * 4 / 1048576.0,
           (unsigned long long)hidf, (double)((size_t)Sdef * E * 4) / 1048576.0, max_cold_bytes / 1048576.0,
           nt * sizeof(CcExpert) / 1024.0, mirror_mib, free_gib());

    /* the cold source: one page-locked buffer of valid blocks, the tier's RAM cache as the source of
     * copies (each cold expert's gate, up and down read from their offsets in it) */
    unsigned char *cold_host = NULL;
    if (cudaHostAlloc((void **)&cold_host, max_cold_bytes, cudaHostAllocPortable) != cudaSuccess) {
        printf("moe-bench: no %zu MiB of page-locked host memory for the cold set\n", max_cold_bytes >> 20);
        return 1;
    }
    fill_blocks(cold_host, max_cold_bytes);
    size_t max_eb = 0;
    for (int l = 0; l < NC; l++) if (expert_bytes(l) > max_eb) max_eb = expert_bytes(l);
    unsigned char *one = NULL;
    cudaHostAlloc((void **)&one, max_eb, cudaHostAllocPortable);
    fill_blocks(one, max_eb);

    /* ---- the table ---- */
    std::vector<CcExpert> th[3];
    for (int b = 0; b < 3; b++) th[b].assign(nt, CcExpert());
    std::vector<float> esh(nt * 2, 1.f);
    std::vector<size_t> slot((size_t)NC * E, 0);
    {
        size_t off = 0;
        for (int l = 0; l < NC; l++) {
            const size_t rbg = rowb(LAYER_GU[l], D);
            for (int e = 0; e < E; e++) {
                const int x = l * E + e;
                esh[2 * x] = 0.5f + frnd() * 0.1f;         /* per-expert scales, as the GGUF's .scale rows hold them */
                esh[2 * x + 1] = 0.5f + frnd() * 0.1f;
                if (!resid[x]) continue;
                char *base = (char *)cc_dev_addr(tabres) + off;
                for (int b = 0; b < 3; b++) { th[b][x].g = base; th[b][x].u = base + (size_t)I * rbg; th[b][x].d = base + 2 * (size_t)I * rbg; }
                slot[x] = off;
                off += expert_bytes(l);
            }
        }
        for (int l = 0; l < NC; l++) {                     /* the initial fill of the residents (untimed) */
            const size_t rbg = rowb(LAYER_GU[l], D), rbd = rowb(LAYER_DN[l], I);
            for (int e = 0; e < E; e++) {
                const int x = l * E + e;
                if (!resid[x]) continue;
                cc_write(tabres, slot[x] / 4, one, (size_t)I * rbg * 2 + (size_t)D * rbd);
            }
        }
        if (!cc_submit(1)) { printf("moe-bench: the resident fill failed\n"); return 1; }
    }
    /* the steps: 24 layer-steps, then the 12 loop layers (6..17) twice more */
    const int nst = nsteps_in == 1 ? 1 : 36;
    std::vector<int> seq(nst);
    /* one layer-step: a loop layer (IQ2_S + NVFP4 down) unless MOEB_LAYER names another, e.g. 20
     * (IQ3_XXS + IQ4_NL down) to measure one down decoder against the other */
    if (nst == 1) { const char *e = getenv("MOEB_LAYER"); seq[0] = e && *e ? atoi(e) : 6;
                    if (seq[0] < 0 || seq[0] >= NC) seq[0] = 6; }
    else { for (int i = 0; i < NC; i++) seq[i] = i;
           for (int i = NC; i < nst; i++) seq[i] = 6 + (i - NC) % 12; }

    /* the routing drawn per step from the layer's histogram, as logits for cc_route */
    std::vector<std::vector<float> > lg(nst);
    for (int i = 0; i < nst; i++) {
        const int l = seq[i];
        double tot = 0; for (int e = 0; e < E; e++) tot += usage_[l][e];
        if (tot <= 0) tot = 1;
        lg[i].resize((size_t)Sdef * E);
        for (int s = 0; s < Sdef; s++)
            for (int e = 0; e < E; e++) {
                /* log(share) plus a Gumbel's noise: the top-8 of these logits is a draw of 8 experts
                 * without replacement from the layer's histogram (the Gumbel-max trick), so the row
                 * histogram the tiles see is the history's own, not a uniform 64 an expert */
                const double sh = (usage_[l][e] + 1.0) / (tot + E);
                const double u1 = (rnd() + 0.5) / 4294967296.0;
                lg[i][(size_t)s * E + e] = (float)(log(sh) - log(-log(u1)));
            }
    }
    {   /* the step's x */
        std::vector<float> xh((size_t)Sdef * D);
        for (size_t i = 0; i < xh.size(); i++) xh[i] = frnd();
        if (!cc_begin() || !cc_write(bx, 0, xh.data(), xh.size() * 4) || !cc_submit(1)) { printf("moe-bench: x upload failed\n"); return 1; }
    }
    CcMoe mp;
    memset(&mp, 0, sizeof mp);
    mp.S = Sdef; mp.K = K; mp.D = D; mp.I = I; mp.act = 1; mp.parts = 3;
    mp.esc_off = (int)esc_off; mp.act_a = 25.f; mp.act_b = 4.f;
    mp.x_row = D; mp.i_row = K; mp.w_row = K; mp.y_row = D; mp.h_off = -1; mp.E = E;
    CcRoute rp;
    memset(&rp, 0, sizeof rp);
    rp.S = Sdef; rp.E = E; rp.K = K; rp.kind = 0; rp.l_row = E; rp.i_row = K; rp.w_row = K;
    rp.norm = 1; rp.temp = 1.f; rp.scale = 1.f;

    /* the cold fetch's regions for layer l into bank b */
    auto fetch_cold = [&](int i, int bank, int which) {   /* which 0 = this step's layer, 1 = the next */
        const int l = seq[i + (which ? 1 : 0)];
        const size_t eb = expert_bytes(l), rbg = rowb(LAYER_GU[l], D), rbd = rowb(LAYER_DN[l], I);
        static std::vector<CcFetch> fr;    /* reused, the engine's CcFetch array is the caller's */
        fr.clear();
        size_t j = 0;
        for (int e = 0; e < E; e++) {
            if (resid[l * E + e]) continue;
            const char *src = (const char *)cold_host + j * eb;
            CcFetch a = {src, j * eb, (size_t)I * rbg};
            CcFetch b2 = {src + (size_t)I * rbg, j * eb + (size_t)I * rbg, (size_t)I * rbg};
            CcFetch c = {src + 2 * (size_t)I * rbg, j * eb + 2 * (size_t)I * rbg, (size_t)D * rbd};
            fr.push_back(a); fr.push_back(b2); fr.push_back(c);
            j++;
        }
        if (fr.empty()) return 1;          /* a layer all of whose experts are resident */
        return cc_fetch_at(stg[bank], fr.data(), (int)fr.size(), bank);
    };
    auto point_cold = [&](int l, int b, int stg_bank) {   /* table b's cold entries -> staging bank stg_bank */
        const size_t rbg = rowb(LAYER_GU[l], D);
        size_t j = 0;
        for (int e = 0; e < E; e++) {
            const int x = l * E + e;
            if (resid[x]) continue;
            char *base = (char *)cc_dev_addr(stg[stg_bank]) + j * expert_bytes(l);
            th[b][x].g = base; th[b][x].u = base + (size_t)I * rbg; th[b][x].d = base + 2 * (size_t)I * rbg;
            j++;
        }
    };
    float *lgp = NULL;                                     /* the logits, page-locked: an async write */
    if (cudaHostAlloc((void **)&lgp, (size_t)Sdef * E * 4, cudaHostAllocPortable) != cudaSuccess) { printf("moe-bench: no pinned logits\n"); return 1; }
    auto tiles = [&](int i, int bank, int mode) {   /* the routing and the experts of step i */
        const int l = seq[i];
        const int dbg = getenv("MOEB_DEBUG") != NULL;
        mp.fmt = LAYER_GU[l]; mp.fmt_d = LAYER_DN[l]; mp.t_off = l * E;
        memcpy(lgp, lg[i].data(), lg[i].size() * 4);
        const int b = cc_write(blg, 0, lgp, lg[i].size() * 4);
        const int c = b && cc_route(blg, NULL, bids, bw, &rp);
        const int d = c && cc_moe(tab[bank], bx, bids, bw, bhid, by, NULL, &mp);
        if (dbg && !d) printf("moe-bench: step %d layer %d: logits %d route %d moe %d (fmt %d/%d)\n", i, l, b, c, d, mp.fmt, mp.fmt_d);
        (void)mode;
        return d;
    };

    struct Cell { const char *name; double ms; };
    std::vector<Cell> cells;

    /* ---- the cold set's rows: the drawn histogram, the top-40 share, the tile padding ---- */
    {
        std::vector<int> ids((size_t)Sdef * K);
        double pad64 = 0, pad128 = 0, act = 0, top40 = 0, resrows = 0;
        for (int i = 0; i < nst; i++) {
            const int l = seq[i];
            mp.fmt = LAYER_GU[l]; mp.fmt_d = LAYER_DN[l]; mp.t_off = l * E;
            if (!cc_begin() || !cc_write(blg, 0, lg[i].data(), lg[i].size() * 4) || !cc_route(blg, NULL, bids, bw, &rp) ||
                !cc_submit(1) || !cc_read(bids, 0, ids.data(), ids.size() * 4)) { printf("moe-bench: route read failed\n"); return 1; }
            std::vector<int> c(E, 0);
            for (int s = 0; s < Sdef; s++) for (int k = 0; k < K; k++) { const int e = ids[(size_t)s * K + k]; if (e >= 0 && e < E) c[e]++; }
            std::vector<int> sc = c;
            std::sort(sc.begin(), sc.end(), std::greater<int>());
            int t64 = 0, t128 = 0, a = 0, t40 = 0, rr = 0;
            for (int e = 0; e < E; e++) { t64 += (c[e] + 63) / 64 * 64; t128 += (c[e] + 127) / 128 * 128; a += c[e]; if (resid[l * E + e]) rr += c[e]; }
            for (int e = 0; e < 40; e++) t40 += sc[e];
            pad64 += t64 - a; pad128 += t128 - a; act += a; top40 += t40; resrows += rr;
            if (i == 0 || i == nst - 1) {
                printf("moe-bench: layer %2d rows/expert:", l);
                for (int e = 0; e < E; e++) if (c[e]) printf(" %d", c[e]);
                printf("\n");
            }
        }
        g_res_share = resrows / act;
        g_cold_mib = 0;
        for (int i = 0; i < nst; i++) g_cold_mib += (double)ncold[seq[i]] * expert_bytes(seq[i]);
        g_cold_mib /= nst * 1048576.0;
        printf("moe-bench: %d steps: %d activated rows (%.0f a step), top-40 share %.1f%%, tile padding %.2f%% at 64 rows, "
               "%.2f%% at 128; the residents carry %.1f%% of the rows, the cold set is %.1f MiB a step "
               "(%.1f MiB a 64K prefill's 2,268 steps)\n",
               nst, (int)act, act / nst, 100.0 * top40 / act, 100.0 * pad64 / act, 100.0 * pad128 / act, 100.0 * resrows / act,
               g_cold_mib, g_cold_mib * 2268.0);
    }

    const double gflop_step_all = 6.0 * (double)Sdef * K * I * D / 1e9;
    /* ---- cell: the tiles alone, all experts / residents only ---- */
    auto cell_tiles = [&](const char *name, int res_only, double flop) {
        const int tb = res_only ? 2 : 0;
        for (int l = 0; l < NC; l++) point_cold(l, tb, 0);
        if (res_only) for (size_t x = 0; x < nt; x++) if (!resid[x]) { th[tb][x].g = th[tb][x].u = th[tb][x].d = NULL; }
        auto push = [&](int b) { return cc_begin() && cc_write(tab[b], 0, th[b].data(), nt * sizeof(CcExpert)) &&
                                        cc_write(tab[b], esc_off / 4, esh.data(), nt * 2 * sizeof(float)) && cc_submit(1); };
        if (!push(tb)) { printf("moe-bench: %s: the table write failed\n", name); return Cell{name, -1}; }
        double best = 1e30, sum = 0;
        int bad = 0;
        for (int r = 0; r < reps && !bad; r++) {
            const double t0 = now_s();
            for (int i = 0; i < nst && !bad; i++) {
                if (!cc_begin() || !tiles(i, tb, 0) || !cc_submit(1)) { bad = 1; }
            }
            const double ms = (now_s() - t0) * 1e3 / nst;
            if (ms < best) best = ms;
            sum += ms;
        }
        if (bad) { printf("moe-bench: %s FAILED\n", name); return Cell{name, -1}; }

        printf("moe-bench: %-24s %8.2f ms/layer-step (mean %7.2f)  %6.2f TFLOP/s  %5.1f%% of 56 ms, %4.1f%% of 70 ms\n",
               name, best, sum / reps, flop / (best * 1e-3) / 1e3, 100.0 * best / 56.0, 100.0 * best / 70.0);
        return Cell{name, best};
    };
    /* ---- cell: the cold set's upload alone ---- */
    auto cell_upload = [&](const char *name) {
        double best = 1e30, sum = 0, b = 0;
        for (int i = 0; i < nst; i++) b += (double)ncold[seq[i]] * expert_bytes(seq[i]);
        for (int r = 0; r < reps; r++) {
            const double t0 = now_s();
            for (int i = 0; i < nst; i++) {
                if (!cc_begin() || !fetch_cold(i, i & 1, 0) || !cc_fetch_wait_at(i & 1) || !cc_submit(1)) {
                    printf("moe-bench: %s FAILED\n", name); return Cell{name, -1};
                }
            }
            const double ms = (now_s() - t0) * 1e3;
            if (ms < best) best = ms;
            sum += ms;
        }
        const double ms_step = best / nst, gib = b / 1073741824.0;
        printf("moe-bench: %-24s %8.2f ms/layer-step (%.1f MiB a step) %6.2f GiB a sequence, %6.2f GB/s, %.2f ms/GiB\n",
               name, ms_step, b / nst / 1048576.0, gib, gib / (best * 1e-3), best / gib);
        (void)sum;
        return Cell{name, ms_step};
    };
    /* ---- cells: the sequence, serial / overlapped / contended ---- */
    auto cell_seq = [&](const char *name, int ov, int cont) {
        double best = 1e30, sum = 0;
        cudaStream_t cs = NULL;
        void *dmirror = NULL;
        unsigned char *hmirror = NULL;
        if (cont) {
            if (cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking) != cudaSuccess) { printf("moe-bench: no stream\n"); return Cell{name, -1}; }
            dmirror = (void *)cc_dev_addr(kvm);
            if (cudaHostAlloc((void **)&hmirror, (size_t)mirror_mib << 20, cudaHostAllocPortable) != cudaSuccess) {
                printf("moe-bench: no %d MiB pinned for the mirror\n", mirror_mib); return Cell{name, -1};
            }
            fill_blocks(hmirror, (size_t)mirror_mib << 20);
        }
        int bad = 0;
        for (int r = 0; r < reps && !bad; r++) {
            int bank = 0;
            /* both tables written once (each bank's cold entries fixed), then the first step's bank
               filled before the sequence (untimed: a prefill's first chunk pays it once) */
            for (int b = 0; b < 2; b++) {
                for (int l = 0; l < NC; l++) point_cold(l, b, b);
                if (!cc_begin() || !cc_write(tab[b], 0, th[b].data(), nt * sizeof(CcExpert)) ||
                    !cc_write(tab[b], esc_off / 4, esh.data(), nt * 2 * sizeof(float)) || !cc_submit(1)) { bad = 1; break; }
            }
            if (bad || !cc_begin() || !fetch_cold(0, 0, 0) || !cc_fetch_wait_at(0) || !cc_submit(1)) { bad = 1; break; }
            const double t0 = now_s();
            for (int i = 0; i < nst && !bad; i++) {
                if (cont) cudaMemcpyAsync(dmirror, hmirror, (size_t)mirror_mib << 20, cudaMemcpyHostToDevice, cs);
                if (!cc_begin()) { bad = 1; break; }
                if (ov) {
                    if (!cc_fetch_wait_at(bank)) { bad = 1; break; }                     /* this bank's data */
                    if (i + 1 < nst && !fetch_cold(i, 1 - bank, 1)) { bad = 1; break; }  /* the next step's, on the copy stream */
                } else if (!fetch_cold(i, bank, 0) || !cc_fetch_wait_at(bank)) { bad = 1; break; }
                if (!tiles(i, bank, 0)) { bad = 1; break; }
                if (ov && !cc_fetch_done_at(bank)) { bad = 1; break; }                    /* this bank's reader is done */
                if (!cc_submit(1)) { bad = 1; break; }
                if (cont) cudaStreamSynchronize(cs);
                bank = 1 - bank;
            }
            const double ms = (now_s() - t0) * 1e3 / nst;
            if (ms < best) best = ms;
            sum += ms;
        }
        if (cs) cudaStreamDestroy(cs);
        if (hmirror) cudaFreeHost(hmirror);
        if (bad) { printf("moe-bench: %s FAILED\n", name); return Cell{name, -1}; }
        double b = 0;
        for (int i = 0; i < nst; i++) b += (double)ncold[seq[i]] * expert_bytes(seq[i]);
        printf("moe-bench: %-24s %8.2f ms/layer-step (mean %7.2f)  %.1f MiB uploaded a step%s\n",
               name, best, sum / reps, b / nst / 1048576.0, cont ? ", + a 1 GiB mirror copy a step (issued, not waited)" : "");
        return Cell{name, best};
    };

    printf("moe-bench: S %d, top-%d, D %d, I %d, E %d; %d steps; overlap %d, contend %d, %d reps, act SiTU-GLU (25, 4)\n",
           Sdef, K, D, I, E, nst, overlap, contend, reps);
    /* bank 0 filled once with real blocks, so the tiles-only cells read real data; then one untimed
     * pass, so a cell's first number is not the first call's kernel module loads */
    if (!cc_begin() || !fetch_cold(0, 0, 0) || !cc_fetch_wait_at(0) || !cc_submit(1)) { printf("moe-bench: first fetch failed\n"); return 1; }
    for (int l = 0; l < NC; l++) point_cold(l, 0, 0);
    { int okw = cc_begin() && cc_write(tab[0], 0, th[0].data(), nt * sizeof(CcExpert)) &&
                        cc_write(tab[0], esc_off / 4, esh.data(), nt * 2 * sizeof(float)) && cc_submit(1);
      for (int i = 0; i < nst && okw; i++) okw = cc_begin() && tiles(i, 0, 0) && cc_submit(1);
      if (!okw) { printf("moe-bench: the warmup pass failed\n"); return 1; } }
    const Cell tiles_all = cell_tiles("tiles, all experts", 0, gflop_step_all);
    /* the residents' share of the tiles: their rows' products (6*I*D a row-slot) */
    const Cell tiles_res = cell_tiles("tiles, residents only", 1, gflop_step_all * (double)G / 3072.0);
    const Cell up = cell_upload("cold upload alone");
    const Cell ser = cell_seq("sequence, serial", 0, 0);
    Cell ovc = Cell{"sequence, overlapped", -1};
    if (overlap) ovc = cell_seq("sequence, overlapped", 1, 0);
    Cell con = Cell{"sequence, contended", -1};
    if (contend) con = cell_seq("sequence, contended", 1, 1);

    /* a 1 GiB pinned->device copy on its own, the mirror upload's price when nothing else runs */
    {
        cudaStream_t s1;
        unsigned char *h1 = NULL;
        void *d1 = (void *)cc_dev_addr(kvm);
        if (cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking) == cudaSuccess &&
            cudaHostAlloc((void **)&h1, (size_t)mirror_mib << 20, cudaHostAllocPortable) == cudaSuccess) {
            fill_blocks(h1, (size_t)mirror_mib << 20);
            const double t0 = now_s();
            cudaMemcpyAsync(d1, h1, (size_t)mirror_mib << 20, cudaMemcpyHostToDevice, s1);
            cudaStreamSynchronize(s1);
            const double ms = (now_s() - t0) * 1e3;
            printf("moe-bench: a %d MiB KV-mirror copy alone: %.1f ms (%.2f GB/s); the tier's 64K prefill moved 10.24 GiB up\n",
                   mirror_mib, ms, mirror_mib / 1024.0 / (ms * 1e-3));
            cudaFreeHost(h1);
            cudaStreamDestroy(s1);
        }
    }

    /* ---- a raw two-stream probe: does a 91 MiB H2D copy from page-locked memory overlap a
     * compute kernel on this device at all? (the staging's overlap depends on it) ---- */
    {
        cudaStream_t a1, a2;
        unsigned char *hp = NULL;
        void *dp = (void *)cc_dev_addr(stg[0]);
        float *sp = NULL;
        const size_t nb = 91u << 20;
        const int N = 8;
        if (cudaStreamCreateWithFlags(&a1, cudaStreamNonBlocking) == cudaSuccess &&
            cudaStreamCreateWithFlags(&a2, cudaStreamNonBlocking) == cudaSuccess &&
            cudaHostAlloc((void **)&hp, nb, cudaHostAllocPortable) == cudaSuccess &&
            cudaMalloc((void **)&sp, 4u << 20) == cudaSuccess) {
            fill_blocks(hp, nb);
            cudaMemsetAsync(sp, 0, 4u << 20, a2);
            double tc = 0, tk = 0;
            for (int r = 0; r < N; r++) {
                const double t0 = now_s();
                cudaMemcpyAsync(dp, hp, nb, cudaMemcpyHostToDevice, a1);
                cudaStreamSynchronize(a1);
                tc += now_s() - t0;
            }
            for (int r = 0; r < N; r++) {
                const double t0 = now_s();
                bench_spin_k<<<(1 << 20) / 256, 256, 0, a2>>>(sp, 1 << 20, 1200);
                cudaStreamSynchronize(a2);
                tk += now_s() - t0;
            }
            double tb = 0;
            for (int r = 0; r < N; r++) {
                const double t0 = now_s();
                cudaMemcpyAsync(dp, hp, nb, cudaMemcpyHostToDevice, a1);
                bench_spin_k<<<(1 << 20) / 256, 256, 0, a2>>>(sp, 1 << 20, 1200);
                cudaStreamSynchronize(a1);
                cudaStreamSynchronize(a2);
                tb += now_s() - t0;
            }
            printf("moe-bench: raw probe: a 91 MiB copy %.2f ms, a spin kernel %.2f ms, both on two streams %.2f ms (max %.2f, "
                   "sum %.2f) -> %s\n", tc * 1e3 / N, tk * 1e3 / N, tb * 1e3 / N, (tc > tk ? tc : tk) * 1e3 / N,
                   (tc + tk) * 1e3 / N, tb < (tc + tk) * 0.85 ? "copies overlap compute" : "copies do NOT overlap compute");
            cudaFree(sp); cudaFreeHost(hp);
        } else (void)dp;
        cudaStreamDestroy(a1); cudaStreamDestroy(a2);
    }

    /* ---- a concurrency probe: the cold upload and one layer-step's tiles on two streams, no waits
     * between them but the ones the chain's staging needs ---- */
    {
        const int N = 20, l = 6;
        cudaStream_t cs;
        if (cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking) == cudaSuccess) {
            double ta = 0, tb = 0;
            for (int r = 0; r < N; r++) {          /* the copy alone */
                const double t0 = now_s();
                for (int k = 0; k < (int)ncold[l]; k++)
                    cudaMemcpyAsync((char *)cc_dev_addr(stg[0]) + (size_t)k * expert_bytes(l), cold_host + (size_t)k * expert_bytes(l),
                                    expert_bytes(l), cudaMemcpyHostToDevice, cs);
                cudaStreamSynchronize(cs);
                ta += now_s() - t0;
            }
            for (int r = 0; r < N; r++) {          /* the tiles alone */
                const double t0 = now_s();
                if (!cc_begin() || !tiles(0, 0, 0) || !cc_submit(1)) break;
                tb += now_s() - t0;
            }
            double tc = 0, tcopy = 0, tframe = 0;
            float cpyms = 0, fms = 0;
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0); cudaEventCreate(&e1);
            for (int r = 0; r < N; r++) {          /* both, the copy issued first and not waited for */
                const double t0 = now_s();
                cudaEventRecord(e0, cs);
                for (int k = 0; k < (int)ncold[l]; k++)
                    cudaMemcpyAsync((char *)cc_dev_addr(stg[1]) + (size_t)k * expert_bytes(l), cold_host + (size_t)k * expert_bytes(l),
                                    expert_bytes(l), cudaMemcpyHostToDevice, cs);
                cudaEventRecord(e1, cs);
                const double tf0 = now_s();
                if (!cc_begin() || !tiles(0, 0, 0) || !cc_submit(1)) break;
                const double tf = now_s() - tf0;
                cudaStreamSynchronize(cs);
                cudaEventElapsedTime(&cpyms, e0, e1);
                tcopy += cpyms; tframe += tf;
                tc += now_s() - t0;
            }
            printf("moe-bench: probe (layer 6, %d reps): copy alone %.2f ms, tiles alone %.2f ms, both on two streams %.2f ms "
                   "(max %.2f, sum %.2f) -> %s; inside it the copy %.2f ms and the frame's submit %.2f ms\n",
                   N, ta * 1e3 / N, tb * 1e3 / N, tc * 1e3 / N, (ta > tb ? ta : tb) * 1e3 / N, (ta + tb) * 1e3 / N,
                   tc < (ta + tb) * 0.85 ? "they overlap" : "they serialize", tcopy / N, tframe * 1e3 / N);
            cudaStreamDestroy(cs);
        }
    }

    /* ---- the decision and the projection (PLAN 15.10's arithmetic) ---- */
    const long steps64 = 63L * 36L, steps250 = 8784L;
    auto proj = [](double ms, long steps, double prefill, double host_s) { return prefill - (host_s - ms * (double)steps / 1000.0); };
    printf("moe-bench: ---- decision ----\n");
    printf("moe-bench: host layer-step measured 56.0 ms (127.0 s/2,268 at 64K) to 70.2 ms (159.1 s chain-reported); "
           "acceptance <35 ms = GPU prompt-MoE, 35-56 = better than the host, >56 = no gain\n");
    const double pick = ovc.ms > 0 ? ovc.ms : ser.ms;
    if (ser.ms > 0)
        printf("moe-bench: serial      %.2f ms -> 64K prefill %.0f s (was 248.0), 250K %.0f s (was 1,678.0; attention+other floor 1,021.8)\n",
               ser.ms, proj(ser.ms, steps64, 248.0, 127.0), proj(ser.ms, steps250, 1678.0, 658.1));
    if (ovc.ms > 0)
        printf("moe-bench: overlapped  %.2f ms -> 64K prefill %.0f s (was 248.0), 250K %.0f s (was 1,678.0; floor 1,021.8)\n",
               ovc.ms, proj(ovc.ms, steps64, 248.0, 127.0), proj(ovc.ms, steps250, 1678.0, 658.1));
    if (con.ms > 0)
        printf("moe-bench: contended   %.2f ms -> 64K prefill %.0f s, 250K %.0f s\n",
               con.ms, proj(con.ms, steps64, 248.0, 127.0), proj(con.ms, steps250, 1678.0, 658.1));
    if (tiles_res.ms > 0) {
        const double f = 1.0 - g_res_share;
        printf("moe-bench: a hybrid (device the residents, host the cold %.0f%%): device %.1f s, host %.1f s at 64K, "
               "bounded by max() = %.1f s of expert work (vs 127.0)\n",
               100 * f, tiles_res.ms * steps64 / 1000.0, 127.0 * f,
               std::max(tiles_res.ms * steps64 / 1000.0, 127.0 * f));
    }
    printf("moe-bench: free VRAM %.2f GiB at the end (%.2f at the start); %.2f GiB of resident weights, "
           "staging %.1f MiB x2, KV mirror %d MiB, buffers %.1f MiB\n",
           free_gib(), free0, res_bytes / 1073741824.0, max_cold_bytes / 1048576.0, mirror_mib, hidf * 4 / 1048576.0 + 2.0 * (size_t)Sdef * D * 4 / 1048576.0);

    const char *jpath = getenv("MOEB_JSON");
    if (jpath && *jpath) {
        FILE *j = fopen(jpath, "w");
        if (j) {
            fprintf(j, "{\n \"cell\": \"moe-bench-%d%%%s%s\",\n \"residency_pct\": %d,\n \"resident\": %d,\n"
                       " \"experts_total\": 3072,\n \"budget_gib\": %.3f,\n \"free_gib_start\": %.2f,\n"
                       " \"free_gib_end\": %.2f,\n \"table_gib\": %.3f,\n \"staging_mib\": %.1f,\n"
                       " \"mirror_mib\": %d,\n \"steps\": %d,\n \"overlap\": %d,\n \"contend\": %d,\n"
                       " \"x_bytes\": %d,\n \"y_bytes\": %d,\n \"hid_floats\": %llu,\n \"hid_bytes\": %llu,\n"
                       " \"ms_tiles_all\": %.3f,\n \"ms_tiles_resident\": %.3f,\n \"ms_upload\": %.3f,\n"
                       " \"ms_serial\": %.3f,\n \"ms_overlap\": %.3f,\n \"ms_contend\": %.3f,\n"
                       " \"gflop_step_all\": %.3f,\n \"proj64_s\": %.1f,\n \"proj250_s\": %.1f,\n"
                       " \"host_step_ms\": [56.0, 70.2],\n \"provenance\": \"measured\"\n}\n",
                    pct, ovc.ms > 0 ? "-ov" : "", con.ms > 0 ? "-cont" : "", pct, G, res_bytes / 1073741824.0, free0,
                    free_gib(), res_bytes / 1073741824.0, max_cold_bytes / 1048576.0, mirror_mib, nst, overlap, contend,
                    Sdef * D * 4, Sdef * D * 4, (unsigned long long)hidf, (unsigned long long)hidf * 4,
                    tiles_all.ms, tiles_res.ms, up.ms, ser.ms, ovc.ms, con.ms, gflop_step_all,
                    pick > 0 ? proj(pick, steps64, 248.0, 127.0) : 0.0, pick > 0 ? proj(pick, steps250, 1678.0, 658.1) : 0.0);
            fclose(j);
            printf("moe-bench: rows written to %s\n", jpath);
        }
    }
    if (cold_host) cudaFreeHost(cold_host);
    if (one) cudaFreeHost(one);
    return 0;
}
