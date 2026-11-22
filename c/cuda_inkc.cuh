/* cuda_inkc.cuh -- Inkling's and Inkling-2's ops on the CUDA chain (cuda_chain.h's sconv, relattn,
 * ink2_mix, ink2_clip and ink2_dconv), included once by cuda_chain.cu after cuda_glm53c.cuh, whose
 * grid helper it uses. Each is the Vulkan chain's op of the same name (shaders/chain_sconv.comp,
 * chain_relattn.comp, chain_ink2.comp) with the same fields in the same order; vk_chain.h has
 * their contracts:
 *   sconv      mode 0 the depthwise causal short convolution with its ring (residual inside),
 *              mode 1 x *= fc, mode 2 x /= fc over n floats
 *   relattn    grouped attention with a relative-position bias bank, a per-row tau, a sliding
 *              window over a ring cache, the step's own rows from kvs, and with has_sink a sink
 *              logit per (row, head) in tau at sink_off that starts the online softmax
 *   ink2_mix   the attention residuals' mix of RMS-normalised block states and the stream
 *   ink2_clip  the L2 clip of each segment after an optional per-element scale
 *   ink2_dconv a global layer's dynamic short convolution with its state */

__global__ static void ccg_sconv_k(float *x, const float *w, float *ring, CcSconv p) {
    const long c = (long)(blockIdx.x + blockIdx.y * gridDim.x) * 64 + threadIdx.x;
    if (p.mode != 0) {
        if (c >= p.n) return;
        const float v = x[p.x_off + c];
        x[p.x_off + c] = p.mode == 1 ? v * p.fc : v / p.fc;
        return;
    }
    if (c >= p.C) return;
    const int nh = p.CK - 1;
    float hist[8];
#pragma unroll
    for (int k = 0; k < 8; k++) hist[k] = k < nh ? ring[p.ring_off + c * nh + k] : 0.f;
    const float *wc = w + p.w_off + c * p.CK;
    const float wl = wc[p.CK - 1];
    for (int s = 0; s < p.S; s++) {
        const long xi = p.x_off + (long)s * p.x_row + c;
        const float cur = x[xi];
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < 8; k++) if (k < nh) acc += wc[k] * hist[k];
        acc += wl * cur;
        x[xi] = acc + cur;
#pragma unroll
        for (int k = 0; k < 7; k++) if (k + 1 < nh) hist[k] = hist[k + 1];
#pragma unroll
        for (int k = 0; k < 8; k++) if (k == nh - 1) hist[k] = cur;
    }
#pragma unroll
    for (int k = 0; k < 8; k++) if (k < nh) ring[p.ring_off + c * nh + k] = hist[k];
}

/* One row of the KV mirror as a float: p.kv_type (0 f32, 1 f16, 2 q8, 3 q4; ink_kv.h's layout),
 * rowp the row's first byte (or float). The typed mirrors hold the host cache's own rows, so a
 * position costs a fifth to a seventh of f32's bytes. */
__device__ __forceinline__ float ccg_kv_at(const void *rowp, int d, int type) {
    if (type == 1) return __half2float(((const __half *)rowp)[d]);
    if (type == 2) {
        const unsigned char *p = (const unsigned char *)rowp + (d >> 5) * 36;
        return (float)(signed char)p[4 + (d & 31)] * *(const float *)p;
    }
    if (type == 3) {
        const unsigned char *p = (const unsigned char *)rowp + (d >> 5) * 20;
        const int v = (p[4 + ((d & 31) >> 1)] >> ((d & 1) * 4)) & 0xF;
        return (float)((v ^ 8) - 8) * *(const float *)p;
    }
    return ((const float *)rowp)[d];
}
/* 32 values of a typed row, one codec block (ink_kv.h's layout), read as the block they are and
 * widened into out[]: the value-at-a-time reads of ccg_kv_at were one or two bytes a thread with
 * the rows 80 bytes apart, which the memory system cannot coalesce. m the values this block holds
 * (32 but a row's last one). */
__device__ __forceinline__ void ccg_kv_blk(const void *rowp, int blk, int m, int type, float *out) {
    if (type == 1) {
        const __half *h = (const __half *)rowp + blk * 32;
        for (int i = 0; i < 32; i++) out[i] = i < m ? __half2float(h[i]) : 0.f;
    } else if (type == 2) {
        const unsigned char *p = (const unsigned char *)rowp + blk * 36;
        const float s = *(const float *)p;
        unsigned w[8];                       /* 32 int8: eight words, four values a word */
        memcpy(w, p + 4, 32);
        for (int i = 0; i < 8; i++) {
            const unsigned x = w[i];
            out[i * 4 + 0] = i * 4 + 0 < m ? (float)(signed char)(x & 0xFF) * s : 0.f;
            out[i * 4 + 1] = i * 4 + 1 < m ? (float)(signed char)((x >> 8) & 0xFF) * s : 0.f;
            out[i * 4 + 2] = i * 4 + 2 < m ? (float)(signed char)((x >> 16) & 0xFF) * s : 0.f;
            out[i * 4 + 3] = i * 4 + 3 < m ? (float)(signed char)(x >> 24) * s : 0.f;
        }
    } else if (type == 3) {
        const unsigned char *p = (const unsigned char *)rowp + blk * 20;
        const float s = *(const float *)p;
        unsigned w[4];
        memcpy(w, p + 4, 16);
        for (int i = 0; i < 8; i++) {
            const unsigned x = w[i >> 1] >> ((i & 1) * 16);
            const int a = (int)((x & 0xF) ^ 8) - 8, b = (int)(((x >> 4) & 0xF) ^ 8) - 8;
            const int c = (int)(((x >> 8) & 0xF) ^ 8) - 8, d = (int)(((x >> 12) & 0xF) ^ 8) - 8;
            out[i * 4 + 0] = i * 4 + 0 < m ? (float)a * s : 0.f;
            out[i * 4 + 1] = i * 4 + 1 < m ? (float)b * s : 0.f;
            out[i * 4 + 2] = i * 4 + 2 < m ? (float)c * s : 0.f;
            out[i * 4 + 3] = i * 4 + 3 < m ? (float)d * s : 0.f;
        }
    } else {
        const float *f = (const float *)rowp + blk * 32;
        for (int i = 0; i < 32; i++) out[i] = i < m ? f[i] : 0.f;
    }
}
/* the mirror's row of (kv head, position) as ccg_kv_at sees it: a byte offset into a typed
 * cache, an element offset into an f32 one */
__device__ __forceinline__ const void *ccg_kv_row(const void *base, const CcRelAttn &p, int off, int kvh, int t) {
    const long row = (long)kvh * p.cap + t % p.cap;
    if (p.kv_type) return (const char *)base + off + row * p.k_rowb;
    return (const float *)base + off + row * p.hd;
}
/* tile b0 (128 positions from b0) of a (head, row)'s attention into its running max m, sum l and
 * weighted V (acc0: dims tid, acc1: tid + 128): chain_relattn.comp's online softmax step */
__device__ __forceinline__ void ccg_relattn_tile(const void *__restrict__ kc, const void *__restrict__ vc, const float *__restrict__ kvs,
        const float *__restrict__ relp, const CcRelAttn &p, const float *qs, const float *rs, float *pe, float *red, int *pt,
        int kvh, int qpos, int t0, int n, float ts, int b0, int tid, float &m, float &l, float &acc0, float &acc1) {
    const int i = b0 + tid;
    float sv = -3.0e38f;
    if (i < n) {
        const int t = t0 + i;
        pt[tid] = t;
        float a = 0.f;
        if (t < p.pos_base) {
            const void *krow = ccg_kv_row(kc, p, p.k_off, kvh, t);
            if (!p.kv_type) {   /* f32: the original loop, so nvcc vectorizes it as it always did */
                const float *kr = (const float *)krow;
                for (int d = 0; d < p.hd; d++) a += qs[d] * kr[d];
            } else for (int d = 0; d < p.hd; d++) a += qs[d] * ccg_kv_at(krow, d, p.kv_type);
        } else { const float *kr = kvs + p.ks_off + (long)(t - p.pos_base) * p.kv_row + kvh * p.hd; for (int d = 0; d < p.hd; d++) a += qs[d] * kr[d]; }
        const int dist = qpos - t;
        float bias = 0.f;
        if (dist < p.ext) for (int j = 0; j < p.d_rel; j++) bias += rs[j] * relp[p.relp_off + j * p.ext + dist];
        sv = ts * (a * p.scale + bias);
    }
    red[tid] = sv;
    __syncthreads();
    for (int off = 64; off > 0; off >>= 1) { if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]); __syncthreads(); }
    const float mn = fmaxf(m, red[0]);
    __syncthreads();
    const float e = i < n ? expf(sv - mn) : 0.f;
    pe[tid] = e; red[tid] = e;
    __syncthreads();
    for (int off = 64; off > 0; off >>= 1) { if (tid < off) red[tid] += red[tid + off]; __syncthreads(); }
    const float corr = expf(m - mn);
    l = l * corr + red[0];
    acc0 *= corr; acc1 *= corr;
    const int cnt = min(128, n - b0), d0 = tid, d1 = tid + 128;
    for (int j = 0; j < cnt; j++) {
        const int t = pt[j];
        const float w = pe[j];
        if (t < p.pos_base) {
            const void *vrow = ccg_kv_row(vc, p, p.v_off, kvh, t);
            if (!p.kv_type) {
                const float *vr = (const float *)vrow;
                if (d0 < p.hd) acc0 += w * vr[d0];
                if (d1 < p.hd) acc1 += w * vr[d1];
            } else {
                if (d0 < p.hd) acc0 += w * ccg_kv_at(vrow, d0, p.kv_type);
                if (d1 < p.hd) acc1 += w * ccg_kv_at(vrow, d1, p.kv_type);
            }
        } else {
            const float *vr = kvs + p.vs_off + (long)(t - p.pos_base) * p.kv_row + kvh * p.hd;
            if (d0 < p.hd) acc0 += w * vr[d0];
            if (d1 < p.hd) acc1 += w * vr[d1];
        }
    }
    m = mn;
    __syncthreads();
}
/* one block of 128 per (head, row): chain_relattn.comp's online softmax over tiles of 128 positions.
 * part (the split, a decode step's rows): block z takes tile z alone, its max, sum and weighted V into
 * part (hd + 2 floats a (row, head, tile)), and ccg_relattn_join_k joins the tiles */
__global__ static void ccg_relattn_k(const float *q, const void *kc, const void *vc, float *o, const float *kvs,
                                     const float *r, const float *relp, const float *tau, CcRelAttn p, float *part) {
    const int h = (int)blockIdx.x, s = (int)blockIdx.y, tid = (int)threadIdx.x;
    const int kvh = h / (p.H / p.KVH);
    __shared__ float qs[256], rs[64], pe[128], red[128];
    __shared__ int pt[128];
    const int qb = p.q_off + s * p.q_row + h * p.hd;
    for (int d = tid; d < p.hd; d += 128) qs[d] = q[qb + d];
    for (int j = tid; j < p.d_rel; j += 128) rs[j] = r[p.r_off + s * p.r_row + h * p.d_rel + j];
    const int qpos = p.pos_base + s;
    const int t0 = p.window > 0 ? max(0, qpos - p.window + 1) : 0;
    const int n = qpos - t0 + 1;
    const float ts = tau[p.tau_off + s];
    __syncthreads();
    float m = -3.0e38f, l = 0.f, acc0 = 0.f, acc1 = 0.f;
    if (part) {
        const int z = (int)blockIdx.z, b0 = z * 128;
        if (b0 < n) ccg_relattn_tile(kc, vc, kvs, relp, p, qs, rs, pe, red, pt, kvh, qpos, t0, n, ts, b0, tid, m, l, acc0, acc1);
        float *pp = part + (((long)s * p.H + h) * gridDim.z + z) * (p.hd + 2);
        if (tid == 0) { pp[0] = m; pp[1] = l; }
        if (tid < p.hd) pp[2 + tid] = acc0;
        if (tid + 128 < p.hd) pp[2 + tid + 128] = acc1;
        return;
    }
    if (p.has_sink) { m = tau[p.sink_off + s * p.H + h]; l = 1.f; }
    for (int b0 = 0; b0 < n; b0 += 128)
        ccg_relattn_tile(kc, vc, kvs, relp, p, qs, rs, pe, red, pt, kvh, qpos, t0, n, ts, b0, tid, m, l, acc0, acc1);
    const float inv = l > 0.f ? 1.f / l : 0.f;
    const int ob = p.o_off + s * p.o_row + h * p.hd;
    if (tid < p.hd) o[ob + tid] = acc0 * inv;
    if (tid + 128 < p.hd) o[ob + tid + 128] = acc1 * inv;
}
/* the split's join: a (head, row)'s nc tiles (and its sink) in tile order into its output */
__global__ static void ccg_relattn_join_k(const float *part, const float *tau, float *o, CcRelAttn p, int nc) {
    const int h = (int)blockIdx.x, s = (int)blockIdx.y, tid = (int)threadIdx.x;
    const float *pp = part + ((long)s * p.H + h) * nc * (p.hd + 2);
    float M = p.has_sink ? tau[p.sink_off + s * p.H + h] : -3.0e38f;
    for (int z = 0; z < nc; z++) if (pp[(long)z * (p.hd + 2) + 1] > 0.f) M = fmaxf(M, pp[(long)z * (p.hd + 2)]);
    float L = p.has_sink ? expf(tau[p.sink_off + s * p.H + h] - M) : 0.f, acc = 0.f;
    for (int z = 0; z < nc; z++) {
        const float *pz = pp + (long)z * (p.hd + 2);
        if (!(pz[1] > 0.f)) continue;
        const float w = expf(pz[0] - M);
        L += pz[1] * w;
        if (tid < p.hd) acc += pz[2 + tid] * w;
    }
    if (tid < p.hd) o[p.o_off + s * p.o_row + h * p.hd + tid] = L > 0.f ? acc / L : 0.f;
}

/* ---- the decode path, GQA-shared (COLI_CUDA_RELATTN_GQA=0: the split walk of ccg_relattn_k) ----
 * ccg_relattn_k launches a block a (query head, row) and every one of them walks, reads and
 * dequantizes the same K/V rows: the G query heads of a KV head do that G times. This kernel has
 * a block a (KV head, CCG_DTB positions): the tile's K and V are dequantized once into shared in
 * CCG_DTP-position pieces and the block's warps, one a query head, keep an online softmax over
 * them, so the mirror is read and dequantized once a KV head and a block writes one partial a
 * (row, head, tile of CCG_DTB) instead of one a 32-position piece. The pieces carry CCG_DTS rows
 * (CCG_DTP plus a decoding row's shift of up to 8), K transposed on its float4 (conflict-free),
 * and the partials are ccg_relattn_join_k's (row, head, tile) layout. */
#define CCG_DTP 32
#define CCG_DTB 128
#define CCG_DTS 40
__global__ static void ccg_relattn_dec_k(const float *__restrict__ q, const void *__restrict__ kc,
                                         const void *__restrict__ vc, const float *__restrict__ kvs,
                                         const float *__restrict__ r, const float *__restrict__ relp,
                                         const float *__restrict__ tau, CcRelAttn p, float *part, int nc) {
    const int kvh = (int)blockIdx.x, z = (int)blockIdx.y, tid = (int)threadIdx.x;
    const int wp = tid >> 5, lane = tid & 31, G = p.H / p.KVH, hd = p.hd, kd4 = hd >> 2;
    const int nthr = (int)blockDim.x, nblk = (hd + 31) >> 5;
    extern __shared__ float sm[];
    float *Kt4 = sm, *Vt = Kt4 + (size_t)kd4 * CCG_DTS * 4, *qs = Vt + (size_t)CCG_DTS * hd,
          *P = qs + (size_t)G * p.S * hd;
    const int t0_0 = p.window > 0 ? (p.pos_base - p.window + 1 > 0 ? p.pos_base - p.window + 1 : 0) : 0;
    const int tb = t0_0 + z * CCG_DTB;
    if (z * CCG_DTB >= (p.window > 0 ? p.window : p.pos_base + p.S)) return;
    for (int idx = tid; idx < G * p.S * hd; idx += nthr) {
        const int g = idx / (p.S * hd), rem = idx - g * p.S * hd, s2 = rem / hd, d = rem - s2 * hd;
        qs[idx] = q[p.q_off + (long)s2 * p.q_row + (long)(kvh * G + g) * hd + d];
    }
    __syncthreads();
    /* every warp stays for the tile's dequantization: returning the spare ones here left the
     * pieces' last rows to a stride no live thread walked */
    const int g = wp, h = kvh * G + g, d0 = lane * 4;
    const bool mine = wp < G;
    const int nmax = p.window > 0 ? p.window : p.pos_base + p.S;
    float m[8], l[8], acc[8][4];
    for (int s = 0; s < 8; s++) {
        m[s] = -3.0e38f; l[s] = 0.f;
        acc[s][0] = acc[s][1] = acc[s][2] = acc[s][3] = 0.f;
    }
    for (int sub = 0; sub < CCG_DTB / CCG_DTP; sub++) {
        const int stb = tb + sub * CCG_DTP;
        if (stb - t0_0 >= nmax) break;              /* no served position left in the block */
        for (int idx = tid; idx < CCG_DTS * nblk; idx += nthr) {
            const int pos = idx / nblk, blk = idx - pos * nblk, t = stb + pos;
            float kb[32], vb[32];
            if (t < p.pos_base) {
                const int mm = hd - blk * 32 < 32 ? hd - blk * 32 : 32;
                ccg_kv_blk(ccg_kv_row(kc, p, p.k_off, kvh, t), blk, mm, p.kv_type, kb);
                ccg_kv_blk(ccg_kv_row(vc, p, p.v_off, kvh, t), blk, mm, p.kv_type, vb);
            } else if (t < p.pos_base + p.S) {
                const long o = (long)(t - p.pos_base) * p.kv_row + (long)kvh * hd + blk * 32;
                for (int i = 0; i < 32 && blk * 32 + i < hd; i++) { kb[i] = kvs[p.ks_off + o + i]; vb[i] = kvs[p.vs_off + o + i]; }
            } else for (int i = 0; i < 32; i++) kb[i] = vb[i] = 0.f;
            for (int i = 0; i < 32 && blk * 32 + i < hd; i += 4) {
                const int d4 = (blk * 32 + i) >> 2;
                *(float4 *)(Kt4 + ((size_t)d4 * CCG_DTS + pos) * 4) = make_float4(kb[i], kb[i + 1], kb[i + 2], kb[i + 3]);
                *(float4 *)(Vt + (size_t)pos * hd + d4 * 4) = make_float4(vb[i], vb[i + 1], vb[i + 2], vb[i + 3]);
            }
        }
        __syncthreads();
        if (mine) {
            #pragma unroll
            for (int s = 0; s < 8; s++) {
                if (s >= p.S) continue;       /* a constant bound keeps m/l/acc in registers */
                const int qpos = p.pos_base + s;
                const int t0 = p.window > 0 ? (qpos - p.window + 1 > 0 ? qpos - p.window + 1 : 0) : 0;
                const int off = t0 - t0_0;   /* 0..7: a row's piece starts at most 7 positions on */
                const int t = stb + off + lane;
                const float ts = tau[p.tau_off + s];
                const bool ok = t >= t0 && t <= qpos;
                float sv = -3.0e38f;
                if (ok) {
                    float4 a = {0.f, 0.f, 0.f, 0.f};
                    const float4 *qq = (const float4 *)(qs + ((size_t)g * p.S + s) * hd);
                    for (int j = 0; j < kd4; j++) {
                        const float4 kk = *(const float4 *)(Kt4 + ((size_t)j * CCG_DTS + off + lane) * 4);
                        const float4 q4 = qq[j];
                        a.x += q4.x * kk.x; a.y += q4.y * kk.y; a.z += q4.z * kk.z; a.w += q4.w * kk.w;
                    }
                    const float dot = ((a.x + a.y) + (a.z + a.w));
                    float bias = 0.f;
                    const int dist = qpos - t;
                    if (dist < p.ext && p.d_rel > 0) {
                        const float *rv = r + p.r_off + (long)s * p.r_row + (long)h * p.d_rel;
                        const float *rp = relp + p.relp_off + dist;
                        for (int j = 0; j < p.d_rel; j++) bias += rv[j] * rp[(long)j * p.ext];
                    }
                    sv = ts * (dot * p.scale + bias);
                }
                float mx = sv;
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
                float e = 0.f;
                if (__ballot_sync(0xffffffffu, ok)) {
                    const float mn = fmaxf(m[s], mx);
                    e = ok ? exp2f((sv - mn) * 1.4426950408889634f) : 0.f;
                    float sum = e;
                    #pragma unroll
                    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
                    const float corr = exp2f((m[s] - mn) * 1.4426950408889634f);
                    l[s] = l[s] * corr + sum;
                    acc[s][0] *= corr; acc[s][1] *= corr; acc[s][2] *= corr; acc[s][3] *= corr;
                    m[s] = mn;
                }
                P[(size_t)g * CCG_DTS + off + lane] = e;
                __syncwarp();
                if (d0 < hd) for (int j = 0; j < CCG_DTP; j++) {
                    const float w = P[(size_t)g * CCG_DTS + off + j];
                    const float4 vv = *(const float4 *)(Vt + (size_t)(off + j) * hd + d0);
                    acc[s][0] += w * vv.x; acc[s][1] += w * vv.y; acc[s][2] += w * vv.z; acc[s][3] += w * vv.w;
                }
                __syncwarp();
            }
        }
        __syncthreads();
    }
    if (mine) {
        #pragma unroll
        for (int s = 0; s < 8; s++) {
            if (s >= p.S) continue;
            float *pp = part + (((long)s * p.H + h) * nc + z) * (p.hd + 2);
            if (lane == 0) { pp[0] = m[s]; pp[1] = l[s]; }
            if (d0 < hd) { float *pd = pp + 2 + d0; pd[0] = acc[s][0]; pd[1] = acc[s][1]; pd[2] = acc[s][2]; pd[3] = acc[s][3]; }
        }
    }
}

/* ---- the prompt path: a tile of query rows, K/V loaded once ----
 * ccg_relattn_k launches a block per (query head, row) and each of them walks the whole cache:
 * a prompt chunk of S rows reads the same K/V row S times. This kernel keeps CCG_QT query rows
 * of one KV head in a block, one warp a row, and walks the cache in CCG_TP-position tiles that
 * the block loads once into shared memory: the cache is read S/CCG_QT times less, and each
 * position's score is a float4 dot with the query heads in shared (broadcast) instead of a
 * serial 128-long chain. The online softmax, the sink, tau, the relative bias, the window and
 * the ring, and the split between the mirror and kvs's own rows, are the scalar kernel's. */
#define CCG_QT 8
#define CCG_TP 32
/* One instantiation a G (the query heads of a KV head, 1..8): the head loop unrolls with a
 * constant bound, so the accumulators stay in registers and a G of 3 does not pay for 8. */
template <int G>
__global__ static void ccg_relattn_qt_t(const float *__restrict__ q, const void *__restrict__ kc,
                                        const void *__restrict__ vc, float *__restrict__ o, const float *__restrict__ kvs,
                                        const float *__restrict__ r, const float *__restrict__ relp,
                                        const float *__restrict__ tau, CcRelAttn p) {
    const int kvh = (int)blockIdx.x, tb0 = (int)blockIdx.y * CCG_QT, tid = (int)threadIdx.x;
    const int wp = tid >> 5, lane = tid & 31, hd = p.hd, kd4 = hd >> 2;
    extern __shared__ float sm[];
    /* Q[QT][G][hd]; Kt4[d4][TP] float4, the K tile transposed on its dim chunks so a lane's
     * float4 is conflict-free (the row-major tile put all 32 lanes' float4 in the same four
     * banks); Vt[TP][hd]; P[G][TP]. */
    float *Q = sm, *Kt4 = Q + (size_t)CCG_QT * G * hd, *Vt = Kt4 + (size_t)kd4 * CCG_TP * 4,
          *P = Vt + (size_t)CCG_TP * hd;
    __shared__ int t_first, t_last;
    if (tid == 0) {
        int lo = 0x7fffffff, hi = -1;
        for (int i = 0; i < CCG_QT; i++) {
            const int s2 = tb0 + i;
            if (s2 >= p.S) continue;
            const int qp = p.pos_base + s2, t0 = p.window > 0 ? max(0, qp - p.window + 1) : 0;
            if (t0 < lo) lo = t0;
            if (qp > hi) hi = qp;
        }
        t_first = lo; t_last = hi;
    }
    for (int idx = tid; idx < CCG_QT * G * hd; idx += (int)blockDim.x) {
        const int rr = idx / (G * hd), rest = idx - rr * G * hd, h = rest / hd, d = rest - h * hd;
        const int s2 = tb0 + rr;
        Q[idx] = s2 < p.S ? q[p.q_off + (long)s2 * p.q_row + (long)(kvh * G + h) * hd + d] : 0.f;
    }
    __syncthreads();
    const int s = tb0 + wp;
    const int qpos = p.pos_base + s, t0 = p.window > 0 ? max(0, qpos - p.window + 1) : 0;
    const float ts = s < p.S ? tau[p.tau_off + s] : 0.f;
    float m[G], l[G], acc[G][4];
    #pragma unroll
    for (int h = 0; h < G; h++) {
        m[h] = -3.0e38f; l[h] = 0.f;
        acc[h][0] = acc[h][1] = acc[h][2] = acc[h][3] = 0.f;
        if (s < p.S && p.has_sink) { m[h] = tau[p.sink_off + s * p.H + kvh * G + h]; l[h] = 1.f; }
    }
    const int d0 = lane * 4;
    for (int tb = (t_first / CCG_TP) * CCG_TP; tb <= t_last; tb += CCG_TP) {
        /* the tile, read and dequantized once: K by dim chunk, V by row */
        for (int idx = tid; idx < kd4 * CCG_TP; idx += (int)blockDim.x) {
            const int d4 = idx / CCG_TP, pos = idx - d4 * CCG_TP, t = tb + pos;
            float4 kv = {0.f, 0.f, 0.f, 0.f}, vv4 = {0.f, 0.f, 0.f, 0.f};
            if (t <= t_last) {
                if (t < p.pos_base) {
                    const void *kr = ccg_kv_row(kc, p, p.k_off, kvh, t), *vr = ccg_kv_row(vc, p, p.v_off, kvh, t);
                    kv.x = ccg_kv_at(kr, d4 * 4 + 0, p.kv_type); vv4.x = ccg_kv_at(vr, d4 * 4 + 0, p.kv_type);
                    kv.y = ccg_kv_at(kr, d4 * 4 + 1, p.kv_type); vv4.y = ccg_kv_at(vr, d4 * 4 + 1, p.kv_type);
                    kv.z = ccg_kv_at(kr, d4 * 4 + 2, p.kv_type); vv4.z = ccg_kv_at(vr, d4 * 4 + 2, p.kv_type);
                    kv.w = ccg_kv_at(kr, d4 * 4 + 3, p.kv_type); vv4.w = ccg_kv_at(vr, d4 * 4 + 3, p.kv_type);
                } else {
                    const long o = (long)(t - p.pos_base) * p.kv_row + (long)kvh * hd + d4 * 4;
                    kv = *(const float4 *)(kvs + p.ks_off + o);
                    vv4 = *(const float4 *)(kvs + p.vs_off + o);
                }
            }
            *(float4 *)(Kt4 + ((size_t)d4 * CCG_TP + pos) * 4) = kv;
            *(float4 *)(Vt + (size_t)pos * hd + d4 * 4) = vv4;
        }
        __syncthreads();
        if (s < p.S) {
            const int t = tb + lane, dist = qpos - t;
            const bool ok = t >= t0 && t <= qpos;
            /* every head's dot from one walk of the lane's K row: the K read is G times cheaper,
             * and the G accumulators give the loop its parallelism */
            float4 a[G];
            #pragma unroll
            for (int h = 0; h < G; h++) a[h].x = a[h].y = a[h].z = a[h].w = 0.f;
            if (ok) {
                for (int j = 0; j < kd4; j++) {
                    const float4 kk = *(const float4 *)(Kt4 + ((size_t)j * CCG_TP + lane) * 4);
                    #pragma unroll
                    for (int h = 0; h < G; h++) {
                        const float4 qq = ((const float4 *)(Q + ((size_t)wp * G + h) * hd))[j];
                        a[h].x += qq.x * kk.x; a[h].y += qq.y * kk.y; a[h].z += qq.z * kk.z; a[h].w += qq.w * kk.w;
                    }
                }
            }
            #pragma unroll
            for (int h = 0; h < G; h++) {
                const float dot = ((a[h].x + a[h].y) + (a[h].z + a[h].w));
                float bias = 0.f;
                if (ok && dist < p.ext && p.d_rel > 0) {
                    const float *rv = r + p.r_off + (long)s * p.r_row + (long)(kvh * G + h) * p.d_rel;
                    const float *rp = relp + p.relp_off + dist;
                    for (int j = 0; j < p.d_rel; j++) bias += rv[j] * rp[(long)j * p.ext];
                }
                const float sv = ok ? ts * (dot * p.scale + bias) : -3.0e38f;
                float mx = sv;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
                const unsigned bal = __ballot_sync(0xffffffffu, ok);
                float e = 0.f;
                if (bal) {
                    const float mn = fmaxf(m[h], mx);
                    e = ok ? exp2f((sv - mn) * 1.4426950408889634f) : 0.f;   /* expf(x) = exp2f(x log2 e) */
                    float sum = e;
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
                    const float corr = exp2f((m[h] - mn) * 1.4426950408889634f);   /* a later tile's larger max */
                    l[h] = l[h] * corr + sum;
                    acc[h][0] *= corr; acc[h][1] *= corr; acc[h][2] *= corr; acc[h][3] *= corr;
                    m[h] = mn;
                }
                P[((size_t)wp * G + h) * CCG_TP + lane] = e;
            }
            __syncwarp();
            if (d0 < hd) {
                /* the V tile's float4 read once a position and used by every head */
                for (int j = 0; j < CCG_TP; j++) {
                    const float4 y = *(const float4 *)(Vt + (size_t)j * hd + d0);
                    #pragma unroll
                    for (int h = 0; h < G; h++) {
                        const float w = P[((size_t)wp * G + h) * CCG_TP + j];
                        acc[h][0] += w * y.x; acc[h][1] += w * y.y; acc[h][2] += w * y.z; acc[h][3] += w * y.w;
                    }
                }
            }
            __syncwarp();
        }
        __syncthreads();
    }
    if (s < p.S && d0 < hd) {
        #pragma unroll
        for (int h = 0; h < G; h++) {
            const float inv = l[h] > 0.f ? 1.f / l[h] : 0.f;
            float *orow = o + p.o_off + (long)s * p.o_row + (long)(kvh * G + h) * hd + d0;
            for (int j = 0; j < 4 && d0 + j < hd; j++) orow[j] = acc[h][j] * inv;
        }
    }
}

/* one block of 256 per row: every vector's sums in one pass, the warps' joined in order */
__global__ static void ccg_ink2_mix_k(const float *x, const float *blk, const float *prm, float *y, CcInk2Mix p) {
    const int s = (int)(blockIdx.x + blockIdx.y * gridDim.x), tid = (int)threadIdx.x, wp = tid >> 5, ln = tid & 31;
    if (s >= p.S) return;
    __shared__ float pms[8 * 16], pdt[8 * 16], sc[16], rr[16];
    const float *xb = x + p.x_off + (long)s * p.x_row, *bb = blk ? blk + p.b_off + (long)s * p.b_row : xb;
    const float *q = prm + p.w_off;
    float ms[16], dt[16];
#pragma unroll
    for (int e = 0; e < 16; e++) { ms[e] = 0.f; dt[e] = 0.f; }
    for (int d = tid; d < p.D; d += 256) {
        const float qv = q[d];
#pragma unroll
        for (int e = 0; e < 16; e++) {
            if (e > p.nb) break;
            const float v = e < p.nb ? bb[(long)e * p.D + d] : xb[d];
            ms[e] += v * v; dt[e] += v * qv;
        }
    }
#pragma unroll
    for (int e = 0; e < 16; e++) {
        if (e > p.nb) break;
        float a = ms[e], b = dt[e];
        for (int off = 16; off > 0; off >>= 1) { a += __shfl_xor_sync(0xffffffffu, a, off); b += __shfl_xor_sync(0xffffffffu, b, off); }
        if (ln == 0) { pms[wp * 16 + e] = a; pdt[wp * 16 + e] = b; }
    }
    __syncthreads();
    if (tid <= p.nb) {
        float a = 0.f, b = 0.f;
        for (int g = 0; g < 8; g++) { a += pms[g * 16 + tid]; b += pdt[g * 16 + tid]; }
        const float r = 1.f / sqrtf(a / (float)p.D + p.eps);
        rr[tid] = r; sc[tid] = (b * r) * p.scale;
    }
    __syncthreads();
    if (tid == 0) {
        float m = sc[0];
        for (int e = 1; e <= p.nb; e++) if (sc[e] > m) m = sc[e];
        float sum = 0.f;
        for (int e = 0; e <= p.nb; e++) { sc[e] = expf(sc[e] - m); sum += sc[e]; }
        for (int e = 0; e <= p.nb; e++) sc[e] /= sum;
    }
    __syncthreads();
    float *yb = y + p.y_off + (long)s * p.y_row;
    for (int d = tid; d < p.D; d += 256) {
        float a = 0.f;
        for (int e = 0; e < p.nb; e++) a += sc[e] * (bb[(long)e * p.D + d] * rr[e]);
        yb[d] = a + sc[p.nb] * (xb[d] * rr[p.nb]);
    }
}

/* one block of 256 per segment */
__global__ static void ccg_ink2_clip_k(float *x, const float *w, CcInk2Clip p) {
    const int g = (int)(blockIdx.x + blockIdx.y * gridDim.x), tid = (int)threadIdx.x;
    if (g >= p.nseg) return;
    __shared__ float red[256];
    const int j = g % p.per_row;
    float *v = x + p.x_off + (long)(g / p.per_row) * p.x_row + (long)j * p.n;
    const float *sw = p.has_w ? w + p.w_off + (long)j * p.n : nullptr;
    float ss = 0.f;
    for (int i = tid; i < p.n; i += 256) { float a = v[i]; if (sw) a *= sw[i]; ss += a * a; }
    red[tid] = ss;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) { if (tid < off) red[tid] += red[tid + off]; __syncthreads(); }
    const float f = p.thr / (sqrtf(red[0]) + 1e-6f);
    for (int i = tid; i < p.n; i += 256) { float a = v[i]; if (sw) a *= sw[i]; v[i] = f < 1.f ? a * f : a; }
}

/* one thread per channel over the step's rows */
__global__ static void ccg_ink2_dconv_k(float *x, const float *dyn, const float *st, float *ring, CcInk2Dconv p) {
    const long c = (long)(blockIdx.x + blockIdx.y * gridDim.x) * 256 + threadIdx.x;
    if (c >= p.C) return;
    const int K = p.K, nh = K - 1, h = (int)(c / p.hd);
    float hist[8];
#pragma unroll
    for (int k = 0; k < 8; k++) hist[k] = k < nh ? ring[p.ring_off + c * nh + k] : 0.f;
    for (int s = 0; s < p.S; s++) {
        const long xi = p.x_off + (long)s * p.x_row + c;
        const float cur = x[xi];
        const float *dw = dyn + p.d_off + (long)s * p.d_row + h * K;
        float acc = cur * (st[p.st_off + c] + dw[0]);
#pragma unroll
        for (int lag = 1; lag < 9; lag++) if (lag < K) acc += hist[nh - lag] * (st[p.st_off + (long)lag * p.C + c] + dw[lag]);
        x[xi] = (acc + cur) / p.fc;
#pragma unroll
        for (int k = 0; k < 7; k++) if (k + 1 < nh) hist[k] = hist[k + 1];
#pragma unroll
        for (int k = 0; k < 8; k++) if (k == nh - 1) hist[k] = cur;
    }
#pragma unroll
    for (int k = 0; k < 8; k++) if (k < nh) ring[p.ring_off + c * nh + k] = hist[k];
}

static int cc_sconv(CcBuf *x, CcBuf *w, CcBuf *ring, const CcSconv *p) {
    CcCtx *c = live();
    if (!c || !p || !x || (p->mode == 0 && (p->CK < 1 || p->CK > 9 || !w || !ring))) return 0;
    const long n = p->mode == 0 ? p->C : p->n;
    if (n <= 0) return 1;
    dim3 g; cg_grid((n + 63) / 64, &g);
    coli_launch(ccg_sconv_k, g, 64, 0, c->stream, fpw(x), p->mode == 0 ? fp(w) : fp(x), p->mode == 0 ? fpw(ring) : fpw(x), *p);
    return launched(c, "sconv");
}
/* the prompt path's launch, one instantiation a G (its kernel's shared memory opt-in) */
template <int G>
static int cc_qt_launch(const float *q, const void *kc, const void *vc, float *o, const float *kvs,
                        const float *r, const float *relp, const float *tau, const CcRelAttn &p,
                        dim3 g, size_t sm, cudaStream_t st) {
    static int attr = -1;
    if (attr < 0)
        attr = cudaFuncSetAttribute(ccg_relattn_qt_t<G>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    96 * 1024) == cudaSuccess ? 1 : 0;
    if (!attr) return 0;
    coli_launch(ccg_relattn_qt_t<G>, g, 256, sm, st, q, kc, vc, o, kvs, r, relp, tau, p);
    return 1;
}
/* COLI_CUDA_RELATTN_GQA=0: the decode step's old walk (a block a (query head, row), the same rows
 * read and dequantized G times) */
static int cc_relattn_gqa(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_CUDA_RELATTN_GQA"); on = !(e && *e == '0'); }
    return on;
}
/* the GQA path's partials, kept (a 250K step's are ~100 MiB; a cudaMallocAsync a call was 65 MiB
 * a layer before) */
static float *cc_relattn_part(size_t bytes) {
    static CcBuf *b; static size_t cap;
    if (cap < bytes) { cc_free(b); b = cc_buf(bytes, CC_DEV); cap = b ? bytes : 0; }
    return b ? (float *)(void *)fp(b) : nullptr;
}
static int cc_relattn(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *kvs, CcBuf *r, CcBuf *relp, CcBuf *tau, const CcRelAttn *p) {
    CcCtx *c = live();
    if (!c || !p || p->hd > 256 || p->hd < 1 || p->d_rel > 64 || p->d_rel < 0 || p->KVH < 1 || p->H % p->KVH || p->cap < 1) return 0;
    if (p->S <= 0) return 1;
    /* the decode step, GQA-shared: one block a (KV head, tile), the tile's K/V read and dequantized
     * once a KV head and every query head of it reading them (P3 of PLAN-colibri.md) */
    {
        const int G = p->H / p->KVH;
        if (cc_relattn_gqa() && p->S <= 8 && p->H % p->KVH == 0 && G >= 1 && G <= 8 && p->hd >= 4 &&
            p->hd <= 128 && (p->hd & 3) == 0 && p->kv_type >= 0 && p->kv_type <= 3) {
            const int nmax = p->window > 0 ? p->window : p->pos_base + p->S;
            const int nc = (nmax + CCG_DTB - 1) / CCG_DTB;   /* a partial a block of CCG_DTB positions */
            const size_t sm = ((size_t)(p->hd >> 2) * CCG_DTS * 4 + (size_t)CCG_DTS * p->hd +
                               (size_t)G * p->S * p->hd + (size_t)G * CCG_DTS) * sizeof(float);
            const size_t pb = (size_t)p->S * p->H * nc * (p->hd + 2) * sizeof(float);
            static int attr = -1;
            if (attr < 0)
                attr = cudaFuncSetAttribute(ccg_relattn_dec_k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            96 * 1024) == cudaSuccess ? 1 : 0;
            float *part = attr ? cc_relattn_part(pb) : nullptr;
            if (part && nc >= 1 && sm <= 96 * 1024) {
                const int nthr = G <= 4 ? 128 : 256;
                coli_launch(ccg_relattn_dec_k, dim3((unsigned)p->KVH, (unsigned)nc), nthr, sm, c->stream,
                            fp(q), (const void *)fp(kc), (const void *)fp(vc), fp(kvs), fp(r), fp(relp), fp(tau), *p, part, nc);
                coli_launch(ccg_relattn_join_k, dim3((unsigned)p->H, (unsigned)p->S), 256, 0, c->stream,
                            (const float *)part, fp(tau), fpw(o), *p, nc);
                return launched(c, "relattn gqa");
            }
            (void)cudaGetLastError();
        }
    }
    /* a decode step's rows (up to 8): each (head, row)'s positions split over blocks of a tile each, then joined
     * (COLI_CUDA_RELATTN_SPLIT=0: a block a (head, row) over every tile). Inkling-2-Mini Q4_K_XL at about 650
     * positions on an RTX PRO 4000: a block a head ran its tiles one after another */
    const int split = cc_relattn_split();
    const int nmax = p->window > 0 ? p->window : p->pos_base + p->S, nc = (nmax + 127) / 128;
    if (split && p->S <= 8 && nc >= 2) {
        float *part = NULL;
        const size_t pb = (size_t)p->S * p->H * nc * (p->hd + 2) * sizeof(float);
        if (cudaMallocAsync((void **)&part, pb, c->stream) == cudaSuccess) {
            coli_launch(ccg_relattn_k, dim3((unsigned)p->H, (unsigned)p->S, (unsigned)nc), 128, 0, c->stream, fp(q), fp(kc), fp(vc), fpw(o),
                        fp(kvs), fp(r), fp(relp), fp(tau), *p, part);
            coli_launch(ccg_relattn_join_k, dim3((unsigned)p->H, (unsigned)p->S), 256, 0, c->stream, (const float *)part, fp(tau), fpw(o), *p, nc);
            (void)cudaFreeAsync(part, c->stream);
            return launched(c, "relattn split");
        }
        (void)cudaGetLastError();
    }
    /* a prompt chunk: a tile of query rows a block, K/V read once a tile (ccg_relattn_qt_k).
     * COLI_CUDA_RELATTN_QT: 0 the per-(head,row) kernel, unset rows from CCG_QT. */
    {
        const char *e = getenv("COLI_CUDA_RELATTN_QT");
        const int G = p->H / p->KVH;
        const int want = e && *e ? atoi(e) : CCG_QT;
        const size_t sm = ((size_t)CCG_QT * G * p->hd + (size_t)CCG_TP * (p->hd + 4) +
                           (size_t)CCG_TP * p->hd + (size_t)CCG_QT * G * CCG_TP) * sizeof(float);
        if (p->S >= want && want > 1 && G >= 1 && G <= 8 && p->hd <= 128 && (p->hd & 3) == 0 &&
            sm <= 96 * 1024 && p->kv_type >= 0 && p->kv_type <= 3) {
            const dim3 g((unsigned)p->KVH, (unsigned)((p->S + CCG_QT - 1) / CCG_QT));
            int on = 0;
            switch (G) {
            case 1: on = cc_qt_launch<1>(fp(q), (const void *)fp(kc), (const void *)fp(vc), fpw(o), fp(kvs), fp(r), fp(relp), fp(tau), *p, g, sm, c->stream); break;
            case 2: on = cc_qt_launch<2>(fp(q), (const void *)fp(kc), (const void *)fp(vc), fpw(o), fp(kvs), fp(r), fp(relp), fp(tau), *p, g, sm, c->stream); break;
            case 3: on = cc_qt_launch<3>(fp(q), (const void *)fp(kc), (const void *)fp(vc), fpw(o), fp(kvs), fp(r), fp(relp), fp(tau), *p, g, sm, c->stream); break;
            case 4: on = cc_qt_launch<4>(fp(q), (const void *)fp(kc), (const void *)fp(vc), fpw(o), fp(kvs), fp(r), fp(relp), fp(tau), *p, g, sm, c->stream); break;
            case 5: on = cc_qt_launch<5>(fp(q), (const void *)fp(kc), (const void *)fp(vc), fpw(o), fp(kvs), fp(r), fp(relp), fp(tau), *p, g, sm, c->stream); break;
            case 6: on = cc_qt_launch<6>(fp(q), (const void *)fp(kc), (const void *)fp(vc), fpw(o), fp(kvs), fp(r), fp(relp), fp(tau), *p, g, sm, c->stream); break;
            case 7: on = cc_qt_launch<7>(fp(q), (const void *)fp(kc), (const void *)fp(vc), fpw(o), fp(kvs), fp(r), fp(relp), fp(tau), *p, g, sm, c->stream); break;
            default: on = cc_qt_launch<8>(fp(q), (const void *)fp(kc), (const void *)fp(vc), fpw(o), fp(kvs), fp(r), fp(relp), fp(tau), *p, g, sm, c->stream); break;
            }
            if (on) return launched(c, "relattn qt");
            (void)cudaGetLastError();
        }
    }
    coli_launch(ccg_relattn_k, dim3((unsigned)p->H, (unsigned)p->S), 128, 0, c->stream, fp(q), (const void *)fp(kc),
                (const void *)fp(vc), fpw(o), fp(kvs), fp(r), fp(relp), fp(tau), *p, (float *)nullptr);
    return launched(c, "relattn");
}
static int cc_ink2_mix(CcBuf *x, CcBuf *blk, CcBuf *prm, CcBuf *y, const CcInk2Mix *p) {
    CcCtx *c = live();
    if (!c || !p || !x || !prm || !y || p->S < 0 || p->D < 1 || p->nb < 0 || p->nb > 15 || (p->nb > 0 && !blk) || y == x || y == blk) return 0;
    if (p->S == 0) return 1;
    dim3 g; cg_grid(p->S, &g);
    coli_launch(ccg_ink2_mix_k, g, 256, 0, c->stream, fp(x), p->nb > 0 ? fp(blk) : nullptr, fp(prm), fpw(y), *p);
    return launched(c, "ink2 mix");
}
static int cc_ink2_clip(CcBuf *x, CcBuf *prm, const CcInk2Clip *p) {
    CcCtx *c = live();
    if (!c || !p || !x || p->nseg < 0 || p->n < 1 || p->per_row < 1 || (p->has_w && !prm)) return 0;
    if (p->nseg == 0) return 1;
    dim3 g; cg_grid(p->nseg, &g);
    coli_launch(ccg_ink2_clip_k, g, 256, 0, c->stream, fpw(x), p->has_w ? fp(prm) : nullptr, *p);
    return launched(c, "ink2 clip");
}
static int cc_ink2_dconv(CcBuf *x, CcBuf *dyn, CcBuf *prm, CcBuf *ring, const CcInk2Dconv *p) {
    CcCtx *c = live();
    if (!c || !p || !x || !dyn || !prm || !ring || p->S < 0 || p->C < 1 || p->K < 1 || p->K > 9 || p->hd < 1 || p->C % p->hd) return 0;
    if (p->S == 0) return 1;
    dim3 g; cg_grid(((long)p->C + 255) / 256, &g);
    coli_launch(ccg_ink2_dconv_k, g, 256, 0, c->stream, fpw(x), fp(dyn), fp(prm), fpw(ring), *p);
    return launched(c, "ink2 dconv");
}
