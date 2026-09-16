// Standalone verification: compare ggml_vec_dot_f16 (existing reference path)
// against ggml_gemv_f16_8x8_f32 / ggml_gemm_f16_8x8_f32 (new repack kernels),
// AND validate the real repack<ggml_fp16_t,8,8> repacking function itself
// (not just a hand-built layout assumed to match it), on identical, known
// input data. Not part of the tracked build.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <functional>
#include <string>

#include "ggml.h"

typedef ggml_fp16_t fp16_t; // ggml.h already provides "typedef uint16_t ggml_fp16_t"

// -- declarations matching repack.h / vec.h (extern "C" linkage) --
extern "C" {
    void ggml_gemv_f16_8x8_f32(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
    void ggml_gemm_f16_8x8_f32(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
    void ggml_vec_dot_f16(int n, float * s, size_t bs, fp16_t * x, size_t bx, fp16_t * y, size_t by, int nrc);
}

// -- declaration matching the internal repack<> template in repack.cpp --
// Same namespace, same template parameter list, same signature: the linker
// resolves this to the explicit specialization repack<ggml_fp16_t,8,8>
// compiled inside repack.cpp.obj, without needing that file's declaration
// to be visible here (only the mangled name has to match).
namespace ggml::cpu::repack {
    template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
    int repack(struct ggml_tensor * t, const void * data, size_t data_size);
}

// -- dead-code stubs --
// arch/x86/repack.cpp.obj and vec.obj are linked as whole translation units
// (loose .obj, not archived in a .lib), which pulls in symbols referenced by
// OTHER unrelated template instantiations (q4_0, q4_K, ...) in the same file.
// None of these are ever reached by this test (we only call the f16 functions
// above directly), so trivial stubs satisfy the linker without touching the
// real code under test.
struct ggml_threadpool;
extern "C" {
    float ggml_table_f32_f16[1 << 16] = {};
    float ggml_table_f32_e8m0_half[1 << 8] = {};
    void ggml_barrier(struct ggml_threadpool *) {}
    void ggml_threadpool_chunk_set(struct ggml_threadpool *, int) {}
    int  ggml_threadpool_chunk_add(struct ggml_threadpool *, int) { return 0; }
}
namespace ggml::cpu {
    class tensor_traits { public: virtual ~tensor_traits() {} };
    class extra_buffer_type { public: virtual ~extra_buffer_type() {} };
}
// force vtable/destructor emission (never used at runtime, linker-only)
static ggml::cpu::tensor_traits g_dummy_tensor_traits;
static ggml::cpu::extra_buffer_type g_dummy_extra_buffer_type;

// -- minimal fp32 <-> fp16 conversion (software, IEEE 754 half) --
static fp16_t fp32_to_fp16(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0) return (fp16_t)sign; // flush to zero, good enough for test values
    if (exp >= 31) return (fp16_t)(sign | 0x7C00);
    return (fp16_t)(sign | (exp << 10) | (mant >> 13));
}

static float fp16_to_fp32(fp16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) { f = sign; }
    else { f = sign | ((exp - 15 + 127) << 23) | (mant << 13); }
    float out; std::memcpy(&out, &f, 4);
    return out;
}

// block_f16x8 layout: 8 rows x 8 cols of raw f16 per block, interleaved
struct block_f16x8 { fp16_t qs[64]; };

// activation pattern cycling every 8 columns; every value here is an exact
// dyadic fraction (power-of-2 denominator) so it round-trips through fp16
// with zero loss -- keeps the vec_dot_f16 (fp16 activations) comparison
// exact rather than introducing its own separate rounding error.
static float base_act(int c) {
    static const float pat[8] = { 0.5f, -1.0f, 2.0f, -0.25f, 1.5f, -2.0f, 0.25f, -0.5f };
    return pat[c % 8];
}

struct TestCase {
    const char * name;
    int nrows;
    int ncols;
    std::function<float(int,int)> weight_fn; // (row, col) -> raw weight value, pre-fp16-rounding
    std::function<float(int,int)> act_fn;    // (m in 0..3, col) -> exact f32 activation value
};

static bool run_test(const TestCase & tc) {
    const int nrows = tc.nrows;
    const int ncols = tc.ncols;
    const int nblk_row = nrows / 8;
    const int nblk_col = ncols / 8;

    printf("=== %s (nrows=%d, ncols=%d) ===\n", tc.name, nrows, ncols);

    bool pass = true;

    // 1. weights, fp16-rounded (this is what a real F16 GGUF tensor stores)
    std::vector<fp16_t> W(nrows * ncols);
    for (int r = 0; r < nrows; r++)
        for (int c = 0; c < ncols; c++)
            W[r * ncols + c] = fp32_to_fp16(tc.weight_fn(r, c));

    // 2. four distinct activation rows, exact f32 (no rounding -- PARAM_TYPE=F32)
    std::vector<float> act4(ncols * 4);
    for (int m = 0; m < 4; m++)
        for (int c = 0; c < ncols; c++)
            act4[m * ncols + c] = tc.act_fn(m, c);

    // 3. fp16 copy of row 0 only, for the ggml_vec_dot_f16 comparison (that
    //    function's signature requires fp16 on both sides)
    std::vector<fp16_t> act0_f16(ncols);
    for (int c = 0; c < ncols; c++) act0_f16[c] = fp32_to_fp16(act4[c]);

    // 4. ground truth: double-precision summation using the FP16-ROUNDED
    //    weight value (matching what the kernel actually reads) times the
    //    EXACT f32 activation value (matching PARAM_TYPE=F32, no rounding)
    std::vector<std::vector<double>> ref(4, std::vector<double>(nrows));
    for (int m = 0; m < 4; m++) {
        for (int r = 0; r < nrows; r++) {
            double s = 0.0;
            for (int c = 0; c < ncols; c++) {
                s += (double) fp16_to_fp32(W[r * ncols + c]) * (double) act4[m * ncols + c];
            }
            ref[m][r] = s;
        }
    }

    // 5. vec_dot_f16 reference, row0 activation only (existing, untouched code)
    std::vector<float> vecdot(nrows);
    for (int r = 0; r < nrows; r++) {
        ggml_vec_dot_f16(ncols, &vecdot[r], 0, &W[r * ncols], 0, act0_f16.data(), 0, 1);
    }

    // 6a. HAND-BUILT packing, matching repack_f16_to_f16x8's documented layout:
    //     blocks for row-group rg occupy a contiguous run of nblk_col blocks,
    //     row-groups laid out one after another (matches how gemv/gemm index
    //     "b_ptr = vx + x*nb" where x = row-group, nb = nblk_col)
    std::vector<block_f16x8> packed_manual(nblk_row * nblk_col);
    for (int rg = 0; rg < nblk_row; rg++) {
        for (int cg = 0; cg < nblk_col; cg++) {
            block_f16x8 & blk = packed_manual[rg * nblk_col + cg];
            for (int rr = 0; rr < 8; rr++) {
                int r = rg * 8 + rr;
                for (int i = 0; i < 8; i++) {
                    int c = cg * 8 + i;
                    blk.qs[rr * 8 + i] = W[r * ncols + c];
                }
            }
        }
    }

    // 6b. REAL packing, via the actual repack<ggml_fp16_t,8,8> function
    //     compiled into repack.cpp.obj -- this is the function used by real
    //     model loading, previously untested.
    std::vector<block_f16x8> packed_real(nblk_row * nblk_col);
    struct ggml_tensor t {};
    t.type   = GGML_TYPE_F16;
    t.ne[0]  = ncols;
    t.ne[1]  = nrows;
    t.ne[2]  = 1;
    t.ne[3]  = 1;
    t.data   = packed_real.data();

    int repack_rc = ggml::cpu::repack::repack<fp16_t, 8, 8>(&t, W.data(), W.size() * sizeof(fp16_t));
    if (repack_rc != 0) {
        printf("  REPACK FAIL: repack<ggml_fp16_t,8,8> returned %d (expected 0)\n", repack_rc);
        pass = false;
    } else if (std::memcmp(packed_manual.data(), packed_real.data(),
                            packed_manual.size() * sizeof(block_f16x8)) != 0) {
        printf("  REPACK FAIL: real repack() output differs from hand-built expected layout\n");
        pass = false;
    } else {
        printf("  repack<ggml_fp16_t,8,8>: byte-identical to expected layout -- PASS\n");
    }

    // From here on, feed the REAL repacked buffer into the kernels -- this
    // exercises the full production chain (repack -> gemv/gemm) together,
    // not just the kernels in isolation against a hand-assumed layout.
    const block_f16x8 * packed = packed_real.data();

    // 7. run the new kernels
    std::vector<float> gemv_out(nrows);
    ggml_gemv_f16_8x8_f32(ncols, gemv_out.data(), nrows, packed, act4.data(), 1, nrows);

    std::vector<float> gemm_out(4 * nrows);
    ggml_gemm_f16_8x8_f32(ncols, gemm_out.data(), nrows, packed, act4.data(), 4, nrows);

    // 8. compare -- row0 across all three implementations
    printf("%-4s %14s %14s %14s %14s  %s\n", "row", "expected", "vec_dot_f16", "gemv_f16", "gemm_f16[m=0]", "status");
    for (int r = 0; r < nrows; r++) {
        double expected = ref[0][r];
        float  gv = gemv_out[r];
        float  gm = gemm_out[0 * nrows + r];
        float  vd = vecdot[r];
        double tol = std::fmax(1e-2, std::fabs(expected) * 1e-4);
        bool row_pass = std::fabs(expected - gv) < tol &&
                         std::fabs(expected - gm) < tol &&
                         std::fabs(expected - vd) < tol;
        pass &= row_pass;
        printf("%-4d %14.4f %14.4f %14.4f %14.4f  %s\n", r, expected, vd, gv, gm, row_pass ? "PASS" : "FAIL");
    }

    // 9. compare -- all 4 gemm rows independently, against their OWN distinct
    //    reference (closes the "does gemm actually use row m's own data" gap)
    for (int m = 0; m < 4; m++) {
        for (int r = 0; r < nrows; r++) {
            double expected = ref[m][r];
            float  gm = gemm_out[m * nrows + r];
            double tol = std::fmax(1e-2, std::fabs(expected) * 1e-4);
            if (std::fabs(expected - gm) >= tol) {
                printf("  GEMM FAIL at m=%d r=%d: expected=%.4f got=%.4f\n", m, r, expected, gm);
                pass = false;
            }
        }
    }

    printf("%s\n\n", pass ? "-> PASS" : "-> FAIL");
    return pass;
}

int main() {
    // ggml_vec_dot_f16 on x86 uses a lookup table (ggml_table_f32_f16) for
    // scalar fp16->fp32, normally populated once by ggml_cpu_init(). We stub
    // that table (never linking the real init to avoid pulling in the whole
    // op dispatcher), so populate it ourselves here.
    for (int i = 0; i < (1 << 16); i++) {
        ggml_table_f32_f16[i] = fp16_to_fp32((fp16_t) i);
    }

    auto four_row_act = [](int m, int c, int ncols) {
        float b = base_act(c);
        if (m == 0) return b;
        if (m == 1) return base_act(ncols - 1 - c); // reversed indexing
        if (m == 2) return 2.0f * b;
        return b + 1.0f;
    };

    std::vector<TestCase> tests;

    // Test 0: baseline -- original case kept for regression continuity.
    // Uniform act=1.0 means the dot product degenerates to "sum of row",
    // so this alone cannot catch a column-ordering bug.
    tests.push_back({
        "baseline (uniform activation)",
        8, 16,
        [](int r, int c) { return (float)(r * 100 + c); },
        [](int /*m*/, int /*c*/) { return 1.0f; }
    });

    // Test 1: non-uniform, distinct-per-row activations.
    // Closes the column-ordering blind spot and the "gemm row m ignored"
    // blind spot: each of the 4 activation rows is genuinely different.
    tests.push_back({
        "non-uniform activations",
        8, 16,
        [](int r, int c) { return r * 100.0f + c * 0.5f; },
        [four_row_act](int m, int c) { return four_row_act(m, c, 16); }
    });

    // Test 2: negative weight values (real model weights are signed).
    tests.push_back({
        "negative weights",
        8, 16,
        [](int r, int c) { return (r - 4) * 10.5f - c * 0.75f; },
        [four_row_act](int m, int c) { return four_row_act(m, c, 16); }
    });

    // Test 3: multiple row-groups (16 rows = 2 groups of 8). Verifies the
    // outer loop advances between row-groups without overlap or skip.
    tests.push_back({
        "multiple row-groups (16 rows / 2 groups)",
        16, 16,
        [](int r, int c) { return (r - 8) * 2.25f - c * 0.5f; },
        [four_row_act](int m, int c) { return four_row_act(m, c, 16); }
    });

    // Test 3b: 3 row-groups (24 rows). Extends test 3 to make sure nothing
    // specific to "exactly 2 groups" was accidentally load-bearing.
    tests.push_back({
        "multiple row-groups (24 rows / 3 groups)",
        24, 16,
        [](int r, int c) { return (r - 12) * 1.75f - c * 0.5f; },
        [four_row_act](int m, int c) { return four_row_act(m, c, 16); }
    });

    // Test 4: larger K (32 cols = 4 column-groups). Stresses the inner loop
    // over more column-groups than the baseline's 2.
    tests.push_back({
        "larger K (32 cols)",
        8, 32,
        [](int r, int c) { return r * 10.25f - c * 0.375f; },
        [four_row_act](int m, int c) { return four_row_act(m, c, 32); }
    });

    // Test 5: fp16 rounding edge case. Weight magnitude (~12345) exceeds
    // fp16's exact-integer range (2048), so fp32_to_fp16 must round. The
    // reference deliberately uses the POST-rounding value (fp16_to_fp32 of
    // the stored fp16), matching exactly what the real kernel reads -- this
    // verifies packing/kernel fidelity to whatever rounding already happened,
    // not the rounding operation itself (repacking is a pure byte-copy, no
    // float conversion; the tensor is already fp16 by the time it's repacked).
    tests.push_back({
        "fp16 rounding edge case (large weights)",
        8, 8,
        [](int r, int c) { return 12345.0f + r * 1000.0f + c; },
        [](int /*m*/, int c) { return (c % 2 == 0) ? 1.0f : -1.0f; }
    });

    bool all_pass = true;
    for (auto & tc : tests) {
        all_pass &= run_test(tc);
    }

    printf("%s\n", all_pass ? "ALL TEST CASES PASSED" : "SOME TEST CASES FAILED");
    return all_pass ? 0 : 1;
}
