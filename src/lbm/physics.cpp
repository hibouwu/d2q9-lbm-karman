#include <lbm/physics.hpp>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <omp.h>

#include <lbm/communications.hpp>
#include <lbm/config.hpp>
#include <lbm/structures.hpp>

#if DIRECTIONS == 9 && DIMENSIONS == 2
/// Definition of the 9 base vectors used to discretize the directions on each mesh.
const Vector direction_matrix[DIRECTIONS] = {
  // clang-format off
  {+0.0, +0.0},
  {+1.0, +0.0}, {+0.0, +1.0}, {-1.0, +0.0}, {+0.0, -1.0},
  {+1.0, +1.0}, {-1.0, +1.0}, {-1.0, -1.0}, {+1.0, -1.0},
  // clang-format on
};
#else
#error Need to define adapted direction matrix.
#endif

#if DIRECTIONS == 9
/// Weights used to compensate the differences in length of the 9 directional vectors.
const double equil_weight[DIRECTIONS] = {
  // clang-format off
  4.0 / 9.0,
  1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
  // clang-format on
};

/// Opposite directions for bounce-back implementation
const int opposite_of[DIRECTIONS] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
#else
#error Need to define adapted equilibrium distribution function
#endif

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

double get_vect_norm_2(Vector const a, Vector const b) {
  double res = 0.0;
  for (size_t k = 0; k < DIMENSIONS; k++) {
    res += a[k] * b[k];
  }
  return res;
}

double get_cell_density(const Mesh* mesh, int x, int y) {
  double res = 0.0;
  for (int k = 0; k < DIRECTIONS; k++) {
    res += Mesh_get_f(mesh, k, x, y);
  }
  return res;
}

void get_cell_velocity(Vector v, const Mesh* mesh, int x, int y, double cell_density) {
  assert(v != NULL);
  for (int d = 0; d < DIMENSIONS; d++) {
    v[d] = 0.0;
    for (int k = 0; k < DIRECTIONS; k++) {
      v[d] += Mesh_get_f(mesh, k, x, y) * direction_matrix[k][d];
    }
    v[d] /= cell_density;
  }
}

double compute_equilibrium_profile(Vector velocity, double density, int direction) {
  const double v2 = get_vect_norm_2(velocity, velocity);
  const double p  = get_vect_norm_2(direction_matrix[direction], velocity);
  const double p2 = p * p;
  double f_eq = 1.0 + (3.0 * p) + ((9.0 / 2.0) * p2) - ((3.0 / 2.0) * v2);
  f_eq *= equil_weight[direction] * density;
  return f_eq;
}

// ---------------------------------------------------------------------------
// A/B experiment: collision implementation dispatch
// ---------------------------------------------------------------------------

typedef enum {
  COLLISION_BASELINE,
  COLLISION_UNROLLED,
  COLLISION_MULTICELL,
  COLLISION_MULTICELL_VEC,
} lbm_collision_impl_t;

static lbm_collision_impl_t get_collision_impl(void) {
  const char* env = getenv("LBM_COLLISION_IMPL");
  if (env == NULL || strcmp(env, "multicell_vec") == 0) return COLLISION_MULTICELL_VEC;
  if (strcmp(env, "multicell") == 0) return COLLISION_MULTICELL;
  if (strcmp(env, "baseline") == 0) return COLLISION_BASELINE;
  if (strcmp(env, "unrolled") == 0) return COLLISION_UNROLLED;
  fprintf(stderr, "[LBM] Unknown LBM_COLLISION_IMPL='%s', using multicell_vec\n", env);
  return COLLISION_MULTICELL_VEC;
}

// ── SoA hand-unrolled D2Q9 BGK (优化19 实验路径) ────────────────────────────
//
// D2Q9 directions (direction_matrix):
//   k=0:(0,0)  k=1:(1,0)  k=2:(0,1)  k=3:(-1,0)  k=4:(0,-1)
//   k=5:(1,1)  k=6:(-1,1) k=7:(-1,-1) k=8:(1,-1)
// Weights: k=0:4/9  k=1-4:1/9  k=5-8:1/36
static inline void compute_cell_collision_unrolled(Mesh* out, const Mesh* in, int x, int y) {
  const double f0 = Mesh_get_f(in, 0, x, y), f1 = Mesh_get_f(in, 1, x, y);
  const double f2 = Mesh_get_f(in, 2, x, y), f3 = Mesh_get_f(in, 3, x, y);
  const double f4 = Mesh_get_f(in, 4, x, y), f5 = Mesh_get_f(in, 5, x, y);
  const double f6 = Mesh_get_f(in, 6, x, y), f7 = Mesh_get_f(in, 7, x, y);
  const double f8 = Mesh_get_f(in, 8, x, y);

  const double density = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
  const double inv_rho = 1.0 / density;
  const double vx      = (f1 - f3 + f5 - f6 - f7 + f8) * inv_rho;
  const double vy      = (f2 - f4 + f5 + f6 - f7 - f8) * inv_rho;
  const double v2      = vx * vx + vy * vy;

  const double A         = 1.0 - RELAX_PARAMETER;
  const double omega_rho = RELAX_PARAMETER * density;
  const double w0        = (4.0 / 9.0)  * omega_rho;
  const double w14       = (1.0 / 9.0)  * omega_rho;
  const double w58       = (1.0 / 36.0) * omega_rho;
  const double base      = 1.0 - 1.5 * v2;

  const double base_vx2 = base + 4.5 * vx * vx;
  const double base_vy2 = base + 4.5 * vy * vy;

  const double ppp      = vx + vy;
  const double pmm      = vy - vx;
  const double base_pp2 = base + 4.5 * ppp * ppp;
  const double base_pm2 = base + 4.5 * pmm * pmm;

  const double vx3  = 3.0 * vx;
  const double vy3  = 3.0 * vy;
  const double ppp3 = 3.0 * ppp;
  const double pmm3 = 3.0 * pmm;

  Mesh_set_f(out, 0, x, y, A * f0 + w0  *  base);
  Mesh_set_f(out, 1, x, y, A * f1 + w14 * (base_vx2 + vx3));
  Mesh_set_f(out, 2, x, y, A * f2 + w14 * (base_vy2 + vy3));
  Mesh_set_f(out, 3, x, y, A * f3 + w14 * (base_vx2 - vx3));
  Mesh_set_f(out, 4, x, y, A * f4 + w14 * (base_vy2 - vy3));
  Mesh_set_f(out, 5, x, y, A * f5 + w58 * (base_pp2 + ppp3));
  Mesh_set_f(out, 6, x, y, A * f6 + w58 * (base_pm2 + pmm3));
  Mesh_set_f(out, 7, x, y, A * f7 + w58 * (base_pp2 - ppp3));
  Mesh_set_f(out, 8, x, y, A * f8 + w58 * (base_pm2 - pmm3));
}

// ── SoA baseline: fused 2-pass (优化16 equivalent) ──────────────────────────

void compute_cell_collision(Mesh* out, const Mesh* in, int x, int y) {
  // Pass 1: fused density + velocity reduction
  double density = 0.0, vx = 0.0, vy = 0.0;
  for (int k = 0; k < DIRECTIONS; k++) {
    const double fk = Mesh_get_f(in, k, x, y);
    density += fk;
    vx      += fk * direction_matrix[k][0];
    vy      += fk * direction_matrix[k][1];
  }
  vx /= density;
  vy /= density;
  const double v2 = vx * vx + vy * vy;

  // Pass 2: equilibrium + BGK update
  for (int k = 0; k < DIRECTIONS; k++) {
    const double fk   = Mesh_get_f(in, k, x, y);
    const double p    = direction_matrix[k][0] * vx + direction_matrix[k][1] * vy;
    const double f_eq = equil_weight[k] * density * (1.0 + 3.0 * p + 4.5 * p * p - 1.5 * v2);
    Mesh_set_f(out, k, x, y, fk - RELAX_PARAMETER * (fk - f_eq));
  }
}

// ── SoA multi-cell: hoisted plane pointers, j-continuous stride-1 (优化21) ──
//
// Plane pointers p0..p8 (read) and q0..q8 (write) are hoisted outside the i
// loop so that inside the j loop, cp_k[j] is a simple stride-1 load.  With
// __restrict__, the compiler can verify non-aliasing across all 18 streams and
// auto-vectorize the j loop: SIMD lanes carry consecutive cells, not directions.
//
// No #pragma omp simd: on Zen 4, vdivpd zmm throughput ≈ 1/24 CPI vs
// 8 pipelined scalar divsd ≈ 2 CPI.  The explicit pragma would force vdivpd;
// leaving it to the auto-vectorizer lets the compiler balance throughput.
static void collision_multicell(Mesh* mesh_out, const Mesh* mesh_in) {
  const int W = (int)mesh_in->width;
  const int H = (int)mesh_in->height;

  // Per-run constants (RELAX_PARAMETER is a compile-time or load-time constant)
  const double A      = 1.0 - RELAX_PARAMETER;
  const double rp_w0  = (4.0 / 9.0)  * RELAX_PARAMETER;
  const double rp_w14 = (1.0 / 9.0)  * RELAX_PARAMETER;
  const double rp_w58 = (1.0 / 36.0) * RELAX_PARAMETER;

  // Plane base pointers — non-overlapping regions of their respective allocations
  const double* __restrict__ p0 = Mesh_get_plane(mesh_in,  0);
  const double* __restrict__ p1 = Mesh_get_plane(mesh_in,  1);
  const double* __restrict__ p2 = Mesh_get_plane(mesh_in,  2);
  const double* __restrict__ p3 = Mesh_get_plane(mesh_in,  3);
  const double* __restrict__ p4 = Mesh_get_plane(mesh_in,  4);
  const double* __restrict__ p5 = Mesh_get_plane(mesh_in,  5);
  const double* __restrict__ p6 = Mesh_get_plane(mesh_in,  6);
  const double* __restrict__ p7 = Mesh_get_plane(mesh_in,  7);
  const double* __restrict__ p8 = Mesh_get_plane(mesh_in,  8);
  double* __restrict__       q0 = Mesh_get_plane(mesh_out, 0);
  double* __restrict__       q1 = Mesh_get_plane(mesh_out, 1);
  double* __restrict__       q2 = Mesh_get_plane(mesh_out, 2);
  double* __restrict__       q3 = Mesh_get_plane(mesh_out, 3);
  double* __restrict__       q4 = Mesh_get_plane(mesh_out, 4);
  double* __restrict__       q5 = Mesh_get_plane(mesh_out, 5);
  double* __restrict__       q6 = Mesh_get_plane(mesh_out, 6);
  double* __restrict__       q7 = Mesh_get_plane(mesh_out, 7);
  double* __restrict__       q8 = Mesh_get_plane(mesh_out, 8);

#pragma omp parallel for schedule(static)
  for (int i = 1; i < W - 1; i++) {
    const size_t col_off = (size_t)i * H;
    // Column pointers — stride-1 in j for every direction plane
    const double* cp0 = p0 + col_off;  const double* cp1 = p1 + col_off;
    const double* cp2 = p2 + col_off;  const double* cp3 = p3 + col_off;
    const double* cp4 = p4 + col_off;  const double* cp5 = p5 + col_off;
    const double* cp6 = p6 + col_off;  const double* cp7 = p7 + col_off;
    const double* cp8 = p8 + col_off;
    double* cq0 = q0 + col_off;  double* cq1 = q1 + col_off;
    double* cq2 = q2 + col_off;  double* cq3 = q3 + col_off;
    double* cq4 = q4 + col_off;  double* cq5 = q5 + col_off;
    double* cq6 = q6 + col_off;  double* cq7 = q7 + col_off;
    double* cq8 = q8 + col_off;

    for (int j = 1; j < H - 1; j++) {
      const double f0 = cp0[j], f1 = cp1[j], f2 = cp2[j];
      const double f3 = cp3[j], f4 = cp4[j], f5 = cp5[j];
      const double f6 = cp6[j], f7 = cp7[j], f8 = cp8[j];

      const double density = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
      const double inv_rho = 1.0 / density;
      const double vx      = (f1 - f3 + f5 - f6 - f7 + f8) * inv_rho;
      const double vy      = (f2 - f4 + f5 + f6 - f7 - f8) * inv_rho;
      const double v2      = vx * vx + vy * vy;

      const double w0  = rp_w0  * density;
      const double w14 = rp_w14 * density;
      const double w58 = rp_w58 * density;
      const double base      = 1.0 - 1.5 * v2;
      const double base_vx2  = base + 4.5 * vx * vx;
      const double base_vy2  = base + 4.5 * vy * vy;
      const double ppp       = vx + vy;
      const double pmm       = vy - vx;
      const double base_pp2  = base + 4.5 * ppp * ppp;
      const double base_pm2  = base + 4.5 * pmm * pmm;
      const double vx3       = 3.0 * vx;
      const double vy3       = 3.0 * vy;
      const double ppp3      = 3.0 * ppp;
      const double pmm3      = 3.0 * pmm;

      cq0[j] = A * f0 + w0  *  base;
      cq1[j] = A * f1 + w14 * (base_vx2 + vx3);
      cq2[j] = A * f2 + w14 * (base_vy2 + vy3);
      cq3[j] = A * f3 + w14 * (base_vx2 - vx3);
      cq4[j] = A * f4 + w14 * (base_vy2 - vy3);
      cq5[j] = A * f5 + w58 * (base_pp2 + ppp3);
      cq6[j] = A * f6 + w58 * (base_pm2 + pmm3);
      cq7[j] = A * f7 + w58 * (base_pp2 - ppp3);
      cq8[j] = A * f8 + w58 * (base_pm2 - pmm3);
    }
  }
}

// ── SoA multi-cell + vectorization fix (优化21 向量化实验) ───────────────────
//
// Same structure as collision_multicell, with two additions:
//   1. Column pointers cp0..cp8 / cq0..cq8 carry __restrict__ so GCC's alias
//      analysis sees them as non-aliasing inside the OMP closure.  Without this,
//      __restrict__ on the plane base pointers is not propagated through the
//      pointer-arithmetic expression (p0 + col_off) and GCC refuses to vectorize.
//   2. #pragma GCC ivdep asserts no loop-carried dependencies on the j loop.
//
// Root cause of non-vectorization in multicell: GCC reports
//   "no vectype for stmt: f0 = *cp0; scalar_type: const double"
// because without __restrict__ on cp0 itself, GCC cannot rule out aliasing
// between cp0..cp8 (reads) and cq0..cq8 (writes) inside the OMP inner function.
static void collision_multicell_vec(Mesh* mesh_out, const Mesh* mesh_in) {
  const int W = (int)mesh_in->width;
  const int H = (int)mesh_in->height;

  const double A      = 1.0 - RELAX_PARAMETER;
  const double rp_w0  = (4.0 / 9.0)  * RELAX_PARAMETER;
  const double rp_w14 = (1.0 / 9.0)  * RELAX_PARAMETER;
  const double rp_w58 = (1.0 / 36.0) * RELAX_PARAMETER;

  const double* __restrict__ p0 = Mesh_get_plane(mesh_in,  0);
  const double* __restrict__ p1 = Mesh_get_plane(mesh_in,  1);
  const double* __restrict__ p2 = Mesh_get_plane(mesh_in,  2);
  const double* __restrict__ p3 = Mesh_get_plane(mesh_in,  3);
  const double* __restrict__ p4 = Mesh_get_plane(mesh_in,  4);
  const double* __restrict__ p5 = Mesh_get_plane(mesh_in,  5);
  const double* __restrict__ p6 = Mesh_get_plane(mesh_in,  6);
  const double* __restrict__ p7 = Mesh_get_plane(mesh_in,  7);
  const double* __restrict__ p8 = Mesh_get_plane(mesh_in,  8);
  double* __restrict__       q0 = Mesh_get_plane(mesh_out, 0);
  double* __restrict__       q1 = Mesh_get_plane(mesh_out, 1);
  double* __restrict__       q2 = Mesh_get_plane(mesh_out, 2);
  double* __restrict__       q3 = Mesh_get_plane(mesh_out, 3);
  double* __restrict__       q4 = Mesh_get_plane(mesh_out, 4);
  double* __restrict__       q5 = Mesh_get_plane(mesh_out, 5);
  double* __restrict__       q6 = Mesh_get_plane(mesh_out, 6);
  double* __restrict__       q7 = Mesh_get_plane(mesh_out, 7);
  double* __restrict__       q8 = Mesh_get_plane(mesh_out, 8);

#pragma omp parallel for schedule(static)
  for (int i = 1; i < W - 1; i++) {
    const size_t col_off = (size_t)i * H;
    // __restrict__ on column pointers: propagates non-aliasing into the j loop
    const double* __restrict__ cp0 = p0 + col_off;
    const double* __restrict__ cp1 = p1 + col_off;
    const double* __restrict__ cp2 = p2 + col_off;
    const double* __restrict__ cp3 = p3 + col_off;
    const double* __restrict__ cp4 = p4 + col_off;
    const double* __restrict__ cp5 = p5 + col_off;
    const double* __restrict__ cp6 = p6 + col_off;
    const double* __restrict__ cp7 = p7 + col_off;
    const double* __restrict__ cp8 = p8 + col_off;
    double* __restrict__ cq0 = q0 + col_off;
    double* __restrict__ cq1 = q1 + col_off;
    double* __restrict__ cq2 = q2 + col_off;
    double* __restrict__ cq3 = q3 + col_off;
    double* __restrict__ cq4 = q4 + col_off;
    double* __restrict__ cq5 = q5 + col_off;
    double* __restrict__ cq6 = q6 + col_off;
    double* __restrict__ cq7 = q7 + col_off;
    double* __restrict__ cq8 = q8 + col_off;

#pragma GCC ivdep
    for (int j = 1; j < H - 1; j++) {
      const double f0 = cp0[j], f1 = cp1[j], f2 = cp2[j];
      const double f3 = cp3[j], f4 = cp4[j], f5 = cp5[j];
      const double f6 = cp6[j], f7 = cp7[j], f8 = cp8[j];

      const double density = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
      const double inv_rho = 1.0 / density;
      const double vx      = (f1 - f3 + f5 - f6 - f7 + f8) * inv_rho;
      const double vy      = (f2 - f4 + f5 + f6 - f7 - f8) * inv_rho;
      const double v2      = vx * vx + vy * vy;

      const double w0  = rp_w0  * density;
      const double w14 = rp_w14 * density;
      const double w58 = rp_w58 * density;
      const double base      = 1.0 - 1.5 * v2;
      const double base_vx2  = base + 4.5 * vx * vx;
      const double base_vy2  = base + 4.5 * vy * vy;
      const double ppp       = vx + vy;
      const double pmm       = vy - vx;
      const double base_pp2  = base + 4.5 * ppp * ppp;
      const double base_pm2  = base + 4.5 * pmm * pmm;
      const double vx3       = 3.0 * vx;
      const double vy3       = 3.0 * vy;
      const double ppp3      = 3.0 * ppp;
      const double pmm3      = 3.0 * pmm;

      cq0[j] = A * f0 + w0  *  base;
      cq1[j] = A * f1 + w14 * (base_vx2 + vx3);
      cq2[j] = A * f2 + w14 * (base_vy2 + vy3);
      cq3[j] = A * f3 + w14 * (base_vx2 - vx3);
      cq4[j] = A * f4 + w14 * (base_vy2 - vy3);
      cq5[j] = A * f5 + w58 * (base_pp2 + ppp3);
      cq6[j] = A * f6 + w58 * (base_pm2 + pmm3);
      cq7[j] = A * f7 + w58 * (base_pp2 - ppp3);
      cq8[j] = A * f8 + w58 * (base_pm2 - pmm3);
    }
  }
}

// ---------------------------------------------------------------------------
// Boundary conditions
// ---------------------------------------------------------------------------

void compute_bounce_back(Mesh* mesh, int x, int y) {
  double tmp[DIRECTIONS];
  for (int k = 0; k < DIRECTIONS; k++) {
    tmp[k] = Mesh_get_f(mesh, opposite_of[k], x, y);
  }
  for (int k = 0; k < DIRECTIONS; k++) {
    Mesh_set_f(mesh, k, x, y, tmp[k]);
  }
}

double helper_compute_poiseuille(const size_t i, const size_t size) {
  const double y = (double)(i - 1);
  const double L = (double)(size - 1);
  return 4.0 * INFLOW_MAX_VELOCITY / (L * L) * (L * y - y * y);
}

void compute_inflow_zou_he_poiseuille_distr(Mesh* mesh, int x, int y, size_t id_y) {
#if DIRECTIONS != 9
#error Implemented only for 9 directions
#endif
  const double v   = helper_compute_poiseuille(id_y, mesh->height);
  const double f0  = Mesh_get_f(mesh, 0, x, y);
  const double f2  = Mesh_get_f(mesh, 2, x, y);
  const double f3  = Mesh_get_f(mesh, 3, x, y);
  const double f4  = Mesh_get_f(mesh, 4, x, y);
  const double f6  = Mesh_get_f(mesh, 6, x, y);
  const double f7  = Mesh_get_f(mesh, 7, x, y);
  const double rho = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - v);

  Mesh_set_f(mesh, 1, x, y, f3);
  Mesh_set_f(mesh, 5, x, y, f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * v);
  Mesh_set_f(mesh, 8, x, y, f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * v);
}

void compute_outflow_zou_he_const_density(Mesh* mesh, int x, int y) {
#if DIRECTIONS != 9
#error Implemented only for 9 directions
#endif
  const double rho = 1.0;
  const double f0  = Mesh_get_f(mesh, 0, x, y);
  const double f1  = Mesh_get_f(mesh, 1, x, y);
  const double f2  = Mesh_get_f(mesh, 2, x, y);
  const double f4  = Mesh_get_f(mesh, 4, x, y);
  const double f5  = Mesh_get_f(mesh, 5, x, y);
  const double f8  = Mesh_get_f(mesh, 8, x, y);
  const double v   = -1.0 + (1.0 / rho) * (f0 + f2 + f4 + 2.0 * (f1 + f5 + f8));

  Mesh_set_f(mesh, 3, x, y, f1 - (2.0 / 3.0) * rho * v);
  Mesh_set_f(mesh, 7, x, y, f5 + 0.5 * (f2 - f4) - (1.0 / 6.0) * rho * v);
  Mesh_set_f(mesh, 6, x, y, f8 + 0.5 * (f4 - f2) - (1.0 / 6.0) * rho * v);
}

// ---------------------------------------------------------------------------
// Main step functions
// ---------------------------------------------------------------------------

void special_cells(Mesh* mesh, lbm_mesh_type_t* mesh_type, const lbm_comm_t* mesh_comm) {
#pragma omp parallel for schedule(static)
  for (size_t i = 1; i < mesh->width - 1; i++) {
    for (size_t j = 1; j < mesh->height - 1; j++) {
      switch (*(lbm_cell_type_t_get_cell(mesh_type, i, j))) {
      case CELL_FUILD:
        break;
      case CELL_BOUNCE_BACK:
        compute_bounce_back(mesh, i, j);
        break;
      case CELL_LEFT_IN:
        compute_inflow_zou_he_poiseuille_distr(mesh, i, j, j + mesh_comm->y);
        break;
      case CELL_RIGHT_OUT:
        compute_outflow_zou_he_const_density(mesh, i, j);
        break;
      }
    }
  }
}

void collision(Mesh* mesh_out, const Mesh* mesh_in) {
  assert(mesh_in->width == mesh_out->width);
  assert(mesh_in->height == mesh_out->height);

  static lbm_collision_impl_t impl = get_collision_impl();

  if (impl == COLLISION_MULTICELL) {
    collision_multicell(mesh_out, mesh_in);
  } else if (impl == COLLISION_MULTICELL_VEC) {
    collision_multicell_vec(mesh_out, mesh_in);
  } else if (impl == COLLISION_UNROLLED) {
#pragma omp parallel for schedule(static)
    for (size_t i = 1; i < mesh_in->width - 1; i++) {
      for (size_t j = 1; j < mesh_in->height - 1; j++) {
        compute_cell_collision_unrolled(mesh_out, mesh_in, i, j);
      }
    }
  } else {
#pragma omp parallel for schedule(static)
    for (size_t i = 1; i < mesh_in->width - 1; i++) {
      for (size_t j = 1; j < mesh_in->height - 1; j++) {
        compute_cell_collision(mesh_out, mesh_in, i, j);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Propagation — k-outer SoA implementation
//
// For direction k with offset (dx, dy):
//   plane_out[i*H + j] = plane_in[(i-dx)*H + (j-dy)]
//
// Outer loop over k keeps each iteration within one pair of W*H planes,
// eliminating the 9-plane scatter/gather of the per-cell baseline.
// The inner j copy is stride-1 → handled as a memcpy per column.
//
// i range for direction k: [max(0,dx), W+min(0,dx))  (source i-dx stays in [0,W))
// j range for direction k: [max(0,dy), H+min(0,dy))  (source j-dy stays in [0,H))
//
// Thread parallelism: one omp parallel region wraps all k iterations;
// omp for distributes the i loop across threads with nowait so fast threads
// advance to the next direction without waiting.  Different k values write to
// disjoint planes, so concurrent execution across directions is safe.
// ---------------------------------------------------------------------------

void propagation(Mesh* mesh_out, const Mesh* mesh_in) {
  const int W = (int)mesh_out->width;
  const int H = (int)mesh_out->height;
#pragma omp parallel
  {
    for (int k = 0; k < DIRECTIONS; k++) {
      const int dx       = (int)direction_matrix[k][0];
      const int dy       = (int)direction_matrix[k][1];
      double*       pout = Mesh_get_plane(mesh_out, k);
      const double* pin  = Mesh_get_plane(mesh_in,  k);
      const int i_lo     = (dx > 0) ? dx : 0;
      const int i_hi     = (dx < 0) ? W + dx : W;
      const int j_lo     = (dy > 0) ? dy : 0;
      const int j_hi     = (dy < 0) ? H + dy : H;
      const size_t nb    = (size_t)(j_hi - j_lo) * sizeof(double);
#pragma omp for schedule(static) nowait
      for (int i = i_lo; i < i_hi; i++) {
        memcpy(pout + i * H + j_lo,
               pin  + (i - dx) * H + j_lo - dy, nb);
      }
    }
  }
}

void propagation_interior(Mesh* mesh_out, const Mesh* mesh_in, bool has_vert_neighbors) {
  const int W = (int)mesh_out->width;
  const int H = (int)mesh_out->height;
#pragma omp parallel
  {
    for (int k = 0; k < DIRECTIONS; k++) {
      const int dx       = (int)direction_matrix[k][0];
      const int dy       = (int)direction_matrix[k][1];
      double*       pout = Mesh_get_plane(mesh_out, k);
      const double* pin  = Mesh_get_plane(mesh_in,  k);
      const int i_lo     = (dx > 0) ? dx : 0;
      const int i_hi     = (dx < 0) ? W + dx : W;
      const int j_lo     = (dy > 0) ? dy : 0;
      const int j_hi     = (dy < 0) ? H + dy : H;

      if (!has_vert_neighbors) {
        const size_t nb = (size_t)(j_hi - j_lo) * sizeof(double);
#pragma omp for schedule(static) nowait
        for (int i = i_lo; i < i_hi; i++) {
          memcpy(pout + i * H + j_lo,
                 pin  + (i - dx) * H + j_lo - dy, nb);
        }
      } else {
        // Interior rows: j=0, j=2..H-3, j=H-1 (skip j=1 and j=H-2).
        // j=1 reads from ghost j=0 (not yet received); j=H-2 reads from ghost j=H-1.
        const int jm_lo      = (j_lo < 2) ? 2 : j_lo;
        const int jm_hi      = (j_hi > H - 2) ? H - 2 : j_hi;
        const size_t nb_mid  = (jm_hi > jm_lo) ? (size_t)(jm_hi - jm_lo) * sizeof(double) : 0;
#pragma omp for schedule(static) nowait
        for (int i = i_lo; i < i_hi; i++) {
          const double* bi = pin  + (i - dx) * H - dy;
          double*       bo = pout + i * H;
          if (j_lo == 0) bo[0] = bi[0];
          if (nb_mid > 0) memcpy(bo + jm_lo, bi + jm_lo, nb_mid);
          if (j_hi == H)  bo[H - 1] = bi[H - 1];
        }
      }
    }
  }
}

void propagation_border(Mesh* mesh_out, const Mesh* mesh_in) {
  // Only j=1 and j=H-2, which read from ghost rows now available after Waitall.
  // For all dy∈{-1,0,1}: sj=1-dy∈{0,1,2} and sj=(H-2)-dy∈{H-3,H-2,H-1} — always valid.
  const int W = (int)mesh_out->width;
  const int H = (int)mesh_out->height;
#pragma omp parallel
  {
    for (int k = 0; k < DIRECTIONS; k++) {
      const int dx       = (int)direction_matrix[k][0];
      const int dy       = (int)direction_matrix[k][1];
      double*       pout = Mesh_get_plane(mesh_out, k);
      const double* pin  = Mesh_get_plane(mesh_in,  k);
      const int i_lo     = (dx > 0) ? dx : 0;
      const int i_hi     = (dx < 0) ? W + dx : W;
#pragma omp for schedule(static) nowait
      for (int i = i_lo; i < i_hi; i++) {
        const double* bi = pin  + (i - dx) * H - dy;
        double*       bo = pout + i * H;
        bo[1]     = bi[1];
        bo[H - 2] = bi[H - 2];
      }
    }
  }
}
