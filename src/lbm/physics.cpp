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

typedef enum { COLLISION_BASELINE, COLLISION_UNROLLED } lbm_collision_impl_t;

static lbm_collision_impl_t get_collision_impl(void) {
  const char* env = getenv("LBM_COLLISION_IMPL");
  if (env == NULL || strcmp(env, "baseline") == 0) return COLLISION_BASELINE;
  if (strcmp(env, "unrolled") == 0) return COLLISION_UNROLLED;
  fprintf(stderr, "[LBM] Unknown LBM_COLLISION_IMPL='%s', using baseline\n", env);
  return COLLISION_BASELINE;
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

  if (impl == COLLISION_UNROLLED) {
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
