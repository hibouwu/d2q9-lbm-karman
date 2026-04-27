#pragma once

#include <lbm/communications.hpp>
#include <lbm/structures.hpp>

extern const int opposite_of[DIRECTIONS];
extern const double equil_weight[DIRECTIONS];
extern const Vector direction_matrix[DIRECTIONS];

/** ------------------------------------------------------------------------ **
 * Helper functions                                                           *
 ** ------------------------------------------------------------------------ **/

/// @brief Computes the product of two vectors.
double get_vect_norm_2(const Vector a, const Vector b);

/// @brief Computes macroscopic density at (x, y) by summing all 9 f_k.
double get_cell_density(const Mesh* mesh, int x, int y);

/// @brief Computes macroscopic velocity at (x, y) given its density.
void get_cell_velocity(Vector v, const Mesh* mesh, int x, int y, double cell_density);

/// @brief Provides the Poiseuille velocity for position i in a tube of given size.
double helper_compute_poiseuille(const size_t i, const size_t size);

/** ------------------------------------------------------------------------ **
 * Collision functions                                                        *
 ** ------------------------------------------------------------------------ **/

/// @brief Computes microscopic equilibrium profile for a given direction.
double compute_equilibrium_profile(Vector velocity, double density, int direction);

/// @brief BGK collision for cell (x, y): reads from in, writes to out.
void compute_cell_collision(Mesh* out, const Mesh* in, int x, int y);

/** ------------------------------------------------------------------------ **
 * Limit conditions                                                           *
 ** ------------------------------------------------------------------------ **/

/// @brief Applies bounce-back reflection on cell (x, y).
void compute_bounce_back(Mesh* mesh, int x, int y);

/// @brief Applies Zou/He inflow boundary at cell (x, y) with global y-index id_y.
void compute_inflow_zou_he_poiseuille_distr(Mesh* mesh, int x, int y, size_t id_y);

/// @brief Applies Zou/He constant-density outflow boundary at cell (x, y).
void compute_outflow_zou_he_const_density(Mesh* mesh, int x, int y);

/** ------------------------------------------------------------------------ **
 * Main functions                                                             *
 ** ------------------------------------------------------------------------ **/

/// @brief Applies special actions linked to boundary/obstacle conditions.
void special_cells(Mesh* mesh, lbm_mesh_type_t* mesh_type, const lbm_comm_t* mesh_comm);

/// @brief Computes BGK collision for all interior cells.
void collision(Mesh* mesh_out, const Mesh* mesh_in);

/// @brief Propagates densities to neighbour cells (full mesh).
void propagation(Mesh* mesh_out, const Mesh* mesh_in);

/// @brief Interior propagation: all rows except j=1 and j=height-2 when has_vert_neighbors.
void propagation_interior(Mesh* mesh_out, const Mesh* mesh_in, bool has_vert_neighbors);

/// @brief Border propagation: only rows j=1 and j=height-2 (need finished halo).
void propagation_border(Mesh* mesh_out, const Mesh* mesh_in);

/// @brief BGK collision for i=1..W-2, j in [j_begin, j_end).
/// Must be called from within an existing #pragma omp parallel region.
/// Uses #pragma omp for (no new parallel region). Ends with implicit barrier.
void collision_rows(Mesh* mesh_out, const Mesh* mesh_in, int j_begin, int j_end);

/// @brief Interior propagation inside an existing #pragma omp parallel region.
/// Uses omp for (no new parallel region). Ends with explicit barrier.
void propagation_interior_omp_region(Mesh* mesh_out, const Mesh* mesh_in, bool has_vert_neighbors);

/// @brief Border propagation inside an existing #pragma omp parallel region.
/// Uses omp for (no new parallel region). Ends with explicit barrier.
void propagation_border_omp_region(Mesh* mesh_out, const Mesh* mesh_in);
