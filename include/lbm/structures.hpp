#pragma once

#include <cstdint>
#include <cstdio>

#include <lbm/config.hpp>

/// @brief A raw double buffer alias (used for MPI comm scratch buffers).
typedef double* lbm_mesh_cell_t;

/// @brief Representation of a vector to manipulate macroscopic velocities.
typedef double Vector[DIMENSIONS];

/// @brief Defines a mesh for the local domain.
///
/// SoA memory layout: cells[k * width * height + x * height + y]
/// All DIRECTIONS=9 direction planes are stored contiguously, plane after plane.
/// Within each plane, x varies slower than y (column-major within each plane).
typedef struct Mesh {
  /// Flat buffer: DIRECTIONS planes of (width * height) doubles.
  double* cells;
  /// Width of the local mesh (phantom/ghost columns included).
  uint32_t width;
  /// Height of the local mesh (phantom/ghost rows included).
  uint32_t height;
} Mesh;

/// @brief Cell types definitions in order to know which process to apply when computing.
typedef enum lbm_cell_type_e {
  /// Standard fluid cell. Applies to collisions.
  CELL_FUILD,
  /// Obstacle of top/bottom border cell. Applies to reflexions.
  CELL_BOUNCE_BACK,
  /// In-border cell. Applies to `Zou/He` with fixed `V`.
  CELL_LEFT_IN,
  /// Out-border cell. Applies to `Zou/He` with constant density gradient.
  CELL_RIGHT_OUT
} lbm_cell_type_t;

/// @brief Array storing the information on the types of cells.
typedef struct lbm_mesh_type_s {
  lbm_cell_type_t* types;
  uint32_t width;
  uint32_t height;
} lbm_mesh_type_t;

/// @brief Header structure for the output file.
typedef struct lbm_file_header_s {
  uint32_t magick;
  uint32_t mesh_width;
  uint32_t mesh_height;
  uint32_t lines;
} lbm_file_header_t;

/// @brief An entry of the output file with both macroscopic quantities.
typedef struct lbm_file_entry_s {
  float v;
  float rho;
} lbm_file_entry_t;

/// @brief Structure to read the output file.
typedef struct lbm_data_file_s {
  FILE* fp;
  lbm_file_header_t header;
  lbm_file_entry_t* entries;
} lbm_data_file_t;

void Mesh_init(Mesh* mesh, uint32_t width, uint32_t height);
void Mesh_release(Mesh* mesh);
void lbm_mesh_type_t_init(lbm_mesh_type_t* mesh, uint32_t width, uint32_t height);
void lbm_mesh_type_t_release(lbm_mesh_type_t* mesh);
void save_frame(FILE* fp, const Mesh* mesh);
void fatal(const char* message);

// ---------------------------------------------------------------------------
// SoA accessors — layout: cells[k * W * H + x * H + y]
// ---------------------------------------------------------------------------

/// Get f_k at grid position (x, y).
static inline double Mesh_get_f(const Mesh* mesh, int k, int x, int y) {
  return mesh->cells[(size_t)k * mesh->width * mesh->height
                     + (size_t)x * mesh->height + (size_t)y];
}

/// Set f_k at grid position (x, y).
static inline void Mesh_set_f(Mesh* mesh, int k, int x, int y, double v) {
  mesh->cells[(size_t)k * mesh->width * mesh->height
              + (size_t)x * mesh->height + (size_t)y] = v;
}

/// Get base pointer for direction-k plane (W*H contiguous doubles).
static inline double* Mesh_get_plane(const Mesh* mesh, int k) {
  return &mesh->cells[(size_t)k * mesh->width * mesh->height];
}

/// @brief Retrieves a pointer on the cell type given coordinates.
static inline lbm_cell_type_t* lbm_cell_type_t_get_cell(
    const lbm_mesh_type_t* meshtype, uint32_t x, uint32_t y) {
  return &meshtype->types[x * meshtype->height + y];
}
