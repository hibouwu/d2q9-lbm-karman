#include <lbm/initialization.hpp>

#include <cassert>
#include <cstddef>

#include <mpi.h>

#include <lbm/physics.hpp>

void init_cond_velocity_0_density_1(Mesh* mesh) {
  assert(mesh != NULL);
  for (size_t i = 0; i < mesh->width; i++) {
    for (size_t j = 0; j < mesh->height; j++) {
      for (int k = 0; k < DIRECTIONS; k++) {
        Mesh_set_f(mesh, k, i, j, equil_weight[k]);
      }
    }
  }
}

void setup_init_state_circle_obstacle(
    Mesh* mesh, lbm_mesh_type_t* mesh_type, const lbm_comm_t* mesh_comm) {
  for (size_t j = mesh_comm->y; j < mesh->height + mesh_comm->y; j++) {
    for (size_t i = mesh_comm->x; i < mesh->width + mesh_comm->x; i++) {
      if (((i - OBSTACLE_X) * (i - OBSTACLE_X))
          + ((j - OBSTACLE_Y) * (j - OBSTACLE_Y)) <= OBSTACLE_R * OBSTACLE_R) {
        *(lbm_cell_type_t_get_cell(mesh_type, i - mesh_comm->x, j - mesh_comm->y)) =
          CELL_BOUNCE_BACK;
      }
    }
  }
}

void setup_init_state_global_poiseuille_profile(
    Mesh* mesh, lbm_mesh_type_t* mesh_type, const lbm_comm_t* mesh_comm) {
  Vector v         = {0.0, 0.0};
  const double rho = 1.0;

  for (size_t i = 0; i < mesh->width; i++) {
    for (size_t j = 0; j < mesh->height; j++) {
      v[0] = helper_compute_poiseuille(j + mesh_comm->y, MESH_HEIGHT);
      for (int k = 0; k < DIRECTIONS; k++) {
        double val = compute_equilibrium_profile(v, rho, k);
        // Cells past column 1 start at rest (null speed)
        if (i > 1) {
          val = equil_weight[k];
        }
        Mesh_set_f(mesh, k, i, j, val);
      }
      *(lbm_cell_type_t_get_cell(mesh_type, i, j)) = CELL_FUILD;
    }
  }
}

void setup_init_state_border(
    Mesh* mesh, lbm_mesh_type_t* mesh_type, const lbm_comm_t* mesh_comm) {
  Vector v         = {0.0, 0.0};
  const double rho = 1.0;

  if (mesh_comm->left_id == -1) {
    for (size_t j = 1; j < mesh->height - 1; j++) {
      *(lbm_cell_type_t_get_cell(mesh_type, 0, j)) = CELL_LEFT_IN;
    }
  }

  if (mesh_comm->right_id == -1) {
    for (size_t j = 1; j < mesh->height - 1; j++) {
      *(lbm_cell_type_t_get_cell(mesh_type, mesh->width - 1, j)) = CELL_RIGHT_OUT;
    }
  }

  if (mesh_comm->top_id == -1) {
    for (size_t i = 0; i < mesh->width; i++) {
      for (int k = 0; k < DIRECTIONS; k++) {
        Mesh_set_f(mesh, k, i, 0, compute_equilibrium_profile(v, rho, k));
      }
      *(lbm_cell_type_t_get_cell(mesh_type, i, 0)) = CELL_BOUNCE_BACK;
    }
  }

  if (mesh_comm->bottom_id == -1) {
    for (size_t i = 0; i < mesh->width; i++) {
      for (int k = 0; k < DIRECTIONS; k++) {
        Mesh_set_f(mesh, k, i, mesh->height - 1,
                   compute_equilibrium_profile(v, rho, k));
      }
      *(lbm_cell_type_t_get_cell(mesh_type, i, mesh->height - 1)) = CELL_BOUNCE_BACK;
    }
  }
}

void setup_init_state(
    Mesh* mesh, lbm_mesh_type_t* mesh_type, const lbm_comm_t* mesh_comm) {
  setup_init_state_global_poiseuille_profile(mesh, mesh_type, mesh_comm);
  setup_init_state_border(mesh, mesh_type, mesh_comm);
  setup_init_state_circle_obstacle(mesh, mesh_type, mesh_comm);
}
