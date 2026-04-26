#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <lbm/communications.hpp>
#include <lbm/physics.hpp>
#include <lbm/profiling.hpp>

// ---------------------------------------------------------------------------
// Frame I/O helpers (output path, not hot)
// ---------------------------------------------------------------------------

void save_frame(FILE* fp, const Mesh* mesh) {
  lbm_file_entry_t buffer[WRITE_BUFFER_ENTRIES];
  size_t cnt = 0;
  for (size_t i = 1; i < mesh->width - 1; i++) {
    for (size_t j = 1; j < mesh->height - 1; j++) {
      const double density = get_cell_density(mesh, i, j);
      Vector v;
      get_cell_velocity(v, mesh, i, j, density);
      const double norm = std::sqrt(get_vect_norm_2(v, v));
      buffer[cnt].rho   = (float)density;
      buffer[cnt].v     = (float)norm;
      cnt++;
      assert(cnt <= WRITE_BUFFER_ENTRIES);
      if (cnt == WRITE_BUFFER_ENTRIES) {
        fwrite(buffer, sizeof(lbm_file_entry_t), cnt, fp);
        cnt = 0;
      }
    }
  }
  if (cnt != 0) {
    fwrite(buffer, sizeof(lbm_file_entry_t), cnt, fp);
  }
}

static int lbm_helper_pgcd(int a, int b) {
  int c;
  while (b != 0) {
    c = a % b;
    a = b;
    b = c;
  }
  return a;
}

static int helper_get_rank_id(int nb_x, int nb_y, int rank_x, int rank_y) {
  if (rank_x < 0 || rank_x >= nb_x) {
    return -1;
  } else if (rank_y < 0 || rank_y >= nb_y) {
    return -1;
  } else {
    return (rank_x + rank_y * nb_x);
  }
}

void lbm_comm_print(const lbm_comm_t* mesh_comm) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  static bool first_call = true;
  if (first_call && rank == RANK_MASTER) {
    first_call = false;
    fprintf(
      stderr,
      "%4s| %8s %8s %8s %8s | %12s %12s %12s %12s | %6s %6s | %6s %6s\n",
      "RANK", "TOP", "BOTTOM", "LEFT", "RIGHT",
      "TOP LEFT", "TOP RIGHT", "BOTTOM LEFT", "BOTTOM RIGHT",
      "POS X", "POS Y", "DIM X", "DIM Y");
  }
  MPI_Barrier(MPI_COMM_WORLD);
  fprintf(
    stderr,
    "%4d| %7d  %7d  %7d  %7d  | %11d  %11d  %11d  %11d  | %5d  %5d  | %5d  %5d \n",
    rank,
    mesh_comm->top_id, mesh_comm->bottom_id,
    mesh_comm->left_id, mesh_comm->right_id,
    mesh_comm->corner_id[CORNER_TOP_LEFT],    mesh_comm->corner_id[CORNER_TOP_RIGHT],
    mesh_comm->corner_id[CORNER_BOTTOM_LEFT], mesh_comm->corner_id[CORNER_BOTTOM_RIGHT],
    mesh_comm->x, mesh_comm->y,
    mesh_comm->width, mesh_comm->height);
}

void lbm_comm_init(lbm_comm_t* mesh_comm, int rank, int comm_size, uint32_t width, uint32_t height) {
  int nb_y = lbm_helper_pgcd(comm_size, width);
  int nb_x = comm_size / nb_y;

  assert(nb_x * nb_y == comm_size);
  if (height % nb_y != 0) {
    fatal("Can't get a 2D cut for current problem size and number of processes.");
  }

  int rank_x = rank % nb_x;
  int rank_y = rank / nb_x;

  mesh_comm->nb_x = nb_x;
  mesh_comm->nb_y = nb_y;

  mesh_comm->width  = width  / nb_x + 2;
  mesh_comm->height = height / nb_y + 2;

  mesh_comm->x = rank_x * width  / nb_x;
  mesh_comm->y = rank_y * height / nb_y;

  mesh_comm->left_id                        = helper_get_rank_id(nb_x, nb_y, rank_x - 1, rank_y);
  mesh_comm->right_id                       = helper_get_rank_id(nb_x, nb_y, rank_x + 1, rank_y);
  mesh_comm->top_id                         = helper_get_rank_id(nb_x, nb_y, rank_x, rank_y - 1);
  mesh_comm->bottom_id                      = helper_get_rank_id(nb_x, nb_y, rank_x, rank_y + 1);
  mesh_comm->corner_id[CORNER_TOP_LEFT]     = helper_get_rank_id(nb_x, nb_y, rank_x - 1, rank_y - 1);
  mesh_comm->corner_id[CORNER_TOP_RIGHT]    = helper_get_rank_id(nb_x, nb_y, rank_x + 1, rank_y - 1);
  mesh_comm->corner_id[CORNER_BOTTOM_LEFT]  = helper_get_rank_id(nb_x, nb_y, rank_x - 1, rank_y + 1);
  mesh_comm->corner_id[CORNER_BOTTOM_RIGHT] = helper_get_rank_id(nb_x, nb_y, rank_x + 1, rank_y + 1);

  mesh_comm->n_requests = 0;
  if (nb_y > 1) {
    mesh_comm->buffer = static_cast<double*>(
      malloc(4 * sizeof(double) * DIRECTIONS * width / nb_x));
  } else {
    mesh_comm->buffer = NULL;
  }
  // Preallocate horizontal pack/unpack buffer (inner_h * DIRECTIONS doubles).
  // Reused for all 4 blocking horizontal send/recv calls per step.
  const size_t inner_h = mesh_comm->height - 2;
  mesh_comm->horiz_buf = static_cast<double*>(
    malloc(inner_h * DIRECTIONS * sizeof(double)));

  lbm_comm_print(mesh_comm);
}

void lbm_comm_release(lbm_comm_t* mesh_comm) {
  mesh_comm->x        = 0;
  mesh_comm->y        = 0;
  mesh_comm->width    = 0;
  mesh_comm->height   = 0;
  mesh_comm->right_id = -1;
  mesh_comm->left_id  = -1;
  if (mesh_comm->buffer != NULL) {
    free(mesh_comm->buffer);
  }
  free(mesh_comm->horiz_buf);
  mesh_comm->horiz_buf = NULL;
}

// ---------------------------------------------------------------------------
// Horizontal ghost-column exchange (SoA: explicit pack/unpack)
//
// Packs column x (y=1..height-2) into a heap buffer as cell-interleaved
// doubles [f0(x,1), f1(x,1), ..., f8(x,1), f0(x,2), ...], then sends or
// receives and unpacks. Blocking.
// ---------------------------------------------------------------------------
static void lbm_comm_sync_ghosts_horizontal(
  lbm_comm_t*       mesh_comm,
  Mesh*             mesh_to_process,
  lbm_comm_type_t   comm_type,
  int               target_rank,
  uint32_t          x
) {
  if (target_rank == -1) return;

  const int inner_h = (int)mesh_comm->height - 2;   // y=1..height-2
  const int count   = inner_h * DIRECTIONS;
  double* buf       = mesh_comm->horiz_buf;           // preallocated, no heap churn

  MPI_Status status;
  if (comm_type == COMM_SEND) {
    for (int j = 1; j <= inner_h; j++) {
      for (int k = 0; k < DIRECTIONS; k++) {
        buf[(j - 1) * DIRECTIONS + k] = Mesh_get_f(mesh_to_process, k, (int)x, j);
      }
    }
    MPI_Send(buf, count, MPI_DOUBLE, target_rank, 0, MPI_COMM_WORLD);
  } else {
    MPI_Recv(buf, count, MPI_DOUBLE, target_rank, 0, MPI_COMM_WORLD, &status);
    for (int j = 1; j <= inner_h; j++) {
      for (int k = 0; k < DIRECTIONS; k++) {
        Mesh_set_f(mesh_to_process, k, (int)x, j, buf[(j - 1) * DIRECTIONS + k]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Diagonal corner-cell exchange (SoA: pack/unpack single cell's 9 values)
// ---------------------------------------------------------------------------
static void lbm_comm_sync_ghosts_diagonal(
  Mesh*           mesh_to_process,
  lbm_comm_type_t comm_type,
  int             target_rank,
  uint32_t        x,
  uint32_t        y
) {
  if (target_rank == -1) return;

  double cell_buf[DIRECTIONS];
  MPI_Status status;

  if (comm_type == COMM_SEND) {
    for (int k = 0; k < DIRECTIONS; k++) {
      cell_buf[k] = Mesh_get_f(mesh_to_process, k, (int)x, (int)y);
    }
    MPI_Send(cell_buf, DIRECTIONS, MPI_DOUBLE, target_rank, 0, MPI_COMM_WORLD);
  } else {
    MPI_Recv(cell_buf, DIRECTIONS, MPI_DOUBLE, target_rank, 0, MPI_COMM_WORLD, &status);
    for (int k = 0; k < DIRECTIONS; k++) {
      Mesh_set_f(mesh_to_process, k, (int)x, (int)y, cell_buf[k]);
    }
  }
}

// ---------------------------------------------------------------------------
// Halo exchange
//
// Phase 1: horizontal (blocking) + async vertical Isend/Irecv.
//
// Vertical buffer sub-layout (count = (width-3)*DIRECTIONS each):
//   buffer + 0*count  sbuf_bot — packed send toward bottom neighbour
//   buffer + 1*count  rbuf_top — receive from top neighbour
//   buffer + 2*count  sbuf_top — packed send toward top neighbour
//   buffer + 3*count  rbuf_bot — receive from bottom neighbour
//
// Pack format (SoA → cell-interleaved):
//   [(f0,f1,..,f8) for x=1], [(f0,f1,..,f8) for x=2], ..., [(f0,f1,..,f8) for x=W-2]
// ---------------------------------------------------------------------------
void lbm_comm_halo_exchange_start(lbm_comm_t* mesh, Mesh* mesh_to_process) {
  LBM_PROF_BEGIN(PROF_HALO_HORIZ);
  lbm_comm_sync_ghosts_horizontal(mesh, mesh_to_process, COMM_SEND, mesh->right_id, mesh->width - 2);
  lbm_comm_sync_ghosts_horizontal(mesh, mesh_to_process, COMM_RECV, mesh->left_id,  0);
  lbm_comm_sync_ghosts_horizontal(mesh, mesh_to_process, COMM_SEND, mesh->left_id,  1);
  lbm_comm_sync_ghosts_horizontal(mesh, mesh_to_process, COMM_RECV, mesh->right_id, mesh->width - 1);
  LBM_PROF_END(PROF_HALO_HORIZ);

  mesh->n_requests = 0;
  if (mesh->nb_y <= 1) return;

  const size_t ncells = mesh_to_process->width - 3;
  const size_t count  = ncells * DIRECTIONS;
  double* sbuf_bot    = mesh->buffer + 0 * count;
  double* rbuf_top    = mesh->buffer + 1 * count;
  double* sbuf_top    = mesh->buffer + 2 * count;
  double* rbuf_bot    = mesh->buffer + 3 * count;

  // Pack both send rows before posting any request
  LBM_PROF_BEGIN(PROF_HALO_VERT_PACK);
  if (mesh->bottom_id != -1) {
    for (size_t x = 1; x < mesh_to_process->width - 2; x++) {
      for (int k = 0; k < DIRECTIONS; k++) {
        sbuf_bot[(x - 1) * DIRECTIONS + k] =
          Mesh_get_f(mesh_to_process, k, (int)x, (int)(mesh->height - 2));
      }
    }
  }
  if (mesh->top_id != -1) {
    for (size_t x = 1; x < mesh_to_process->width - 2; x++) {
      for (int k = 0; k < DIRECTIONS; k++) {
        sbuf_top[(x - 1) * DIRECTIONS + k] =
          Mesh_get_f(mesh_to_process, k, (int)x, 1);
      }
    }
  }
  LBM_PROF_END(PROF_HALO_VERT_PACK);

  LBM_PROF_BEGIN(PROF_HALO_VERT_POST);
  if (mesh->top_id != -1) {
    MPI_Irecv(rbuf_top, count, MPI_DOUBLE, mesh->top_id,    0, MPI_COMM_WORLD,
              &mesh->requests[mesh->n_requests++]);
  }
  if (mesh->bottom_id != -1) {
    MPI_Irecv(rbuf_bot, count, MPI_DOUBLE, mesh->bottom_id, 0, MPI_COMM_WORLD,
              &mesh->requests[mesh->n_requests++]);
  }
  if (mesh->bottom_id != -1) {
    MPI_Isend(sbuf_bot, count, MPI_DOUBLE, mesh->bottom_id, 0, MPI_COMM_WORLD,
              &mesh->requests[mesh->n_requests++]);
  }
  if (mesh->top_id != -1) {
    MPI_Isend(sbuf_top, count, MPI_DOUBLE, mesh->top_id,    0, MPI_COMM_WORLD,
              &mesh->requests[mesh->n_requests++]);
  }
  LBM_PROF_END(PROF_HALO_VERT_POST);
}

// Phase 2: Waitall + unpack vertical ghost rows + diagonal exchanges.
void lbm_comm_halo_exchange_finish(lbm_comm_t* mesh, Mesh* mesh_to_process) {
  if (mesh->n_requests > 0) {
    LBM_PROF_BEGIN(PROF_HALO_WAITALL);
    MPI_Waitall(mesh->n_requests, mesh->requests, MPI_STATUSES_IGNORE);
    mesh->n_requests = 0;
    LBM_PROF_END(PROF_HALO_WAITALL);

    const size_t ncells = mesh_to_process->width - 3;
    const size_t count  = ncells * DIRECTIONS;
    double* rbuf_top    = mesh->buffer + 1 * count;
    double* rbuf_bot    = mesh->buffer + 3 * count;

    LBM_PROF_BEGIN(PROF_HALO_VERT_UNPACK);
    if (mesh->top_id != -1) {
      for (size_t x = 1; x < mesh_to_process->width - 2; x++) {
        for (int k = 0; k < DIRECTIONS; k++) {
          Mesh_set_f(mesh_to_process, k, (int)x, 0,
                     rbuf_top[(x - 1) * DIRECTIONS + k]);
        }
      }
    }
    if (mesh->bottom_id != -1) {
      for (size_t x = 1; x < mesh_to_process->width - 2; x++) {
        for (int k = 0; k < DIRECTIONS; k++) {
          Mesh_set_f(mesh_to_process, k, (int)x, (int)(mesh->height - 1),
                     rbuf_bot[(x - 1) * DIRECTIONS + k]);
        }
      }
    }
    LBM_PROF_END(PROF_HALO_VERT_UNPACK);
  }

  LBM_PROF_BEGIN(PROF_HALO_DIAG);
  lbm_comm_sync_ghosts_diagonal(mesh_to_process, COMM_SEND, mesh->corner_id[CORNER_TOP_LEFT],     1,               1);
  lbm_comm_sync_ghosts_diagonal(mesh_to_process, COMM_RECV, mesh->corner_id[CORNER_BOTTOM_RIGHT], mesh->width - 1, mesh->height - 1);
  lbm_comm_sync_ghosts_diagonal(mesh_to_process, COMM_SEND, mesh->corner_id[CORNER_BOTTOM_LEFT],  1,               mesh->height - 2);
  lbm_comm_sync_ghosts_diagonal(mesh_to_process, COMM_RECV, mesh->corner_id[CORNER_TOP_RIGHT],    mesh->width - 1, 0);
  lbm_comm_sync_ghosts_diagonal(mesh_to_process, COMM_SEND, mesh->corner_id[CORNER_TOP_RIGHT],    mesh->width - 2, 1);
  lbm_comm_sync_ghosts_diagonal(mesh_to_process, COMM_RECV, mesh->corner_id[CORNER_BOTTOM_LEFT],  0,               mesh->height - 1);
  lbm_comm_sync_ghosts_diagonal(mesh_to_process, COMM_SEND, mesh->corner_id[CORNER_BOTTOM_RIGHT], mesh->width - 2, mesh->height - 2);
  lbm_comm_sync_ghosts_diagonal(mesh_to_process, COMM_RECV, mesh->corner_id[CORNER_TOP_LEFT],     0,               0);
  LBM_PROF_END(PROF_HALO_DIAG);
}

void lbm_comm_halo_exchange(lbm_comm_t* mesh, Mesh* mesh_to_process) {
  lbm_comm_halo_exchange_start(mesh, mesh_to_process);
  lbm_comm_halo_exchange_finish(mesh, mesh_to_process);
}

// ---------------------------------------------------------------------------
// Frame output: build compact (rho, v) buffer from mesh interior cells
// ---------------------------------------------------------------------------

static lbm_file_entry_t* build_frame_buf(const Mesh* mesh, int* out_ncells) {
  const int iw     = (int)mesh->width  - 2;
  const int ih     = (int)mesh->height - 2;
  const int ncells = iw * ih;
  lbm_file_entry_t* buf =
    static_cast<lbm_file_entry_t*>(malloc((size_t)ncells * sizeof(lbm_file_entry_t)));
  int idx = 0;
  for (int i = 1; i <= iw; i++) {
    for (int j = 1; j <= ih; j++) {
      const double density = get_cell_density(mesh, i, j);
      Vector v;
      get_cell_velocity(v, mesh, i, j, density);
      buf[idx].rho = (float)density;
      buf[idx].v   = (float)std::sqrt(get_vect_norm_2(v, v));
      idx++;
    }
  }
  *out_ncells = ncells;
  return buf;
}

void save_frame_all_domain(FILE* fp, Mesh* source_mesh) {
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (comm_size == 1) {
    save_frame(fp, source_mesh);
    return;
  }

  LBM_PROF_BEGIN(PROF_FRAME_BUILD);
  int ncells;
  lbm_file_entry_t* local_buf = build_frame_buf(source_mesh, &ncells);
  LBM_PROF_END(PROF_FRAME_BUILD);

  if (rank == RANK_MASTER) {
    LBM_PROF_BEGIN(PROF_FRAME_FWRITE);
    fwrite(local_buf, sizeof(lbm_file_entry_t), ncells, fp);
    LBM_PROF_END(PROF_FRAME_FWRITE);
    free(local_buf);

    lbm_file_entry_t* recv_buf =
      static_cast<lbm_file_entry_t*>(malloc((size_t)ncells * sizeof(lbm_file_entry_t)));
    for (int i = 1; i < comm_size; i++) {
      LBM_PROF_BEGIN(PROF_FRAME_MPI);
      MPI_Recv(recv_buf, ncells * 2, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      LBM_PROF_END(PROF_FRAME_MPI);

      LBM_PROF_BEGIN(PROF_FRAME_FWRITE);
      fwrite(recv_buf, sizeof(lbm_file_entry_t), ncells, fp);
      LBM_PROF_END(PROF_FRAME_FWRITE);
    }
    free(recv_buf);
  } else {
    LBM_PROF_BEGIN(PROF_FRAME_MPI);
    MPI_Send(local_buf, ncells * 2, MPI_FLOAT, RANK_MASTER, 0, MPI_COMM_WORLD);
    LBM_PROF_END(PROF_FRAME_MPI);
    free(local_buf);
  }
}
