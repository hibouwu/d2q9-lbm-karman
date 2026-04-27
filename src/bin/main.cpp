#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <mpi.h>
#include <omp.h>

#include <lbm/lib.hpp>

/// @brief Writes the output file's header.
/// @param fp File descriptor to write to.
/// @param mesh_comm Domain to save.
static void write_file_header(FILE* fp, lbm_comm_t* mesh_comm) {
  // Setup header values
  lbm_file_header_t header = {
    .magick      = RESULT_MAGICK,
    .mesh_width  = MESH_WIDTH,
    .mesh_height = MESH_HEIGHT,
    .lines       = static_cast<uint32_t>(mesh_comm->nb_y),
  };

  // Write file
  fwrite(&header, sizeof(header), 1, fp);
}

/// @brief Opens the output file's header in write mode.
/// @return File descriptor to write to.
static FILE* open_output_file() {
  // No output if empty filename
  if (RESULT_FILENAME == NULL) {
    return NULL;
  }

  // Open result file
  FILE* fp = fopen(RESULT_FILENAME, "wb");
  if (fp == NULL) {
    perror(RESULT_FILENAME);
    abort();
  }

  return fp;
}

static inline void close_file(FILE* fp) {
  fclose(fp);
}

int main(int argc, char* argv[]) {
  // Init MPI, get current rank and communicator size.
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  // Get config filename
  char* config_filename;
  if (argc == 2) {
    config_filename = strdup(argv[1]);
  } else {
    fprintf(stderr, "Usage: %s <CONFIG_FILE>\n", argv[0]);
    return -1;
  }

  // Load config file and display it on master node
  load_config(config_filename);
  if (rank == RANK_MASTER) {
    print_config();
  }

  // Init structures, allocate memory...
  lbm_comm_t mesh_comm;
  lbm_comm_init(&mesh_comm, rank, comm_size, MESH_WIDTH, MESH_HEIGHT);

  Mesh mesh;
  Mesh_init(&mesh, lbm_comm_width(&mesh_comm), lbm_comm_height(&mesh_comm));

  Mesh temp;
  Mesh_init(&temp, lbm_comm_width(&mesh_comm), lbm_comm_height(&mesh_comm));

  lbm_mesh_type_t mesh_type;
  lbm_mesh_type_t_init(&mesh_type, lbm_comm_width(&mesh_comm), lbm_comm_height(&mesh_comm));

  // Master open the output file
  FILE* fp = NULL;
  if (rank == RANK_MASTER) {
    fp = open_output_file();
    // Write header
    write_file_header(fp, &mesh_comm);
  }

  // Setup initial conditions on mesh
  setup_init_state(&mesh, &mesh_type, &mesh_comm);
  setup_init_state(&temp, &mesh_type, &mesh_comm);

  // Write initial condition in output file
  if (lbm_gbl_config.output_filename != NULL) {
    save_frame_all_domain(fp, &mesh);
  }

  // Barrier to wait for all processes before starting
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == RANK_MASTER) {
    putc('\n', stdout);
  }

  LBM_PROF_INIT();

  // Time steps
  const double start_time = MPI_Wtime();
  for (ssize_t i = 1; i <= ITERATIONS; i++) {
    if (rank == RANK_MASTER) {
      fprintf(stderr, "\rStep: %6d/%6d", i, ITERATIONS);
    }

    LBM_PROF_BEGIN(PROF_LOOP_TOTAL);

    // Compute special actions (border, obstacle...)
    LBM_PROF_BEGIN(PROF_SPECIAL_CELLS);
    special_cells(&mesh, &mesh_type, &mesh_comm);
    LBM_PROF_END(PROF_SPECIAL_CELLS);

    const bool has_vn = (mesh_comm.nb_y > 1);

    if (!has_vn) {
      // Single-rank vertical path: no vertical halo needed, simple serial structure.
      LBM_PROF_BEGIN(PROF_COLLISION);
      collision(&temp, &mesh);
      LBM_PROF_END(PROF_COLLISION);

      LBM_PROF_BEGIN(PROF_HALO_START);
      lbm_comm_halo_exchange_start(&mesh_comm, &temp);
      LBM_PROF_END(PROF_HALO_START);

      LBM_PROF_BEGIN(PROF_PROPAGATION_INTERIOR);
      propagation_interior(&mesh, &temp, false);
      LBM_PROF_END(PROF_PROPAGATION_INTERIOR);

      LBM_PROF_BEGIN(PROF_HALO_FINISH);
      lbm_comm_halo_exchange_finish(&mesh_comm, &temp);
      LBM_PROF_END(PROF_HALO_FINISH);
    } else {
      // Extended overlap path: one omp parallel region spans collision through
      // propagation_border, maximising the window where interior work overlaps
      // with in-flight MPI transfers.
      //
      // Execution order inside the region:
      //   1. border collision (j=1, j=H-2)          — all threads
      //   2. halo_start (Isend/Irecv)                — master only
      //   3. interior collision (j=2..H-3)           — all threads, overlapped
      //   4. propagation_interior                    — all threads, overlapped
      //   5. halo_finish (Waitall + unpack)          — master only
      //   6. propagation_border (j=1, j=H-2)        — all threads
      //
      // All LBM_PROF_* calls inside the region are wrapped in omp master to
      // avoid races on the shared profiling accumulators.

      const int H = (int)temp.height;

#pragma omp parallel
      {
        // ── 1. Border collision ──────────────────────────────────────────────
#pragma omp master
        { LBM_PROF_BEGIN(PROF_COLLISION); }

        collision_rows(&temp, &mesh, 1, 2);       // j=1;   implicit barrier
        collision_rows(&temp, &mesh, H - 2, H - 1); // j=H-2; implicit barrier
        // All threads synchronized here; border row data ready to pack.

        // ── 2. Post async vertical halo (master only) ────────────────────────
        // Non-master threads skip omp master and proceed immediately to step 3.
        // Master joins step 3 after posting Isend/Irecv (fast, non-blocking).
#pragma omp master
        {
          LBM_PROF_BEGIN(PROF_HALO_START);
          lbm_comm_halo_exchange_start(&mesh_comm, &temp);
          LBM_PROF_END(PROF_HALO_START);
        }
        // No barrier: interior collision does not read ghost rows.

        // ── 3. Interior collision (overlapped with in-flight MPI) ───────────
        collision_rows(&temp, &mesh, 2, H - 2);   // implicit barrier

#pragma omp master
        { LBM_PROF_END(PROF_COLLISION); }

        // ── 4. Propagation interior (overlapped with in-flight MPI) ─────────
#pragma omp master
        { LBM_PROF_BEGIN(PROF_PROPAGATION_INTERIOR); }

        propagation_interior_omp_region(&mesh, &temp, true);
        // ends with explicit #pragma omp barrier

#pragma omp master
        { LBM_PROF_END(PROF_PROPAGATION_INTERIOR); }

        // ── 5. Wait for halo transfers (master only) ─────────────────────────
#pragma omp master
        {
          LBM_PROF_BEGIN(PROF_HALO_FINISH);
          lbm_comm_halo_exchange_finish(&mesh_comm, &temp);
          LBM_PROF_END(PROF_HALO_FINISH);
        }
        // Barrier: propagation_border reads ghost rows filled by halo_finish.
#pragma omp barrier

        // ── 6. Border propagation (uses now-available ghost rows) ────────────
#pragma omp master
        { LBM_PROF_BEGIN(PROF_PROPAGATION_BORDER); }

        propagation_border_omp_region(&mesh, &temp);
        // ends with explicit #pragma omp barrier

#pragma omp master
        { LBM_PROF_END(PROF_PROPAGATION_BORDER); }
      } // end #pragma omp parallel
    }

    // Save step
    if (i % WRITE_STEP_INTERVAL == 0 && lbm_gbl_config.output_filename != NULL) {
      LBM_PROF_BEGIN(PROF_SAVE_FRAME);
      save_frame_all_domain(fp, &mesh);
      LBM_PROF_END(PROF_SAVE_FRAME);
    }

    LBM_PROF_END(PROF_LOOP_TOTAL);
  }
  const double end_time      = MPI_Wtime();
  const double elapsed_time  = end_time - start_time;
  const uint64_t total_cells = static_cast<uint64_t>(MESH_WIDTH) * MESH_HEIGHT;
  const double mlups         = (static_cast<double>(total_cells) * ITERATIONS) / (elapsed_time * 1e6);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == RANK_MASTER) {
    if (fp != NULL) {
      close_file(fp);
    }
    fprintf(stderr, "\rSIMULATION COMPLETED.\n\n");
    fprintf(stderr, "FOM:  %.2f MLUPS\n", mlups);
  }

  LBM_PROF_REPORT();

  // Free memory
  lbm_comm_release(&mesh_comm);
  Mesh_release(&mesh);
  Mesh_release(&temp);
  lbm_mesh_type_t_release(&mesh_type);

  // Close MPI
  MPI_Finalize();

  return EXIT_SUCCESS;
}
