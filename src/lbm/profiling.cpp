#include <lbm/profiling.hpp>

#ifdef LBM_ENABLE_PROFILING

#include <cassert>
#include <cstdio>
#include <cstring>

#include <mpi.h>

// ---------------------------------------------------------------------------
// Per-rank accumulators (file-scope, no external linkage)
// ---------------------------------------------------------------------------

static double s_total[PROF_COUNT];  // wall seconds accumulated per phase
static long   s_count[PROF_COUNT];  // invocation count per phase
static double s_t0[PROF_COUNT];     // MPI_Wtime() snapshot at last lbm_prof_begin

// Human-readable labels — must stay in sync with lbm_prof_phase_t order.
static const char* const k_names[PROF_COUNT] = {
    // Phase 1
    "special_cells",
    "collision",
    "halo_start",
    "propagation_interior",
    "halo_finish",
    "propagation_border",
    "save_frame",
    "--- loop total ---",
    // Phase 2: halo_start internals
    "  halo: horizontal",
    "  halo: vert_pack",
    "  halo: vert_post",
    // Phase 2: halo_finish internals
    "  halo: waitall",
    "  halo: vert_unpack",
    "  halo: diagonal",
    // Phase 2: save_frame_all_domain internals
    "  frame: build_buf",
    "  frame: mpi",
    "  frame: fwrite",
};

// Compile-time check: array length == PROF_COUNT
static_assert(sizeof(k_names) / sizeof(k_names[0]) == PROF_COUNT,
              "k_names length mismatch with lbm_prof_phase_t");

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

void lbm_prof_init(void) {
    memset(s_total, 0, sizeof(s_total));
    memset(s_count, 0, sizeof(s_count));
    memset(s_t0,    0, sizeof(s_t0));
}

void lbm_prof_begin(lbm_prof_phase_t phase) {
    s_t0[phase] = MPI_Wtime();
}

void lbm_prof_end(lbm_prof_phase_t phase) {
    s_total[phase] += MPI_Wtime() - s_t0[phase];
    s_count[phase]++;
}

void lbm_prof_report(void) {
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // Two MPI_Reduce calls — one for sum (→ avg), one for max.
    // No extra MPI_Barrier: MPI_Reduce itself provides collective synchronisation.
    double sum_t[PROF_COUNT];
    double max_t[PROF_COUNT];
    MPI_Reduce(s_total, sum_t, PROF_COUNT, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(s_total, max_t, PROF_COUNT, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        return;
    }

    // Denominator for percentage: average loop total across ranks.
    const double loop_avg = (s_count[PROF_LOOP_TOTAL] > 0)
                            ? sum_t[PROF_LOOP_TOTAL] / comm_size
                            : 1.0;

    // Table header
    fprintf(stderr, "\n");
    fprintf(stderr, "=== LBM Phase Profiling  ranks=%d ===\n", comm_size);
    fprintf(stderr, "%-26s  %12s  %12s  %12s  %9s\n",
            "Phase", "avg_total_s", "max_total_s", "avg/call", "pct_loop");
    fprintf(stderr, "%-26s  %12s  %12s  %12s  %9s\n",
            "-------------------------",
            "------------", "------------", "------------", "---------");

    // Print one row per phase; skip phases that were never called.
    for (int p = 0; p < PROF_COUNT; p++) {
        if (s_count[p] == 0) {
            continue;
        }

        const double avg_t   = sum_t[p] / comm_size;
        const double per_c   = avg_t / (double)s_count[p];  // seconds per call
        const double pct     = avg_t / loop_avg * 100.0;

        if (p == PROF_LOOP_TOTAL) {
            // Separator before the total row
            fprintf(stderr, "%-26s  %12s  %12s  %12s  %9s\n",
                    "-------------------------",
                    "------------", "------------", "------------", "---------");
            fprintf(stderr, "%-26s  %12.5f  %12.5f  %9.4fms  %8.2f%%\n",
                    k_names[p], avg_t, max_t[p], per_c * 1e3, pct);
        } else {
            fprintf(stderr, "%-26s  %12.5f  %12.5f  %9.4fms  %8.2f%%\n",
                    k_names[p], avg_t, max_t[p], per_c * 1e3, pct);
        }
    }
    fprintf(stderr, "\n");
    fprintf(stderr,
            "Notes:\n"
            "  avg_total_s  : mean across ranks of total time spent in this phase\n"
            "  max_total_s  : slowest rank's total — primary bottleneck indicator\n"
            "  avg/call     : avg_total_s / call_count (local rank 0 count)\n"
            "  pct_loop     : avg_total_s / avg loop_total * 100\n"
            "  Phase-2 rows : sub-timings inside halo_start/finish and save_frame;\n"
            "                 their sum approximates the enclosing Phase-1 row.\n"
            "\n");
}

#endif  // LBM_ENABLE_PROFILING
