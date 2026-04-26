#pragma once

/// @file profiling.hpp
/// @brief Phase-level wall-clock profiling for the LBM main loop.
///
/// Enabled by defining LBM_ENABLE_PROFILING at compile time:
///   cmake -DLBM_ENABLE_PROFILING=ON ...
///
/// When the macro is NOT defined, all four macros expand to ((void)0).
/// No branches, no global state, no link-time symbols — zero overhead.
///
/// Typical usage pattern:
///   LBM_PROF_INIT();                  // once, before the loop
///   for (int i = 0; i < N; i++) {
///       LBM_PROF_BEGIN(PROF_LOOP_TOTAL);
///       LBM_PROF_BEGIN(PROF_COLLISION);
///       collision(...);
///       LBM_PROF_END(PROF_COLLISION);
///       LBM_PROF_END(PROF_LOOP_TOTAL);
///   }
///   LBM_PROF_REPORT();                // once, after the loop

/// Phase identifiers.
/// Numeric order determines the row order in the printed report table.
typedef enum lbm_prof_phase_e {
    // --- Phase 1: coarse main-loop phases ---
    PROF_SPECIAL_CELLS = 0,       ///< special_cells()
    PROF_COLLISION,               ///< collision()
    PROF_HALO_START,              ///< lbm_comm_halo_exchange_start()
    PROF_PROPAGATION_INTERIOR,    ///< propagation_interior()
    PROF_HALO_FINISH,             ///< lbm_comm_halo_exchange_finish()
    PROF_PROPAGATION_BORDER,      ///< propagation_border() — only when nb_y > 1
    PROF_SAVE_FRAME,              ///< save_frame_all_domain() — only on write steps
    PROF_LOOP_TOTAL,              ///< full time-step wall time; denominator for pct

    // --- Phase 2: halo_exchange_start internals ---
    PROF_HALO_HORIZ,              ///< horizontal blocking send/recv (no-ops when nb_x=1)
    PROF_HALO_VERT_PACK,          ///< memcpy packing of vertical send buffers
    PROF_HALO_VERT_POST,          ///< MPI_Irecv + MPI_Isend posting

    // --- Phase 2: halo_exchange_finish internals ---
    PROF_HALO_WAITALL,            ///< MPI_Waitall on posted vertical requests
    PROF_HALO_VERT_UNPACK,        ///< memcpy unpacking of vertical recv buffers
    PROF_HALO_DIAG,               ///< diagonal corner blocking send/recv

    // --- Phase 2: save_frame_all_domain internals ---
    PROF_FRAME_BUILD,             ///< build_frame_buf(): local density+velocity compute
    PROF_FRAME_MPI,               ///< MPI_Send / MPI_Recv of packed float frame data
    PROF_FRAME_FWRITE,            ///< fwrite() to disk

    PROF_COUNT                    ///< sentinel — total number of tracked phases
} lbm_prof_phase_t;

#ifdef LBM_ENABLE_PROFILING

/// Zero all per-rank accumulators. Call exactly once before the main loop.
void lbm_prof_init(void);

/// Record the wall-clock start of @p phase on this rank.
/// Not reentrant per phase: always pair with lbm_prof_end before calling again
/// for the same phase id.
void lbm_prof_begin(lbm_prof_phase_t phase);

/// Record the wall-clock end of @p phase; accumulate elapsed time and call count.
void lbm_prof_end(lbm_prof_phase_t phase);

/// Reduce per-rank statistics across all MPI ranks (MPI_Reduce, no extra barrier)
/// and print a formatted summary table on rank 0.
/// Must be called after the main loop and before MPI_Finalize.
void lbm_prof_report(void);

#define LBM_PROF_INIT()         lbm_prof_init()
#define LBM_PROF_BEGIN(phase)   lbm_prof_begin(phase)
#define LBM_PROF_END(phase)     lbm_prof_end(phase)
#define LBM_PROF_REPORT()       lbm_prof_report()

#else   // LBM_ENABLE_PROFILING not defined — all macros are zero-cost no-ops

#define LBM_PROF_INIT()         ((void)0)
#define LBM_PROF_BEGIN(phase)   ((void)0)
#define LBM_PROF_END(phase)     ((void)0)
#define LBM_PROF_REPORT()       ((void)0)

#endif  // LBM_ENABLE_PROFILING
