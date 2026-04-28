#!/usr/bin/env bash
# bench.sh — official D2Q9 LBM benchmark entry point (opt26 final code)
#
# Runs FOM + profiling in one pass. All environment preconditions are hard
# requirements; any failure exits non-zero and stops all further work.
#
# Output: scripts/bench_runs/YYYYMMDD-HHMMSS/
#   meta.txt               machine, compiler, MPI, git rev
#   fom.tsv                7 configs × run1/run2/run3/median (MLUPS)
#   profiling_raw.txt      3 profiling runs (full stdout)
#   profiling_selected.txt best run (max FOM = proxy for min loop_total avg)
#   console.log            all build + run output
#   status.txt             SUCCESS or FAILED

set -euo pipefail

# ── Constants ─────────────────────────────────────────────────────────────────

MPICXX=/usr/lib64/mpich/bin/mpicxx
MPICC=/usr/lib64/mpich/bin/mpicc
MPIEXEC=/usr/lib64/mpich/bin/mpiexec
MPI_LIBDIR=/usr/lib64/mpich/lib
MPI_ENV=(env -u MPI_HOME LD_LIBRARY_PATH="$MPI_LIBDIR")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
CFG="$ROOT/bench_config.txt"

TIMESTAMP=$(date '+%Y%m%d-%H%M%S')
RUN_DIR="$SCRIPT_DIR/bench_runs/$TIMESTAMP"

BUILD_FOM="$ROOT/build-bench-fom"
BUILD_PROF="$ROOT/build-bench-prof"
LIB_FOM="$BUILD_FOM/lib/libtop.lbm-lib.so"
LIB_PROF="$BUILD_PROF/lib/libtop.lbm-lib.so"
BIN_FOM="$BUILD_FOM/top.lbm-exe"
BIN_PROF="$BUILD_PROF/top.lbm-exe"

NRUNS=3

# FOM config matrix: "np:omp"
FOM_CONFIGS=(
  "1:4"
  "1:16"
  "2:4"
  "2:8"
  "2:16"
  "4:4"
  "4:8"
)

PROF_NP=2
PROF_OMP=8

# ── State ─────────────────────────────────────────────────────────────────────

FREQ_LOCKED=false
LOG=""   # set after RUN_DIR is created

# ── Cleanup trap ──────────────────────────────────────────────────────────────

cleanup() {
  local ec=$?
  if $FREQ_LOCKED; then
    if sudo -n cpupower frequency-set -g schedutil &>/dev/null; then
      echo "[cleanup] CPU governor restored to schedutil."
    else
      echo "WARNING: Failed to restore CPU governor." >&2
      echo "         Run manually: sudo cpupower frequency-set -g schedutil" >&2
      ec=1
    fi
  fi
  if [[ -d "$RUN_DIR" ]]; then
    local status="FAILED"
    [[ $ec -eq 0 ]] && status="SUCCESS"
    echo "$status" > "$RUN_DIR/status.txt"
    echo "[bench] $status — results in: $RUN_DIR"
  fi
  exit $ec
}
trap cleanup EXIT

# ── Logging ───────────────────────────────────────────────────────────────────

log()  { echo "$*" | tee -a "$LOG"; }
logn() { printf '%s' "$*" | tee -a "$LOG"; }
die()  {
  echo "FATAL: $*" | tee -a "$LOG" >&2
  exit 1
}

# ── Setup output dir ──────────────────────────────────────────────────────────

mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/console.log"
: > "$LOG"
: > "$RUN_DIR/fom.tsv"
: > "$RUN_DIR/profiling_raw.txt"

log "[bench] $(date '+%Y-%m-%d %H:%M:%S') — output: $RUN_DIR"

# ── Prechecks ─────────────────────────────────────────────────────────────────

log ""
log "=== Prechecks ==="

for bin in "$MPICXX" "$MPICC" "$MPIEXEC"; do
  [[ -x "$bin" ]] || die "Not found: $bin"
done
log "[ok] MPI tools"

[[ -f "$CFG" ]] || die "Config not found: $CFG"
log "[ok] Config: $CFG"

command -v cpupower &>/dev/null || die "cpupower not found — required for frequency locking"
log "[ok] cpupower found"

sudo -n cpupower frequency-set -g performance &>/dev/null \
  || die "sudo -n cpupower frequency-set -g performance failed — check /etc/sudoers"
log "[ok] sudo cpupower"

sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' &>/dev/null \
  || die "sudo -n drop_caches failed — check /etc/sudoers"
log "[ok] sudo drop_caches"

log "All prechecks passed."

# ── meta.txt ──────────────────────────────────────────────────────────────────

{
  echo "timestamp:   $TIMESTAMP"
  echo "machine:     $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
  echo "kernel:      $(uname -r)"
  echo "cores:       $(nproc --all) logical / $(nproc) physical"
  echo "compiler:    $("$MPICXX" --version 2>&1 | head -1)"
  echo "mpiexec:     $("$MPIEXEC" --version 2>&1 | head -1)"
  echo "git_rev:     $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo N/A)"
  echo "build_fom:   $BUILD_FOM"
  echo "build_prof:  $BUILD_PROF"
} > "$RUN_DIR/meta.txt"
cat "$RUN_DIR/meta.txt" >> "$LOG"

# ── Lock CPU frequency ────────────────────────────────────────────────────────

log ""
log "=== Lock CPU frequency ==="
sudo -n cpupower frequency-set -g performance >> "$LOG" 2>&1
FREQ_LOCKED=true
log "[ok] Governor set to performance."

# ── Build ─────────────────────────────────────────────────────────────────────

do_build() {
  local build_dir=$1 profiling=$2
  log ""
  log "=== Build: $(basename "$build_dir") (PROFILING=$profiling) ==="
  rm -f "$build_dir/CMakeCache.txt"
  cmake -S "$ROOT" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-march=native" \
    -DLBM_ENABLE_PROFILING="$profiling" \
    -DCMAKE_CXX_COMPILER="$MPICXX" \
    -DCMAKE_C_COMPILER="$MPICC" \
    -DMPIEXEC_EXECUTABLE="$MPIEXEC" \
    >> "$LOG" 2>&1 || die "cmake configure failed: $build_dir"
  cmake --build "$build_dir" --parallel "$(nproc)" >> "$LOG" 2>&1 \
    || die "cmake build failed: $build_dir"
  log "Build done."
}

do_build "$BUILD_FOM" "OFF"
do_build "$BUILD_PROF" "ON"

# ── Post-build checks ─────────────────────────────────────────────────────────

log ""
log "=== Post-build checks ==="

check_ldd() {
  local lib=$1
  if ! ldd "$lib" 2>/dev/null | grep "libmpi" | grep -q "$MPI_LIBDIR"; then
    die "ldd: $lib does not link to MPI from $MPI_LIBDIR"
  fi
  log "[ok] ldd: $(basename "$lib") → $MPI_LIBDIR"
}

check_zmm() {
  local lib=$1
  local count
  count=$(objdump -d "$lib" | grep -c "zmm" || true)
  [[ "$count" -gt 0 ]] \
    || die "AVX-512: 0 zmm instructions in $lib — rebuild with -march=native failed"
  log "[ok] AVX-512: $count zmm in $(basename "$lib")"
}

for lib in "$LIB_FOM" "$LIB_PROF"; do
  [[ -f "$lib" ]] || die "Library missing after build: $lib"
  check_ldd "$lib"
  check_zmm "$lib"
done

log "Post-build checks passed."

# ── Helpers ───────────────────────────────────────────────────────────────────

drop_caches() {
  sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null \
    || die "drop_caches failed during run"
}

median() {
  sort -n | awk '{a[NR]=$1} END{
    n=NR
    if (n%2==1) print a[(n+1)/2]
    else        printf "%.2f\n", (a[n/2]+a[n/2+1])/2
  }'
}

extract_fom() {
  grep -oP '[0-9]+\.[0-9]+ MLUPS' | grep -oP '[0-9]+\.[0-9]+' || true
}

# ── FOM benchmark ─────────────────────────────────────────────────────────────

log ""
log "==================================================================="
log " FOM benchmark (PROFILING=OFF)"
log "==================================================================="

printf 'np\tomp\trun1\trun2\trun3\tmedian\n' >> "$RUN_DIR/fom.tsv"

run_fom() {
  local np=$1 omp=$2
  log ""
  log "--- np=$np OMP=$omp ---"

  # Warmup
  logn "  [warmup] "
  drop_caches
  OMP_NUM_THREADS=$omp "${MPI_ENV[@]}" "$MPIEXEC" -n "$np" "$BIN_FOM" "$CFG" \
    >> "$LOG" 2>&1 || die "Warmup failed: np=$np OMP=$omp"
  log "done"

  # Formal runs
  local foms=()
  logn "  [fom   ] "
  for (( i=1; i<=NRUNS; i++ )); do
    drop_caches
    local out
    out=$(OMP_NUM_THREADS=$omp "${MPI_ENV[@]}" "$MPIEXEC" -n "$np" "$BIN_FOM" "$CFG" 2>&1) \
      || die "Run $i failed: np=$np OMP=$omp"
    echo "$out" >> "$LOG"

    local f
    f=$(echo "$out" | extract_fom)
    [[ -n "$f" ]] || die "No MLUPS in output: np=$np OMP=$omp run $i"
    foms+=("$f")
    logn "$f  "
  done
  log ""

  local med
  med=$(printf '%s\n' "${foms[@]}" | median)
  log "  median: $med MLUPS"

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$np" "$omp" "${foms[0]}" "${foms[1]}" "${foms[2]}" "$med" \
    >> "$RUN_DIR/fom.tsv"
}

for cfg in "${FOM_CONFIGS[@]}"; do
  IFS=: read -r np omp <<< "$cfg"
  run_fom "$np" "$omp"
done

log ""
log "FOM complete. fom.tsv written."

# ── Profiling benchmark ───────────────────────────────────────────────────────

log ""
log "==================================================================="
log " Profiling benchmark (PROFILING=ON, np=$PROF_NP OMP=$PROF_OMP)"
log "==================================================================="

prof_foms=()
prof_outputs=()

# Warmup
logn "  [warmup] "
drop_caches
OMP_NUM_THREADS=$PROF_OMP "${MPI_ENV[@]}" "$MPIEXEC" -n "$PROF_NP" "$BIN_PROF" "$CFG" \
  >> "$LOG" 2>&1 || die "Profiling warmup failed"
log "done"

for (( i=1; i<=NRUNS; i++ )); do
  log ""
  log "--- Profiling run $i/$NRUNS ---"
  drop_caches

  local_out=$(OMP_NUM_THREADS=$PROF_OMP "${MPI_ENV[@]}" \
    "$MPIEXEC" -n "$PROF_NP" "$BIN_PROF" "$CFG" 2>&1) \
    || die "Profiling run $i failed"

  echo "$local_out" >> "$LOG"

  # Require phase table
  echo "$local_out" | grep -qE "collision|halo_start|propagation" \
    || die "Profiling run $i: no phase table in output"

  local pf
  pf=$(echo "$local_out" | extract_fom)
  [[ -n "$pf" ]] || die "Profiling run $i: no MLUPS in output"

  prof_foms+=("$pf")
  prof_outputs+=("$local_out")
  log "  FOM: $pf MLUPS"

  {
    printf '=== Profiling run %d (FOM: %s MLUPS) ===\n' "$i" "$pf"
    echo "$local_out"
    echo ""
  } >> "$RUN_DIR/profiling_raw.txt"
done

# Select run with max FOM (= min loop_total avg)
best_idx=0
best_fom=${prof_foms[0]}
for (( i=1; i<NRUNS; i++ )); do
  if (( $(echo "${prof_foms[$i]} > $best_fom" | bc -l) )); then
    best_fom=${prof_foms[$i]}
    best_idx=$i
  fi
done

{
  printf '# Selected: run %d of %d (max FOM = %s MLUPS)\n' \
    "$((best_idx+1))" "$NRUNS" "$best_fom"
  echo "# Criterion: max FOM across $NRUNS runs (proxy for min loop_total avg)"
  echo ""
  echo "${prof_outputs[$best_idx]}"
} > "$RUN_DIR/profiling_selected.txt"

log ""
log "Profiling complete. Selected run $((best_idx+1)) (FOM=$best_fom MLUPS)."

# ── Final summary ─────────────────────────────────────────────────────────────

log ""
log "==================================================================="
log " Summary"
log "==================================================================="
log ""
column -t -s $'\t' "$RUN_DIR/fom.tsv" | sed 's/^/  /' | tee -a "$LOG"
log ""
log "  Profiling: run $((best_idx+1)) selected, FOM=$best_fom MLUPS"
log ""
log "  Run dir: $RUN_DIR"
log "==================================================================="
