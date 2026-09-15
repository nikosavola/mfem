# Apple Metal / OCCA skeleton: progress and handoff (Linux-side agent)

**Status:** WP2 (build-system skeleton) and the Linux-doable slice of WP3
(OCCA Metal PA prototype -- everything except actually running on Metal)
implemented and verified on a Linux x86 VM with no Apple hardware.

**Companion to:** `doc/apple-metal-mlx-support.md` (the original research and
implementation handoff document). Read that first; this document assumes it.

**Audience:** the next implementation agent, running on real Apple Silicon
hardware, picking up WP0/WP1 and extending this skeleton to actually exercise
`occa-metal`.

**Branch:** `occa-metal-apple-support`, commits `cac9ad430`..`4eed60b2f`
(5 commits, one per numbered scope item below plus item 0). `git log
cac9ad430^..HEAD` in this repo has the full commit messages, each with the
verification commands and their real output inline -- this document
summarizes and cross-references them rather than repeating everything.

## 0. The headline finding: the "existing OCCA support" baseline itself did not build against OCCA 2.0.0 -- and it was worse than two bugs

Niko's pre-staging found 2 build errors in `general/device.cpp`. Actually
fixing "the OCCA adapter" and getting it to genuinely run end-to-end (not
just compile) surfaced **6** separate OCCA-1.x-to-2.x API breaks, only 2 of
which a compile catches -- the other 4 only show up once you actually
configure a device and run a kernel through it:

| # | File | Break | Caught by |
|---|---|---|---|
| 1 | `general/device.cpp` | `occa::device::setup(const char*)` ambiguous between `setup(const std::string&)`/`setup(const occa::json&)` overloads | compile |
| 2 | `general/device.cpp` | `occa::loadKernels("mfem")` removed; no bulk kernel-preload API in OCCA 2.x | compile |
| 3 | `general/occa.cpp` | `occa::cpu::wrapMemory()` / `occa::cuda::wrapMemory()` free functions removed; no `occa::cpu` namespace, no public `occa/modes/cuda/utils.hpp` | compile |
| 4 | `linalg/vector.cpp` | `occa::linalg::{dot,min,max}` (OCCA 1.x's reduction module) removed entirely, not renamed | compile |
| 5 | `general/device.cpp` | `occaDevice.setup("mode: 'X'")` (un-braced) -- OCCA 2.x's `occa::json::load()` only recognizes `{`, `[`, a quote, a digit, or true/false/null at the top level; needs `"{mode: 'X'}"` | **runtime only** (`ex1 -d occa-cpu`) |
| 6 | `fem/occa.okl` | `@barrier("some-label")` -- OCCA 2.x's `@barrier` attribute only accepts no argument or the literal string `"warp"`, not an arbitrary label | **runtime only**, and only when the *whole* `.okl` file is exercised (see finding below) |

This is exactly the "OCCA Metal stability... must be proven" concern
section 4.3 of the research doc flags, landing one layer earlier than
anticipated: even OCCA Serial/OpenMP/CUDA support, which the doc treated as
a known-working baseline to preserve, needed real fixes first. See commits
`cac9ad430` (breaks 1-4) and `6d8bd1407` (breaks 5 partially -- the setup()
JSON-brace half is actually in `ddb9b81f5` -- and 6).

**Practical implication for you:** do not trust "it compiles" as evidence
that an OCCA code path works. Breaks 5 and 6 above only surfaced when
actually running `examples/ex1 -pa -d occa-cpu` end-to-end. If you hit a
similar "compiles fine, aborts/mis-transforms at runtime" issue while
extending this to `occa-metal`, that pattern -- correct on the C++ side,
broken against OCCA 2.x's actual runtime semantics -- is the first thing to
suspect.

## 1. What was implemented, file by file

### Item 0: OCCA 2.0.0 compatibility (prerequisite for everything else)
- `general/device.cpp`: `std::string(...)`-wrap the two ambiguous `setup()`
  calls; remove `occa::loadKernels("mfem")` (kernels build lazily via
  `buildKernel()` now; `addLibraryPath("mfem", ...)`, already present, is
  what makes `occa://mfem/...` paths resolve).
- `general/occa.cpp`: replace the removed `occa::cpu::wrapMemory` /
  `occa::cuda::wrapMemory` free-function calls with the generic
  `occa::device::wrapMemory()` member, which dispatches internally to
  whatever mode the device was `setup()` with. Also removes the
  `#include <occa/modes/cuda/utils.hpp>` guard block entirely -- that header
  does not exist in OCCA 2.0.0's public `include/` tree at all (see finding
  below), so it was already broken for any `OCCA_CUDA` build, just untested
  on this CUDA-less VM.
- `linalg/vector.cpp`: `Vector::operator*`/`Min()`/`Max()` no longer call
  `occa::linalg::{dot,min,max}` (removed upstream); they fall through
  unconditionally to the existing generic `compute_dot`/`compute_min`/
  `compute_max()` closures, which already handle every backend OCCA could
  have selected here correctly.

### Items 1-2: `OCCA_METAL` backend ID and capability check
- `general/device.hpp`: `Backend::OCCA_METAL = 1 << 15` (appended, not
  inserted among existing values; `NUM_BACKENDS` 15 -> 16). Added to
  `Backend::OCCA_MASK`. **Not** added to `Backend::DEVICE_MASK` -- this is
  the hard rule from the task brief; see the enum comment in the file for
  the full rationale (`Device::UpdateMemoryTypeAndClass` keys off
  `Allows(DEVICE_MASK)` to hand out device pointers to generic CPU-loop code
  paths, and no generic path has been proven Metal-safe).
- `general/device.cpp`: name `"occa-metal"` in `backend_list`/`backend_name`
  (priority: right after `occa-cuda`); mutual-exclusion check extended;
  `OccaDeviceSetup()` gets an `else if (metal)` branch with **two
  independent, separately-diagnosed** rejection conditions (see below).
  Also fixes API break #5 above (JSON braces) for all four `setup()` calls,
  not just the metal one.
- `general/occa.hpp`: `DeviceCanUseOcca()` now also permits
  `Backend::OCCA_METAL`, parallel to the existing `OCCA_CUDA` branch (both
  device-class OCCA backends, both kept out of `DEVICE_MASK`). This is what
  would let MFEM's existing OCCA-preferring PA dispatch
  (`fem/integ/bilininteg_diffusion_pa.cpp` etc.) reach an `occa-metal`
  device once one is actually configured. Marked
  `// TODO(apple-metal):` -- nothing here can exercise this without real
  Metal hardware.

### Item 3: non-Apple rejection (the one thing genuinely provable here)
Two independent checks in `OccaDeviceSetup()`'s new `metal` branch, each
with its own diagnostic:
1. `#if !defined(__APPLE__)` -> `"the OCCA Metal backend ('occa-metal') is
   only available on Apple platforms!"`
2. `#elif !OCCA_METAL_ENABLED` -> `"...requires OCCA built with Metal
   support (OCCA_METAL_ENABLED)!"`

Only past both does it reach the (unverified, `TODO(apple-metal)`-marked)
`occaDevice.setup("{mode: 'Metal', ...}")` call.

**Verified** with `tests/unit/miniapps/test_occa_metal_rejection.cpp`
(commit `4eed60b2f`): on this Linux, non-Apple, OCCA-Metal-disabled build,
requesting `mfem::Device device("occa-metal");` aborts with diagnostic #1
above and exit code 134 (SIGABRT) -- not a silent fallback, not an
unexplained crash. Diagnostic #2 (Apple platform, but OCCA itself lacks
Metal support) is **not independently verified** -- there is no way to be
"on Apple" from this VM. It is straightforward C preprocessor logic
(`#elif !OCCA_METAL_ENABLED`, the exact same pattern as the existing
`OCCA_CUDA_ENABLED`/`OCCA_OPENMP_ENABLED` checks this file already had), so
risk is low, but exercise it explicitly on real hardware with an
Metal-disabled OCCA build as a first sanity check.

### Item 4: `fem/occa.okl` precision
- New `okl_real_t` **macro** (not typedef -- see below), `float` under
  `MFEM_OKL_SINGLE`, else `double`. All 179 occurrences of the word
  `double` in the file replaced with `okl_real_t`.
- `general/occa.hpp`: new `OccaSetRealTypeDefine(occa::properties&)` sets
  `defines/MFEM_OKL_SINGLE` only under `MFEM_USE_SINGLE`. Called at all 6
  `buildKernel()` call sites in
  `fem/integ/bilininteg_{mass,diffusion}_kernels.cpp`, alongside the
  existing `D1D`/`Q1D` defines.
- This was a **real correctness bug**, not just precision-hygiene: before
  this fix, an `MFEM_PRECISION=single` build would pass `real_t`-sized
  (4-byte) buffers into OCCA kernels that read/wrote them as `double`
  (8-byte) -- a buffer-layout mismatch, silently wrong, not a crash.

### Items 5-7: CMake/GNU make skeleton and config reporting
New experimental, off-by-default `MFEM_USE_OCCA_METAL` in both build
systems (`config/defaults.cmake`, `CMakeLists.txt`, `config/cmake/
config.hpp.in`, `config/cmake/MFEMConfig.cmake.in`, `config/defaults.mk`,
`config/config.hpp.in`, `config/config.mk.in`, top-level `makefile`, and
`general/version.cpp`'s `GetConfigStr()`). It does **not** select any
additional dependency or source file -- OCCA is still the only external
dependency, and the `occa-metal` runtime backend is always compiled in
(items 1-2 above), gated at `Device::Setup()` time regardless of this flag.
It exists only to (a) fail the configure step early and explicitly for an
obviously-impossible platform/dependency combination, and (b) advertise the
capability in MFEM's config report. See commit `f3a74a69b` for the full
list of touch points and why each was needed (in particular:
`MFEM_USE_OCCA_METAL` must be added to the `MFEM_DEFINES` list in the
top-level `makefile`, or the `#define` silently never reaches
`config.hpp`/`config.mk` -- easy to miss, nothing errors if you do).

## 2. What is stubbed / left as TODO, and why

Everything below is marked `// TODO(apple-metal):` at its exact location in
the code.

- **`OccaDeviceSetup()`'s actual `occaDevice.setup("{mode: 'Metal', ...}")`
  call** (`general/device.cpp`): syntactically plausible (matches the CUDA
  branch's now-fixed pattern) but **completely unverified** -- there is no
  Metal runtime on this machine to run it against. This is WP0/WP1
  territory.
- **`DeviceCanUseOcca()` returning true for `OCCA_METAL`**
  (`general/occa.hpp`): means MFEM's OCCA-preferring PA dispatch would
  attempt to route through `occa-metal` once one is configured. Whether
  that actually produces correct results is entirely untested.
- **MFEM-pointer-to-OCCA-Metal-memory-object mapping** (research doc
  section 11, open question #6): not attempted here at all. See the OCCA
  findings section below for what `~/dev/occa-ref` actually shows about
  this, which meaningfully refines (in a good way) what the doc assumed.
- **Dedicated test-suite wiring**
  (`tests/unit/miniapps/test_occa_metal_rejection.cpp` and
  `test_occa_pa_kernels.cpp`): both are real, runnable, standalone
  executables (see their file-level comments for exact build commands) but
  are **not** wired into `tests/unit/CMakeLists.txt` / `tests/unit/makefile`
  as permanent targets (mirroring `debug_device_tests` /
  `DEBUG_DEVICE_TEST`). This was a deliberate time-boxing choice, not a
  discovered blocker -- doing it properly is cheap once you're set up to
  actually run these against real hardware, and is a natural thing to do
  alongside whatever WP3/WP8 hardware-specific test targets you add.

## 3. WP2 / WP3 acceptance criteria

### WP2 (optional build-system skeleton)
- "Default Linux and macOS builds, including GNU make, remain unchanged."
  **verified-here** (Linux only; macOS default build not checked -- no Mac
  available). Default CMake configure (`cmake .. -DMFEM_USE_MPI=NO`):
  succeeds, `config/_config.hpp` has `/* #undef MFEM_USE_OCCA_METAL */`.
  Default GNU make (`make config && make -j4`, no new flags): builds
  `libmfem.a` unchanged.
- "The enabled CMake build reports the experimental capability
  accurately." **verified-here**, partially: `cmake ..
  -DMFEM_USE_OCCA=YES -DMFEM_USE_OCCA_METAL=YES -DOCCA_DIR=~/dev/occa-ref`
  fails cleanly at configure time on this platform (that *is* accurate
  reporting -- it cannot be enabled here) with a clear diagnostic; the
  "successfully enabled and reports itself in `GetConfigStr()`" path is
  **needs-Apple-hardware** (nothing here can get past the `NOT APPLE`
  check to test it).
- "Relocated installed test executables find required shader resources."
  **needs-Apple-hardware** (no shader resources exist yet; nothing to
  relocate-test).
- "Clear failure on unsupported platforms, precision, or dependency
  versions." **verified-here** for platform (both CMake and GNU make, see
  commit `f3a74a69b`'s message for the exact commands and output).

### WP3 (OCCA Metal PA prototype)
- "Existing OCCA Serial/OpenMP/CUDA behavior is unchanged."
  **verified-here** for Serial/OpenMP (no CUDA toolchain on this VM, so
  CUDA is **needs-Apple-hardware** -- or rather, needs any CUDA machine).
  Double precision: `tests/unit/miniapps/test_occa_pa_kernels.cpp` across
  `cpu`/`occa-cpu`/`occa-omp`, 2D/3D, orders 1-3, `ne=5` (not divisible by
  `M2_ELEMENT_BATCH`=32) -- Mass and Diffusion PA operator-action
  fingerprints identical to 12 decimal digits. Single precision (`
  MFEM_PRECISION=single`): same test, `sumsq`/`y[0]`/`y[last]` agree to
  ~1.4e-7 relative (float machine epsilon; summation-order noise, not a
  bug) across all combinations, and Mass matches to `0.0` relative diff
  exactly; confirmed via the OCCA-generated source in `~/.occa/cache/*/
  occa.source.cpp` that `MFEM_OKL_SINGLE` genuinely reached the kernel
  (`typedef float * DofToQuad_t;`, not `double`) rather than just assuming
  the define plumbing worked. Also directly verified via `examples/ex1 -pa`
  (`star.mesh` 2D, `fichera.mesh` 3D): `-d cpu`/`-d occa-cpu`/`-d occa-omp`
  converge to bit-identical CG iteration histories in double precision.
- "Correct resource wrapping and alias offsets." **needs-Apple-hardware**
  for Metal specifically. For Serial/OpenMP/CUDA, unchanged (still routes
  through `occa::device::wrapMemory()`, now the *only* route -- see finding
  below).
- "Metal-compatible scalar handling in `fem/occa.okl`." **verified-here**:
  `okl_real_t` correctly resolves to `float`/`double` based on
  `MFEM_OKL_SINGLE`, confirmed by direct inspection of generated source.
  Whether *Metal specifically* (MSL) accepts this is **needs-Apple-hardware**.
- "Tested OCCA Metal runtime selection without claiming generic device
  coverage." **verified-here** for the rejection path (item 3 above);
  **needs-Apple-hardware** for actual selection succeeding.
- "PA mass and diffusion setup/apply/diagonal tests where currently
  implemented." Setup/apply: **verified-here** for Serial/OpenMP (see
  above). Diagonal assembly (`AssembleDiagonalPA`): **not exercised** by
  `test_occa_pa_kernels.cpp` (it only calls `Mult()`) -- worth adding if you
  extend that test.
- "Standalone OCCA test results for any failure reproduced through MFEM."
  N/A here -- see the OCCA-level findings below (the `@barrier` and JSON
  syntax breaks were found and fixed by reading OCCA's own source directly,
  not by reproducing a standalone OCCA test failure).

## 4. Findings from `~/dev/occa-ref` that refine or contradict the research doc

- **No public `occa/modes/metal/*.hpp` exists at all** in OCCA 2.0.0's
  `include/` tree (confirmed: `find include -iname "*metal*"` and
  `find include -iname "*cuda*"` are both empty -- and note this means the
  *existing* `#include <occa/modes/cuda/utils.hpp>` this repo had before
  item 0's fix was **also** broken, just never compiled on this CUDA-less
  VM). Section 7's "wrap actual Metal resources rather than CPU pointers"
  implies a mode-specific API akin to `occa::cuda::wrapMemory`; that API
  never existed publicly for either CUDA or Metal in OCCA 2.x. The actual
  route, and the one item 0's fix to `general/occa.cpp` now uses
  unconditionally for every mode, is the generic
  `occa::device::wrapMemory(ptr, bytes)` member (`include/occa/core/
  device.hpp:728`, dispatching internally to `modeDevice->wrapMemory(...)`).
  This is **better** than the doc anticipated -- one code path already
  covers Metal with no Metal-specific header -- but is **completely
  unverified for Metal** (see below).
- **Open question #6** ("Can the selected OCCA version reliably wrap shared
  `MTLBuffer` storage with nonzero offsets?") is still genuinely open, but
  here is what the *internal* (non-public,
  `src/occa/internal/modes/metal/*.hpp`) headers show, which was not
  available to the original research pass: `occa::metal::memory` (
  `memory.hpp`) carries `api::metal::buffer_t metalBuffer` and
  `udim_t bufferOffset` directly as public members, and
  `occa::metal::buffer::slice(offset, bytes)` (`buffer.hpp`) exists and
  returns a `modeMemory_t*` -- i.e. the internal representation is already
  shaped exactly like what section 3.2's "reverse registry" design wants
  (buffer + offset, alias-able by slicing). That's encouraging evidence
  the alias model *can* work, not proof that it does. Do not treat this as
  "verified" -- it is "the internal data structure looks like it was
  designed for this," found by reading source, not by running anything.
- **`occa::linalg` (OCCA 1.x's dot/min/max reduction module) does not exist
  in OCCA 2.x at all** -- not renamed, removed. `Vector::operator*`,
  `Vector::Min()`, `Vector::Max()` have had **no OCCA fast path of any
  kind** since this fix (they fall through to MFEM's own generic reduce()).
  This is a real, permanent behavior change from whatever this repo's OCCA
  integration did against OCCA 1.x, not specific to Metal -- worth knowing
  before benchmarking anything and wondering where the OCCA reduction
  kernels went.
- **OCCA parses/transforms every `@kernel` in a `.okl` file when building
  any single one of them.** Discovered because fixing the `@barrier` syntax
  in `DiffusionApply3D_GPU`/`MassApply3D_GPU` (which only run under
  `occa-cuda`) was necessary to get `DiffusionApply2D_CPU` (which is what
  `occa-cpu` actually uses) to build at all -- a syntax error anywhere in
  `fem/occa.okl` breaks kernel building for every kernel in the file,
  including ones nowhere near the requested one. This is exactly the shape
  a Metal-mode OKL-to-MSL translation failure in one kernel will present:
  as a failure to build a completely unrelated kernel from the same file.
  Non-obvious; worth remembering when debugging translation failures on
  real Metal hardware.
- `OCCA_METAL_ENABLED` is a real, always-defined macro (0 or 1, from
  `include/occa/defines/compiledDefines.hpp`) exactly analogous to the
  `OCCA_CUDA_ENABLED`/`OCCA_OPENMP_ENABLED` this repo already checked --
  confirmed `0` on this Linux OCCA build, which is what makes the item-3
  rejection path exercisable and verified here at all.
- OCCA 2.0.0 was built here via its own GNU `Makefile` (not CMake), with
  `OCCA_OPENMP_ENABLED=1`, `OCCA_CUDA_ENABLED=0`, `OCCA_METAL_ENABLED=0`.
  `./bin/occa info` runs (CPU info only, no GPU section, as expected with
  no GPU present) -- this is the "standalone OCCA sanity baseline" the task
  brief asked to check; it was already pre-built and working before this
  session started.

## 5. Exact commands to run first (in order)

All from `~/dev/mfem` on the `occa-metal-apple-support` branch, assuming an
OCCA 2.0.0 checkout is available (the reference used here,
`~/dev/occa-ref`, is a **separate, read-only clone** on this Linux VM --
you will need your own OCCA 2.0.0 build with Metal enabled on your Mac; the
`OCCA_METAL_ENABLED` check in `general/device.cpp` needs it).

```sh
# 1. Sanity: confirm your OCCA has Metal enabled.
#    grep OCCA_METAL_ENABLED <your-occa>/include/occa/defines/compiledDefines.hpp
#    must show "#define OCCA_METAL_ENABLED 1"

# 2. Known-green baseline: default builds must be unaffected by this branch.
cmake -S . -B build-default -DMFEM_USE_MPI=NO && cmake --build build-default -j
make config MFEM_USE_MPI=NO && make -j

# 3. Existing OCCA Serial/OpenMP/CUDA baseline (this session's fixes; should
#    already be green -- if not, something about your OCCA differs from the
#    OCCA 2.0.0 this was validated against).
make clean
make config MFEM_USE_MPI=NO MFEM_USE_OCCA=YES OCCA_DIR=<your-occa-dir> && make -j
cd examples && make ex1 -j
./ex1 -m ../data/star.mesh -pa -d occa-cpu    # should match -d cpu exactly
./ex1 -m ../data/star.mesh -pa -d occa-omp    # ditto
cd ..
g++ -O3 -std=c++17 -I. -I<your-occa-dir>/include \
    tests/unit/miniapps/test_occa_pa_kernels.cpp \
    -L. -lmfem -L<your-occa-dir>/lib -locca -lrt -o test_occa_pa_kernels
for d in cpu occa-cpu occa-omp; do ./test_occa_pa_kernels -d $d > pa.$d.txt; done
diff pa.cpu.txt pa.occa-cpu.txt   # only the --device/Device-config header lines should differ

# 4. The actual WP0/WP1 starting point: confirm occa-metal is now selectable
#    at all (this is expected to differ from this session's result -- on
#    real Apple hardware with Metal-enabled OCCA, this must NOT abort).
g++ -O3 -std=c++17 -I. -I<your-occa-dir>/include \
    tests/unit/miniapps/test_occa_metal_rejection.cpp \
    -L. -lmfem -L<your-occa-dir>/lib -locca -lrt -o test_occa_metal_rejection
./test_occa_metal_rejection   # this session's Linux result: SIGABRT,
                               # "only available on Apple platforms"
                               # your result should be: PASS/FAIL per the
                               # program's own criterion (does NOT throw ==
                               # backend accepted -- see the file's
                               # top comment for exact expected behavior,
                               # which is now "occa-metal should NOT be
                               # rejected here")

# 5. Then: WP0 (pin toolchain versions) and WP1 (memory interop experiments,
#    starting from the occa::metal::memory/buffer findings in section 4
#    above) as described in doc/apple-metal-mlx-support.md, before trying
#    to make DiffusionSetup2D/etc. actually run on occa-metal.

# 6. Once occa-metal is confirmed selectable: extend
#    tests/unit/miniapps/test_occa_pa_kernels.cpp with a "metal" case and
#    compare against the cpu fingerprint the same way, in single precision
#    only (MFEM_PRECISION=single) -- occa.okl's okl_real_t plumbing (item 4
#    above) is already wired for this.

# 7. Wire test_occa_metal_rejection.cpp and test_occa_pa_kernels.cpp into
#    tests/unit/CMakeLists.txt / tests/unit/makefile as real targets
#    (mirror debug_device_tests / DEBUG_DEVICE_TEST) once you have real
#    hardware results to protect with them.
```

## 6. Things this session deliberately did not touch

Per the task's hard rules -- confirmed not touched, `git diff
25555cfa7..HEAD` in this repo:
- `Backend::DEVICE_MASK` (OCCA_METAL is not in it).
- `general/forall.hpp` (no `MetalWrap`, no dispatch changes).
- `MFEM_USE_CUDA_OR_HIP` (not referenced anywhere in this branch's changes).
- `linalg/hypre.cpp`, `general/communication.cpp` (untouched).
- No claim of generic `MFEM_FORALL` / device-memory support for
  `OCCA_METAL` anywhere, in code or in this document.
