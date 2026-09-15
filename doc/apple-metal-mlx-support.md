# Apple GPU acceleration for MFEM: Metal, MLX, and OCCA

**Status:** research and implementation handoff

**Research snapshot:** 2026-09-15

**Audience:** MFEM maintainers and implementation agents

**Versions examined:** MFEM at this checkout, MLX 0.32.2, and OCCA 2.0.0

## 1. Executive summary

Apple GPU acceleration is feasible for selected MFEM operations, but it is not
feasible to add as a mechanical CUDA/HIP runtime substitution.

The central problem is kernel compilation. MFEM's CUDA and HIP implementations
compile arbitrary captured C++ lambdas from `MFEM_FORALL` as device kernels.
Metal Shading Language (MSL) cannot consume those C++ closures through a normal
C++ or Objective-C++ build. A new implementation therefore needs one of:

1. explicitly named MSL kernels and typed host wrappers;
2. a maintained C++-to-MSL translator;
3. a portability layer that emits MSL; or
4. coarse-grained offload to a framework such as MLX.

The other critical constraint is precision. Apple Metal GPU kernels do not
provide a native `double` type, and MLX rejects `float64` GPU execution. MFEM
defaults to `real_t == double`. The first prototype must therefore build MFEM
with `MFEM_PRECISION=single`; it must not silently change precision or execute
double-precision data on the GPU.

### Recommendation

Do not initially add a general `Backend::METAL` to `Backend::DEVICE_MASK`, and
do not claim that `MFEM_FORALL` supports Metal. Instead, run three
single-precision proof-of-concept tracks behind experimental build options:

1. **OCCA Metal, first priority:** extend MFEM's existing OCCA operator boundary
   to run the current mass and diffusion OKL kernels through OCCA 2's Metal
   mode. This has the smallest MFEM kernel-source change, but OCCA's Metal
   stability and buffer interoperability must be proven.
2. **MLX, in parallel:** wrap MFEM storage as MLX arrays, implement one vector
   operation with MLX primitives, and implement one partial-assembly (PA)
   operator with `mlx::core::fast::metal_kernel`. This tests whether MLX's
   allocation, lifetime, synchronization, and C++20 requirements fit MFEM.
3. **Direct Metal baseline:** implement the same small vector kernel with a
   named MSL function. This gives a performance and integration baseline and is
   the fallback if neither OCCA nor MLX is suitable.

After these experiments, select an operator-level route. A full native Metal
backend should be considered only after named-kernel coverage, memory aliases,
reductions, restrictions, and safe fallback behavior have been demonstrated.

### Non-negotiable guardrails

- Do not expose Metal as a complete device backend when unported operations can
  fall through to CPU loops with GPU storage.
- Do not broaden `MFEM_USE_CUDA_OR_HIP` mechanically. In many locations it
  means that a CUDA/HIP-specific library or kernel exists.
- Do not enable CUDA/HIP Hypre device mode, GPU-aware MPI, UVM, CUB/hipCUB,
  cuSPARSE/hipSPARSE, or vendor BLAS merely because an Apple GPU is present.
- Start with synchronous execution and host-staged MPI.
- Retain MFEM's ownership of MPI; do not initialize MLX's distributed MPI
  backend.
- Keep Objective-C and Objective-C++ types out of MFEM public headers.
- Treat MLX and OCCA support as optional and distinct from a future native
  Metal backend.

## 2. Scope and success definition

The long-term goal is to accelerate useful MFEM simulations on Apple Silicon
GPUs while preserving MFEM's existing API, memory correctness, and CPU
behavior.

The first useful milestone is deliberately smaller:

- an optional Apple-only build;
- single precision;
- vector operations required by a minimal iterative solve;
- PA mass and diffusion setup/apply/diagonal kernels;
- correct serial execution and MPI with host staging;
- explicit diagnostics for unsupported paths;
- no regression in builds that do not enable Apple GPU support; and
- a measured end-to-end benefit for at least one PA example.

The first milestone does **not** include:

- generic `MFEM_FORALL` source compatibility;
- double-precision GPU execution;
- private Metal storage and fully asynchronous scheduling;
- GPU-aware MPI;
- Metal-resident Hypre, PETSc, SUNDIALS, or other solver TPLs;
- feature parity with CUDA/HIP; or
- support for Intel Macs or non-Apple Metal devices.

## 3. Existing MFEM execution model

### 3.1 Backends and dispatch

Backend bit values, family masks, and `DEVICE_MASK` are defined in
`general/device.hpp:29-109`. Names and fixed dispatch priority are duplicated in
`general/device.cpp:48-64`; parsing and aliases are in
`Device::Configure` at `general/device.cpp:191-274`.

`Device::Setup` validates compiled support and initializes selected runtimes
(`general/device.cpp:554-620`). `Device::Print` reports the result
(`general/device.cpp:319-355`). A new backend ID must be appended rather than
renumbering the existing public bit values.

The important safety property is that `Device::UpdateMemoryTypeAndClass` uses
`Device::Allows(DEVICE_MASK)` to select device memory and
`MemoryClass::DEVICE` (`general/device.cpp:357-423`). `Read`, `Write`, and
`ReadWrite` then request that configured class (`general/device.hpp:295-388`).
It is therefore unsafe to add a Metal bit to `DEVICE_MASK` before all generic
fallback paths are made Metal-safe.

Device queries in `general/device.cpp:426-477` and
`general/device.cpp:622-801` assume CUDA or HIP. Metal implementations must
define conservative semantics for:

- device count and selection;
- memory reporting;
- device identifier;
- compute-unit count; and
- thread execution width.

These are not all direct analogues of CUDA properties. For example,
`Device::NumMultiprocessors` influences reduction launch sizing
(`general/reducers.hpp:547-565`).

### 3.2 Memory and ownership

`Memory<T>` stores a host pointer and metadata. The global `MemoryManager` maps
the host pointer to an internal device pointer, validity flags, ownership, and
aliases (`general/mem_manager.hpp:116-205` and
`general/mem_manager.cpp:157-196`).

The existing device extension point is `internal::DeviceMemorySpace`, whose
implementations supply allocation, deallocation, and H-to-D, D-to-D, and D-to-H
copies (`general/mem_manager.cpp:205-235`). CUDA and HIP adapters are in
`general/mem_manager.cpp:466-519`; controller selection is in
`general/mem_manager.cpp:778-810`.

MFEM assumes that device storage can be represented by a raw `void *` and that
aliases can be produced with byte-offset pointer arithmetic
(`general/mem_manager.cpp:1580-1593`). Metal APIs instead bind an `MTLBuffer`
plus an offset. An initial direct-Metal design can use shared-storage buffers,
expose `MTLBuffer.contents` as the MFEM device pointer, and maintain an internal
reverse registry:

```text
contents address range -> retained Metal buffer + byte offset
```

That preserves MFEM alias arithmetic while letting a named kernel recover the
resource handle. It does not support private storage. A later private-storage
implementation requires a deeper internal representation of a resource handle
and offset; opaque fake pointers are unsafe because MFEM performs pointer
arithmetic before reaching a backend wrapper.

`MemoryType::MANAGED` currently means CUDA/HIP managed memory with specific
pointer-identity and synchronization behavior
(`general/mem_manager.cpp:425-450` and `general/mem_manager.cpp:521-558`).
Metal shared storage is not automatically equivalent. An initial Apple backend
must reject `metal:uvm` and must not participate in existing `gpu_uvm` tests.

### 3.3 `MFEM_FORALL` is the architectural boundary

The public launch family is defined in `general/forall.hpp:186-223` and
`general/forall.hpp:1130-1259`. Dispatch converges in `ForallWrap`, currently
trying RAJA CUDA, RAJA HIP, native CUDA, native HIP, debug, RAJA/OpenMP, and CPU
(`general/forall.hpp:1038-1110`).

CUDA and HIP wrappers instantiate the captured C++ closure as a templated GPU
kernel and invoke its body (`general/forall.hpp:712-1035`). Ordinary Metal host
compilation cannot convert that closure, MFEM templates, tensor views, and
helper functions into MSL. A `MetalWrap<BODY>` cannot simply mirror
`CuWrap<BODY>`.

PA kernels additionally rely on:

- multidimensional thread and block indices;
- threadgroup barriers;
- statically sized shared arrays;
- device atomics;
- launch bounds; and
- device-specific polynomial-order limits.

Examples are in
`fem/integ/bilininteg_diffusion_kernels.hpp:172-220` and
`fem/integ/bilininteg_diffusion_kernels.hpp:988-1067`. A native route will need
a conservative `DofQuadLimits_METAL` selected both by runtime dispatch and by
the MSL generation/registration path.

### 3.4 Representative end-to-end path

For `examples/ex1.cpp` with PA diffusion:

1. the example constructs `Device` from `-d`;
2. `BilinearForm` creates `PABilinearFormExtension`;
3. diffusion allocates maps, geometric factors, and `pa_data`;
4. assembly runs setup kernels;
5. apply chooses libCEED, OCCA, ragged-simplex, or MFEM native dispatch;
6. registered kernels select shared-memory specializations or a generic
   fallback; and
7. `Read`/`ReadWrite` obtains pointers before `forall_3D` launches.

The relevant call path is
`examples/ex1.cpp:79-224`,
`fem/bilinearform_ext.cpp:332-368`,
`fem/integ/bilininteg_diffusion_pa.cpp:43-173`, and
`fem/integ/bilininteg_diffusion_kernels.hpp:1002-1017`.

Consequently, accelerating `ex1 -pa` needs more than diffusion apply. It needs
setup, restrictions, vector operations, reductions used by the solver, memory
transitions, and safe handling of every unported operation.

### 3.5 CUDA/HIP assumptions that need explicit audit

| Area | Current assumption | Apple-safe initial policy |
|---|---|---|
| Reductions | Device path is CUDA/HIP-specific and host fallback receives existing pointers (`general/reducers.hpp:444-593`). | Add selected kernels or stage before pointer acquisition. |
| Scans and selection | Device implementation uses CUB/hipCUB (`general/scan.hpp:18-75`, `general/scan.hpp:291-408`). | Keep on host until named Metal implementations exist. |
| Atomics | `AtomicAdd` is CUDA/HIP-only on device (`general/backends.hpp:90-129`). | Define operation- and dtype-specific Metal capability checks. |
| Hypre | Broad `DEVICE_MASK` checks can select GPU execution (`linalg/hypre.cpp:50-71`). | Keep Hypre memory and execution on host. |
| MPI | GPU-aware paths pass pointers directly to MPI (`general/communication.cpp:2090-2182`). | Synchronize and host-stage. |
| Sparse matrices | Vendor paths are cuSPARSE/hipSPARSE (`linalg/sparsemat.cpp:27-102`). | Use a proven host path first. |
| Batched BLAS | GPU BLAS selection is CUDA/HIP-specific (`linalg/batched/batched.cpp:21-47`). | Use host code unless its native kernel dependencies are ported. |
| DFEM launcher | Emits CUDA/HIP global kernels directly (`fem/dfem/util.hpp:589-645`). | Mark unsupported until separately ported. |

## 4. Technology assessment

### 4.1 Direct Metal

Direct Metal provides explicit control over devices, command queues, buffers,
compute pipeline states, command buffers, and synchronization. MSL can be
compiled into a `.metallib` at build time or compiled from source at runtime.
MFEM would need a C++ facade with an Objective-C++ implementation, or the
official metal-cpp headers.

**Strengths**

- Direct buffer-and-offset binding matches the semantics needed by explicit
  MFEM kernels.
- Outputs can write into existing MFEM-owned buffers.
- Pipeline specialization, resource lifetime, diagnostics, and scheduling are
  under MFEM's control.
- It has no additional runtime framework dependency.

**Limitations**

- It cannot compile arbitrary `MFEM_FORALL` C++ closures.
- Every accelerated operation needs MSL source or a translator.
- Apple GPU MSL has no native double-precision floating-point type.
- MFEM must own pipeline caching, shader packaging, queue scheduling, error
  reporting, and buffer interoperability.
- Supporting private storage is incompatible with the current raw-pointer alias
  model without deeper memory-manager changes.

Direct Metal is the most controllable long-term route if MFEM accepts an
explicit named-kernel backend. It is also the right baseline for measuring the
overhead of MLX and OCCA.

### 4.2 MLX

[MLX](https://github.com/ml-explore/mlx) is an MIT-licensed array framework
designed for Apple Silicon. It provides a C++ API, lazy graphs, CPU and GPU
streams, unified-memory-aware arrays, common array operations, and custom Metal
kernels through `mlx::core::fast::metal_kernel`.

The custom-kernel API accepts MSL source, input arrays, output shapes and
dtypes, launch dimensions, threadgroup dimensions, template parameters, and
compile options. It is sufficient to express explicit PA kernels with static
threadgroup memory, barriers, and supported atomics. It does not translate an
MFEM C++ lambda.

**Useful properties**

- Apple GPU discovery, command submission, compilation, caching, and stream
  handling are already implemented.
- MLX primitives can prototype elementwise operations, reductions,
  gather/scatter, and tensor contractions.
- Custom MSL kernels allow FEM-specific work that is not naturally represented
  as an ML graph.
- Unified-memory-aware inputs may avoid explicit copies on Apple Silicon.

**Integration constraints**

- MLX's Metal backend does not support `float64`.
- MLX 0.32.2 requires C++20 for its C++ build, while MFEM's normal requirement
  is C++17.
- Metal builds require a recent Xcode, macOS SDK, and deployment target; the
  examined release requires CMake 3.25 or newer and macOS 14 or newer.
- Standalone C++ installation is source-based. A CMake package from the Python
  distribution may also be located using `python -m mlx --cmake-dir`, but that
  introduces Python packaging into dependency discovery.
- A statically linked executable must be able to find MLX's `mlx.metallib`,
  normally beside the executable or through `METAL_PATH`.
- Constructing an MLX array from an arbitrary pointer is only zero-copy when
  MLX can create a Metal buffer over that allocation; otherwise MLX copies.
  The adapter must detect and test this rather than assume it.
- MLX custom kernels return newly allocated output arrays. They do not directly
  bind an arbitrary existing MFEM output buffer. This conflicts with
  `ReadWrite`, in-place updates, and MFEM's existing ownership model.
- MLX arrays carry graph and lifetime state that cannot be represented by a raw
  MFEM pointer alone.
- Lazy execution requires explicit evaluation or synchronization before MFEM
  host access, MPI, destruction, or a non-MLX consumer.
- No stable public C++ constructor for an arbitrary existing `MTLBuffer`, public
  C++ DLPack bridge, or sparse matrix/SpMV/solver layer was found.
- `fast::metal_kernel` exposes static threadgroup arrays but not a dynamic
  threadgroup-memory byte count. MLX's precompiled custom-primitive example
  offers lower-level control, including dynamic threadgroup memory, by using
  backend APIs whose stability is weaker.
- GPU scatter does not bounds-check indices. Assembly-like scatter use needs
  MFEM-side validation plus contention and determinism tests.
- MFEM's tensor helpers generally use first-index-contiguous conventions,
  whereas MLX defaults to row-major, last-index-contiguous arrays. A zero-copy
  tensor adapter must reverse shapes, provide validated strides, or convert
  layout explicitly.
- MLX's own MPI backend initializes and finalizes Open MPI and has no public
  route for attaching MFEM's existing communicator. It must not be used by the
  prototype.
- Some float32 matrix and convolution paths may use reduced precision by
  default. Strict numerical comparisons must test with `MLX_ENABLE_TF32=0`.
- MLX is a fast-moving 0.x dependency, so pinning and API-isolation are
  important.

MLX is therefore best evaluated as an **optional operator-level backend**, not
as the implementation of generic `MFEM_FORALL`. The prototype should answer:

1. Can ordinary MFEM host allocations be wrapped without a copy consistently?
2. Can an MLX output be copied, donated, or safely rebound to MFEM without
   breaking aliases and ownership?
3. What graph and synchronization overhead remains for PA-sized kernels?
4. Can the required `mlx.metallib` be found from build-tree, installed, static,
   shared, test, and MPI executables?
5. Is requiring C++20 acceptable only when `MFEM_USE_MLX=ON`?
6. Can a separately compiled C++20 adapter with a C or C++17-compatible facade
   contain the language-version change?

The least invasive first implementation keeps MFEM memory host-accessible,
wraps inputs at the operator boundary, evaluates eagerly at that boundary, and
copies outputs back when rebinding cannot be proved safe. This may not be fast;
its purpose is to measure the cost and validate semantics before modifying
`MemoryManager`.

### 4.3 OCCA Metal

MFEM already has optional OCCA integration:

- backend IDs for Serial, OpenMP, and CUDA;
- setup in `general/device.cpp:479-526`;
- pointer wrapping in `general/occa.cpp:29-40`;
- capability checks in `general/occa.hpp:67-76`; and
- PA mass and diffusion kernels in `fem/occa.okl`.

OCCA 2 advertises a Metal mode and translates OKL kernels to MSL. It compiles
Metal sources with Apple's command-line tools and provides Metal memory and
stream abstractions.

**Why this is the first prototype**

- The MFEM operator dispatch boundary already prefers OCCA before native PA
  kernels (`fem/integ/bilininteg_diffusion_pa.cpp:43-94`).
- Existing OKL kernels cover mass and diffusion.
- It avoids introducing a second array/graph ownership model.
- It can show whether an external source translator is sufficient for MFEM's
  explicit FEM kernels.

**Work and risks**

- Add an experimental `OCCA_METAL` selection and OCCA Metal setup.
- Update `DeviceCanUseOcca` and the mass/diffusion OCCA wrappers.
- Replace hard-coded `double` in `fem/occa.okl` with a Metal-compatible
  precision strategy.
- Determine how MFEM raw pointers and aliases map to OCCA Metal memory objects.
  MFEM's current non-CUDA branch wraps pointers as CPU memory.
- Package or cache generated kernels robustly.
- Test OCCA independently before interpreting an MFEM failure. OCCA's open
  Metal test work includes documented crashes, so advertised mode support is
  not sufficient evidence of production readiness.

OCCA only accelerates operations with OKL implementations. It does not solve
generic MFEM `forall`, restrictions, vector algebra, solver reductions, MPI,
or unported integrators.

### 4.4 Comparison

| Criterion | Direct Metal | MLX | OCCA Metal |
|---|---|---|---|
| Generic MFEM C++ lambda support | No | No | No; OKL kernels only |
| Existing MFEM integration | None | None | Partial |
| Existing MFEM FEM kernels | None | None | PA mass/diffusion |
| Existing output-buffer binding | Yes | Custom kernels allocate outputs | Via OCCA memory object, subject to bridge |
| Runtime/compiler ownership | MFEM | MLX | OCCA |
| C++ requirement | C++17 facade is possible | C++20 | C++17 |
| Native FP64 on Apple GPU | No | No | No viable MSL `double` |
| Dependency maturity risk | Apple API stable; MFEM code is new | Fast-moving 0.x API | Metal backend requires validation |
| Best role | Long-term named-kernel backend and baseline | Coarse operator experiments | Fastest existing-kernel prototype |

Metal Performance Shaders and MPSGraph may be useful later for dense or sparse
linear algebra, but they do not address MFEM's custom PA kernels or generic
lambda compilation and are not part of the initial recommendation.

## 5. Feasibility gates and decision record

No user-visible production backend should be merged until all five gates have
an explicit answer.

### Gate A: precision

**Required experiment:** configure MFEM with `MFEM_PRECISION=single`, compile
and run CPU references plus all three Apple GPU prototypes.

**Pass:** selected kernels meet operation-specific single-precision tolerances
and an upstream proposal clearly documents that the Apple GPU route is
single-only.

**Fail:** the target applications require native FP64 or mixed-precision
behavior that has not been designed and validated.

### Gate B: kernel model

**Required experiment:** implement the same vector operation and PA diffusion
apply using OCCA Metal, MLX custom Metal, and direct named MSL.

**Pass:** at least one route has maintainable source, diagnostics, pipeline
caching, and competitive warm execution.

**Fail:** the only working route requires duplicating an unsustainable portion
of MFEM without a generation or registration plan.

### Gate C: memory, aliases, and output ownership

**Required experiment:** cover zero-byte storage, nonzero-offset aliases,
external wrapping, repeated allocation/address reuse, in-place updates, host
access after launch, and destruction with work in flight.

**Pass:** all ownership and synchronization behavior can be expressed without
undefined raw-pointer use or hidden copies in steady state.

**Fail:** MLX/OCCA requires pointer rebinding or handles that cannot preserve
MFEM aliases, and no contained `MemoryManager` extension solves it.

### Gate D: fallback and parallel behavior

**Required experiment:** run an example containing both ported and unported
operations, plus a two-rank MPI case.

**Pass:** unsupported operations fail early or stage before selecting a pointer;
MPI stages through synchronized host-accessible storage; Hypre stays on host.

**Fail:** any CPU loop, MPI call, or TPL receives a Metal-only pointer.

### Gate E: build and deployment

**Required experiment:** test build-tree and installed consumers, static and
shared MFEM, CMake and GNU make, and executable relocation.

**Pass:** optional support is isolated, downstream requirements are exported,
and all shader/metallib resources are found without source-tree assumptions.

**Fail:** executables only run from the build tree or enabling the option
silently changes unrelated builds.

## 6. Proposed architecture after a successful prototype

### 6.1 Keep capability layers separate

Use distinct compile-time and runtime identities:

- `MFEM_USE_MLX` for the optional MLX dependency;
- existing `MFEM_USE_OCCA` plus a verified `occa-metal` mode;
- `MFEM_USE_METAL` only for an eventual native runtime;
- distinct runtime backend names such as `mlx`, `occa-metal`, and `metal`.

Do not label MLX or OCCA as native Metal internally. They have different
ownership, packaging, and capability constraints.

During the experimental operator-level phase, keep the new IDs out of
`DEVICE_MASK` or otherwise keep `Device::GetDeviceMemoryType()` host-safe.
Accelerated operators should opt in with explicit capability checks. Only a
backend that can safely execute or stage every generic path should join
`DEVICE_MASK`.

### 6.2 Operator capability registry

Record capabilities by operation rather than assuming that one GPU bit implies
full support. The minimum dimensions are:

- backend implementation;
- operation or kernel family;
- scalar dtype;
- dimension and `(D1D,Q1D)` limits;
- required threadgroup memory;
- atomics requirement;
- input/output storage interoperability; and
- serial/MPI policy.

Existing `MFEM_REGISTER_KERNELS` dispatch
(`fem/kernel_dispatch.hpp:26-177`) can remain the specialization mechanism.
Initially, branch inside mass/diffusion host functions and call an Apple
implementation. A later cleanup can add backend identity to dispatch keys if
multiple explicit backends become permanent.

### 6.3 Native runtime boundary

If direct Metal is selected, add a pure C++ facade, for example
`general/metal.hpp`. Its implementation can use Objective-C++ in
`general/metal.mm` or remain C++ with metal-cpp; `.mm` build rules are needed
only for the former. Responsibilities should include:

- device and queue creation;
- shared-buffer allocation and reverse lookup;
- explicit copy and fill operations;
- command submission and completion;
- named pipeline lookup and caching;
- launch-limit validation;
- buffer/offset binding;
- diagnostic translation; and
- teardown that waits for in-flight work.

MSL source should be grouped by stable operation families. Decide through an
architecture decision record whether to embed source, embed a compiled
metallib, or install a resource. Runtime compilation is useful for the
prototype; a release design must handle cache invalidation and installed
executables.

### 6.4 Synchronization

Begin with one queue and conservative commit-and-wait at:

- D-to-H transitions;
- a host `Read` after GPU writes;
- MPI or host TPL calls;
- buffer deallocation;
- runtime teardown; and
- explicit `Device::Synchronize`.

Batching and asynchronous overlap can be added only after validity and lifetime
tests pass. MLX evaluation and OCCA streams must be brought under the same
observable synchronization contract.

### 6.5 Unsupported operations

An accelerated operation must choose one of three behaviors before acquiring
storage:

1. run a proven Apple GPU implementation;
2. synchronize and execute a deliberate host-staged implementation; or
3. fail with an actionable unsupported-operation diagnostic.

Implicitly reaching an existing CPU fallback after calling `Read()` or
`Write()` with a device memory class is not an acceptable fallback.

## 7. File-by-file implementation map

This is an inventory, not a request to change every file in the first patch.

| Area | Files | Expected work |
|---|---|---|
| Backend IDs and API | `general/device.hpp` | Append IDs, define narrowly scoped masks, document priority and device-query semantics. |
| Runtime setup | `general/device.cpp` | Names, aliases, validation, setup, reporting, device selection, and conservative properties. |
| Kernel macros | `general/backends.hpp` | Add only abstractions supported by the selected explicit-kernel route; do not redefine CUDA/HIP feature macros. |
| Dispatch | `general/forall.hpp` | Add Metal limits or launch hooks only after there is a real implementation; keep unsupported generic closures safe. |
| Memory contracts | `general/mem_manager.hpp`, `general/mem_manager.cpp` | Add controller/bridge, alias tests, validity transitions, and synchronization. Reuse generic `MemoryType::DEVICE` only for a complete native backend. |
| Native runtime | new private C++/Objective-C++ files under `general/` | Encapsulate Metal API types, buffers, queue, pipelines, and diagnostics; choose `.cpp` with metal-cpp or `.mm` deliberately. |
| MLX adapter | new private files under `general/` or `fem/` | Isolate MLX headers, array wrapping, eval/sync, output ownership, and dtype checks. |
| OCCA adapter | `general/occa.hpp`, `general/occa.cpp`, `general/device.cpp` | Recognize tested Metal mode and wrap actual Metal resources rather than CPU pointers. |
| OCCA kernels | `fem/occa.okl` | Remove hard-coded `double`, validate generated MSL, and preserve CPU/CUDA behavior. |
| PA mass | `fem/integ/bilininteg_mass_pa.cpp`, `fem/integ/bilininteg_mass_kernels.*` | Add explicit capability branch and setup/apply/diagonal coverage. |
| PA diffusion | `fem/integ/bilininteg_diffusion_pa.cpp`, `fem/integ/bilininteg_diffusion_kernels.*` | Add explicit capability branch, specializations, fallback policy, and Metal limits. |
| Restrictions | `fem/restriction.cpp`, `fem/prestriction.cpp` and related kernels | Port or host-stage every restriction used by the target example. |
| Vector algebra | `linalg/vector.cpp` | Add the minimal elementwise and reduction operations required by the selected solver. |
| Reductions/scans | `general/reducers.hpp`, `general/scan.hpp` | Add explicit Apple kernels or stage safely before pointer acquisition. |
| MPI | `general/communication.cpp` | Force synchronized host staging; reject GPU-aware mode initially. |
| Hypre | `linalg/hypre.cpp` | Gate device policy on a compatible CUDA/HIP backend, not generic Apple GPU presence. |
| CMake options | `config/defaults.cmake`, root and component `CMakeLists.txt` | Optional dependency discovery, C++/Objective-C++ language handling, frameworks, sources, shader resources, and mutual-exclusion checks. |
| CMake package export | `config/cmake/config.hpp.in`, `config/cmake/MFEMConfig.cmake.in` | Export enabled features and downstream link/resource requirements. |
| GNU make | `config/defaults.mk`, `config/config.hpp.in`, `config/config.mk.in`, top-level `makefile` | Options, compiler flags, `.mm`/shader rules, dependencies, exports, shortcuts, and status output. |
| Configuration report | `general/version.cpp` | Include Apple feature flags in `GetConfigStr`. |
| Tests | `tests/unit/CMakeLists.txt`, `tests/unit/makefile`, targeted test sources | Add dedicated Apple test executables rather than treating the current `[GPU]` suite as automatically supported. |
| Examples/benchmarks | `examples/CMakeLists.txt`, `examples/makefile`, `tests/benchmarks/CMakeLists.txt` | Add only proven contexts and preserve CPU/CUDA/HIP matrices. |
| User documentation | `INSTALL`, `doc/CodeDocumentation.dox`, examples, `CHANGELOG` | State hardware, precision, build, packaging, unsupported combinations, and performance scope. |

## 8. Work packages for implementation agents

Each work package should produce a small reviewable change. Agents must report
verified evidence, unresolved assumptions, and exact tests. They must not
activate a public backend early to make a narrow test pass.

### WP0: reproducible feasibility harness

**Dependencies:** none

**Purpose:** establish one Apple Silicon machine and pinned toolchain as the
reference environment.

**Deliverables**

- Record chip, macOS, Xcode, SDK, CMake, compiler, MLX, and OCCA versions.
- Build MFEM in single precision with both CMake and GNU make before enabling a
  new dependency.
- Create out-of-tree experiments for vector addition using direct Metal, MLX
  primitives/custom Metal, and OCCA Metal.
- Capture cold compile, warm launch, synchronization, and numerical results.

**Acceptance**

- CPU and all viable GPU outputs agree at a stated tolerance.
- Unsupported `double` behavior fails clearly.
- Results distinguish compilation time from steady-state execution.

**Stop condition**

- If the chosen hardware cannot execute a stable custom Metal kernel, do not
  modify MFEM backend IDs.

### WP1: memory interoperability experiments

**Dependencies:** WP0

**Purpose:** answer the highest-risk ownership questions before operator work.

**Deliverables**

- Direct Metal shared-buffer reverse-lookup prototype with nonzero offsets.
- MLX pointer-wrap experiment that records whether the resulting data pointer
  aliases the original allocation for multiple sizes and alignments.
- MLX output lifetime/rebinding experiment.
- OCCA Metal wrap/slice experiment using the actual handle type expected by
  OCCA.
- A synchronization/lifetime stress test for each route.

**Acceptance**

- Zero-size, offset alias, in-place, copy, address reuse, and destruction tests
  are deterministic under sanitizers available on macOS.
- Every hidden input or output copy is measured and documented.

**Stop condition**

- Reject a route if correct aliasing requires fabricating arithmetic pointers
  or retaining owners through an unbounded global leak.

### WP2: optional build-system skeleton

**Dependencies:** WP0 and a decision to continue a route

**Purpose:** make experiments reproducible without changing default builds.

**Deliverables**

- Apple-only experimental CMake option for the selected dependency/runtime.
- Dependency and toolchain version diagnostics.
- Feature macros and installed-package exports.
- Static/shared and build-tree/installed resource discovery.
- Clear failure on unsupported platforms, precision, or dependency versions.
- For MLX, consume the exported target named `mlx`; do not assume an
  `MLX::mlx` target. Defer GNU-make dependency support until WP6 selects MLX as
  a maintained route.

**Acceptance**

- Default Linux and macOS builds, including GNU make, remain unchanged.
- The enabled CMake build reports the experimental capability accurately.
- Relocated installed test executables find required shader resources.

### WP3: OCCA Metal PA prototype

**Dependencies:** WP1 and WP2

**Purpose:** test the shortest path to existing PA kernels.

**Deliverables**

- Tested OCCA Metal runtime selection without claiming generic device coverage.
- Correct resource wrapping and alias offsets.
- Metal-compatible scalar handling in `fem/occa.okl`.
- PA mass and diffusion setup/apply/diagonal tests where currently implemented.
- Standalone OCCA test results for any failure reproduced through MFEM.

**Acceptance**

- Existing OCCA Serial/OpenMP/CUDA behavior is unchanged.
- CPU versus OCCA Metal operator action agrees across 2D/3D order cases,
  including sizes not divisible by workgroup dimensions.
- No steady-state host copy is hidden inside the measured apply path.

**Stop condition**

- Stop using OCCA as the leading route if current OCCA Metal cannot pass memory,
  barrier, and repeated-launch tests on the supported OS/GPU matrix.

### WP4: MLX operator prototype

**Dependencies:** WP1 and WP2

**Purpose:** determine whether MLX is a maintainable MFEM operator backend.

**Deliverables**

- Private adapter that isolates MLX headers and enforces `real_t == float`.
- Evaluate both an MLX-enabled C++20 MFEM target and a separate C++20 adapter
  with a C/C++17-compatible facade.
- One MLX primitive vector operation and one FEM custom Metal kernel.
- Explicit evaluation/synchronization boundaries.
- Measured input wrapping, output allocation/copy/rebinding, and graph overhead.
- Packaging tests for `mlx.metallib`.
- Strict-float tests with `MLX_ENABLE_TF32=0`; do not initialize MLX MPI.

**Acceptance**

- MFEM storage remains valid after all temporary MLX arrays are destroyed.
- Aliased vectors and `ReadWrite` cases have defined behavior.
- Cold and warm timings are reported separately.
- A moved installed executable finds MLX resources without source-tree paths.
- MFEM retains sole ownership of MPI initialization and finalization.

**Stop condition**

- Keep MLX as a research/prototyping dependency only if output ownership or
  per-operator overhead prevents an end-to-end benefit.

### WP5: direct named-Metal baseline

**Dependencies:** WP1 and WP2

**Purpose:** provide the control implementation and fallback architecture.

**Deliverables**

- Private C++ facade and Objective-C++ runtime.
- Shared-buffer registry, one queue, synchronous completion, and diagnostics.
- Named vector and PA kernels equivalent to WP3/WP4.
- Pipeline cache and selected shader packaging approach.

**Acceptance**

- No Metal API type leaks into installed MFEM headers.
- Offset aliases bind the correct buffer and byte offset.
- Pipeline and command errors include kernel identity and Metal diagnostics.
- Resource teardown is clean with work in flight.

### WP6: architecture decision

**Dependencies:** WP3, WP4, and WP5 results

**Purpose:** select one production direction rather than maintaining three
equivalent implementations.

Compare:

- correctness and supported orders;
- source duplication and maintainability;
- input/output copies and allocation counts;
- cold-start and warm operator performance;
- end-to-end example performance;
- dependency/toolchain burden;
- installed-resource reliability;
- error quality; and
- upstream project maturity.

Possible outcomes:

- OCCA for selected PA operators, with direct Metal for missing infrastructure;
- MLX as an optional coarse operator backend;
- direct named Metal as the main route;
- research-only support because precision or ownership requirements fail.

Publish the result as an architecture decision record before adding a complete
`metal` device alias.

### WP7: safe simulation milestone

**Dependencies:** WP6

**Purpose:** run a complete, useful MFEM solve.

**Deliverables**

- Required restrictions and vector kernels.
- Dot/norm reductions required by the selected iterative solver.
- PA mass and diffusion setup/apply/diagonal.
- Deliberate staging for every remaining operation.
- `ex1 -pa` and `ex1p -pa` with synchronized host-staged MPI.

**Acceptance**

- An instrumentation mode shows which operations ran on GPU, staged, or failed.
- No CPU fallback receives an Apple-GPU-only pointer.
- Serial and two-rank solutions match the single-precision CPU reference.
- Warm end-to-end runtime improves on a stated problem-size range.

### WP8: production hardening

**Dependencies:** WP7

**Purpose:** prepare an upstream-supported feature.

**Deliverables**

- Dedicated physical-Apple-GPU CI plus compile-only checks where necessary.
- Expanded FEM, memory, error, installation, and MPI tests.
- Performance baselines and regression thresholds.
- `INSTALL`, Doxygen, examples, and `CHANGELOG` updates.
- Explicit compatibility table for OS, hardware, precision, CMake/make, MPI,
  and TPLs.

**Acceptance**

- All supported configurations pass for both build systems.
- Unsupported combinations fail during configuration or device setup.
- The implementation follows MFEM contribution, style, testing, and AI-use
  disclosure requirements in `CONTRIBUTING.md`.

## 9. Test and performance matrix

### 9.1 Build and packaging

- CMake and GNU make.
- Debug and Release.
- Static and shared MFEM.
- Build-tree and installed downstream consumers.
- Apple Silicon baseline and a newer Apple GPU family.
- Minimum supported macOS/Xcode/SDK and the current supported versions.
- `MFEM_PRECISION=single` success and `double` rejection for GPU execution.
- Explicit failure on non-Apple platforms when native Metal is requested.
- Executables launched outside the build/install directory.

Current GitHub CI has macOS GNU-make coverage but not equivalent macOS CMake
coverage in `.github/workflows/builds-and-tests.yml:52-69`. Compilation jobs
must be distinguished from tests that have access to a physical Apple GPU.
The current GPU unit-test entry point also skips single-precision builds
(`tests/unit/gpu_unit_test_main.cpp:16-27`), so it cannot validate the initial
MLX/Metal target without a dedicated single-precision executable.

### 9.2 Memory

Adapt patterns from `tests/unit/miniapps/test_debug_device.cpp`:

- zero and nonzero allocations;
- H-to-D, D-to-H, and D-to-D copies;
- `Read`, `Write`, and `ReadWrite` validity transitions;
- aliases at zero and nonzero offsets;
- external pointer wrapping;
- ownership transfer and destruction;
- command completion before host access or free;
- repeated allocation and address reuse;
- MLX/OCCA wrapper lifetime;
- leak and use-after-free checks; and
- large allocations and allocation failure diagnostics.

### 9.3 Launch semantics

- `N = 0`, `1`, workgroup-size minus/plus one, and a large value.
- 1D, 2D, batched-Z, 3D, and explicit-grid mappings as they are introduced.
- Threadgroup memory and barriers.
- Launch-limit violation diagnostics.
- Supported integer/floating atomic cases.
- Compile and pipeline failure diagnostics.
- Repeated deterministic launches.

### 9.4 FEM correctness

- PA mass and diffusion in 2D and 3D.
- All registered `(D1D,Q1D)` cases allowed by Metal-specific limits.
- Generic/non-specialized order.
- Element counts not divisible by launch dimensions.
- Apply and diagonal assembly.
- Coefficient variants exercised by existing PA tests.
- CPU-reference operator action and end-to-end solution norms.
- MLX strict-float (`MLX_ENABLE_TF32=0`) and default-mode comparisons for any
  operation that may use reduced precision.
- Restrictions and prolongation required by the target examples.

Existing starting points include
`tests/unit/fem/test_pa_kernels.cpp:118-140`,
`tests/unit/general/test_reduction.cpp:23-84`, and
`tests/unit/general/test_scan.cpp:23-178`.

Create dedicated Apple test executables initially. Current CMake and GNU make
define GPU suites in terms of CUDA or HIP and include UVM assumptions
(`tests/unit/CMakeLists.txt:236-455` and `tests/unit/makefile:39-88`).

### 9.5 MPI and TPL safety

- Two-rank host-staged `ex1p`.
- MPI with and without an explicit GPU-aware request; the latter must be
  rejected for Apple backends initially.
- Hypre configured for host memory/execution.
- No accidental CUDA/HIP vendor library selection.
- Mixed accelerated/staged operation sequence with synchronization checks.

### 9.6 Performance

Report:

- cold shader/JIT compilation;
- warm launch latency;
- allocation count and bytes;
- input/output copies and bytes;
- kernel-only time;
- operator setup and apply time;
- synchronization time;
- end-to-end solve time; and
- CPU/GPU crossover problem size.

Compare OCCA, MLX, and direct Metal on identical kernels and launch geometry.
Include MFEM CPU and OpenMP baselines. Performance claims should use medians and
variance over repeated warm runs, while reporting cold start separately.

## 10. Risk register

| Risk | Severity | Mitigation |
|---|---:|---|
| No sustainable conversion of arbitrary C++ closures to MSL | Critical | Use explicit operator kernels; do not promise generic `forall`. |
| MFEM applications require FP64 | Critical | Single-precision gate and explicit support statement; stop if FP64 is required. |
| CPU fallback dereferences Metal-only storage | Critical | Operator capability checks and staging before pointer acquisition. |
| MLX outputs cannot preserve MFEM ownership/aliases | High | Measure copy/rebind options in WP1/WP4; keep MLX operator-level or reject it. |
| OCCA Metal backend is unstable | High | Require standalone OCCA tests and pin a proven version/commit. |
| Raw pointer cannot identify Metal/OCCA resource and offset | High | Shared-buffer registry for prototype; deeper memory record before private storage. |
| Incorrect async lifetime or host visibility | High | One queue and conservative waits, followed by stress tests. |
| Hypre enters CUDA/HIP device policy | High | Gate on actually compatible backends, not broad device presence. |
| MPI receives unsupported pointers | High | Disable GPU-aware mode and host-stage with synchronization. |
| Threadgroup, atomic, or launch assumptions differ | High | Metal-specific limits and per-kernel capability tests. |
| C++20 requirement leaks into all MFEM builds | Medium | Apply only when MLX is enabled and isolate MLX headers. |
| Shader or `mlx.metallib` missing after installation | Medium | Embed or install explicitly and test relocated consumers. |
| Optional dependency changes default builds | Medium | Options off by default; non-Apple CI and package-export tests. |
| Cold compilation dominates short simulations | Medium | Cache pipelines and report cold/warm timings separately. |
| Three prototypes become three permanent implementations | Medium | Mandatory WP6 architecture decision and removal of rejected paths. |

## 11. Open questions

1. Which MFEM applications motivate Apple GPU support, and can they use
   single precision throughout?
2. What minimum Apple GPU family and macOS version should be supported?
3. Is an optional C++20 mode acceptable for MLX-enabled MFEM and downstream
   applications?
4. Can MLX guarantee or expose whether pointer construction was zero-copy using
   a supported public API?
5. Can MLX custom-kernel output storage be donated or adopted without relying
   on unstable backend internals?
6. Can the selected OCCA version reliably wrap shared `MTLBuffer` storage with
   nonzero offsets?
7. Is OCCA's OKL-to-MSL output maintainable and performant for all PA
   specializations required by MFEM?
8. Should native shader source be generated from a common representation or
   maintained as explicit MSL?
9. Is runtime shader compilation acceptable, or must releases embed a
   precompiled metallib?
10. What observable synchronization guarantee should MFEM expose across native
    Metal, MLX, and OCCA?
11. Which solver and preconditioner provide the first useful end-to-end target
    without Metal-resident Hypre?
12. Who can provide physical Apple GPU CI and performance monitoring?

## 12. References

### MFEM sources in this checkout

- `general/device.hpp` and `general/device.cpp` — backend configuration.
- `general/mem_manager.hpp` and `general/mem_manager.cpp` — memory contracts.
- `general/forall.hpp` and `general/backends.hpp` — kernel language and launch.
- `general/occa.hpp`, `general/occa.cpp`, and `fem/occa.okl` — existing OCCA
  adapter and PA kernels.
- `fem/kernel_dispatch.hpp` — specialized kernel registration.
- `fem/integ/bilininteg_mass_*` and `fem/integ/bilininteg_diffusion_*` — target
  PA operators.
- `general/reducers.hpp`, `general/scan.hpp`, and `linalg/vector.cpp` — solver
  infrastructure.
- `CONTRIBUTING.md` — development, test, documentation, changelog, and review
  requirements.

### Apple Metal

- [Metal overview](https://developer.apple.com/metal/)
- [Metal Shading Language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Performing calculations on a GPU](https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu)
- [Metal shader libraries](https://developer.apple.com/documentation/metal/metal-shader-libraries)
- [Metal-cpp](https://developer.apple.com/metal/cpp/)
- [Resource storage modes](https://developer.apple.com/documentation/metal/mtlstoragemode)

### MLX

- [MLX repository](https://github.com/ml-explore/mlx)
- [MLX 0.32.2 release](https://github.com/ml-explore/mlx/releases/tag/v0.32.2)
- [Build and installation guide](https://ml-explore.github.io/mlx/build/html/install.html)
- [Using MLX from C++](https://ml-explore.github.io/mlx/build/html/dev/mlx_in_cpp.html)
- [Custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
- [Memory management](https://ml-explore.github.io/mlx/build/html/dev/memory_management.html)
- [Precision](https://ml-explore.github.io/mlx/build/html/usage/precision.html)
- [C++ API source: arrays](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/array.h)
- [C++ API source: custom kernels](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/fast.h)
- [Metal backend source](https://github.com/ml-explore/mlx/tree/v0.32.2/mlx/backend/metal)

### OCCA

- [OCCA repository](https://github.com/libocca/occa)
- [OCCA 2.0.0 release](https://github.com/libocca/occa/releases/tag/v2.0.0)
- [OCCA modes documentation](https://occa.readthedocs.io/en/latest/api.html)
- [OCCA Metal sources](https://github.com/libocca/occa/tree/v2.0.0/src/occa/internal/modes/metal)
- [Draft Metal test-fix work](https://github.com/libocca/occa/pull/766)

## 13. Handoff checklist

Before an implementation agent opens a patch, it should be able to answer:

- [ ] Is the work native Metal, MLX, or OCCA Metal?
- [ ] Is MFEM configured in single precision, with double rejected explicitly?
- [ ] Does the change avoid claiming generic `MFEM_FORALL` support?
- [ ] Are memory ownership, alias offsets, and synchronization defined?
- [ ] Are unported operations staged before pointer acquisition or rejected?
- [ ] Do Hypre and MPI remain on supported host paths?
- [ ] Are default and non-Apple builds unaffected?
- [ ] Are shader and framework requirements exported to installed consumers?
- [ ] Are cold and warm performance measured separately?
- [ ] Does the patch include focused correctness and failure-mode tests?
- [ ] Are limitations documented in user-facing text?
- [ ] Is every technical claim tied to source, test output, or an authoritative
      external reference?
