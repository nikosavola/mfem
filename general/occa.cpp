// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "occa.hpp"

#ifdef MFEM_USE_OCCA
#include "device.hpp"

namespace mfem
{

// This variable is defined in device.cpp:
namespace internal { extern occa::device occaDevice; }

occa::device &OccaDev() { return internal::occaDevice; }

occa::memory OccaMemoryWrap(void *ptr, std::size_t bytes)
{
   // OCCA 2.x no longer exposes the per-mode free functions this used to
   // call (occa::cpu::wrapMemory, occa::cuda::wrapMemory) as public API --
   // only occa::device::wrapMemory() member, which internally dispatches to
   // whatever mode the device was setup() with (Serial, OpenMP, CUDA,
   // Metal, ...). A single generic call replaces the old per-mode
   // branching and, as a side effect, is also what an occa-metal device
   // would use here without needing OCCA-Metal-specific headers -- OCCA
   // 2.0.0 does not ship a public occa/modes/metal/*.hpp analogous to the
   // (also no-longer-public) occa/modes/cuda/utils.hpp the old code
   // included.
   return internal::occaDevice.wrapMemory(ptr, bytes);
}

} // namespace mfem

#endif // MFEM_USE_OCCA
