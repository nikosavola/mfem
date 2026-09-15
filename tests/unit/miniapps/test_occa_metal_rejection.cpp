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
//
// Proves that requesting the experimental "occa-metal" backend on a
// non-Apple platform (or a build without MFEM_USE_OCCA, or an OCCA build
// without Metal support) fails loudly and explicitly, instead of silently
// falling back to another backend or crashing without a diagnostic. See
// doc/apple-metal-mlx-support-progress.md, WP2/item-3.
//
// This is a standalone, dedicated executable (not part of the shared
// unit_tests Catch2 binary) on purpose: mfem::Device is a process-wide
// singleton that can only be configured once, and a rejected
// Device::Configure() call can leave it in a partially-configured state
// (see general/device.cpp, Device::Setup()) that would be unsafe to reuse
// for any other test running in the same process.
//
// TODO(apple-metal): not yet wired into tests/unit/CMakeLists.txt /
// tests/unit/makefile as a dedicated test target (mirroring
// debug_device_tests / DEBUG_DEVICE_TEST in both files) -- left for the
// Mac-side agent alongside any real occa-metal hardware test targets from
// WP3/WP8. Build and run manually for now, e.g. (Serial/OpenMP OCCA build):
//   g++ -O3 -std=c++17 -I<mfem-src-dir> -I<occa-include-dir> \
//       tests/unit/miniapps/test_occa_metal_rejection.cpp \
//       -L<mfem-build-dir> -lmfem -L<occa-lib-dir> -locca -lrt \
//       -o test_occa_metal_rejection
//   ./test_occa_metal_rejection ; echo "exit code: $?"
// Expected on this (Linux, non-Apple) platform: nonzero exit, with a
// diagnostic mentioning "Apple platforms" if MFEM was built with
// MFEM_USE_OCCA=YES (this file's actual target), or "MFEM_USE_OCCA=YES" if
// it was not -- never a clean, silent return.

#include "mfem.hpp"

#include <cstdlib>
#include <iostream>

int main()
{
   std::cout << "Requesting mfem::Device(\"occa-metal\") on this platform;"
             " this must not return normally.\n";

#ifdef MFEM_USE_EXCEPTIONS
   try
   {
      mfem::Device device("occa-metal");
      std::cerr << "FAIL: occa-metal was silently accepted (device count = "
                << mfem::Device::GetDeviceCount() << ")!\n";
      return EXIT_FAILURE;
   }
   catch (const std::exception &e)
   {
      std::cout << "PASS: rejected with an exception: " << e.what() << '\n';
      return EXIT_SUCCESS;
   }
#else
   // Without MFEM_USE_EXCEPTIONS, MFEM_ABORT() terminates the process
   // directly (e.g. via abort()) instead of throwing -- there is nothing
   // here to catch. That is itself the expected, passing behavior: any
   // nonzero/abnormal exit from this program (this line included) is PASS;
   // only a clean, successful return is FAIL.
   mfem::Device device("occa-metal");
   std::cerr << "FAIL: occa-metal was silently accepted (device count = "
             << mfem::Device::GetDeviceCount() << ")!\n";
   return EXIT_FAILURE;
#endif
}
