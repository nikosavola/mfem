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
// Prints a fixed-precision fingerprint of the MassIntegrator and
// DiffusionIntegrator partial-assembly operator action, in 2D and 3D, over
// several polynomial orders and an element count *not* divisible by
// fem/occa.okl's M2_ELEMENT_BATCH (32), for whichever --device backend is
// requested. Run it once per backend and diff the output: identical output
// across "cpu", "occa-cpu", and "occa-omp" is the evidence that fixing
// MFEM's OCCA adapter for OCCA 2.0.0 (see
// doc/apple-metal-mlx-support-progress.md) did not change the existing
// OCCA Serial/OpenMP PA operator behavior.
//
// This has to run as three separate processes -- one per backend -- rather
// than as one program/test looping over device strings, because
// mfem::Device is a process-wide singleton that can only be Configure()'d
// once (general/device.cpp, Device::Setup()).
//
// TODO(apple-metal): not yet wired into tests/unit/CMakeLists.txt /
// tests/unit/makefile as a dedicated test target; see the same TODO in
// test_occa_metal_rejection.cpp. Build and run manually for now, e.g.:
//   g++ -O3 -std=c++17 -I<mfem-src-dir> -I<occa-include-dir> \
//       tests/unit/miniapps/test_occa_pa_kernels.cpp \
//       -L<mfem-build-dir> -lmfem -L<occa-lib-dir> -locca -lrt \
//       -o test_occa_pa_kernels
//   for d in cpu occa-cpu occa-omp; do
//     ./test_occa_pa_kernels --device $d > /tmp/pa.$d.txt
//   done
//   diff /tmp/pa.cpu.txt /tmp/pa.occa-cpu.txt
//   diff /tmp/pa.cpu.txt /tmp/pa.occa-omp.txt

#include "mfem.hpp"

#include <iomanip>
#include <iostream>

using namespace mfem;

namespace
{

// Deterministic, non-symmetric "random" field so the fingerprint is
// sensitive to indexing/ordering bugs, not just overall scale.
real_t TestField(const Vector &x)
{
   real_t v = 1.0;
   for (int d = 0; d < x.Size(); d++)
   {
      v += (d + 1) * std::sin((d + 2.3) * x(d)) - 0.5 * x(d) * x(d);
   }
   return v;
}

// Runs MassIntegrator and DiffusionIntegrator PA for one (dim, order, ne)
// combination and prints a fingerprint of each operator's action.
void RunOne(int dim, int order, int ne)
{
   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(ne, ne, Element::QUADRILATERAL)
               : Mesh::MakeCartesian3D(ne, ne, ne, Element::HEXAHEDRON);
   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);

   GridFunction gf(&fes);
   FunctionCoefficient coeff(TestField);
   gf.ProjectCoefficient(coeff);

   for (const char *name : {"Mass", "Diffusion"})
   {
      BilinearForm form(&fes);
      if (std::string(name) == "Mass")
      {
         form.AddDomainIntegrator(new MassIntegrator);
      }
      else
      {
         form.AddDomainIntegrator(new DiffusionIntegrator);
      }
      form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      form.Assemble();

      Vector y(fes.GetVSize());
      form.Mult(gf, y);

      real_t sum = 0.0, sumsq = 0.0;
      for (int i = 0; i < y.Size(); i++)
      {
         sum += y(i);
         sumsq += y(i) * y(i);
      }

      std::cout << std::fixed << std::setprecision(12)
                << name << " dim=" << dim << " order=" << order
                << " ne=" << ne << " ndof=" << y.Size()
                << " sum=" << sum << " sumsq=" << sumsq
                << " y[0]=" << y(0) << " y[last]=" << y(y.Size() - 1)
                << '\n';
   }
}

} // namespace

int main(int argc, char *argv[])
{
   const char *device_config = "cpu";
   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device", "Device configuration.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(std::cout);
      return 1;
   }
   args.PrintOptions(std::cout);

   Device device(device_config);
   device.Print();

   // ne=5 is not a multiple of fem/occa.okl's M2_ELEMENT_BATCH (32), so the
   // *_GPU-shaped OCCA kernels (which batch elements) exercise their
   // "leftover" tail path, not just full batches.
   for (int dim : {2, 3})
   {
      for (int order : {1, 2, 3})
      {
         RunOne(dim, order, /*ne=*/5);
      }
   }

   return 0;
}
