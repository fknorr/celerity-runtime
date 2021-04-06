<p align="center">
<img src="docs/celerity_logo.png" alt="Celerity Logo">
</p>

# Celerity Runtime - [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/celerity/celerity-runtime/blob/master/LICENSE) [![Semver 2.0](https://img.shields.io/badge/semver-2.0.0-blue)](https://semver.org/spec/v2.0.0.html) [![PRs # Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/celerity/celerity-runtime/blob/master/CONTRIBUTING.md)

The Celerity distributed runtime and API aims to bring the power and ease of
use of [SYCL](https://sycl.tech) to distributed memory clusters.

> If you want a step-by-step introduction on how to set up dependencies and
> implement your first Celerity application, check out the
> [tutorial](docs/tutorial.md)!

## Overview

Programming modern accelerators is already challenging in and of itself.
Combine it with the distributed memory semantics of a cluster, and the
complexity can become so daunting that many leave it unattempted. Celerity
wants to relieve you of some of this burden, allowing you to target
accelerator clusters with programs that look like they are written for a
single device.

### High-level API based on SYCL

Celerity makes it a priority to stay as close to the SYCL API as possible. If
you have an existing SYCL application, you should be able to migrate it to
Celerity without much hassle. If you know SYCL already, this will probably
look very familiar to you:

```cpp
celerity::buffer<float, 1> buf(sycl::range<1>(1024));
queue.submit([=](celerity::handler& cgh) {
  auto acc = buf.get_access<sycl::access::mode::discard_write>(
    cgh,
    celerity::access::one_to_one<1>()           // 1
  );
  cgh.parallel_for<class MyKernel>(
    sycl::range<1>(1024),                       // 2
    [=](sycl::item<1> item) {                   // 3
      acc[item] = sycl::sin(item[0] / 1024.f);  // 4
    });
});
```

1. Provide a [range-mapper](docs/range-mappers.md) to tell Celerity which
   parts of the buffer will be accessed by the kernel.

2. Submit a kernel to be executed by 1024 parallel _work items_. This kernel
   may be split across any number of nodes.

3. Kernels can be expressed as C++11 lambda functions, just like in SYCL. In
   fact, no changes to your existing kernels are required \*.

4. Access your buffers as if they reside on a single device -- even though
   they might be scattered throughout the cluster.

\* There are currently some limitations to what types of kernels Celerity
supports - see [Issues & Limitations](docs/issues-and-limitations.md).

### Run it like any other MPI application

The kernel shown above can be run on a single GPU, just like in SYCL, or on a
whole cluster -- without having to change anything about the program itself.

For example, if we were to run it on two GPUs using `mpirun -n 2 ./my_example`,
the first GPU might compute the range `0-512` of the kernel, while the second
one computes `512-1024`. However, as the user, you don't have to care how
exactly your computation is being split up.

To see how you can use the result of your computation, look at some of our
fully-fledged [examples](examples), or follow the
[tutorial](docs/tutorial.md)!

## Building Celerity

Celerity uses CMake as its build system. The build process itself is rather
simple, however you have to make sure that you have a few dependencies
installed first.

### Dependencies

- A supported SYCL implementation, either
  - [hipSYCL](https://github.com/illuhad/hipsycl), or
  - [ComputeCpp](https://www.codeplay.com/products/computesuite/computecpp)
- A MPI 2 implementation (tested with OpenMPI 4.0, MPICH 3.3 should work as well)
- [CMake](https://www.cmake.org) (3.5.1 or newer)
- A C++17 compiler

Building can be as simple as calling `cmake && make`, depending on your setup
you might however also have to provide some library paths etc.
See our [installation guide](docs/installation.md) for more information.

The runtime comes with several [examples](examples) that are built
automatically when the `CELERITY_BUILD_EXAMPLES` CMake option is set (true by
default).

## Using Celerity as a Library

Simply run `make install` (or equivalent, depending on build system) to copy
all relevant header files and libraries to the `CMAKE_INSTALL_PREFIX`. This
includes a CMake [package configuration file](https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html#package-configuration-file)
which is placed inside the `lib/cmake` directory. You can then use
`find_package(Celerity CONFIG)` to include Celerity into your CMake project.
Once included, you can use the `add_celerity_to_target(TARGET target SOURCES source1 source2...)`
function to set up the required dependencies for a target (no need to link manually).

## Running a Celerity Application

Celerity is built on top of MPI, which means a Celerity application can be
executed like any other MPI application (i.e., using `mpirun` or equivalent).
There are several environment variables that you can use to influence
Celerity's runtime behavior:

### Environment Variables

- `CELERITY_LOG_LEVEL` controls the logging output level. One of `trace`, `debug`,
  `info`, `warn`, `err`, `critical`, or `off`.
- `CELERITY_DEVICES` can be used to assign different compute devices to Celerity worker
  nodes on a single host. The syntax is as follows:
  `CELERITY_DEVICES="<platform_id> <first device_id> <second device_id> ... <nth device_id>"`.
  Note that this should normally not be required, as Celerity will attempt to
  automatically assign a unique device to each worker on a host.
- `CELERITY_FORCE_WG=<work_group_size>` can be used to force a particular work
  group size for _every kernel_ and _every dimension_. This currently exists
  as a workaround until Celerity supports ND-range kernels.
- `CELERITY_PROFILE_OCL` controls whether OpenCL-level profiling information
  should be queried (currently not supported when using hipSYCL).

## Disclaimer

Celerity is a research project first and foremost, and is still in
early development. While it does work for certain applications, it probably
does not fully support your use case just yet. We'd however love for you to
give it a try and tell us about how you could imagine using Celerity for your
projects in the future!
