..
 SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0
..
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
..
 http://www.apache.org/licenses/LICENSE-2.0
..
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

##############
PVA Solutions
##############

PVA Solutions is an open-source collection of computer-vision and signal-processing
operators, samples, and pipelines optimized for the NVIDIA Programmable Vision
Accelerator (PVA). Refer to the `PVA Solutions documentation <https://docs.nvidia.com/pva/solutions/index.html>`__ for more information.

Getting Started
===============

Set up your local PVA Solutions project
------------------------------------------

Install the dependencies needed to set up the project:

- git

On Ubuntu >= 22.04, install the following packages using `apt`:

.. code-block:: shell

    sudo apt install -y git

Initialize CV-CUDA explicitly after cloning the repository:

.. code-block:: shell

    git submodule update --init

Build PVA Solutions
-------------------

Prerequisites
+++++++++++++

- NVIDIA PVA SDK version 2.5.4 or later is required, with version 2.10.0 set as the default recommended version. Please follow the `PVA SDK documentation <https://docs.nvidia.com/pva/sdk/index.html>`__ for installing the SDK and its prerequisites.
  If multiple versions of PVA SDK are installed, the ``FORCE_PVA_SDK_VERSION`` CMake flag can be set to select between them, e.g. ``-DFORCE_PVA_SDK_VERSION=2.5.4``.

- CUDA Toolkit 12.5 (or later) and cross-platform packages
- CMake 3.20 (or later)
- To build the DLInference samples:
    - Python >= 3.10.12 and < 3.13, and pip3 >= 22.0.2 are required.
    - LLVM 16.0.6 or later and Clang 16.0.6 or later are required.
    - Install the Python packages listed in ``samples/assets/onnx/requirements.txt``.
    - Download the network models listed in ``samples/assets/onnx/networks.txt``.

- To build the YOLOv5 pipelines: the same prerequisites as the DLInference
  samples, plus TensorRT 10 (10.3 or later) AArch64 cross-development packages.
  TensorRT 11 is not supported.

  - The YOLOv5 pipelines build for DRIVE OS Linux and Jetson Linux (L4T) targets only.
  - The PVA+GPU pipeline requires PVA SDK 2.7.0 or later for CUDA-stream submission; when built with an older SDK, it reports an error and exits.
  - Set the ``TENSORRT_ROOT`` CMake variable to select a TensorRT installation. By default, ``/usr`` is used when it contains the AArch64 TensorRT headers.
  - If TensorRT 10 is not found, the YOLOv5 pipelines are skipped. Set ``-DPVA_SOLUTIONS_REQUIRE_TENSORRT=ON`` to make a missing TensorRT 10 a configuration error instead.
  - The YOLOv5 pipelines need the TensorRT 10 runtime on the target.

For a consistent development environment, we recommend creating a development container (devcontainer).
This allows you to install all required dependencies in an isolated environment, helping to avoid version conflicts and platform-specific issues.

Building for DRIVE OS Linux or Jetson Linux (L4T)
+++++++++++++++++++++++++++++++++++++++++++++++++

To build PVA Solutions for DRIVE OS Linux or Jetson Linux (L4T), you need to set some environment variables. Refer to the `PVA SDK documentation <https://docs.nvidia.com/pva/sdk/index.html>`__ for detailed instructions.
Below is an example of how to build for Linux platforms:

.. code-block:: shell

    mkdir build_l4t && cd build_l4t
    cmake -DPVA_BUILD_MODE=L4T ..
    make package -j16

``pva_dl_lib`` static libraries are downloaded automatically at configure time. Set ``IMPORT_DEVICE_TARGETS=ON`` to use pre-compiled device binaries; they are downloaded at configure time.

To build NvSci interop examples, set the ``NVSCI_INCLUDE_PATH`` and ``NVSCI_LIBRARY_PATH`` CMake flags to the paths to the NvSci include and library directories, respectively.

.. code-block:: shell

    cmake -DPVA_BUILD_MODE=L4T -DNVSCI_INCLUDE_PATH=/path/to/nvsci/include -DNVSCI_LIBRARY_PATH=/path/to/nvsci/lib ..


Building for DRIVE OS QNX
+++++++++++++++++++++++++

The procedure is similar to building for Linux platforms; however some QNX toolchain environment variables are also required. Refer to the `PVA SDK documentation <https://docs.nvidia.com/pva/sdk/index.html>`__ for detailed instructions.
QNX CUDA Toolkit and cross-compile packages are required to build for QNX. Path to the QNX CUDA Toolkit should be set using the ``CUDA_TOOLKIT_DIR`` CMake flag, similar to the example below.

.. code-block:: shell

    mkdir build_qnx && cd build_qnx
    cmake -DPVA_BUILD_MODE=QNX -DCUDA_TOOLKIT_DIR=/usr/local/cuda-safe-12.5 ..
    make package -j16

Building for EMU
++++++++++++++++

PVA SDK 2.10 and later include EMU, an x86 emulator that runs NVVPU-compiled VPU programs on the development host.
EMU lets you build and test PVA Solutions and samples without deploying to target hardware.
Below is an example of how to build for EMU:

.. code-block:: shell

    mkdir build_emu && cd build_emu
    cmake -DPVA_BUILD_MODE=EMU ..
    make -j16

.. note::

    EMU requires PVA SDK 2.10 or later. With older PVA SDK versions, use ``-DPVA_BUILD_MODE=NATIVE`` instead.

Deploying PVA Solutions
-----------------------

On Linux Platforms
++++++++++++++++++

PVA Operator sample applications require `pva_operator` and `nvcv_types` dynamic libraries on the target device, in addition to cuPVA runtime libraries. Ensure that cuPVA host runtime libraries (and compatible BSP) are installed on the target device.

PVA Solutions debian packages can be used to conveniently install dynamic libraries, sample application binaries, and asset files.

Copy Debian Packages to Target
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Copy the runtime library and sample Debian packages from the root build directory (build_l4t) to the target system.

.. code-block:: shell

    pva-solutions-<version>-l4t.deb
    pva-solutions-<version>-l4t-samples.deb
    pva-solutions-<version>-sample-assets.deb

Install Debian Packages on Target
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    sudo apt install ./pva-solutions-<version>-l4t.deb ./pva-solutions-<version>-l4t-samples.deb ./pva-solutions-<version>-sample-assets.deb

On QNX Platforms
++++++++++++++++

For automotive platforms with read-only filesystems (such as DRIVE OS), PVA Solutions libraries, samples applications and cuPVA runtime libraries need to be deployed to the filesystem prior to flashing the board.
Refer to the DRIVE OS SDK/PDK documentation for details about how to deploy binaries to the target filesystem.

Install Debian Packages on Host
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For QNX platforms, the following packages contain PVA Solutions libraries and samples:

.. code-block:: shell

    pva-solutions-<version>-qnx-cross.deb
    pva-solutions-<version>-qnx-cross-samples.deb
    pva-solutions-<version>-sample-assets.deb

PVA Solutions cross-compiled for QNX can be installed to the host system using the following commands:

.. code-block:: shell

    sudo apt install ./pva-solutions-<version>-qnx-cross.deb ./pva-solutions-<version>-qnx-cross-samples.deb ./pva-solutions-<version>-sample-assets.deb

Deploy VPU Application Signature Allowlist
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PVA drivers allow system administrators to limit the VPU executables which may be run on the platform by registering cryptographic signatures of VPU ELFs in a secure allowlist database.
The allowlist for the PVA Operators (`cupva_allowlist_pva_operators_all`) must be deployed to the target filesystem.
Refer to the `PVA SDK documentation <https://docs.nvidia.com/pva/sdk/index.html>`__ for detailed instructions on deploying and registering the allowlist file.

Running PVA Solutions Sample Applications
-----------------------------------------

On Target Device
++++++++++++++++

Sample applications will be installed in the target filesystem at a location specified by your SDK/PDK configuration. The exact path depends on your deployment settings.
For Linux platforms, sample applications will be installed in:

.. code-block:: shell

    /opt/nvidia/pva-solutions-<version>/bin/aarch64-linux-gnu/

Some sample applications require input files. For Linux platforms, these assets will be installed in:

.. code-block:: shell

    /opt/nvidia/pva-solutions-<version>/share/sample_assets/

You can run the sample applications using commands similar to the following:

.. code-block:: shell

    /opt/nvidia/pva-solutions-<version>/bin/aarch64-linux-gnu/cornersubpix -a /opt/nvidia/pva-solutions-<version>/share/sample_assets/

For QNX platforms, PVA Solutions and PVA SDK library paths should be added to the ``LD_LIBRARY_PATH`` environment variable.

For PVA SDK 2.6.x and later:

.. code-block:: shell

    export LD_LIBRARY_PATH=<pva-sdk-install-dir>/lib/aarch64-qnx800:<pva-solutions-install-dir>/lib/aarch64-qnx800:$LD_LIBRARY_PATH

For PVA SDK 2.5.x:

.. code-block:: shell

    export LD_LIBRARY_PATH=<pva-sdk-install-dir>/lib/aarch64-qnx710:<pva-solutions-install-dir>/lib/aarch64-qnx710:$LD_LIBRARY_PATH

Development Host
++++++++++++++++

You can run the PVA Operator sample applications on your development host using an EMU or NATIVE build.
Select the emulated PVA generation at run time with ``PVA_GEN`` (``GEN2`` for Orin, ``GEN3`` for Thor).

.. code-block:: shell

    cd build_emu
    export PVA_GEN=GEN2
    ./samples/image_processing/cornersubpix/cornersubpix -a ../samples/assets/

Contributing
============

`PVA-Solutions <https://docs.nvidia.com/pva/solutions/index.html>`__ is an open source project. As part of the Open Source Community, we are committed to the cycle of learning, improving, and updating that makes this community thrive. However, PVA-Solutions is not yet ready for external contributions.

To report a bug, request a new feature, or ask a general question, please file a GitHub issue.

To understand our commitment to the Open Source Community, and providing an environment that both supports and respects the efforts of all contributors, please read our `Code of Conduct <./CODE_OF_CONDUCT.md>`__.

License
=======

PVA-Solutions operates under the `Apache-2.0 <./LICENSE>`__ license.

Third‑Party License/Notice
==========================

Refer `THIRD-PARTY-LICENSES <./THIRD-PARTY-LICENSES.txt>`__ for all 3rd Party OSS licenses and Notices

Submodules
----------

This repository includes the following third-party submodule:

- **CV-CUDA**: path `3rdparty/cvcuda <./3rdparty/cvcuda>`__, source `https://github.com/CVCUDA/CV-CUDA.git <https://github.com/CVCUDA/CV-CUDA.git>`__, licensed under Apache-2.0, see `3rdparty/cvcuda/LICENSE.md <./3rdparty/cvcuda/LICENSE.md>`__

  CV-CUDA has its own nested submodules. PVA Solutions builds only the NVCV part of CV-CUDA, which does not use them, and ``git submodule update --init`` does not fetch them. They are present only after a recursive clone, each with its own license file:

  - **pybind11**: path `3rdparty/cvcuda/3rdparty/pybind11 <./3rdparty/cvcuda/3rdparty/pybind11>`__, source `https://github.com/pybind/pybind11.git <https://github.com/pybind/pybind11.git>`__, licensed under BSD-3-Clause, see `3rdparty/cvcuda/3rdparty/pybind11/LICENSE <./3rdparty/cvcuda/3rdparty/pybind11/LICENSE>`__
  - **googletest**: path `3rdparty/cvcuda/3rdparty/googletest <./3rdparty/cvcuda/3rdparty/googletest>`__, source `https://github.com/google/googletest.git <https://github.com/google/googletest.git>`__, licensed under BSD-3-Clause, see `3rdparty/cvcuda/3rdparty/googletest/LICENSE <./3rdparty/cvcuda/3rdparty/googletest/LICENSE>`__
  - **dlpack**: path `3rdparty/cvcuda/3rdparty/dlpack <./3rdparty/cvcuda/3rdparty/dlpack>`__, source `https://github.com/dmlc/dlpack.git <https://github.com/dmlc/dlpack.git>`__, licensed under Apache-2.0, see `3rdparty/cvcuda/3rdparty/dlpack/LICENSE <./3rdparty/cvcuda/3rdparty/dlpack/LICENSE>`__
  - **nvbench**: path `3rdparty/cvcuda/3rdparty/nvbench <./3rdparty/cvcuda/3rdparty/nvbench>`__, source `https://github.com/NVIDIA/nvbench.git <https://github.com/NVIDIA/nvbench.git>`__, licensed under Apache-2.0 with LLVM Exceptions, see `3rdparty/cvcuda/3rdparty/nvbench/LICENSE <./3rdparty/cvcuda/3rdparty/nvbench/LICENSE>`__

Security
========

PVA-Solutions, as a NVIDIA program, is committed to secure development practices.
Please read our `Security <./SECURITY.md>`__ page to learn more.