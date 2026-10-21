#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# ---------------------------------------------------------------------------
# Configure-time fetch of precompiled artifacts.
#
# This module downloads exactly the set of prebuilt artifacts the current
# configuration needs and drops them into the on-disk locations the existing
# importers already expect:
#
#   * device binaries -> src/operator/device_binaries/<ver>/<target>__merged.bin
#                        + src/operator/device_binaries/<ver>/pva_allowlist_default_all
#                        (consumed by pva_solutions_device() when IMPORT_DEVICE_TARGETS=ON)
#   * pva_dl_lib       -> src/operator/deep_learning/pva-dl-lib/lib/<arch>/libpva_dl_<ver>.a
#                        (<arch> is the target arch, or <arch>-emu for EMU)
#                        + matching allowlist + include/pva_lib_defs.h
#   * onnx compiler    -> tools/pva-onnx-compiler
#
# ---------------------------------------------------------------------------

# Tri-state gate:
#   AUTO (default) : fetch only when a release tag is pinned
#   ON             : force fetch; error if nothing is pinned
#   OFF            : never fetch.
set(PVA_FETCH_PREBUILT_BINARIES "AUTO" CACHE STRING
    "Fetch precompiled device/library/tool artifacts from Artifactory: AUTO | ON | OFF")
set_property(CACHE PVA_FETCH_PREBUILT_BINARIES PROPERTY STRINGS AUTO ON OFF)

# Override the artifact location.
set(PVA_PREBUILT_BASE_URL_OVERRIDE "" CACHE STRING
    "Override the manifest's PVA_PREBUILT_BASE_URL")

if(NOT PVA_FETCH_PREBUILT_BINARIES)
    message(STATUS "[pva-fetch] disabled (PVA_FETCH_PREBUILT_BINARIES=OFF) - building from source")
    return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/prebuilt_manifest.cmake")

if(PVA_PREBUILT_BASE_URL_OVERRIDE)
    set(PVA_PREBUILT_BASE_URL "${PVA_PREBUILT_BASE_URL_OVERRIDE}")
endif()

string(TOUPPER "${PVA_FETCH_PREBUILT_BINARIES}" _pva_fetch_mode)
if(NOT PVA_PREBUILT_TAG)
    # AUTO with an unstamped manifest: build from source.
    if(NOT _pva_fetch_mode STREQUAL "AUTO")
        message(FATAL_ERROR
            "[pva-fetch] PVA_FETCH_PREBUILT_BINARIES=ON but no release tag is pinned. "
            "Check out a stamped release.")
    endif()
    message(STATUS "[pva-fetch] AUTO: no release tag pinned - building from source")
    return()
endif()

set(_pva_ver   "${CUPVA_RUNTIME_VERSION}")            # 2005..2010
set(_pva_arch  "${CMAKE_LIBRARY_ARCHITECTURE}")       # aarch64-linux-gnu | aarch64-qnx710/800 | x86_64-linux-gnu
set(_pva_base  "${PVA_PREBUILT_BASE_URL}/${PVA_PREBUILT_TAG}")
set(_pva_cache "${CMAKE_CURRENT_SOURCE_DIR}/.pva_prebuilt_cache")
file(MAKE_DIRECTORY "${_pva_cache}")

message(STATUS "[pva-fetch] tag=${PVA_PREBUILT_TAG} runtime=${_pva_ver} arch=${_pva_arch}")
message(STATUS "[pva-fetch] base=${_pva_base}")

# _pva_fetch_and_extract(<objname> <url_subdir> <sha> <dest_dir> <stamp_key>)
#   objname   : basename of the object (a <objname>.tar.gz is expected on the server)
#   url_subdir: server sub-path under <tag>/ that holds <objname>.tar.gz
#   sha       : expected SHA256 (required; empty -> fatal error)
#   dest_dir  : directory the tarball is extracted into
#   stamp_key : unique key for the idempotency stamp written under dest_dir
function(_pva_fetch_and_extract objname url_subdir sha dest_dir stamp_key)
    set(_url   "${_pva_base}/${url_subdir}/${objname}.tar.gz")
    # Cache is scoped by tag so a tree re-stamped to a different release can
    # never reuse a previous tag's tarball.
    set(_tar   "${_pva_cache}/${PVA_PREBUILT_TAG}/${objname}.tar.gz")
    set(_stamp "${dest_dir}/.pva_fetched_${stamp_key}")
    get_filename_component(_tar_dir "${_tar}" DIRECTORY)
    file(MAKE_DIRECTORY "${_tar_dir}")

    # Idempotent: skip if this exact (tag,object) was already extracted here.
    if(EXISTS "${_stamp}")
        file(READ "${_stamp}" _stamp_tag)
        string(STRIP "${_stamp_tag}" _stamp_tag)
        if(_stamp_tag STREQUAL PVA_PREBUILT_TAG)
            message(STATUS "[pva-fetch] ${objname}: already present (tag ${PVA_PREBUILT_TAG})")
            return()
        endif()
    endif()

    if(NOT sha)
        message(FATAL_ERROR
            "[pva-fetch] no SHA256 pinned for ${objname} in the stamped manifest. "
            "A release must pin every object's SHA256 - re-stamp the manifest.")
    endif()

    # A cached tarball is reused only after re-checking its hash: a truncated or
    # tampered cache entry must never be extracted.
    if(EXISTS "${_tar}" AND sha)
        file(SHA256 "${_tar}" _have_sha)
        if(NOT _have_sha STREQUAL sha)
            message(STATUS "[pva-fetch] cached ${objname}.tar.gz failed SHA256 - re-downloading")
            file(REMOVE "${_tar}")
        endif()
    endif()

    if(NOT EXISTS "${_tar}")
        message(STATUS "[pva-fetch] downloading ${_url}")
        # Download to a .part file and publish it into the cache only after it
        # passes the hash + gzip checks, so an interrupted or corrupt download can
        # never be left behind as a "valid" cache entry for a later configure.
        set(_part "${_tar}.part")
        file(REMOVE "${_part}")
        set(_hash_args "")
        if(sha)
            set(_hash_args EXPECTED_HASH "SHA256=${sha}")
        endif()
        file(DOWNLOAD "${_url}" "${_part}"
            STATUS _dl_status
            TLS_VERIFY ON
            ${_hash_args})
        list(GET _dl_status 0 _dl_code)
        list(GET _dl_status 1 _dl_msg)
        if(NOT _dl_code EQUAL 0)
            set(_body "")
            if(EXISTS "${_part}")
                file(READ "${_part}" _body LIMIT 512)
                file(REMOVE "${_part}")
            endif()
            message(FATAL_ERROR
                "[pva-fetch] failed to download ${_url}\n"
                "  status: ${_dl_code} ${_dl_msg}\n"
                "  body:   ${_body}\n"
                "  If this host is unreachable, override the source with "
                "-DPVA_PREBUILT_BASE_URL_OVERRIDE=<url>.")
        endif()
        # validate the gzip magic so a failed download gives a clear error
        # instead of a confusing ARCHIVE_EXTRACT failure.
        file(READ "${_part}" _magic LIMIT 2 HEX)
        if(NOT _magic STREQUAL "1f8b")
            file(READ "${_part}" _body LIMIT 512)
            file(REMOVE "${_part}")
            message(FATAL_ERROR
                "[pva-fetch] ${_url} is not a gzip archive (likely a 404 / error page):\n${_body}")
        endif()
        file(RENAME "${_part}" "${_tar}")
    endif()

    file(MAKE_DIRECTORY "${dest_dir}")
    file(ARCHIVE_EXTRACT INPUT "${_tar}" DESTINATION "${dest_dir}")
    file(WRITE "${_stamp}" "${PVA_PREBUILT_TAG}\n")
    message(STATUS "[pva-fetch] ${objname}: extracted into ${dest_dir}")
endfunction()

set(_pva_dl_dir "${CMAKE_CURRENT_SOURCE_DIR}/src/operator/deep_learning/pva-dl-lib")

# --- Device binaries: VPU-only, keyed by runtime version. Fetched only when the
#     user opts in with IMPORT_DEVICE_TARGETS=ON (default OFF: device code compiles
#     from source). Not valid on NATIVE (the SDK cannot import cross ELFs into an
#     x86 device build). pva_dl_lib / onnx / assets below are fetched regardless. ---
if(IMPORT_DEVICE_TARGETS AND NOT CUPVA_BUILD_MODE STREQUAL "NATIVE")
    _pva_fetch_and_extract(
        "device_binaries_${_pva_ver}"
        "device_binaries"
        "${PVA_PREBUILT_SHA256_device_binaries_${_pva_ver}}"
        "${DEVICE_BINARIES_ROOT}"                 # tarball root is <ver>/...
        "dev_${_pva_ver}")
endif()

# --- pva_dl_lib ---
set(_pva_dl_arch "${_pva_arch}")
if(CUPVA_BUILD_MODE STREQUAL "EMU")
    string(APPEND _pva_dl_arch "-emu")
endif()
_pva_fetch_and_extract(
    "pva_dl_lib_${_pva_ver}_${_pva_dl_arch}"
    "pva_dl_lib"
    "${PVA_PREBUILT_SHA256_pva_dl_lib_${_pva_ver}_${_pva_dl_arch}}"
    "${_pva_dl_dir}"                              # tarball root is lib/<arch>/... + include/...
    "dl_${_pva_ver}_${_pva_dl_arch}")

# --- onnx compiler: x86_64 build-host tool, runtime-version independent ---
_pva_fetch_and_extract(
    "onnx_compiler_x86_64-linux-gnu"
    "host-tools"
    "${PVA_PREBUILT_SHA256_onnx_compiler_x86_64-linux-gnu}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools"       # tarball root is pva-onnx-compiler
    "onnx")

# --- Sample/pipeline data assets: runtime- and arch-independent ---
_pva_fetch_and_extract(
    "assets"
    "assets"
    "${PVA_PREBUILT_SHA256_assets}"
    "${CMAKE_CURRENT_SOURCE_DIR}"             # tarball root is samples/... + pipelines/...
    "assets")
