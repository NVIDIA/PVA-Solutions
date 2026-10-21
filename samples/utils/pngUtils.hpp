/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef PNG_UTILS_HPP
#define PNG_UTILS_HPP

#include "lodepng.h" // Adjusted path for lodepng.h

#include <cstring>  // Needed for memcpy
#include <iostream> // Needed for std::cout
#include <string>
#include <vector>

/**
 * @brief Load a grayscale PNG image into a buffer
 *
 * @tparam T The type of the buffer to load the image into
 * @param filePath The path to the PNG file
 * @param buf The buffer to load the image into
 * @param height The height of the image
 * @param width The width of the image
 * @return 0 on success, error code on failure
 */
template<typename T>
int32_t loadGrayscalePNG(std::string filePath, T *buf, int32_t height, int32_t width)
{
    std::string suffixStr = filePath.substr(filePath.find_last_of('.') + 1);

    if (suffixStr == "png")
    {
        std::vector<unsigned char>
            png_data; // Renamed from 'png' to avoid confusion if a 'width' variable named 'png' existed
        std::vector<unsigned char> image_pixels; // Renamed from 'image'
        unsigned dec_width, dec_height;          // Renamed from width, height to avoid shadowing parameters

        //load and decode
        unsigned error = lodepng::load_file(png_data, filePath); // Use filePath directly
        if (!error)
        {
            lodepng::State state;
            state.info_raw.colortype = LCT_GREY;
            state.info_raw.bitdepth  = sizeof(T) * 8;
            error                    = lodepng::decode(image_pixels, dec_width, dec_height, state, png_data);

            // Verify that decode was successful
            if (!error)
            {
                // Verify that the decoded image is actually grayscale
                if (state.info_png.color.colortype != LCT_GREY)
                {
                    std::cout << "Input image " << filePath << " is not grayscale." << std::endl;
                    return -1;
                }
                // Verify that the decoded image is not smaller than the expected dimensions
                if (width * height > dec_width * dec_height)
                {
                    std::cout << "Decoded image image size (" << dec_width << "x" << dec_height
                              << ") is smaller than expected dimensions (" << width << "x" << height << ")."
                              << std::endl;
                    return -1;
                }
                memcpy(buf, image_pixels.data(), width * height * sizeof(T));
            }
            else
            {
                std::cout << "Failed to decode " << filePath << " error " << error << ": " << lodepng_error_text(error)
                          << std::endl;
                return error;
            }
        }
        else
        {
            std::cout << "Failed to load " << filePath << " error " << error << ": " << lodepng_error_text(error)
                      << std::endl;
            return error;
        }
    }
    else
    {
        std::cout << "Unsupported file type: " << suffixStr << std::endl;
        return -1;
    }

    return 0; // Success
}

/**
 * @brief Write a grayscale PNG image from a buffer
 *
 * @tparam T The type of the buffer to write the image from
 * @param filePath The path to the PNG file
 * @param buf The buffer to write the image from
 * @param height The height of the image
 * @param width The width of the image
 * @return 0 on success, error code on failure
 */
template<typename T>
int32_t writeGrayscalePNG(std::string filePath, T *buf, int32_t height, int32_t width)
{
    std::vector<unsigned char> image_data_vec(width * height * sizeof(T)); // Use unsigned char for LodePNG
    memcpy(image_data_vec.data(), buf, width * height * sizeof(T));

    lodepng::State state;
    state.info_raw.colortype = LCT_GREY;
    state.info_raw.bitdepth  = sizeof(T) * 8; // Bit depth of the raw input buffer

    // For PNG output, LodePNG often works best if you aim for common formats,
    // or let it auto-convert if the raw bit depth isn't standard for PNG (e.g. 16-bit for LCT_GREY).
    // If T is uint8_t, then sizeof(T)*8 is 8. If T is uint16_t, it's 16. Both are valid for LCT_GREY.
    state.info_png.color.colortype = LCT_GREY;
    state.info_png.color.bitdepth  = sizeof(T) * 8; // Match raw bit depth for directness

    state.encoder.auto_convert = 0; // Do not auto-convert, use the settings from info_png.color

    std::vector<unsigned char> png_buffer;
    unsigned error = lodepng::encode(png_buffer, image_data_vec, width, height, state);
    if (error)
    {
        std::cout << "LodePNG encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
        return error;
    }
    else
    {
        error = lodepng::save_file(png_buffer, filePath.c_str()); // save_file needs const char*
        if (error)
        {
            std::cout << "LodePNG save_file error " << error << ": " << lodepng_error_text(error) << std::endl;
            return error;
        }
    }
    return 0; // Success
}

#endif // PNG_UTILS_HPP
