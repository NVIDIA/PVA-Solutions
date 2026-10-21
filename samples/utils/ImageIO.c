/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ImageIO.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

int GetAssetsDirectory(int argc, char **argv, char *assetsDirectory, int maxPathLength)
{
    int err = -1;

    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "-a") || !strcmp(argv[i], "--assets"))
        {
            if (i + 1 < argc)
            {
                strncpy(assetsDirectory, argv[++i], maxPathLength - 1);
                assetsDirectory[maxPathLength - 1] = '\0';
                printf("Using asset directory: %s\n", assetsDirectory);
                err = 0;
            }
            else
            {
                printf("--assets option requires one argument.\n");
            }
            break;
        }
    }

    if (err != 0)
    {
        printf("Please specify assets directory using -a or --assets option\n");
    }

    return err;
}

int WriteImageBuffer(const char *filename, const char *directory, void *buf, size_t size)
{
    int retval = 0;
    char outputImagePath[MAX_PATH_LENGTH];
    snprintf(outputImagePath, MAX_PATH_LENGTH, "%s/%s", directory, filename);
    FILE *fp = fopen(outputImagePath, "w");
    if (fp == NULL)
    {
        printf("Failed to open %s for writing!\n", outputImagePath);
        return -1;
    }

    size_t s = fwrite(buf, sizeof(char), size, fp);
    if (s != size)
    {
        printf("Failed to write %lu bytes to %s\n", size, outputImagePath);
        retval = -1;
    }
    else
    {
        printf("Wrote %lu bytes to %s\n", size, outputImagePath);
    }
    fclose(fp);
    return retval;
}

int ReadImageBuffer(const char *filename, const char *directory, void *buf, size_t size)
{
    int retval = 0;
    char inputImagePath[MAX_PATH_LENGTH];
    snprintf(inputImagePath, MAX_PATH_LENGTH, "%s/%s", directory, filename);
    FILE *fp = fopen(inputImagePath, "r");
    if (fp == NULL)
    {
        printf("Failed to open %s for reading!\n", inputImagePath);
        return -1;
    }

    size_t s = fread(buf, sizeof(char), size, fp);
    if (s != size)
    {
        printf("Failed to read %lu bytes from %s\n", size, inputImagePath);
        retval = -1;
    }
    fclose(fp);
    return retval;
}

// "size" is the number of source pixels in bytes to keep aligned with ReadImageBuffer()
int ReadImageBufferIN16OUT8(const char *filename, const char *directory, void *buf, size_t size)
{
    int retval = 0;
    char inputImagePath[MAX_PATH_LENGTH];
    snprintf(inputImagePath, MAX_PATH_LENGTH, "%s/%s", directory, filename);
    FILE *fp = fopen(inputImagePath, "r");
    if (fp == NULL)
    {
        printf("Failed to open %s for reading!\n", inputImagePath);
        return -1;
    }

    const size_t numPixels = size / sizeof(uint16_t);
    uint16_t pixel;
    size_t elementsRead = 0;
    uint8_t *dst        = (uint8_t *)buf;
    while (elementsRead < numPixels && fread(&pixel, sizeof(uint16_t), 1, fp) == 1)
    {
        dst[elementsRead++] = pixel >> 8;
    }

    if (elementsRead != numPixels)
    {
        printf("Failed to read %lu pixels from %s (read %lu pixels)\n", numPixels, inputImagePath, elementsRead);
        retval = -1;
    }

    fclose(fp);
    return retval;
}

int ReadCvtImageBuffer(const char *filename, const char *directory, int srcBytesPerPixel, int dstBytesPerPixel,
                       size_t numPixels, void *dst)
{
    if (srcBytesPerPixel == dstBytesPerPixel)
    {
        return ReadImageBuffer(filename, directory, dst, numPixels * srcBytesPerPixel);
    }
    else if (srcBytesPerPixel == 2 && dstBytesPerPixel == 1)
    {
        return ReadImageBufferIN16OUT8(filename, directory, dst, numPixels * srcBytesPerPixel);
    }
    else
    {
        printf("Unsupported srcBytesPerPixel: %d, dstBytesPerPixel: %d\n", srcBytesPerPixel, dstBytesPerPixel);
        return -1;
    }
}

int ReadCSVFloatBuffer(const char *filename, const char *directory, float *buf, int size)
{
    int retval = 0, i = 0;
    char inputImagePath[MAX_PATH_LENGTH];
    snprintf(inputImagePath, MAX_PATH_LENGTH, "%s/%s", directory, filename);
    FILE *fp = fopen(inputImagePath, "r");
    if (fp == NULL)
    {
        printf("Failed to open %s for reading!\n", inputImagePath);
        return -1;
    }

    while (i < size && fscanf(fp, "%f", &buf[i]) != EOF)
    {
        i++;
        if (fscanf(fp, ",") < 0)
        {
            retval = -1;
        }
    };

    if (i != size)
    {
        printf("Failed to read %d entries from %s (%d entries were read)\n", size, inputImagePath, i);
        retval = -1;
    }
    else
    {
        printf("Read %d entries from %s\n", size, inputImagePath);
    }
    fclose(fp);
    return retval;
}
