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

#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include <cstdint>

template<class T>
struct Point2
{
    T x;
    T y;
    Point2() {}
    Point2(T x, T y)
        : x{x}
        , y{y}
    {
    }
};
using Point2f = Point2<float>;
using Point2i = Point2<int32_t>;

template<class T>
struct Rect
{
    T x;
    T y;
    T width;
    T height;
    Rect() {}
    Rect(T x, T y, T w, T h)
        : x{x}
        , y{y}
        , width{w}
        , height{h}
    {
    }
    bool contains(const Point2<T> &pt) const
    {
        return x <= pt.x && pt.x < x + width && y <= pt.y && pt.y < y + height;
    }
};

#endif