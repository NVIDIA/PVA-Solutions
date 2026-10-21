/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "bundle_adjustment_ref.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

constexpr float kFloatMin = 1.17549435e-38f;
constexpr float kPiHalf   = 1.5707963267948966f;

struct Vec3
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

inline Vec3 operator+(const Vec3 &a, const Vec3 &b)
{
    return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 operator-(const Vec3 &a, const Vec3 &b)
{
    return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 operator*(float s, const Vec3 &a)
{
    return Vec3{s * a.x, s * a.y, s * a.z};
}

inline Vec3 Cross(const Vec3 &a, const Vec3 &b)
{
    return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// Row-major 3x3 matrix.
struct Mat3
{
    float m[3][3] = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};

    float &At(int i, int j)
    {
        return m[i][j];
    }
    float At(int i, int j) const
    {
        return m[i][j];
    }
};

inline Mat3 MatMul(const Mat3 &a, const Mat3 &b)
{
    Mat3 r;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < 3; ++k)
            {
                sum += a.At(i, k) * b.At(k, j);
            }
            r.At(i, j) = sum;
        }
    }
    return r;
}

inline Vec3 MatVec(const Mat3 &r, const Vec3 &v)
{
    return Vec3{r.At(0, 0) * v.x + r.At(0, 1) * v.y + r.At(0, 2) * v.z,
                r.At(1, 0) * v.x + r.At(1, 1) * v.y + r.At(1, 2) * v.z,
                r.At(2, 0) * v.x + r.At(2, 1) * v.y + r.At(2, 2) * v.z};
}

// Rig-to-camera-frame (or world-to-camera-frame) SE(3) rigid transform.
struct SE3
{
    Mat3 R;
    Vec3 t;

    static SE3 FromRowMajor4x4(const float *m)
    {
        SE3 se3;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                se3.R.At(i, j) = m[i * 4 + j];
            }
        }
        se3.t = Vec3{m[0 * 4 + 3], m[1 * 4 + 3], m[2 * 4 + 3]};
        return se3;
    }

    void ToRowMajor4x4(float *m) const
    {
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                m[i * 4 + j] = R.At(i, j);
            }
        }
        m[0 * 4 + 3] = t.x;
        m[1 * 4 + 3] = t.y;
        m[2 * 4 + 3] = t.z;
        m[3 * 4 + 0] = 0.0f;
        m[3 * 4 + 1] = 0.0f;
        m[3 * 4 + 2] = 0.0f;
        m[3 * 4 + 3] = 1.0f;
    }

    SE3 Compose(const SE3 &other) const
    {
        SE3 result;
        result.R = MatMul(R, other.R);
        result.t = MatVec(R, other.t) + t;
        return result;
    }

    Vec3 TransformPoint(const Vec3 &p) const
    {
        return MatVec(R, p) + t;
    }
};

// Small dense row-major matrix.
class Matrix
{
public:
    Matrix()
        : rows_(0)
        , cols_(0)
    {
    }
    Matrix(int32_t rows, int32_t cols)
        : rows_(rows)
        , cols_(cols)
        , data_(static_cast<size_t>(rows) * cols, 0.0f)
    {
    }

    int32_t Rows() const
    {
        return rows_;
    }
    int32_t Cols() const
    {
        return cols_;
    }

    float &At(int32_t i, int32_t j)
    {
        return data_[static_cast<size_t>(i) * cols_ + j];
    }
    float At(int32_t i, int32_t j) const
    {
        return data_[static_cast<size_t>(i) * cols_ + j];
    }

    void SetZero()
    {
        std::fill(data_.begin(), data_.end(), 0.0f);
    }

    Matrix Transpose() const
    {
        Matrix result(cols_, rows_);
        for (int32_t i = 0; i < rows_; ++i)
        {
            for (int32_t j = 0; j < cols_; ++j)
            {
                result.At(j, i) = At(i, j);
            }
        }
        return result;
    }

    Matrix operator*(const Matrix &rhs) const
    {
        Matrix result(rows_, rhs.cols_);
        for (int32_t i = 0; i < rows_; ++i)
        {
            for (int32_t j = 0; j < rhs.cols_; ++j)
            {
                float sum = 0.0f;
                for (int32_t k = 0; k < cols_; ++k)
                {
                    sum += At(i, k) * rhs.At(k, j);
                }
                result.At(i, j) = sum;
            }
        }
        return result;
    }

    Matrix operator+(const Matrix &rhs) const
    {
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i)
        {
            result.data_[i] = data_[i] + rhs.data_[i];
        }
        return result;
    }

    Matrix operator-(const Matrix &rhs) const
    {
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i)
        {
            result.data_[i] = data_[i] - rhs.data_[i];
        }
        return result;
    }

    Matrix operator*(float s) const
    {
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i)
        {
            result.data_[i] = data_[i] * s;
        }
        return result;
    }

    Matrix Block(int32_t i, int32_t j, int32_t m, int32_t n) const
    {
        Matrix result(m, n);
        for (int32_t mi = 0; mi < m; ++mi)
        {
            for (int32_t mj = 0; mj < n; ++mj)
            {
                result.At(mi, mj) = At(i + mi, j + mj);
            }
        }
        return result;
    }

    void SetBlock(const Matrix &src, int32_t i, int32_t j)
    {
        for (int32_t mi = 0; mi < src.Rows(); ++mi)
        {
            for (int32_t mj = 0; mj < src.Cols(); ++mj)
            {
                At(i + mi, j + mj) = src.At(mi, mj);
            }
        }
    }

private:
    int32_t rows_;
    int32_t cols_;
    std::vector<float> data_;
};

inline Matrix operator*(float s, const Matrix &m)
{
    return m * s;
}

// Solves A * x = B for symmetric positive-(semi)definite A.
Matrix CholeskySolve(const Matrix &A, const Matrix &B, int32_t n)
{
    Matrix L(n, n);
    for (int32_t i = 0; i < n; ++i)
    {
        for (int32_t j = 0; j <= i; ++j)
        {
            float val = A.At(i, j);
            for (int32_t k = 0; k < j; ++k)
            {
                val -= L.At(i, k) * L.At(j, k);
            }

            if (i == j)
            {
                // Clamp rather than assert so a degenerate system still returns a value.
                L.At(i, i) = std::sqrt(std::max(val, 1e-12f));
            }
            else
            {
                L.At(i, j) = val / L.At(j, j);
            }
        }
    }

    Matrix y(n, 1);
    for (int32_t i = 0; i < n; ++i)
    {
        float val = B.At(i, 0);
        for (int32_t j = 0; j < i; ++j)
        {
            val -= L.At(i, j) * y.At(j, 0);
        }
        y.At(i, 0) = val / L.At(i, i);
    }

    Matrix x(n, 1);
    for (int32_t i = n - 1; i >= 0; --i)
    {
        float val = y.At(i, 0);
        for (int32_t j = i + 1; j < n; ++j)
        {
            val -= L.At(j, i) * x.At(j, 0);
        }
        x.At(i, 0) = val / L.At(i, i);
    }

    return x;
}

// Returns false and pins the residual to (1, 1) when the point is closer than minDistance.
bool ComputeResidual(const Vec3 &pointCamera, float measU, float measV, float minDistance, Matrix &residual)
{
    const bool inFront = pointCamera.z >= minDistance;
    if (!inFront)
    {
        residual.At(0, 0) = 1.0f;
        residual.At(1, 0) = 1.0f;
    }
    else
    {
        residual.At(0, 0) = pointCamera.x / pointCamera.z - measU;
        residual.At(1, 0) = pointCamera.y / pointCamera.z - measV;
    }
    return inFront;
}

// d(residual)/d(point) in camera frame; cameraRotation is the rotation of rig2cam * pose.
Matrix ComputeJacobianPoint(const Mat3 &cameraRotation, const Vec3 &pointCamera)
{
    const float x     = pointCamera.x;
    const float y     = pointCamera.y;
    const float z     = pointCamera.z;
    const float zinv  = 1.0f / z;
    const float zinv2 = zinv * zinv;

    Matrix jp(2, 3);
    jp.At(0, 0) = -x * cameraRotation.At(2, 0) * zinv2 + cameraRotation.At(0, 0) * zinv;
    jp.At(0, 1) = -x * cameraRotation.At(2, 1) * zinv2 + cameraRotation.At(0, 1) * zinv;
    jp.At(0, 2) = -x * cameraRotation.At(2, 2) * zinv2 + cameraRotation.At(0, 2) * zinv;
    jp.At(1, 0) = -y * cameraRotation.At(2, 0) * zinv2 + cameraRotation.At(1, 0) * zinv;
    jp.At(1, 1) = -y * cameraRotation.At(2, 1) * zinv2 + cameraRotation.At(1, 1) * zinv;
    jp.At(1, 2) = -y * cameraRotation.At(2, 2) * zinv2 + cameraRotation.At(1, 2) * zinv;
    return jp;
}

// d(residual)/d(pose) on the se(3) tangent, ordered [translation; rotation].
Matrix ComputeJacobianPose(const Vec3 &pointCamera, const Vec3 &pointRig, const Mat3 &rig2camR)
{
    const float a = pointCamera.x;
    const float b = pointCamera.y;
    const float c = pointCamera.z;
    const float d = pointRig.x;
    const float e = pointRig.y;
    const float f = pointRig.z;

    const float r0 = rig2camR.At(0, 0);
    const float r1 = rig2camR.At(0, 1);
    const float r2 = rig2camR.At(0, 2);
    const float r3 = rig2camR.At(1, 0);
    const float r4 = rig2camR.At(1, 1);
    const float r5 = rig2camR.At(1, 2);
    const float r6 = rig2camR.At(2, 0);
    const float r7 = rig2camR.At(2, 1);
    const float r8 = rig2camR.At(2, 2);

    const float cinv  = 1.0f / c;
    const float cinv2 = cinv * cinv;

    const float c1 = r5 * cinv - (b * r8) * cinv2;
    const float c2 = r4 * cinv - (b * r7) * cinv2;
    const float c3 = r3 * cinv - (b * r6) * cinv2;
    const float c4 = r2 * cinv - (a * r8) * cinv2;
    const float c5 = r1 * cinv - (a * r7) * cinv2;
    const float c6 = r0 * cinv - (a * r6) * cinv2;

    Matrix jc(2, 6);
    jc.At(0, 0) = c6;
    jc.At(0, 1) = c5;
    jc.At(0, 2) = c4;
    jc.At(1, 0) = c3;
    jc.At(1, 1) = c2;
    jc.At(1, 2) = c1;
    jc.At(0, 3) = e * c4 - f * c5;
    jc.At(0, 4) = f * c6 - d * c4;
    jc.At(0, 5) = d * c5 - e * c6;
    jc.At(1, 3) = e * c1 - f * c2;
    jc.At(1, 4) = f * c3 - d * c1;
    jc.At(1, 5) = d * c2 - e * c3;
    return jc;
}

// Multiplicative LM damping: diag(m) *= (1 + lambda).
void DiagonalScale(Matrix &m, float lambda, int32_t n)
{
    for (int32_t i = 0; i < n; ++i)
    {
        m.At(i, i) = m.At(i, i) + lambda * m.At(i, i);
    }
}

// Inverse of a symmetric 3x3; returns false (invA untouched) if near-singular.
bool InverseSymmetric3x3(const Matrix &A, Matrix &invA, float threshold)
{
    const float a00 = A.At(0, 0);
    const float a10 = A.At(1, 0);
    const float a20 = A.At(2, 0);
    const float a11 = A.At(1, 1);
    const float a21 = A.At(2, 1);
    const float a22 = A.At(2, 2);

    const float cof0 = a11 * a22 - a21 * a21;
    const float cof1 = a20 * a21 - a10 * a22;
    const float cof2 = a10 * a21 - a20 * a11;
    const float detA = a00 * cof0 + a10 * cof1 + a20 * cof2;

    if (std::fabs(detA) < threshold)
    {
        return false;
    }

    const float invDetA = 1.0f / detA;
    invA.At(0, 0)       = cof0 * invDetA;
    invA.At(1, 0)       = cof1 * invDetA;
    invA.At(2, 0)       = cof2 * invDetA;
    invA.At(1, 1)       = (a00 * a22 - a20 * a20) * invDetA;
    invA.At(2, 1)       = (a10 * a20 - a00 * a21) * invDetA;
    invA.At(2, 2)       = (a00 * a11 - a10 * a10) * invDetA;
    invA.At(0, 1)       = invA.At(1, 0);
    invA.At(0, 2)       = invA.At(2, 0);
    invA.At(1, 2)       = invA.At(2, 1);
    return true;
}

// Exp-map coefficients; Taylor expansion below thetaSq = 1e-4 avoids 0/0.
void CalcExpMapCoef(float thetaSq, float &A, float &B, float &C)
{
    if (thetaSq < 1.0e-4f)
    {
        const float fac = 1.0f / 6.0f;
        A               = 1.0f - thetaSq * fac;
        B               = 0.5f - thetaSq * (fac * 0.25f);
        C               = fac - thetaSq * (fac * 0.05f);
    }
    else
    {
        const float theta      = std::sqrt(thetaSq);
        const float thetaInv   = 1.0f / theta;
        const float thetaSqInv = thetaInv * thetaInv;

        A = thetaInv * std::sin(theta);
        B = (theta < kPiHalf) ? (A * A / (1.0f + std::cos(theta))) : (thetaSqInv * (1.0f - std::cos(theta)));
        C = thetaSqInv * (1.0f - A);
    }
}

// Rodrigues rotation matrix for w, from CalcExpMapCoef coefficients.
Mat3 CalcTaylorExpansion(const Vec3 &w, float A, float B)
{
    Mat3 rot;
    rot.At(0, 0) = 1.0f - B * (w.y * w.y + w.z * w.z);
    rot.At(1, 1) = 1.0f - B * (w.x * w.x + w.z * w.z);
    rot.At(2, 2) = 1.0f - B * (w.x * w.x + w.y * w.y);

    rot.At(0, 1) = B * w.x * w.y - A * w.z;
    rot.At(1, 0) = B * w.x * w.y + A * w.z;

    rot.At(0, 2) = B * w.x * w.z + A * w.y;
    rot.At(2, 0) = B * w.x * w.z - A * w.y;

    rot.At(1, 2) = B * w.y * w.z - A * w.x;
    rot.At(2, 1) = B * w.y * w.z + A * w.x;
    return rot;
}

// SE(3) exponential map of a 6-vector [u (translation); w (rotation)].
SE3 ExponentialMap(const Matrix &dc)
{
    const Vec3 u{dc.At(0, 0), dc.At(1, 0), dc.At(2, 0)};
    const Vec3 w{dc.At(3, 0), dc.At(4, 0), dc.At(5, 0)};

    const float thetaSq = w.x * w.x + w.y * w.y + w.z * w.z;
    float A             = 0.0f;
    float B             = 0.0f;
    float C             = 0.0f;
    CalcExpMapCoef(thetaSq, A, B, C);

    const Mat3 rot = CalcTaylorExpansion(w, A, B);
    const Vec3 wxu = Cross(w, u);
    const Vec3 t   = u + B * wxu + C * Cross(w, wxu);

    SE3 pose;
    pose.R = rot;
    pose.t = t;
    return pose;
}

// Cauchy robustifier: returns the cost and outputs the IRLS weight.
float ComputeRobustifierCostAndWeight(float &weight, float squareError, float squareScale)
{
    const float c   = 1.0f / squareScale;
    const float sum = 1.0f + squareError * c;
    weight          = std::max(kFloatMin, 1.0f / sum);
    return squareScale * std::log(sum);
}

struct TrackPoseBlock
{
    int32_t cameraIndex;
    Matrix tPc; // Hpc^T * Hpp^-1, 6x3
};

bool IsCameraActive(int32_t cameraIndex, int32_t activeCameraBegin, int32_t activeCameraCount)
{
    const int32_t idx = cameraIndex - activeCameraBegin;
    return idx >= 0 && idx < activeCameraCount;
}

int32_t ActivePoseIndex(int32_t cameraIndex, int32_t activeCameraBegin)
{
    return cameraIndex - activeCameraBegin;
}

// Solves the Schur-reduced pose system A * dC = B, back-substitutes point updates dP,
// and returns the total cost at the pre-update state.
float ComputeUpdateAndCost(const std::vector<SE3> &rigPoses, const std::vector<Vec3> &trackPoints,
                           const std::vector<uint32_t> &trackObservedMask, const std::vector<bool> &trackFixed,
                           const float *measurements, int32_t cameraCount, int32_t trackCount, const SE3 &rig2cam,
                           int32_t activeCameraBegin, int32_t activeCameraCount, const BundleAdjustmentSetting &setting,
                           float lambda, std::vector<Vec3> &dP, Matrix &dC)
{
    const int32_t dimc = 6 * activeCameraCount;
    Matrix A(dimc, dimc);
    Matrix B(dimc, 1);

    std::vector<Vec3> tP(trackCount);
    std::vector<std::vector<TrackPoseBlock>> trackPoseBlocks(trackCount);

    const float squareScale = setting.robustifierScaleMeas * setting.robustifierScaleMeas;
    float cost              = 0.0f;

    for (int32_t i = 0; i < trackCount; ++i)
    {
        Matrix Hpp(3, 3);
        Matrix bp(3, 1);
        struct HpcEntry
        {
            int32_t cameraIndex;
            Matrix hpc;
        };
        std::vector<HpcEntry> hpcByCamera;

        for (int32_t camIdx = 0; camIdx < cameraCount; ++camIdx)
        {
            if (((trackObservedMask[i] >> camIdx) & 1u) == 0u)
            {
                continue;
            }

            const SE3 cameraPose   = rig2cam.Compose(rigPoses[camIdx]);
            const Vec3 pointCamera = cameraPose.TransformPoint(trackPoints[i]);

            const size_t measOffset = (static_cast<size_t>(i) * cameraCount + camIdx) * 2;
            const float measU       = measurements[measOffset + 0];
            const float measV       = measurements[measOffset + 1];

            Matrix residual(2, 1);
            const bool inFront = ComputeResidual(pointCamera, measU, measV, setting.minDistance, residual);

            float weight            = 1.0f;
            const float squareError = residual.At(0, 0) * residual.At(0, 0) + residual.At(1, 0) * residual.At(1, 0);
            cost += ComputeRobustifierCostAndWeight(weight, squareError, squareScale);

            if (!inFront)
            {
                continue;
            }

            const Matrix jp       = ComputeJacobianPoint(cameraPose.R, pointCamera);
            Matrix jtjPoint       = (weight * setting.rinvMeas) * (jp.Transpose() * jp);
            const Matrix jtePoint = (weight * setting.rinvMeas) * (jp.Transpose() * residual);
            DiagonalScale(jtjPoint, lambda, 3);
            Hpp = Hpp + jtjPoint;
            bp  = bp - jtePoint;

            if (IsCameraActive(camIdx, activeCameraBegin, activeCameraCount))
            {
                const Vec3 pointRig  = rigPoses[camIdx].TransformPoint(trackPoints[i]);
                const Matrix jc      = ComputeJacobianPose(pointCamera, pointRig, rig2cam.R);
                Matrix jtjPose       = (weight * setting.rinvMeas) * (jc.Transpose() * jc);
                const Matrix jtePose = (weight * setting.rinvMeas) * (jc.Transpose() * residual);
                DiagonalScale(jtjPose, lambda, 6);

                const Matrix hpc = (weight * setting.rinvMeas) * (jp.Transpose() * jc);
                hpcByCamera.push_back(HpcEntry{camIdx, hpc});

                const int32_t cidx = ActivePoseIndex(camIdx, activeCameraBegin);
                A.SetBlock(A.Block(cidx * 6, cidx * 6, 6, 6) + jtjPose, cidx * 6, cidx * 6);
                B.SetBlock(B.Block(cidx * 6, 0, 6, 1) - jtePose, cidx * 6, 0);
            }
        }

        Matrix HppInv(3, 3);
        const bool hasInverse = InverseSymmetric3x3(Hpp, HppInv, 1e-18f);
        (void)hasInverse; // singular Hpp leaves HppInv == 0, i.e. no point update

        const Matrix tp = HppInv * bp;
        tP[i]           = Vec3{tp.At(0, 0), tp.At(1, 0), tp.At(2, 0)};

        for (size_t mi = 0; mi < hpcByCamera.size(); ++mi)
        {
            const int32_t camI  = hpcByCamera[mi].cameraIndex;
            const Matrix &hpcI  = hpcByCamera[mi].hpc;
            const int32_t cidxI = ActivePoseIndex(camI, activeCameraBegin);

            const Matrix tPc = hpcI.Transpose() * HppInv;
            trackPoseBlocks[i].push_back(TrackPoseBlock{camI, tPc});

            B.SetBlock(B.Block(cidxI * 6, 0, 6, 1) - hpcI.Transpose() * tp, cidxI * 6, 0);

            for (size_t mj = mi; mj < hpcByCamera.size(); ++mj)
            {
                const int32_t camJ  = hpcByCamera[mj].cameraIndex;
                const Matrix &hpcJ  = hpcByCamera[mj].hpc;
                const int32_t cidxJ = ActivePoseIndex(camJ, activeCameraBegin);

                const Matrix schurBlock = tPc * hpcJ;
                A.SetBlock(A.Block(cidxI * 6, cidxJ * 6, 6, 6) - schurBlock, cidxI * 6, cidxJ * 6);
                if (mi != mj)
                {
                    A.SetBlock(A.Block(cidxJ * 6, cidxI * 6, 6, 6) - schurBlock.Transpose(), cidxJ * 6, cidxI * 6);
                }
            }
        }
    }

    dC = (dimc > 0) ? CholeskySolve(A, B, dimc) : Matrix(0, 1);

    dP.assign(trackCount, Vec3{});
    for (int32_t i = 0; i < trackCount; ++i)
    {
        if (trackFixed[i])
        {
            continue;
        }

        Vec3 dp = tP[i];
        for (const TrackPoseBlock &block : trackPoseBlocks[i])
        {
            const int32_t cidx   = ActivePoseIndex(block.cameraIndex, activeCameraBegin);
            const Matrix dcBlock = dC.Block(cidx * 6, 0, 6, 1);
            const Matrix delta   = block.tPc.Transpose() * dcBlock;
            dp                   = dp - Vec3{delta.At(0, 0), delta.At(1, 0), delta.At(2, 0)};
        }
        dP[i] = dp;
    }

    return cost;
}

// Cost-only evaluation, used to score a candidate LM step.
float ComputeCostOnly(const std::vector<SE3> &rigPoses, const std::vector<Vec3> &trackPoints,
                      const std::vector<uint32_t> &trackObservedMask, const float *measurements, int32_t cameraCount,
                      int32_t trackCount, const SE3 &rig2cam, const BundleAdjustmentSetting &setting)
{
    const float squareScale = setting.robustifierScaleMeas * setting.robustifierScaleMeas;
    float cost              = 0.0f;

    for (int32_t i = 0; i < trackCount; ++i)
    {
        for (int32_t camIdx = 0; camIdx < cameraCount; ++camIdx)
        {
            if (((trackObservedMask[i] >> camIdx) & 1u) == 0u)
            {
                continue;
            }

            const SE3 cameraPose   = rig2cam.Compose(rigPoses[camIdx]);
            const Vec3 pointCamera = cameraPose.TransformPoint(trackPoints[i]);

            const size_t measOffset = (static_cast<size_t>(i) * cameraCount + camIdx) * 2;
            const float measU       = measurements[measOffset + 0];
            const float measV       = measurements[measOffset + 1];

            Matrix residual(2, 1);
            ComputeResidual(pointCamera, measU, measV, setting.minDistance, residual);

            float weight            = 1.0f;
            const float squareError = residual.At(0, 0) * residual.At(0, 0) + residual.At(1, 0) * residual.At(1, 0);
            cost += ComputeRobustifierCostAndWeight(weight, squareError, squareScale);
        }
    }

    return cost;
}

// Applies the LM step: points += dP (fixed tracks untouched), poses = exp(dC) * pose.
void ApplyUpdate(const std::vector<SE3> &basePoses, const std::vector<Vec3> &basePoints,
                 const std::vector<bool> &trackFixed, const std::vector<Vec3> &dP, const Matrix &dC,
                 int32_t activeCameraBegin, int32_t activeCameraCount, std::vector<SE3> &outPoses,
                 std::vector<Vec3> &outPoints)
{
    outPoints.resize(basePoints.size());
    for (size_t i = 0; i < basePoints.size(); ++i)
    {
        outPoints[i] = trackFixed[i] ? basePoints[i] : (basePoints[i] + dP[i]);
    }

    outPoses.resize(basePoses.size());
    for (size_t c = 0; c < basePoses.size(); ++c)
    {
        const int32_t cameraIndex = static_cast<int32_t>(c);
        if (IsCameraActive(cameraIndex, activeCameraBegin, activeCameraCount))
        {
            const int32_t cidx   = ActivePoseIndex(cameraIndex, activeCameraBegin);
            const Matrix dcBlock = dC.Block(cidx * 6, 0, 6, 1);
            const SE3 delta      = ExponentialMap(dcBlock);
            outPoses[c]          = delta.Compose(basePoses[c]);
        }
        else
        {
            outPoses[c] = basePoses[c];
        }
    }
}

} // namespace

int32_t BundleAdjustmentRef(const float *poses, const float *points, const float *measurements, const int32_t *bitmask,
                            const float *rig2cam, int32_t cameraCount, int32_t trackCount, int32_t activeCameraCount,
                            int32_t activeCameraBegin, const BundleAdjustmentSetting &setting, float *updatedPoses,
                            float *updatedPoints, float *finalCost, int32_t *iterationsUsed)
{
    std::vector<SE3> rigPoses(cameraCount);
    for (int32_t c = 0; c < cameraCount; ++c)
    {
        rigPoses[c] = SE3::FromRowMajor4x4(poses + static_cast<size_t>(c) * 16);
    }

    std::vector<Vec3> trackPoints(trackCount);
    for (int32_t i = 0; i < trackCount; ++i)
    {
        const float w    = points[static_cast<size_t>(i) * 4 + 3];
        const float invW = (w != 0.0f) ? (1.0f / w) : 1.0f;
        trackPoints[i] =
            Vec3{points[static_cast<size_t>(i) * 4 + 0] * invW, points[static_cast<size_t>(i) * 4 + 1] * invW,
                 points[static_cast<size_t>(i) * 4 + 2] * invW};
    }

    std::vector<uint32_t> trackObservedMask(trackCount);
    std::vector<bool> trackFixed(trackCount);
    for (int32_t i = 0; i < trackCount; ++i)
    {
        const uint32_t bm    = static_cast<uint32_t>(bitmask[i]);
        trackObservedMask[i] = bm & 0x7FFFFFFFu;
        // Bit 31 set = active track.
        trackFixed[i] = ((bm >> 31) & 1u) == 0u;
    }

    const SE3 rig2camPose = SE3::FromRowMajor4x4(rig2cam);

    std::vector<SE3> bestPoses   = rigPoses;
    std::vector<Vec3> bestPoints = trackPoints;

    float lambda           = setting.lambdaInit;
    float bestCost         = 1.0e9f;
    float costChangeFactor = 1.0f;

    std::vector<Vec3> dP;
    Matrix dC;

    int32_t iter = 0;
    while (iter < setting.maxIterations && costChangeFactor > setting.minCostChange && bestCost > setting.minCost)
    {
        bestCost = ComputeUpdateAndCost(bestPoses, bestPoints, trackObservedMask, trackFixed, measurements, cameraCount,
                                        trackCount, rig2camPose, activeCameraBegin, activeCameraCount, setting, lambda,
                                        dP, dC);

        std::vector<SE3> newPoses;
        std::vector<Vec3> newPoints;
        ApplyUpdate(bestPoses, bestPoints, trackFixed, dP, dC, activeCameraBegin, activeCameraCount, newPoses,
                    newPoints);

        const float newCost = ComputeCostOnly(newPoses, newPoints, trackObservedMask, measurements, cameraCount,
                                              trackCount, rig2camPose, setting);

        costChangeFactor = std::fabs(bestCost - newCost) / bestCost;

        if (newCost < bestCost)
        {
            if (lambda > setting.lambdaMin)
            {
                lambda *= setting.lambdaDecFactor;
            }
            bestPoses.swap(newPoses);
            bestPoints.swap(newPoints);
            bestCost = newCost;
        }
        else
        {
            if (lambda < setting.lambdaMax)
            {
                lambda *= setting.lambdaIncFactor;
            }
        }

        ++iter;
    }

    for (int32_t c = 0; c < cameraCount; ++c)
    {
        bestPoses[c].ToRowMajor4x4(updatedPoses + static_cast<size_t>(c) * 16);
    }

    for (int32_t i = 0; i < trackCount; ++i)
    {
        updatedPoints[static_cast<size_t>(i) * 4 + 0] = bestPoints[i].x;
        updatedPoints[static_cast<size_t>(i) * 4 + 1] = bestPoints[i].y;
        updatedPoints[static_cast<size_t>(i) * 4 + 2] = bestPoints[i].z;
        updatedPoints[static_cast<size_t>(i) * 4 + 3] = 1.0f;
    }

    *finalCost      = (iter > 0) ? bestCost : 0.0f;
    *iterationsUsed = iter;
    return 0;
}
