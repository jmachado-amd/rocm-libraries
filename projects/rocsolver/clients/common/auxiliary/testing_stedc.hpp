/* **************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "common/matrix_utils/matrix_utils.hpp"
#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"
#include "common/misc/rocsolver_timer.hpp"

template <typename T, typename S, typename U>
void stedc_checkBadArgs(const rocblas_handle handle,
                        const rocblas_evect evect,
                        const rocblas_int n,
                        S dD,
                        S dE,
                        T dC,
                        const rocblas_int ldc,
                        U dInfo)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_stedc(nullptr, evect, n, dD, dE, dC, ldc, dInfo),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_stedc(handle, rocblas_evect(0), n, dD, dE, dC, ldc, dInfo),
                          rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_stedc(handle, evect, n, (S) nullptr, dE, dC, ldc, dInfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stedc(handle, evect, n, dD, (S) nullptr, dC, ldc, dInfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stedc(handle, evect, n, dD, dE, (T) nullptr, ldc, dInfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stedc(handle, evect, n, dD, dE, dC, ldc, (U) nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_stedc(handle, evect, 0, (S) nullptr, (S) nullptr, (T) nullptr, ldc, dInfo),
        rocblas_status_success);
}

template <typename T>
void testing_stedc_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_evect evect = rocblas_evect_original;
    rocblas_int n = 2;
    rocblas_int ldc = 2;

    // memory allocations
    device_strided_batch_vector<S> dD(1, 1, 1, 1);
    device_strided_batch_vector<S> dE(1, 1, 1, 1);
    device_strided_batch_vector<T> dC(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dD.memcheck());
    CHECK_HIP_ERROR(dE.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    // check bad arguments
    stedc_checkBadArgs(handle, evect, n, dD.data(), dE.data(), dC.data(), ldc, dInfo.data());
}

// For `n > 1`, creates a symmetrized `n` by `n` tridiagonal, Clement matrix T of the following form:
//
//             (   sqrt(n - 1)   sqrt(2(n - 2))     sqrt((n - 2)2)   sqrt(n - 1)   )
// T = tridiag ( 0             0                ...                0             0 )
//             (   sqrt(n - 1)   sqrt(2(n - 2))     sqrt((n - 2)2)   sqrt(n - 1)   )
//
// were the `i-th` off-diagonal entry is sqrt(i(n - i)), 1 <= i < n.
//
template <bool CPU, bool GPU, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void stedc_clement_initData(const rocblas_handle handle,
                            const rocblas_evect evect,
                            const rocblas_int n,
                            Sd& dD,
                            Sd& dE,
                            Td& dC,
                            const rocblas_int ldc,
                            Ud& /* dInfo */,
                            Sh& hD,
                            Sh& hE,
                            Th& hC,
                            Uh& /* hInfo */)
{
    using S = decltype(std::real(T{}));
    rocblas_int bc = 1;
    double eps = (std::getenv("TEST_SCALE") != nullptr) ? atof(std::getenv("TEST_SCALE")) : 1E-7;

    if(CPU)
    {
        rocblas_init<T>(hC, true);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            // New matrix initialization
            using HMatT = HostMatrix<T, rocblas_int>;
            using HMatS = HostMatrix<S, rocblas_int>;
            using BDesc = typename HMatT::BlockDescriptor;

            auto hCw = HMatT::Wrap(hC[b], ldc, n);
            hCw->set_to_zero();
            auto hDw = HMatS::Wrap(hD[b], n, 1);
            hDw->set_to_zero();
            auto hEw = HMatS::Wrap(hE[b], n, 1);
            hEw->set_to_zero();

            if(hCw && hDw && hEw) // update matrices if n >= 1
            {
                auto C = HMatT::Eye(n, n);
                auto D = HMatS::Zeros(n, 1);
                auto E = HMatS::Ones(n - 1, 1);

                for(rocblas_int i = 1; i < n; ++i)
                {
                    E[i - 1] = std::sqrt(i * (n - i));
                }

                hCw->copy_data_from(C);
                hDw->copy_data_from(eps*D);
                hEw->copy_data_from(eps*E);
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dD.transfer_from(hD));
        CHECK_HIP_ERROR(dE.transfer_from(hE));

        if(evect == rocblas_evect_original)
            CHECK_HIP_ERROR(dC.transfer_from(hC));
    }
}

// Creates an `n` by `n` tridiagonal, Toeplitz matrix T of the following form:
//
//             (   1         1         1    )
// T = tridiag ( 2   2 ... 2   2 ... 2    2 )
//             (   1         1         1    )
//
template <bool CPU, bool GPU, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void stedc_toeplitz_initData(const rocblas_handle handle,
                             const rocblas_evect evect,
                             const rocblas_int n,
                             Sd& dD,
                             Sd& dE,
                             Td& dC,
                             const rocblas_int ldc,
                             Ud& /* dInfo */,
                             Sh& hD,
                             Sh& hE,
                             Th& hC,
                             Uh& /* hInfo */)
{
    using S = decltype(std::real(T{}));
    rocblas_int bc = 1;
    double eps = (std::getenv("TEST_SCALE") != nullptr) ? atof(std::getenv("TEST_SCALE")) : 1E-7;

    if(CPU)
    {
        rocblas_init<T>(hC, true);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            // New matrix initialization
            using HMatT = HostMatrix<T, rocblas_int>;
            using HMatS = HostMatrix<S, rocblas_int>;
            using BDesc = typename HMatT::BlockDescriptor;

            auto hCw = HMatT::Wrap(hC[b], ldc, n);
            hCw->set_to_zero();
            auto hDw = HMatS::Wrap(hD[b], n, 1);
            hDw->set_to_zero();
            auto hEw = HMatS::Wrap(hE[b], n, 1);
            hEw->set_to_zero();

            if(hCw && hDw && hEw) // update matrices if n >= 1
            {
                auto C = HMatT::Eye(n, n);
                auto D = 2 * HMatS::Ones(n, 1);
                auto E = HMatS::Ones(n - 1, 1);

                hCw->copy_data_from(C);
                hDw->copy_data_from(eps*D);
                hEw->copy_data_from(eps*E);
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dD.transfer_from(hD));
        CHECK_HIP_ERROR(dE.transfer_from(hE));

        if(evect == rocblas_evect_original)
            CHECK_HIP_ERROR(dC.transfer_from(hC));
    }
}

// Creates an `n` by `n` identity matrix.
//
template <bool CPU, bool GPU, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void stedc_identity_initData(const rocblas_handle handle,
                             const rocblas_evect evect,
                             const rocblas_int n,
                             Sd& dD,
                             Sd& dE,
                             Td& dC,
                             const rocblas_int ldc,
                             Ud& /* dInfo */,
                             Sh& hD,
                             Sh& hE,
                             Th& hC,
                             Uh& /* hInfo */)
{
    using S = decltype(std::real(T{}));
    rocblas_int bc = 1;
    double eps = (std::getenv("TEST_SCALE") != nullptr) ? atof(std::getenv("TEST_SCALE")) : 1E-7;

    if(CPU)
    {
        rocblas_init<T>(hC, true);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            // New matrix initialization
            using HMatT = HostMatrix<T, rocblas_int>;
            using HMatS = HostMatrix<S, rocblas_int>;
            using BDesc = typename HMatT::BlockDescriptor;

            auto hCw = HMatT::Wrap(hC[b], ldc, n);
            hCw->set_to_zero();
            auto hDw = HMatS::Wrap(hD[b], n, 1);
            hDw->set_to_zero();
            auto hEw = HMatS::Wrap(hE[b], n, 1);
            hEw->set_to_zero();

            if(hCw && hDw && hEw) // update matrices if n >= 1
            {
                auto C = HMatT::Eye(n, n);
                auto D = HMatS::Ones(n, 1);
                auto E = HMatS::Zeros(n - 1, 1);

                hCw->copy_data_from(C);
                hDw->copy_data_from(eps*D);
                hEw->copy_data_from(eps*E);
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dD.transfer_from(hD));
        CHECK_HIP_ERROR(dE.transfer_from(hE));

        if(evect == rocblas_evect_original)
            CHECK_HIP_ERROR(dC.transfer_from(hC));
    }
}

// Creates an `n` by `n` tridiagonal, Wilkinson matrix, which is formed as follows:
//
// 1. If `n` is even:
//                      (   1          1            1          1   )
// W_{2m + 1} = tridiag ( m   (m - 1) ... 0.5 0.5 ... (m - 1)    m )
//                      (   1          1            1          1   )
//
// 2. If `n` is odd:
//                      (   1          1         1          1   )
// W_{2m + 1} = tridiag ( m   (m - 1) ... 1 0 1 ... (m - 1)   m )
//                      (   1          1         1          1   )
//
// where `n = 2m + 1`.
//
template <bool CPU, bool GPU, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void stedc_wilkinson_initData(const rocblas_handle handle,
                              const rocblas_evect evect,
                              const rocblas_int n,
                              Sd& dD,
                              Sd& dE,
                              Td& dC,
                              const rocblas_int ldc,
                              Ud& /* dInfo */,
                              Sh& hD,
                              Sh& hE,
                              Th& hC,
                              Uh& /* hInfo */)
{
    using S = decltype(std::real(T{}));
    rocblas_int bc = 1;
    double eps = (std::getenv("TEST_SCALE") != nullptr) ? atof(std::getenv("TEST_SCALE")) : 1E-7;

    if(CPU)
    {
        rocblas_init<T>(hC, true);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            // New matrix initialization
            using HMatT = HostMatrix<T, rocblas_int>;
            using HMatS = HostMatrix<S, rocblas_int>;
            using BDesc = typename HMatT::BlockDescriptor;

            auto hCw = HMatT::Wrap(hC[b], ldc, n);
            hCw->set_to_zero();
            auto hDw = HMatS::Wrap(hD[b], n, 1);
            hDw->set_to_zero();
            auto hEw = HMatS::Wrap(hE[b], n, 1);
            hEw->set_to_zero();

            if(hCw && hDw && hEw) // update matrices if n >= 1
            {
                S m = (n - 1) / S(2);
                auto C = HMatT::Eye(n, n);
                auto D = HMatS::Zeros(n, 1);
                auto E = HMatS::Ones(n - 1, 1);

                for(rocblas_int i = 0; i < n / 2; ++i)
                {
                    D[i] = m - i;
                    D[n - 1 - i] = m - i;
                }

                hCw->copy_data_from(C);
                hDw->copy_data_from(eps*D);
                hEw->copy_data_from(eps*E);
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dD.transfer_from(hD));
        CHECK_HIP_ERROR(dE.transfer_from(hE));

        if(evect == rocblas_evect_original)
            CHECK_HIP_ERROR(dC.transfer_from(hC));
    }
}

template <bool CPU, bool GPU, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void stedc_random_initData(const rocblas_handle handle,
                           const rocblas_evect evect,
                           const rocblas_int n,
                           Sd& dD,
                           Sd& dE,
                           Td& dC,
                           const rocblas_int ldc,
                           Ud& dInfo,
                           Sh& hD,
                           Sh& hE,
                           Th& hC,
                           Uh& hInfo)
{
    if(CPU)
    {
        using S = decltype(std::real(T{}));

        rocblas_init<S>(hD, true);
        rocblas_init<S>(hE, true);

        for(int i = 0; i < n - 1; ++i)
            hE[0][i] -= 4;

        // initialize C to the identity matrix
        if(evect == rocblas_evect_original)
        {
            for(rocblas_int j = 0; j < n; j++)
            {
                for(rocblas_int i = 0; i < n; i++)
                {
                    if(i == j)
                        hC[0][i + j * ldc] = 1;
                    else
                        hC[0][i + j * ldc] = 0;
                }
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dD.transfer_from(hD));
        CHECK_HIP_ERROR(dE.transfer_from(hE));

        if(evect == rocblas_evect_original)
            CHECK_HIP_ERROR(dC.transfer_from(hC));
    }
}

template <bool CPU, bool GPU, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void stedc_default_initData(const rocblas_handle handle,
                            const rocblas_evect evect,
                            const rocblas_int n,
                            Sd& dD,
                            Sd& dE,
                            Td& dC,
                            const rocblas_int ldc,
                            Ud& /* dInfo */,
                            Sh& hD,
                            Sh& hE,
                            Th& hC,
                            Uh& /* hInfo */)
{
    if(CPU)
    {
        using S = decltype(std::real(T{}));

        // if the matrix is too small (n < 4), simply initialize D and E
        if(n < 4)
        {
            rocblas_init<S>(hD, true);
            rocblas_init<S>(hE, true);
        }

        // otherwise, the marix will be divided in exactly 2 independent blocks, if the size is even,
        // or 3 if the size is odd. The 2 main independent blocks will have the same eigenvalues.
        // The last block, when the size is odd, will have eigenvalue equal 1.
        else
        {
            rocblas_int N1 = n / 2;
            rocblas_int E = n - 2 * N1;

            // a. initialize the eigenvalues for the uppermost sub-blocks of the main independent blocks.
            // The second sub-block will have some repeated eigenvalues in order to test the deflation process
            S d;
            rocblas_int NN1 = N1 / 2;
            rocblas_int NN2 = N1 - NN1;
            rocblas_int s1 = NN1 * NN1;
            rocblas_int s2 = NN2 * NN2;
            rocblas_int sw = NN2 * 32;
            std::vector<S> A1(s1);
            std::vector<S> A2(s2);
            for(rocblas_int i = 0; i < NN1; ++i)
            {
                for(rocblas_int j = 0; j < NN1; ++j)
                {
                    if(i == j)
                    {
                        d = (i + 1) / S(NN1);
                        A1[i + i * NN1] = d;
                        A2[i + i * NN2] = (i % 2 == 0) ? d : -d;
                    }
                    else
                    {
                        A1[i + j * NN1] = 0;
                        A2[i + j * NN2] = 0;
                    }
                }
            }
            if(NN2 > NN1)
            {
                for(rocblas_int i = 0; i < NN1; ++i)
                {
                    A2[NN1 + i * NN2] = 0;
                    A2[i + NN1 * NN2] = 0;
                }
                A2[NN1 + NN1 * NN2] = 0;
            }

            // b. find the corresponding tridiagonal matrices containing the setup eigenvalues of each sub-block
            // first find random orthogonal matrices Q1 and Q2
            Sh Q1(s1, 1, s1, 1);
            Sh Q2(s2, 1, s2, 1);
            rocblas_init<S>(Q1, true);
            rocblas_init<S>(Q2, true);
            std::vector<S> hW(sw);
            std::vector<S> ipiv1(NN1);
            std::vector<S> ipiv2(NN2);
            cpu_geqrf<S>(NN1, NN1, Q1.data(), NN1, ipiv1.data(), hW.data(), sw);
            cpu_geqrf<S>(NN2, NN2, Q2.data(), NN2, ipiv2.data(), hW.data(), sw);
            // now multiply the orthogonal matrices by the diagonals A1 and A2 to hide the eigenvalues
            cpu_ormqr_unmqr<S>(rocblas_side_left, rocblas_operation_transpose, NN1, NN1, NN1,
                               Q1.data(), NN1, ipiv1.data(), A1.data(), NN1, hW.data(), sw);
            cpu_ormqr_unmqr<S>(rocblas_side_right, rocblas_operation_none, NN1, NN1, NN1, Q1.data(),
                               NN1, ipiv1.data(), A1.data(), NN1, hW.data(), sw);
            cpu_ormqr_unmqr<S>(rocblas_side_left, rocblas_operation_transpose, NN2, NN2, NN2,
                               Q2.data(), NN2, ipiv2.data(), A2.data(), NN2, hW.data(), sw);
            cpu_ormqr_unmqr<S>(rocblas_side_right, rocblas_operation_none, NN2, NN2, NN2, Q2.data(),
                               NN2, ipiv2.data(), A2.data(), NN2, hW.data(), sw);
            // finally, perform tridiagonalization
            cpu_sytrd_hetrd<S>(rocblas_fill_upper, NN1, A1.data(), NN1, hD[0], hE[0], ipiv1.data(),
                               hW.data(), sw);
            cpu_sytrd_hetrd<S>(rocblas_fill_upper, NN2, A2.data(), NN2, hD[0] + NN1, hE[0] + NN1,
                               ipiv2.data(), hW.data(), sw);

            // c. integrate blocks into final matrix
            // integrate the 2 sub-blocks into the first independent block
            hE[0][NN1 - 1] = 1;
            hD[0][NN1 - 1] += 1;
            hD[0][NN1] += 1;
            // copy the independent block over
            for(rocblas_int i = 0; i < N1; ++i)
            {
                hD[0][N1 + i] = hD[0][i];
                hE[0][N1 + i] = hE[0][i];
            }
            hE[0][N1 - 1] = 0;
            hE[0][2 * N1 - 1] = 0;
            // integrate the 2 sub-blocks into the second independent block
            // (using negative p to test secular eqn algorithm)
            hE[0][N1 + NN1 - 1] = -1;
            hD[0][N1 + NN1 - 1] -= 2;
            hD[0][N1 + NN1] -= 2;
            // if there is a third independent block, initialize it with 1
            if(E == 1)
                hD[0][n - 1] = 1;
        }

        // initialize C to the identity matrix
        if(evect == rocblas_evect_original)
        {
            for(rocblas_int j = 0; j < n; j++)
            {
                for(rocblas_int i = 0; i < n; i++)
                {
                    if(i == j)
                        hC[0][i + j * ldc] = 1;
                    else
                        hC[0][i + j * ldc] = 0;
                }
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dD.transfer_from(hD));
        CHECK_HIP_ERROR(dE.transfer_from(hE));

        if(evect == rocblas_evect_original)
            CHECK_HIP_ERROR(dC.transfer_from(hC));
    }
}

// Creates the tridiagonal reduction of an `n` by `n` matrix with spectrum
//
//   { l_i = ulp * i, for 1 <= i <= n - 1; l_n = 1 }
//
// i.e. the EIG7 spectrum, but reduced to tridiagonal form with cpu_sytrd so that the
// exact same D, E can be handed to both rocSOLVER and LAPACK. ||A|| = 1, so dstedc's
// ORGNRM normalization is a no-op. `ulp` defaults to the working precision's epsilon
// but can be pinned with TEST_ULP so that a float and a double run share one matrix.
template <bool CPU, bool GPU, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void stedc_eig7_initData(const rocblas_handle handle,
                         const rocblas_evect evect,
                         const rocblas_int n,
                         Sd& dD,
                         Sd& dE,
                         Td& dC,
                         const rocblas_int ldc,
                         Ud& dInfo,
                         Sh& hD,
                         Sh& hE,
                         Th& hC,
                         Uh& hInfo)
{
    using S = decltype(std::real(T{}));
    rocblas_int bc = 1;

    if(CPU)
    {
        rocblas_init<T>(hC, true);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            using HMatT = HostMatrix<T, rocblas_int>;
            using HMatS = HostMatrix<S, rocblas_int>;
            using BDescT = typename HMatT::BlockDescriptor;

            auto hCw = HMatT::Wrap(hC[b], ldc, n);
            hCw->set_to_zero();
            auto hDw = HMatS::Wrap(hD[b], n, 1);
            hDw->set_to_zero();
            auto hEw = HMatS::Wrap(hE[b], n, 1);
            hEw->set_to_zero();

            if(hCw && hDw && hEw && n > 1)
            {
                S ulp = (std::getenv("TEST_ULP") != nullptr)
                    ? (S)atof(std::getenv("TEST_ULP"))
                    : std::numeric_limits<S>::epsilon();

                // dense A = Q * diag(eigs) * Q^H with ||A|| = 1
                std::vector<T> A((size_t)n * n);
                auto Aw = HMatT::Wrap(A.data(), n, n);
                rocblas_init<T>(A.data(), n, n, n);
                auto seed = HMatT::Wrap(A.data(), n, n)->block(BDescT().nrows(n).ncols(n));
                auto [Q, _] = qr(seed);
                auto eigs = ulp * HMatT::FromRange(1, n - 1, n - 1);
                eigs = cat(eigs, HMatT::Ones(1, 1));
                Aw->copy_data_from(Q * HMatT::Zeros(n).diag(eigs) * adjoint(Q));

                // reduce to tridiagonal form; D and E then define the test problem
                std::vector<T> tau(n), work((size_t)n * n);
                cpu_sytrd_hetrd(rocblas_fill_lower, n, A.data(), n, hD[b], hE[b], tau.data(),
                                work.data(), (rocblas_int)work.size());

                hCw->copy_data_from(HMatT::Eye(n, n));
            }
        }
    }

    if(GPU)
    {
        CHECK_HIP_ERROR(dD.transfer_from(hD));
        CHECK_HIP_ERROR(dE.transfer_from(hE));

        if(evect == rocblas_evect_original)
            CHECK_HIP_ERROR(dC.transfer_from(hC));
    }
}

// Block diagonal tridiagonal with a small but non-negligible coupling off-diagonal.
//
//   [ s * Toeplitz(2,1)   c              ]        s = MIXED_SCALE   (top block scale)
//   [ c                   EIG7 reduction ]        c = MIXED_COUPLING
//
// The bottom block is O(1) and fixes ||T|| ~ 1, so a ScaLAPACK-style global ORGNRM
// rescale is a no-op. The top block is still tiny in absolute terms, so every merge
// internal to it sees maxd ~ s while maxz stays O(1) -- the same mismatch as the
// uniformly scaled case, untouched by a global rescale. The coupling is kept above
// LAPACK's splitting threshold eps*sqrt(|d_i d_i+1|) so that sstedc treats this as one
// irreducible block and performs a single global rescale, matching ScaLAPACK.
template <bool CPU, bool GPU, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void stedc_mixed_initData(const rocblas_handle handle,
                          const rocblas_evect evect,
                          const rocblas_int n,
                          Sd& dD,
                          Sd& dE,
                          Td& dC,
                          const rocblas_int ldc,
                          Ud& dInfo,
                          Sh& hD,
                          Sh& hE,
                          Th& hC,
                          Uh& hInfo)
{
    using S = decltype(std::real(T{}));
    rocblas_int bc = 1;

    if(CPU)
    {
        rocblas_init<T>(hC, true);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            using HMatT = HostMatrix<T, rocblas_int>;
            using HMatS = HostMatrix<S, rocblas_int>;
            using BDescT = typename HMatT::BlockDescriptor;

            auto hCw = HMatT::Wrap(hC[b], ldc, n);
            hCw->set_to_zero();
            auto hDw = HMatS::Wrap(hD[b], n, 1);
            hDw->set_to_zero();
            auto hEw = HMatS::Wrap(hE[b], n, 1);
            hEw->set_to_zero();

            if(hCw && hDw && hEw && n > 3)
            {
                // MIXED_BOT=one reduces the bottom block to a single entry equal to 1.
                // That alone pins ORGNRM = 1 (so a global rescale is a no-op) while the
                // whole rest of the matrix stays at scale s, with no structure of its own
                // to confound the measurement.
                const char* botenv = std::getenv("MIXED_BOT");
                const bool bot_one = (botenv != nullptr) && (std::string(botenv) == "one");

                rocblas_int n1 = (std::getenv("MIXED_N1") != nullptr)
                    ? (rocblas_int)atoi(std::getenv("MIXED_N1"))
                    : (bot_one ? n - 1 : n / 2);
                n1 = std::max((rocblas_int)2, std::min(n - 1, n1));
                rocblas_int n2 = n - n1;

                S s = (std::getenv("MIXED_SCALE") != nullptr)
                    ? (S)atof(std::getenv("MIXED_SCALE"))
                    : (S)5e-7;
                S c = (std::getenv("MIXED_COUPLING") != nullptr)
                    ? (S)atof(std::getenv("MIXED_COUPLING"))
                    : (S)1e-9;

                // ---- top block: scaled Toeplitz, tridiag(1, 2, 1) * s
                for(rocblas_int i = 0; i < n1; ++i)
                    hD[b][i] = s * S(2);
                for(rocblas_int i = 0; i + 1 < n1; ++i)
                    hE[b][i] = s;

                // ---- coupling between the two blocks
                hE[b][n1 - 1] = c;

                // ---- bottom block
                S ulp = (std::getenv("TEST_ULP") != nullptr)
                    ? (S)atof(std::getenv("TEST_ULP"))
                    : std::numeric_limits<S>::epsilon();

                if(n2 == 1)
                {
                    // single entry; its value alone fixes ||T||
                    S bv = (std::getenv("MIXED_BOTVAL") != nullptr)
                        ? (S)atof(std::getenv("MIXED_BOTVAL"))
                        : S(1);
                    hD[b][n - 1] = bv;
                }
                else
                {
                std::vector<T> A2((size_t)n2 * n2);
                auto A2w = HMatT::Wrap(A2.data(), n2, n2);
                rocblas_init<T>(A2.data(), n2, n2, n2);
                auto seed = HMatT::Wrap(A2.data(), n2, n2)->block(BDescT().nrows(n2).ncols(n2));
                auto [Q, _] = qr(seed);
                auto eigs = ulp * HMatT::FromRange(1, n2 - 1, n2 - 1);
                eigs = cat(eigs, HMatT::Ones(1, 1));
                A2w->copy_data_from(Q * HMatT::Zeros(n2).diag(eigs) * adjoint(Q));

                std::vector<S> d2(n2), e2(n2);
                std::vector<T> tau(n2), work((size_t)n2 * n2);
                cpu_sytrd_hetrd(rocblas_fill_lower, n2, A2.data(), n2, d2.data(), e2.data(),
                                tau.data(), work.data(), (rocblas_int)work.size());

                for(rocblas_int i = 0; i < n2; ++i)
                    hD[b][n1 + i] = d2[i];
                for(rocblas_int i = 0; i + 1 < n2; ++i)
                    hE[b][n1 + i] = e2[i];
                }

                hCw->copy_data_from(HMatT::Eye(n, n));

                if(std::getenv("MIXED_VERBOSE") != nullptr)
                {
                    S maxall = 0, maxtop = 0;
                    for(rocblas_int i = 0; i < n; ++i)
                        maxall = std::max(maxall, std::abs(hD[b][i]));
                    for(rocblas_int i = 0; i + 1 < n; ++i)
                        maxall = std::max(maxall, std::abs(hE[b][i]));
                    for(rocblas_int i = 0; i < n1; ++i)
                        maxtop = std::max(maxtop, std::abs(hD[b][i]));
                    printf("[mixed] n=%d n1=%d n2=%d s=%.3e c=%.3e  ORGNRM=%.3e  top_scale=%.3e  "
                           "split_thresh~%.3e\n",
                           (int)n, (int)n1, (int)n2, (double)s, (double)c, (double)maxall,
                           (double)maxtop,
                           (double)(std::numeric_limits<S>::epsilon()
                                    * std::sqrt(std::abs(hD[b][n1 - 1]) * std::abs(hD[b][n1]))));
                }
            }
        }
    }

    if(GPU)
    {
        CHECK_HIP_ERROR(dD.transfer_from(hD));
        CHECK_HIP_ERROR(dE.transfer_from(hE));

        if(evect == rocblas_evect_original)
            CHECK_HIP_ERROR(dC.transfer_from(hC));
    }
}

template <bool CPU, bool GPU, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void stedc_initData(const rocblas_handle handle,
                    const rocblas_evect evect,
                    const rocblas_int n,
                    Sd& dD,
                    Sd& dE,
                    Td& dC,
                    const rocblas_int ldc,
                    Ud& dInfo,
                    Sh& hD,
                    Sh& hE,
                    Th& hC,
                    Uh& hInfo)
{
    if((std::getenv("TEST_MIXED") != nullptr) || (std::getenv("STEDC_TEST_MIXED") != nullptr))
    {
        stedc_mixed_initData<CPU, GPU, T>(handle, evect, n, dD, dE, dC, ldc, dInfo, hD, hE, hC,
                                          hInfo);
    }
    else if((std::getenv("TEST_EIG7") != nullptr) || (std::getenv("STEDC_TEST_EIG7") != nullptr))
    {
        stedc_eig7_initData<CPU, GPU, T>(handle, evect, n, dD, dE, dC, ldc, dInfo, hD, hE, hC,
                                         hInfo);
    }
    else if((std::getenv("TEST_WILKINSON") != nullptr)
            || (std::getenv("STEDC_TEST_WILKINSON") != nullptr))
    {
        stedc_wilkinson_initData<CPU, GPU, T>(handle, evect, n, dD, dE, dC, ldc, dInfo, hD, hE, hC,
                                              hInfo);
    }
    else if((std::getenv("TEST_CLEMENT") != nullptr)
            || (std::getenv("STEDC_TEST_CLEMENT") != nullptr))
    {
        stedc_clement_initData<CPU, GPU, T>(handle, evect, n, dD, dE, dC, ldc, dInfo, hD, hE, hC,
                                            hInfo);
    }
    else if((std::getenv("TEST_TOEPLITZ") != nullptr)
            || (std::getenv("STEDC_TEST_TOEPLITZ") != nullptr))
    {
        stedc_toeplitz_initData<CPU, GPU, T>(handle, evect, n, dD, dE, dC, ldc, dInfo, hD, hE, hC,
                                             hInfo);
    }
    else if((std::getenv("TEST_IDENTITY") != nullptr)
            || (std::getenv("STEDC_TEST_IDENTITY") != nullptr))
    {
        stedc_identity_initData<CPU, GPU, T>(handle, evect, n, dD, dE, dC, ldc, dInfo, hD, hE, hC,
                                             hInfo);
    }
    else if((std::getenv("TEST_RANDOM") != nullptr) || (std::getenv("STEDC_TEST_RANDOM") != nullptr))
    {
        stedc_random_initData<CPU, GPU, T>(handle, evect, n, dD, dE, dC, ldc, dInfo, hD, hE, hC,
                                           hInfo);
    }
    else
    {
        stedc_default_initData<CPU, GPU, T>(handle, evect, n, dD, dE, dC, ldc, dInfo, hD, hE, hC,
                                            hInfo);
    }

    return;
}

template <typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void stedc_getError(const rocblas_handle handle,
                    const rocblas_evect evect,
                    const rocblas_int n,
                    Sd& dD,
                    Sd& dE,
                    Td& dC,
                    const rocblas_int ldc,
                    Ud& dInfo,
                    Sh& hD,
                    Sh& hDRes,
                    Sh& hE,
                    Sh& hERes,
                    Th& hC,
                    Th& hCRes,
                    Uh& hInfo,
                    Uh& hInfoRes,
                    double* max_err,
                    double* max_errv)
{
    constexpr bool COMPLEX = rocblas_is_complex<T>;
    using S = decltype(std::real(T{}));

    using HMatT = HostMatrix<T, rocblas_int>;
    using HMatS = HostMatrix<S, rocblas_int>;
    using BDescT = typename HMatT::BlockDescriptor;
    using BDescS = typename HMatS::BlockDescriptor;

    int lgn = floor(log(n - 1) / log(2)) + 1;
    size_t lwork = (COMPLEX) ? n * n : 0;
    size_t lrwork = (evect == rocblas_evect_none || n <= 1) ? 1 : 1 + 3 * n + 4 * n * n + 2 * n * lgn;
    size_t liwork = (evect == rocblas_evect_none || n <= 1) ? 1 : 6 + 6 * n + 5 * n * lgn;
    std::vector<T> work(lwork);
    std::vector<S> rwork(lrwork);
    std::vector<rocblas_int> iwork(liwork);

    // input data initialization
    stedc_initData<true, true, T>(handle, evect, n, dD, dE, dC, ldc, dInfo, hD, hE, hC, hInfo);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(
        rocsolver_stedc(handle, evect, n, dD.data(), dE.data(), dC.data(), ldc, dInfo.data()));
    CHECK_HIP_ERROR(hDRes.transfer_from(dD));
    CHECK_HIP_ERROR(hERes.transfer_from(dE));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));
    if(evect != rocblas_evect_none)
        CHECK_HIP_ERROR(hCRes.transfer_from(dC));

    // if eigenvectors were required, prepare matrix A (upper triangular) for implicit tests
    rocblas_int lda = n;
    size_t size_A = lda * n;
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    if(evect != rocblas_evect_none)
    {
        for(rocblas_int i = 0; i < n; i++)
        {
            for(rocblas_int j = i; j < n; j++)
            {
                if(i == j)
                    hA[0][i + j * lda] = hD[0][i];
                else if(i + 1 == j)
                    hA[0][i + j * lda] = hE[0][i];
                else
                    hA[0][i + j * lda] = 0;
            }
        }
    }

    // CPU lapack
    // Temporary: keep the input tridiagonal so a double-precision reference can be run
    // on the exact same D, E after cpu_stedc overwrites them.
    std::vector<S> refD, refE;
    const bool want_evec_ref
        = (std::getenv("TEST_EVEC_REF") != nullptr) && (evect != rocblas_evect_none) && (n > 1);
    if(want_evec_ref)
    {
        refD.assign(hD[0], hD[0] + n);
        refE.assign(hE[0], hE[0] + n);
    }

    cpu_stedc(evect, n, hD[0], hE[0], hC[0], ldc, work.data(), lwork, rwork.data(), lrwork,
              iwork.data(), liwork, hInfo[0]);

    // Temporary: three-way eigenvector comparison. Solve the same tridiagonal in double,
    // whose deflation tolerance is ~1e9 times finer than the eigenvalue spacing and so
    // sits far outside the regime under investigation, and use it as the reference for
    // both the rocSOLVER result and LAPACK's own single-precision result. If both float
    // results deviate equally the eigenvectors are simply ill-conditioned; if rocSOLVER
    // deviates further, the deflation criterion is implicated.
    if(want_evec_ref)
    {
        if constexpr(!rocblas_is_complex<T>)
        {
            std::vector<double> dD(refD.begin(), refD.end()), dE(refE.begin(), refE.end());
            std::vector<double> dC((size_t)n * n, 0.0);
            for(rocblas_int i = 0; i < n; ++i)
                dC[(size_t)i + (size_t)i * n] = 1.0;

            int lgn2 = (int)std::floor(std::log((double)(n - 1)) / std::log(2.0)) + 1;
            std::vector<double> dwork(1), drwork(1 + 4 * n + 4 * (size_t)n * n + 2 * (size_t)n * lgn2);
            std::vector<rocblas_int> diwork(6 + 6 * n + 5 * (size_t)n * lgn2);
            rocblas_int dinfo = 0;
            cpu_stedc(rocblas_evect_tridiagonal, n, dD.data(), dE.data(), dC.data(), n,
                      dwork.data(), (rocblas_int)dwork.size(), drwork.data(),
                      (rocblas_int)drwork.size(), diwork.data(), (rocblas_int)diwork.size(), &dinfo);

            auto angle = [&](const T* M, rocblas_int ldm, rocblas_int col) {
                double dot = 0, nm = 0;
                for(rocblas_int r = 0; r < n; ++r)
                {
                    double v = (double)std::real(M[(size_t)r + (size_t)col * ldm]);
                    dot += v * dC[(size_t)r + (size_t)col * n];
                    nm += v * v;
                }
                nm = std::sqrt(nm);
                double c = (nm > 0) ? std::abs(dot) / nm : 0.0;
                c = std::min(1.0, std::max(0.0, c));
                return std::sqrt(std::max(0.0, 1.0 - c * c)); // sin(angle)
            };

            // QR control on the identical tridiagonal, in the same single precision.
            // steqr performs no deflation, so if it resolves pairs that stedc loses the
            // information is present in float and the loss is attributable to D&C.
            std::vector<S> qD(refD.begin(), refD.end()), qE(refE.begin(), refE.end());
            std::vector<T> qC((size_t)ldc * n, T(0));
            for(rocblas_int i = 0; i < n; ++i)
                qC[(size_t)i + (size_t)i * ldc] = T(1);
            std::vector<S> qwork((size_t)std::max(1, 2 * n - 2));
            rocblas_int qinfo = 0;
            cpu_steqr(rocblas_evect_tridiagonal, n, qD.data(), qE.data(), qC.data(), ldc,
                      qwork.data(), &qinfo);

            auto angleq = [&](const T* M, rocblas_int ldm, rocblas_int col) {
                double dot = 0, nm = 0;
                for(rocblas_int r = 0; r < n; ++r)
                {
                    double v = (double)std::real(M[(size_t)r + (size_t)col * ldm]);
                    dot += v * dC[(size_t)r + (size_t)col * n];
                    nm += v * v;
                }
                nm = std::sqrt(nm);
                double c = (nm > 0) ? std::abs(dot) / nm : 0.0;
                c = std::min(1.0, std::max(0.0, c));
                return std::sqrt(std::max(0.0, 1.0 - c * c));
            };

            printf("[ref] dinfo=%d qinfo=%d   col  lambda_ref     roc_eval_rel  lap_eval_rel  "
                   "qr_eval_rel   roc_evec_sin  lap_evec_sin  qr_evec_sin\n",
                   dinfo, (int)qinfo);
            for(rocblas_int i = 0; i < n; ++i)
            {
                double le = dD[i];
                double den = std::max(std::abs(le), 1e-300);
                printf("[ref] %5d  %.6e  %.4e  %.4e  %.4e  %.4e  %.4e  %.4e\n", (int)i, le,
                       std::abs((double)hDRes[0][i] - le) / den,
                       std::abs((double)hD[0][i] - le) / den,
                       std::abs((double)qD[i] - le) / den, angle(hCRes[0], ldc, i),
                       angle(hC[0], ldc, i), angleq(qC.data(), ldc, i));
            }
        }
    }

    // Depending on `evect`, stedc can return the eigenvectors of the original
    // matrix (A) or the eigenvectors of the tridiagonal matrix (T).  Thus,
    // given a pair U, D of eigenvectors and eigenvalues computed by stedc the
    // reconstructed matrix
    //
    // U * D * adjoint(U)
    //
    // can be either A or T.  The following code uses lapack to compute
    //
    // AorT = U * D * adjoint(U),
    //
    // which will be later used to compare with the eigenvectors and
    // eigenvalues computed by rocSOLVER.
    //
    auto AorT = HMatT::Empty();
    if((evect != rocblas_evect_none) && (n > 0))
    {
        auto C = HMatT::Wrap(hC[0], ldc, n)->block(BDescT().nrows(n).ncols(n));
        auto d = HMatT::Convert(hD[0], 1, n)->block(BDescT().nrows(1).ncols(n));
        auto D = HMatT::Zeros(n, n).diag(d);
        AorT = C * D * adjoint(C);
    }

    // check info
    EXPECT_EQ(hInfo[0][0], hInfoRes[0][0]);
    if(hInfo[0][0] != hInfoRes[0][0])
        *max_err = 1;
    else
        *max_err = 0;

    double err;
    *max_errv = 0;

    if((hInfo[0][0] == 0) && (n > 0))
    {
        // check that eigenvalues are correct and in order
        // error is ||hD - hDRes|| / ||hD||
        // using frobenius norm
        err = norm_error('F', 1, n, 1, hD[0], hDRes[0]);
        *max_err = err > *max_err ? err : *max_err;

        // check eigenvectors if required
        if(evect != rocblas_evect_none)
        {
            // Input matrix
            auto C = HMatT::Wrap(hCRes[0], ldc, n)->block(BDescT().nrows(n).ncols(n));
            // Computed eigenvalues
            auto d = HMatT::Convert(hDRes[0], 1, n)->block(BDescT().nrows(1).ncols(n));
            // Diagonal matrix of size n by n with computed eigenvalues
            auto D = HMatT::Zeros(n, n).diag(d);

            // Orthogonal error
            auto OE = C * adjoint(C) - HMatT::Eye(n, n);
            err = OE.max_col_norm();
            *max_errv = err > *max_err ? err : *max_err;

            // Residual error
            auto RE = AorT - C * D * adjoint(C);
            err = RE.norm() / AorT.norm();
            *max_err = err > *max_err ? err : *max_err;
        }
    }
}

template <typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void stedc_getPerfData(const rocblas_handle handle,
                       const rocblas_evect evect,
                       const rocblas_int n,
                       Sd& dD,
                       Sd& dE,
                       Td& dC,
                       const rocblas_int ldc,
                       Ud& dInfo,
                       Sh& hD,
                       Sh& hE,
                       Th& hC,
                       Uh& hInfo,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool profile_kernels,
                       const bool perf)
{
    constexpr bool COMPLEX = rocblas_is_complex<T>;
    using S = decltype(std::real(T{}));

    int lgn = floor(log(n - 1) / log(2)) + 1;
    size_t lwork = (COMPLEX) ? n * n : 0;
    size_t lrwork = (evect == rocblas_evect_none || n <= 1) ? 1 : 1 + 3 * n + 4 * n * n + 2 * n * lgn;
    size_t liwork = (evect == rocblas_evect_none || n <= 1) ? 1 : 6 + 6 * n + 5 * n * lgn;
    std::vector<T> work(lwork);
    std::vector<S> rwork(lrwork);
    std::vector<rocblas_int> iwork(liwork);

    if(!perf)
    {
        stedc_initData<true, false, T>(handle, evect, n, dD, dE, dC, ldc, dInfo, hD, hE, hC, hInfo);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_stedc(evect, n, hD[0], hE[0], hC[0], ldc, work.data(), lwork, rwork.data(), lrwork,
                  iwork.data(), liwork, hInfo[0]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    stedc_initData<true, false, T>(handle, evect, n, dD, dE, dC, ldc, dInfo, hD, hE, hC, hInfo);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        stedc_initData<false, true, T>(handle, evect, n, dD, dE, dC, ldc, dInfo, hD, hE, hC, hInfo);

        CHECK_ROCBLAS_ERROR(
            rocsolver_stedc(handle, evect, n, dD.data(), dE.data(), dC.data(), ldc, dInfo.data()));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    rocsolver_timer timer;

    if(profile > 0)
    {
        if(profile_kernels)
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile
                                         | rocblas_layer_mode_ex_log_kernel);
        else
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile);
        rocsolver_log_set_max_levels(profile);
    }

    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        stedc_initData<false, true, T>(handle, evect, n, dD, dE, dC, ldc, dInfo, hD, hE, hC, hInfo);

        timer.start(stream);
        rocsolver_stedc(handle, evect, n, dD.data(), dE.data(), dC.data(), ldc, dInfo.data());
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <typename T>
void testing_stedc(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    char evectC = argus.get<char>("evect");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int ldc = argus.get<rocblas_int>("ldc", n);

    rocblas_evect evect = char2rocblas_evect(evectC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_D = n;
    size_t size_E = n;
    size_t size_C = ldc * n;
    double max_err = 0, max_errv = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_DRes = (argus.unit_check || argus.norm_check) ? size_D : 0;
    size_t size_ERes = (argus.unit_check || argus.norm_check) ? size_E : 0;
    size_t size_CRes = (argus.unit_check || argus.norm_check) ? size_C : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || (evect != rocblas_evect_none && ldc < n));
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_stedc(handle, evect, n, (S*)nullptr, (S*)nullptr,
                                              (T*)nullptr, ldc, (rocblas_int*)nullptr),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_stedc(handle, evect, n, (S*)nullptr, (S*)nullptr, (T*)nullptr,
                                          ldc, (rocblas_int*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<S> hD(size_D, 1, size_D, 1);
    host_strided_batch_vector<S> hDRes(size_DRes, 1, size_DRes, 1);
    host_strided_batch_vector<S> hE(size_E, 1, size_E, 1);
    host_strided_batch_vector<S> hERes(size_ERes, 1, size_ERes, 1);
    host_strided_batch_vector<T> hC(size_C, 1, size_C, 1);
    host_strided_batch_vector<T> hCRes(size_CRes, 1, size_CRes, 1);
    host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, 1);
    host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, 1);
    device_strided_batch_vector<S> dD(size_D, 1, size_D, 1);
    device_strided_batch_vector<S> dE(size_E, 1, size_E, 1);
    device_strided_batch_vector<T> dC(size_C, 1, size_C, 1);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
    if(size_D)
        CHECK_HIP_ERROR(dD.memcheck());
    if(size_E)
        CHECK_HIP_ERROR(dE.memcheck());
    if(size_C)
        CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_stedc(handle, evect, n, dD.data(), dE.data(), dC.data(), ldc, dInfo.data()),
            rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        stedc_getError<T>(handle, evect, n, dD, dE, dC, ldc, dInfo, hD, hDRes, hE, hERes, hC, hCRes,
                          hInfo, hInfoRes, &max_err, &max_errv);

    // collect performance data
    if(argus.timing && hot_calls > 0)
        stedc_getPerfData<T>(handle, evect, n, dD, dE, dC, ldc, dInfo, hD, hE, hC, hInfo,
                             &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                             argus.profile_kernels, argus.perf);

    // validate results for rocsolver-test
    // using n * machine_precision as tolerance
    if(argus.unit_check)
    {
        ROCSOLVER_TEST_CHECK(T, max_err, n);
        if(evect != rocblas_evect_none)
            ROCSOLVER_TEST_CHECK(T, max_errv, n);
    }

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("evect", "n", "ldc");
            rocsolver_bench_output(evectC, n, ldc);

            rocsolver_bench_header("Results:");
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, std::max(max_err, max_errv));
            }
            else
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us");
                rocsolver_bench_output(cpu_time_used, gpu_time_used);
            }
            rocsolver_bench_endl();
        }
        else
        {
            if(argus.norm_check)
                rocsolver_bench_output(gpu_time_used, std::max(max_err, max_errv));
            else
                rocsolver_bench_output(gpu_time_used);
        }
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}

#define EXTERN_TESTING_STEDC(...) extern template void testing_stedc<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_STEDC, FOREACH_SCALAR_TYPE, APPLY_STAMP)
