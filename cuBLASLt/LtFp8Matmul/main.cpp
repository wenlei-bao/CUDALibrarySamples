/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <vector>

#include <cuda_runtime_api.h>
#include <cuda_fp8.h>
#include <cublasLt.h>

#include "sample_cublasLt_LtFp8Matmul.h"
#include "helpers.h"


void printAlgo(const cublasLtMatmulAlgo_t& algo) {
    int algoId, tile, swizzle, customOption, numSplitsK, reductionScheme;

    checkCublasStatus(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), NULL));
    checkCublasStatus(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), NULL));
    checkCublasStatus(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK, sizeof(numSplitsK), NULL));
    checkCublasStatus(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme), NULL));
    checkCublasStatus(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), NULL));
    checkCublasStatus(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption), NULL));

    // printf("algo={ Id=%d, tileIdx=%d splitK=%d reduc=%d swizzle=%d custom=%d }\n",
    //     algoId, tile, numSplitsK, reductionScheme, swizzle, customOption);    
}

int main() {
    int M = 4096;
    int N = 12288;
    int K = 12288;

    cublasLtMatmulAlgo_t algo;

    TestBench<__nv_fp8_e4m3, __nv_fp8_e4m3, float> props(M, N, K, 1.0f, 0.0f /* ignored */, 32ULL * 1024 * 1024);

    props.run([&props] {
        LtFp8Matmul(props.ltHandle,
                    props.m,
                    props.n,
                    props.k,
                    &props.alpha,
                    props.AscaleDev,
                    props.Adev,
                    props.k,
                    props.BscaleDev,
                    props.Bdev,
                    props.k,
                    props.CscaleDev,
                    props.Cdev,
                    props.m,
                    props.DscaleDev,
                    props.DamaxDev,
                    props.workspace,
                    props.workspaceSize);
    });

    printf("timer: %f ms\n", props.seconds * 1000);
    printf("timer: %f s\n", props.seconds);
    float product = float(M) * float(N) * float(K);
    printf("product: %f \n", product);
    float gflops = 2.0f * product * 10/ float(1.0e9) / props.seconds;
    printf("GFLOPS: %f\n", gflops);
    printf("TFLOPS: %f\n", gflops / 1000.f);

    
    printAlgo(algo);
    return 0;
}