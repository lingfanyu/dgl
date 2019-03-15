/* TODOs
 * - segment_reduce_forward, segment_reduce_backward;
 * - switch backend from aten to dlpack
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Type.h>
#include <c10/util/Exception.h>
#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define AT_CASE_ITYPE(enum_type, type, DTYPE, NAME, ...)                    \
  case enum_type: {                                                         \
    const at::Type& dtype = DTYPE;                                          \
    using idx_t = type;                                                     \
    switch (dtype.scalarType()) {                                           \
      case at::ScalarType::Float: {                                         \
        using data_t = float;                                               \
        return __VA_ARGS__();                                               \
      }                                                                     \
      case at::ScalarType::Double: {                                        \
        using data_t = double;                                              \
        return __VA_ARGS__();                                               \
      }                                                                     \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", dtype.toString(), "'");   \
    }                                                                       \
  }                

#define AT_DISPATCH_IDX_DATA_TYPES(ITYPE, DTYPE, NAME, ...)                             \
  [&] {                                                                                 \
    const at::Type& itype = ITYPE;                                                      \
    switch (itype.scalarType()) {                                                       \
      AT_CASE_ITYPE(at::ScalarType::Int, int32_t, DTYPE, NAME, __VA_ARGS__)             \
      AT_CASE_ITYPE(at::ScalarType::Long, int64_t, DTYPE, NAME, __VA_ARGS__)            \
      default:                                                                          \
        AT_ERROR(#NAME, " not implemented for '", itype.toString(), "'");               \
    }                                                                                   \
  }()

namespace {

/*
 * CUDA Kernel of the forward function for Source Multiply Edge Function.
 * For `src_mul_edge` operation, the arguments are csr(column-major)
 * representations.
 */
template <typename idx_t, typename data_t>
__global__ void vector_spmm_forward_kernel(const idx_t* __restrict__ indptr,
        const idx_t* __restrict__ eid, const idx_t* __restrict__ indices, const
        data_t* __restrict__ edata, const data_t* __restrict__ x, data_t*
        __restrict__ y, const int d, const int n, const int h) {
    int i = blockIdx.x;
    int tx = threadIdx.x;
    if (i < n) {
        for (int j = tx; j < d * h; j += blockDim.x) {
            data_t sum = 0;
            for (int k = indptr[i]; k < indptr[i + 1]; ++k) {
                sum += edata[eid[k] * h + j / d] * x[indices[k] * d * h + j];
            }
            y[i * d * h + j] = sum;
        }
   }
}

/*
 * CUDA Kernel of the backward function for Source Multiply Edge Function.
 */
template <typename idx_t, typename data_t>
__global__ void vector_spmm_backward_kernel_0(const idx_t* __restrict__ indptr,
        const idx_t* __restrict__ eid, const idx_t* __restrict__ indices, const
        data_t* __restrict__ dy, const data_t* __restrict__ xt, data_t*
        __restrict__ dedata, const int d, const int n, const int h) {
    int i = blockIdx.x; 
    int tx = threadIdx.x;
    if (i < n) {
        for (int j = indptr[i] + tx; j < indptr[i + 1]; j += blockDim.x) {
            for (int ko = 0; ko < h; ++ko) {
                data_t sum = 0;
                for (int ki = 0; ki < d; ++ki) {
                    sum += dy[(i * h + ko) * d + ki] * xt[(ko * d + ki) * n +
                        indices[j]];
                }
                dedata[eid[j] * h + ko] = sum;
            }
        }
    }
}

template <typename idx_t, typename data_t>
__global__ void vector_spmm_backward_kernel_1(const idx_t* __restrict__ indptr,
        const idx_t* __restrict__ eid, const idx_t* __restrict__ indices, const
        data_t* __restrict__ edata, const data_t* __restrict__ dy, data_t*
        __restrict__ dx, const int d, const int n, const int h) {
    int i = blockIdx.x; 
    int tx = threadIdx.x;
    if (i < n) {
        for (int j = tx; j < d * h; j += blockDim.x) {
            data_t sum = 0;
            for (int k = indptr[i]; k < indptr[i + 1]; ++k) {
                sum += edata[eid[k] * h + j / d] * dy[indices[k] * d * h + j];
            }
            dx[i * d * h + j] = sum;
        }
    }
}

} // End of namespace


at::Tensor vector_spmm_cuda_forward(
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& indices,
    const at::Tensor& edata,
    const at::Tensor& x) {
    // indptr: (n + 1); eid, indices: (e); edata: (e) or (e, h); x: (n, d) or
    // (n, h, d);
    cudaSetDevice(indptr.get_device());

    const auto n = indptr.size(0) - 1;
    const auto h = (edata.dim() == 2) ? edata.size(1): 1;
    const auto d = x.size(-1); 
    
    const int threads = 128;
    const dim3 blocks(n);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const auto y = at::zeros_like(x, x.options());
    
    AT_DISPATCH_IDX_DATA_TYPES(eid.type(), x.type(), "vector_spmm_forward", ([&] {
        vector_spmm_forward_kernel<idx_t, data_t><<<blocks, threads, 0, stream>>>(
            indptr.data<idx_t>(),
            eid.data<idx_t>(),
            indices.data<idx_t>(),
            edata.data<data_t>(),
            x.data<data_t>(),
            y.data<data_t>(),
            d, n, h);
    }));
    THCudaCheck(cudaGetLastError());
    return y;
}

std::vector<at::Tensor> vector_spmm_cuda_backward(
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& indices,
    const at::Tensor& indptr_t,
    const at::Tensor& eid_t,
    const at::Tensor& indices_t,
    const at::Tensor& edata,
    const at::Tensor& dy,
    const at::Tensor& x) {
    // indptr: (n + 1); eid, indices: (e); edata: (e) or (e, h); dy, x: (n, d) or (n, h, d); 
    cudaSetDevice(indptr.get_device());

    const auto n = indptr.size(0) - 1;
    const auto h = (edata.dim() == 2) ? edata.size(1): 1;
    const auto d = x.size(-1);

    int threads = 32;
    const dim3 blocks(n);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    const auto xt = (h == 1) ? x.transpose(0, 1).contiguous(): x.permute({1, 2, 0}).contiguous();

    const auto dx = at::zeros_like(x, x.options());
    const auto dedata = at::zeros_like(edata, edata.options());

    AT_DISPATCH_IDX_DATA_TYPES(eid.type(), x.type(), "vector_spmm_backward_0", ([&] {
        vector_spmm_backward_kernel_0<idx_t, data_t><<<blocks, threads, 0, stream>>>(
            indptr.data<idx_t>(),
            eid.data<idx_t>(),
            indices.data<idx_t>(),
            dy.data<data_t>(),
            xt.data<data_t>(),
            dedata.data<data_t>(),
            d, n, h);
    }));

    threads = 128;
    AT_DISPATCH_IDX_DATA_TYPES(eid.type(), x.type(), "vector_spmm_backward_1", ([&] {
        vector_spmm_backward_kernel_1<idx_t, data_t><<<blocks, threads, 0, stream>>>(
            indptr_t.data<idx_t>(),
            eid_t.data<idx_t>(),
            indices_t.data<idx_t>(),
            edata.data<data_t>(),
            dy.data<data_t>(),
            dx.data<data_t>(),
            d, n, h);
    }));
    THCudaCheck(cudaGetLastError());
    return {dedata, dx};
}

