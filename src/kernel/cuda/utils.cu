/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cuda/utils.cu
 * \brief Utility function implementations on CUDA
 */
#include "../../runtime/cuda/cuda_common.h"
#include "../utils.h"

namespace dgl {
namespace kernel {
namespace utils {

template <typename DType>
__global__ void _FillKernel(DType* ptr, size_t length, DType val) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    ptr[tx] = val;
    tx += stride_x;
  }
}

template <int XPU, typename DType>
void Fill(const DLContext& ctx, DType* ptr, size_t length, DType val) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int nt = utils::FindNumThreads(length, 1024);
  int nb = (length + nt - 1) / nt;
  _FillKernel<<<nb, nt, 0, thr_entry->stream>>>(ptr, length, val);
}

template void Fill<kDLGPU, float>(const DLContext& ctx, float* ptr, size_t length, float val);
template void Fill<kDLGPU, double>(const DLContext& ctx, double* ptr, size_t length, double val);

template <typename Idx, typename DType>
__global__ void _Take1DKernel(DType* ptr, const DType* src, const Idx* indices, size_t length) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    ptr[tx] = src[indices[tx]];
    tx += stride_x;
  }
}

template <int XPU, typename Idx, typename DType>
void Take1D(const DLContext& ctx, DType* ptr, const DType* src, const Idx* indices, size_t length) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int nt = utils::FindNumThreads(length, 1024);
  int nb = (length + nt - 1) / nt;
  _Take1DKernel<<<nb, nt, 0, thr_entry->stream>>>(ptr, src, indices, length);
}

template void Take1D<kDLGPU, int32_t, float>(const DLContext&, float*, const float*, const int32_t*, size_t);
template void Take1D<kDLGPU, int32_t, double>(const DLContext&, double*, const double*, const int32_t*, size_t);

}  // namespace utils
}  // namespace kernel
}  // namespace dgl
