/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cuda/binary_reduce_sum.cu
 * \brief CUDA kernels for binary reduce sum
 */
#include <dgl/runtime/device_api.h>

#include "../../runtime/cuda/cuda_common.h"
#include "./binary_reduce_impl.cuh"
#include "./backward_binary_reduce_impl.cuh"
#include "../utils.h"
#include "../csr_interface.h"
#include "./cusparse.cuh"

using minigun::advance::RuntimeConfig;

namespace dgl {
namespace kernel {
namespace cuda {

// forward

template <typename DType>
void FallbackCallBinaryReduce(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    GData<int32_t, DType>* gdata) {
  constexpr int XPU = kDLGPU;
  typedef int32_t Idx;
  typedef SelectSrc LeftSelector;
  typedef SelectNone RightSelector;
  typedef BinaryUseLhs<DType> BinaryOp;
  typedef ReduceSum<kDLGPU, DType> Reducer;
  typedef cuda::FunctorsTempl<Idx, DType, LeftSelector,
                        RightSelector, BinaryOp, Reducer>
          Functors;
  typedef cuda::BinaryReduce<Idx, DType, Functors> UDF;
  // csr
  auto outcsr = graph.GetOutCSRMatrix();
  minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(outcsr.indptr, outcsr.indices);
  // If the user-given mapping is none and the target is edge data, we need to
  // replace the mapping by the edge ids in the csr graph so that the edge
  // data is correctly read/written.
  if (LeftSelector::target == binary_op::kEdge && gdata->lhs_mapping == nullptr) {
    gdata->lhs_mapping = static_cast<Idx*>(outcsr.data->data);
  }
  if (RightSelector::target == binary_op::kEdge && gdata->rhs_mapping == nullptr) {
    gdata->rhs_mapping = static_cast<Idx*>(outcsr.data->data);
  }
  if (OutSelector<Reducer>::Type::target == binary_op::kEdge
      && gdata->out_mapping == nullptr) {
    gdata->out_mapping = static_cast<Idx*>(outcsr.data->data);
  }
  // TODO(minjie): allocator
  minigun::advance::Advance<XPU, Idx, cuda::AdvanceConfig, GData<Idx, DType>, UDF>(
        rtcfg, csr, gdata, minigun::IntArray1D<Idx>());
}

template <typename DType>
void FallbackCallBackwardBinaryReduce(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<int32_t, DType>* gdata) {
  constexpr int XPU = kDLGPU;
  constexpr int Mode = binary_op::kGradLhs;
  typedef int32_t Idx;
  typedef SelectSrc LeftSelector;
  typedef SelectNone RightSelector;
  typedef BinaryUseLhs<DType> BinaryOp;
  typedef ReduceSum<kDLGPU, DType> Reducer;
  // For backward computation, we use reverse csr and switch dst and src.
  // This benefits the most common src_op_edge or copy_src case, because the
  // gradients of src are now aggregated into destination buffer to reduce
  // competition of atomic add.
  auto incsr = graph.GetInCSRMatrix();
  minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(incsr.indptr, incsr.indices);
  typedef cuda::BackwardFunctorsTempl<Idx, DType,
          typename SwitchSrcDst<LeftSelector>::Type,
          typename SwitchSrcDst<RightSelector>::Type,
          BinaryOp, Reducer> Functors;
  typedef cuda::BackwardBinaryReduce<Mode, Idx, DType, Functors> UDF;
  // If the user-given mapping is none and the target is edge data, we need to
  // replace the mapping by the edge ids in the csr graph so that the edge
  // data is correctly read/written.
  if (LeftSelector::target == binary_op::kEdge
      && gdata->lhs_mapping == nullptr) {
    gdata->lhs_mapping = static_cast<Idx*>(incsr.data->data);
  }
  if (RightSelector::target == binary_op::kEdge
      && gdata->rhs_mapping == nullptr) {
    gdata->rhs_mapping = static_cast<Idx*>(incsr.data->data);
  }
  if (OutSelector<Reducer>::Type::target == binary_op::kEdge
      && gdata->out_mapping == nullptr) {
    gdata->out_mapping = static_cast<Idx*>(incsr.data->data);
  }
  // TODO(minjie): allocator
  minigun::advance::Advance<XPU, Idx, cuda::AdvanceConfig, BackwardGData<Idx, DType>, UDF>(
        rtcfg, csr, gdata, minigun::IntArray1D<Idx>());
}

}  // namespace cuda

template <>
void CallBinaryReduce<kDLGPU, int32_t, float, SelectSrc, SelectNone,
                      BinaryUseLhs<float>, ReduceSum<kDLGPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    GData<int32_t, float>* gdata) {
  if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
    cuda::FallbackCallBinaryReduce<float>(rtcfg, graph, gdata);
  } else {
    // cusparse use rev csr for csrmm
    auto csr = graph.GetInCSRMatrix();
    cuda::CusparseCsrmm2(rtcfg, csr, static_cast<float*>(nullptr), gdata->lhs_data, gdata->out_data,
        gdata->x_length);
  }
}

template <>
void CallBinaryReduce<kDLGPU, int32_t, double, SelectSrc, SelectNone,
                      BinaryUseLhs<double>, ReduceSum<kDLGPU, double>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    GData<int32_t, double>* gdata) {
  if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
    cuda::FallbackCallBinaryReduce<double>(rtcfg, graph, gdata);
  } else {
    // cusparse use rev csr for csrmm
    auto csr = graph.GetInCSRMatrix();
    cuda::CusparseCsrmm2(rtcfg, csr, static_cast<double*>(nullptr), gdata->lhs_data, gdata->out_data,
        gdata->x_length);
  }
}

// backward

template <>
void CallBackwardBinaryReduce<kDLGPU, binary_op::kGradLhs, int32_t, float,
                              SelectSrc, SelectNone,
                              BinaryUseLhs<float>, ReduceSum<kDLGPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<int32_t, float>* gdata) {
  if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
    cuda::FallbackCallBackwardBinaryReduce<float>(rtcfg, graph, gdata);
  } else {
    auto csr = graph.GetOutCSRMatrix();
    cuda::CusparseCsrmm2(rtcfg, csr, static_cast<float*>(nullptr), gdata->grad_out_data, gdata->grad_lhs_data,
        gdata->x_length);
  }
}

template <>
void CallBackwardBinaryReduce<kDLGPU, binary_op::kGradLhs, int32_t, double,
                              SelectSrc, SelectNone,
                              BinaryUseLhs<double>, ReduceSum<kDLGPU, double>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<int32_t, double>* gdata) {
  if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
    cuda::FallbackCallBackwardBinaryReduce<double>(rtcfg, graph, gdata);
  } else {
    auto csr = graph.GetOutCSRMatrix();
    cuda::CusparseCsrmm2(rtcfg, csr, static_cast<double*>(nullptr), gdata->grad_out_data, gdata->grad_lhs_data,
        gdata->x_length);
  }
}

// generate definitions

#define REDUCER ReduceSum
#define XPU kDLGPU
#define IDX int32_t

EVAL(GEN_DTYPE, GEN_OP_TARGET, GEN_DEFINE);
EVAL(GEN_BACKWARD_MODE, GEN_DTYPE, GEN_OP_TARGET, GEN_BACKWARD_DEFINE);

}  // namespace kernel
}  // namespace dgl
