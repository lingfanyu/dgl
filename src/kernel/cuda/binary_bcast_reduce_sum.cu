/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cuda/binary_bcast_reduce_sum.cu
 * \brief CUDA kernels for braodcasting binary reduce sum
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
void FallbackCallBinaryReduceBcast(
  const minigun::advance::RuntimeConfig& rtcfg,
  const CSRWrapper& graph,
  BcastGData<2, int32_t, DType>* gdata) {
  LOG(INFO) << "***************" << "FallbackCallBinaryReduceBcast" << "*******************";
  constexpr int XPU = kDLGPU;
  constexpr int NDim = 2;
  typedef int32_t Idx;
  typedef SelectSrc LeftSelector;
  typedef SelectEdge RightSelector;
  typedef BinaryMul<DType> BinaryOp;
  typedef ReduceSum<kDLGPU, DType> Reducer;
  typedef cuda::FunctorsTempl<Idx, DType, LeftSelector,
                        RightSelector, BinaryOp, Reducer>
          Functors;
  typedef cuda::BinaryReduceBcast<NDim, Idx, DType, Functors> UDF;
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
  minigun::advance::Advance<XPU, Idx, cuda::AdvanceConfig,
    BcastGData<NDim, Idx, DType>, UDF>(
        rtcfg, csr, gdata, minigun::IntArray1D<Idx>());
}

// backward

template <typename DType>
void FallbackCallBackwardBinaryReduceBcast(
    const minigun::advance::RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardBcastGData<2, int32_t, DType>* gdata) {
  LOG(INFO) << "***************" << "FallbackCallBackwardBinaryReduceBcast" << "*******************";
  constexpr int XPU = kDLGPU;
  constexpr int Mode = binary_op::kGradLhs;
  constexpr int NDim = 2;
  typedef int32_t Idx;
  typedef SelectSrc LeftSelector;
  typedef SelectEdge RightSelector;
  typedef BinaryMul<DType> BinaryOp;
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
  typedef cuda::BackwardBinaryReduceBcast<Mode, NDim, Idx, DType, Functors> UDF;
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
  minigun::advance::Advance<XPU, Idx, cuda::AdvanceConfig,
    BackwardBcastGData<NDim, Idx, DType>, UDF>(
        rtcfg, csr, gdata, minigun::IntArray1D<Idx>());
}

}  // namespace cuda

template <>
void CallBinaryReduceBcast<kDLGPU, 2, int32_t, float, SelectSrc, SelectEdge,
                      BinaryMul<float>, ReduceSum<kDLGPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BcastGData<2, int32_t, float>* gdata) {
  if ((gdata->rhs_shape[0] != 1 && gdata->rhs_shape[1] != 1) || 
      gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
    cuda::FallbackCallBinaryReduceBcast<float>(rtcfg, graph, gdata);
  } else {
    // cusparse use rev csr for csrmm
    auto csr = graph.GetInCSRMatrix();
    // XXX: this is a hack because we did not reorder edges
    cuda::CusparseCsrmm2(rtcfg, csr, gdata->rhs_data, gdata->lhs_data, 
        gdata->out_data, gdata->out_len);
  }
}

template <>
void CallBinaryReduceBcast<kDLGPU, 2, int32_t, double, SelectSrc, SelectEdge,
                      BinaryMul<double>, ReduceSum<kDLGPU, double>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BcastGData<2, int32_t, double>* gdata) {
  if ((gdata->rhs_shape[0] != 1 && gdata->rhs_shape[1] != 1) || 
      gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
    cuda::FallbackCallBinaryReduceBcast<double>(rtcfg, graph, gdata);
  } else {
    // cusparse use rev csr for csrmm
    auto csr = graph.GetInCSRMatrix();
    // XXX: this is a hack because we did not reorder edges
    cuda::CusparseCsrmm2(rtcfg, csr, gdata->rhs_data, gdata->lhs_data, 
        gdata->out_data, gdata->out_len);
  }
}

// backward

template <>
void CallBackwardBinaryReduceBcast<kDLGPU, binary_op::kGradLhs, 2, int32_t, 
                                   float, SelectSrc, SelectEdge,
                              BinaryMul<float>, ReduceSum<kDLGPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardBcastGData<2, int32_t, float>* gdata) {
  if ((gdata->rhs_shape[0] != 1 && gdata->rhs_shape[1] != 1) || 
      gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
    cuda::FallbackCallBackwardBinaryReduceBcast<float>(rtcfg, graph, gdata);
  } else {
    auto csr = graph.GetOutCSRMatrix();
    // XXX: this is a hack because we did not reorder edges
    cuda::CusparseCsrmm2(rtcfg, csr, gdata->rhs_data, gdata->grad_out_data, 
        gdata->grad_lhs_data, gdata->out_len);
  }
}

template <>
void CallBackwardBinaryReduceBcast<kDLGPU, binary_op::kGradLhs, 2, int32_t, 
                                   double, SelectSrc, SelectEdge,
                              BinaryMul<double>, ReduceSum<kDLGPU, double>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardBcastGData<2, int32_t, double>* gdata) {
  if ((gdata->rhs_shape[0] != 1 && gdata->rhs_shape[1] != 1) || 
      gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
    cuda::FallbackCallBackwardBinaryReduceBcast<double>(rtcfg, graph, gdata);
  } else {
    auto csr = graph.GetOutCSRMatrix();
    // XXX: this is a hack because we did not reorder edges
    cuda::CusparseCsrmm2(rtcfg, csr, gdata->rhs_data, gdata->grad_out_data, 
        gdata->grad_lhs_data, gdata->out_len);
  }
}

// generate definitions
#define REDUCER ReduceSum
#define XPU kDLGPU
#define IDX int32_t

EVAL(GEN_NDIM, GEN_DTYPE, GEN_OP_TARGET, GEN_BCAST_DEFINE);
EVAL(GEN_BACKWARD_MODE, GEN_NDIM, GEN_DTYPE, GEN_OP_TARGET,
     GEN_BACKWARD_BCAST_DEFINE);

}  // namespace kernel
}  // namespace dgl
