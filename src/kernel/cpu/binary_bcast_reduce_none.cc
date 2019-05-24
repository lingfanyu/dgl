/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cpu/binary_bcast_reduce_none.cc
 * \brief CPU kernels for braodcasting binary reduce none
 */
#include "./binary_reduce_impl.h"
#include "./backward_binary_reduce_impl.h"

namespace dgl {
namespace kernel {

#define REDUCER ReduceNone
#define XPU kDLCPU

#define IDX int32_t
EVAL(GEN_NDIM, GEN_DTYPE, GEN_TARGET, GEN_BINARY_OP, GEN_BCAST_DEFINE);
#undef IDX

#define IDX int64_t
EVAL(GEN_NDIM, GEN_DTYPE, GEN_TARGET, GEN_BINARY_OP, GEN_BCAST_DEFINE);
#undef IDX

EVAL(GEN_BACKWARD_MODE, GEN_NDIM, GEN_DTYPE, GEN_TARGET, GEN_BINARY_OP,
     GEN_BACKWARD_BCAST_DEFINE);

}  // namespace kernel
}  // namespace dgl
