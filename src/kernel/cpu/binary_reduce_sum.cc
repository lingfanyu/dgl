/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cpu/binary_reduce_sum.cc
 * \brief CPU kernels for binary reduce sum
 */
#include "./binary_reduce_impl.h"
#include "./backward_binary_reduce_impl.h"

namespace dgl {
namespace kernel {

#define REDUCER ReduceSum
#define XPU kDLCPU

#define IDX int32_t
EVAL(GEN_DTYPE, GEN_TARGET, GEN_BINARY_OP, GEN_DEFINE);
#undef IDX

#define IDX int64_t
EVAL(GEN_DTYPE, GEN_TARGET, GEN_BINARY_OP, GEN_DEFINE);
#undef IDX

EVAL(GEN_BACKWARD_MODE, GEN_DTYPE, GEN_TARGET, GEN_BINARY_OP, GEN_BACKWARD_DEFINE);

}  // namespace kernel
}  // namespace dgl
