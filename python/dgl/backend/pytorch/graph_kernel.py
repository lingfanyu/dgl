from torch.utils.cpp_extension import load
import os
from torch.autograd import Function

dgl_path = os.environ.get("DGL_PATH", "")
if not dgl_path:
    raise RuntimeError("Must set environment variable DGL_PATH")
path = os.path.join(dgl_path, "python/dgl/backend/pytorch")
sources=["graph_kernel.cu", "graph_kernel.cpp"]
sources=[os.path.join(path, f) for f in sources]
graph_kernel = load(name="graph_kernel", sources=sources)

class VectorSPMM(Function):
    @staticmethod
    def forward(ctx, indptr, eid, indices, ptr_t, eid_t, indices_t, edata, x):
        y = graph_kernel.vector_spmm_forward(indptr, eid, indices, edata, x)
        ctx.save_for_backward(indptr, eid, indices, ptr_t, eid_t, indices_t,
                              edata, x)
        return y

    @staticmethod
    def backward(ctx, dy):
        indptr, eid, indices, ptr_t, eid_t, indices_t, edata, x = ctx.saved_tensors
        dedata, dx = graph_kernel.vector_spmm_backward(indptr, eid, indices, ptr_t, eid_t,
                                          indices_t, edata, dy, x)
        return None, None, None, None, None, None, dedata, dx
