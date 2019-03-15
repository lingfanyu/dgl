#include <torch/torch.h>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor vector_spmm_cuda_forward(
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& indices,
    const at::Tensor& edata,
    const at::Tensor& x);

at::Tensor vector_spmm_forward(
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& indices,
    const at::Tensor& edata,
    const at::Tensor& x) {
    CHECK_INPUT(indptr);
    CHECK_INPUT(eid);
    CHECK_INPUT(indices);
    CHECK_INPUT(edata);
    CHECK_INPUT(x);
    return vector_spmm_cuda_forward(indptr, eid, indices, edata, x);
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
    const at::Tensor& x);

std::vector<at::Tensor> vector_spmm_backward(
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& indices,
    const at::Tensor& indptr_t,
    const at::Tensor& eid_t,
    const at::Tensor& indices_t,
    const at::Tensor& edata,
    const at::Tensor& dy,
    const at::Tensor& x) {
    CHECK_INPUT(indptr);
    CHECK_INPUT(eid);
    CHECK_INPUT(indices);
    CHECK_INPUT(indptr_t);
    CHECK_INPUT(eid_t);
    CHECK_INPUT(indices_t);
    CHECK_INPUT(edata);
    CHECK_INPUT(dy);
    CHECK_INPUT(x);
    return vector_spmm_cuda_backward(indptr, eid, indices, indptr_t, eid_t, indices_t, edata, dy, x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_spmm_forward", &vector_spmm_forward, "Vectorized SPMM forward");
    m.def("vector_spmm_backward", &vector_spmm_backward, "Vectorized SPMM backward");
}
