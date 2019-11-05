#include "./cusparse_common.cuh"

namespace dgl {
namespace kernel {
namespace cuda {

template <typename DType>
cusparseStatus_t Xcsrmm2(cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const DType* alpha, const cusparseMatDescr_t descrA,
    const DType* csrValA, const int* csrRowPtrA, const int* csrColIndA,
    const DType* B, int ldb, const DType* beta, DType* C, int ldc) {
  LOG(INFO) << "Not supported dtype";
  return CUSPARSE_STATUS_EXECUTION_FAILED;
}

template <>
cusparseStatus_t Xcsrmm2<float>(cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const float* alpha, const cusparseMatDescr_t descrA,
    const float* csrValA, const int* csrRowPtrA, const int* csrColIndA,
    const float* B, int ldb, const float* beta, float* C, int ldc) {
  return cusparseScsrmm2(handle, transA, transB, m, n, k, nnz,
      alpha, descrA, csrValA, csrRowPtrA, csrColIndA,
      B, ldb, beta, C, ldc);
}

template <>
cusparseStatus_t Xcsrmm2<double>(cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const double* alpha, const cusparseMatDescr_t descrA,
    const double* csrValA, const int* csrRowPtrA, const int* csrColIndA,
    const double* B, int ldb, const double* beta, double* C, int ldc) {
  return cusparseDcsrmm2(handle, transA, transB, m, n, k, nnz,
      alpha, descrA, csrValA, csrRowPtrA, csrColIndA,
      B, ldb, beta, C, ldc);
}

template <typename DType>
cublasStatus_t Xgeam(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,
    const DType* alpha, const DType* A, int lda,
    const DType* beta, const DType* B, int ldb,
    DType* C, int ldc) {
  LOG(INFO) << "Not supported dtype";
  return CUBLAS_STATUS_EXECUTION_FAILED;
}

template <>
cublasStatus_t Xgeam<float>(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,
    const float* alpha, const float* A, int lda,
    const float* beta, const float* B, int ldb,
    float* C, int ldc) {
  return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda,
      beta, B, ldb, C, ldc);
}

template <>
cublasStatus_t Xgeam<double>(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,
    const double* alpha, const double* A, int lda,
    const double* beta, const double* B, int ldb,
    double* C, int ldc) {
  return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda,
      beta, B, ldb, C, ldc);
}

} // namespace cuda
} // namespace kernel
} // namespace dgl
