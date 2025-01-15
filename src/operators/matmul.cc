#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        auto A_shape = inputs[0]->getDims();
        auto B_shape = inputs[1]->getDims();
        if(transA){
            std::swap(A_shape[A_shape.size()-1],A_shape[A_shape.size()-2]);
        }
        if(transB){
            std::swap(B_shape[B_shape.size()-1],B_shape[B_shape.size()-2]);
        }
        auto output_shape = A_shape;
        output_shape[output_shape.size()-1] = B_shape[B_shape.size()-1];
        return {{output_shape}};
    }

} // namespace infini