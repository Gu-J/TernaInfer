#pragma once

/*****************************************************************************
 * gemm_scale_ws.h
 *
 * 组装层：把以下四个组件拼装成对外可用的 GemmScaleWs 类
 *
 *   1. DefaultGemmKernel          —— 标准 CUTLASS int8 GEMM 骨架
 *   2. EpilogueVisitorScaleWs     —— 我们自定义的 Visitor
 *   3. EpilogueWithVisitor        —— 把 Visitor 注入到标准 Epilogue 里
 *   4. GemmWithEpilogueVisitor    —— 驱动 Mma + 带 Visitor 的 Epilogue
 *
 * 对比 example 35 的 gemm_with_softmax.h，本文件更简单：
 *   - 无 reduction，无多 kernel 调度
 *   - 只有一次 kernel launch
 *****************************************************************************/

#include "cutlass/cutlass.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"
#include "cutlass/device_kernel.h"

#include "examples/35_gemm_softmax/gemm_with_epilogue_visitor.h"    
#include "epilogue_visitor_scale_ws.h"       // 我们自己的 Visitor
#include "cutlass/bfloat16.h"
using bf16 = cutlass::bfloat16_t;

namespace cutlass {

// =============================================================================
// GemmScaleWs
//
// 用法示例（见 main.cu）：
//   GemmScaleWs gemm;
//   gemm.initialize(args);
//   gemm(stream);
// =============================================================================

template <
  // ---- 矩阵元素类型 ----
  typename ElementA_     = int8_t,
  typename LayoutA_      = cutlass::layout::RowMajor,
  typename ElementB_     = int8_t,
  typename LayoutB_      = cutlass::layout::ColumnMajor,

  // ---- 输出类型----
  typename ElementOutput_ = bf16,

  // ---- 中间计算精度 ----
  typename ElementCompute_ = float,

  // ---- 硬件配置 ----
  typename OperatorClass_    = cutlass::arch::OpClassTensorOp,
  typename ArchTag_          = cutlass::arch::Sm80,   // A6000 = Ampere = Sm86，兼容 Sm80

  // ---- Tile 形状 ----
  typename ThreadblockShape_ = cutlass::gemm::GemmShape<128, 128, 64>,
  typename WarpShape_        = cutlass::gemm::GemmShape<64,  64,  64>,
  typename InstructionShape_ = cutlass::gemm::GemmShape<16,  8,   32>, // int8 指令

  int kStages_ = 3
>
class GemmScaleWs {
public:

  // --------------------------------------------------------------------------
  // 类型别名
  // --------------------------------------------------------------------------

  using ElementA       = ElementA_;
  using LayoutA        = LayoutA_;
  using ElementB       = ElementB_;
  using LayoutB        = LayoutB_;
  using ElementOutput  = ElementOutput_;
  using ElementCompute = ElementCompute_;

  // int8 GEMM 的累加器固定为 int32
  using ElementAccumulator = int32_t;

  using OperatorClass    = OperatorClass_;
  using ArchTag          = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape        = WarpShape_;
  using InstructionShape = InstructionShape_;

  static int const kStages = kStages_;

  // 输出矩阵固定 RowMajor（scale 和 ws 的索引逻辑依赖此假设）
  using LayoutOutput = cutlass::layout::RowMajor;

  // 向量化访问宽度：128 bit / sizeof(ElementOutput)
  // float: 4 个元素；int8: 16 个元素
  static int const kAlignmentA      = 16; // int8，128bit/8bit
  static int const kAlignmentB      = 16;
  static int const kAlignmentOutput =
      128 / cutlass::sizeof_bits<ElementOutput>::value;

  using ThreadblockSwizzle =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  // --------------------------------------------------------------------------
  // Step 1: 构造标准 DefaultGemmKernel
  //
  // 目的：借用它的 Mma 和 Epilogue::OutputTileIterator
  // 它的 EpilogueFunctorOp（LinearCombination）在后续会被 Visitor 替换掉，
  // 但 DefaultGemm 模板要求必须传一个，所以这里给个占位符
  // --------------------------------------------------------------------------

  // 占位用的 EpilogueFunctorOp（决定 OutputTileIterator 的形状，不参与实际计算）
  using PlaceholderEpilogueFunctor = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,       // 输出类型
    kAlignmentOutput,    // 向量化宽度
    ElementAccumulator,  // 累加器类型
    ElementCompute       // 计算精度
  >;

  using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementOutput,
    LayoutOutput,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    PlaceholderEpilogueFunctor,
    ThreadblockSwizzle,
    kStages,
    true,    // SplitKSerial
    typename cutlass::gemm::device::DefaultGemmConfiguration<
      OperatorClass, ArchTag, ElementA, ElementB, ElementOutput, ElementCompute
    >::Operator
  >::GemmKernel;

  // --------------------------------------------------------------------------
  // Step 2: 定义我们的 EpilogueVisitor
  // --------------------------------------------------------------------------

  using EpilogueVisitor = cutlass::epilogue::threadblock::EpilogueVisitorScaleWs<
    ThreadblockShape,
    DefaultGemmKernel::kThreadCount,
    typename DefaultGemmKernel::Epilogue::OutputTileIterator,  // 复用标准的迭代器
    ElementAccumulator,   // int32
    ElementOutput,        // bf16
    ElementCompute        // float
  >;

  // --------------------------------------------------------------------------
  // Step 3: 把 Visitor 注入到标准 Epilogue 结构中
  //
  // EpilogueWithVisitorFromExistingEpilogue 的作用：
  //   - 复用 DefaultGemmKernel::Epilogue 的 OutputTileIterator、SharedStorage 等
  //   - 把 Visitor 的 visit()/begin_row() 等回调接入 tile 迭代循环
  // --------------------------------------------------------------------------

  using Epilogue = typename cutlass::epilogue::threadblock::
    EpilogueWithVisitorFromExistingEpilogue<
      EpilogueVisitor,
      typename DefaultGemmKernel::Epilogue
    >::Epilogue;

  // --------------------------------------------------------------------------
  // Step 4: 组装最终的 GemmKernel
  // --------------------------------------------------------------------------

  using GemmKernel = cutlass::gemm::kernel::GemmWithEpilogueVisitor<
    typename DefaultGemmKernel::Mma,
    Epilogue,
    ThreadblockSwizzle
  >;

  // --------------------------------------------------------------------------
  // 便于外部使用的 TensorRef 类型
  // --------------------------------------------------------------------------

  using TensorRefA = TensorRef<ElementA,      LayoutA>;
  using TensorRefB = TensorRef<ElementB,      LayoutB>;
  using TensorRefD = TensorRef<ElementOutput, LayoutOutput>;

  // ==========================================================================
  // Arguments：用户在 host 端填写
  // ==========================================================================

  struct Arguments {

    cutlass::gemm::GemmCoord  problem_size;  // {M, N, K}

    TensorRefA                ref_A;
    TensorRefB                ref_B;
    TensorRefD                ref_D;         // 输出矩阵（无 C，beta=0）

    ElementOutput const     *ptr_scale;     // device 指针，shape [M]
    ElementOutput const     *ptr_ws;        // device 指针，shape [ws_num]
    int                       ws_num;        // 列分组数，demo 中为 3

    Arguments() = default;

    Arguments(
      cutlass::gemm::GemmCoord  problem_size_,
      TensorRefA                ref_A_,
      TensorRefB                ref_B_,
      TensorRefD                ref_D_,
      ElementOutput const     *ptr_scale_,
      ElementOutput const     *ptr_ws_,
      int                       ws_num_
    ) :
      problem_size(problem_size_),
      ref_A(ref_A_),
      ref_B(ref_B_),
      ref_D(ref_D_),
      ptr_scale(ptr_scale_),
      ptr_ws(ptr_ws_),
      ws_num(ws_num_)
    {}
  };

  // ==========================================================================
  // Params：host 端预计算，传入 device kernel
  // ==========================================================================

  struct Params {
    typename GemmKernel::Params gemm_kernel_params;

    Params() = default;

    Params(Arguments const &args) {

      // 构造 EpilogueVisitor 的参数
      typename EpilogueVisitor::Arguments visitor_args(
        args.ptr_scale,
        args.ptr_ws,
        args.ws_num,
        args.problem_size.n()   // problem_n，用于计算 group_width
      );

      // 构造 GemmKernel 的参数
      // GemmWithEpilogueVisitor::Arguments 的签名（见 gemm_with_epilogue_visitor.h）：
      //   mode, problem_size, batch_count,
      //   ref_A, ref_B, ref_C, ref_D,
      //   ptr_Max, ptr_Sum,           ← 我们不用，传 nullptr
      //   batch_stride_A, batch_stride_B,
      //   epilogue_visitor_args
      gemm_kernel_params = typename GemmKernel::Params(
        typename GemmKernel::Arguments(
          cutlass::gemm::GemmUniversalMode::kGemm,
          args.problem_size,
          1,               // batch_count = 1
          args.ref_A,
          args.ref_B,
          args.ref_D,      // ref_C（框架签名要求，本 Visitor 不读 C）
          args.ref_D,      // ref_D（实际写出目标）
          nullptr,         // ptr_Max，不使用
          nullptr,         // ptr_Sum，不使用
          0,               // batch_stride_A
          0,               // batch_stride_B
          visitor_args
        )
      );
    }
  };

  // ==========================================================================
  // 公开方法
  // ==========================================================================

private:
  Params params_;

public:

  GemmScaleWs() = default;

  // 初始化（host 端调用）
  cutlass::Status initialize(Arguments const &args) {
    params_ = Params(args);
    return cutlass::Status::kSuccess;
  }

  // 启动 kernel（可指定 stream）
  cutlass::Status run(cudaStream_t stream = nullptr) {

    // 计算 grid 形状
    ThreadblockSwizzle swizzle;
    dim3 grid = swizzle.get_grid_shape(params_.gemm_kernel_params.grid_tiled_shape);
    dim3 block(GemmKernel::kThreadCount, 1, 1);

    int smem_size = static_cast<int>(sizeof(typename GemmKernel::SharedStorage));

    // CUDA 的 <<<>>> 语法不支持 namespace 限定的模板函数，必须先 using
    using cutlass::Kernel;

    // 如果 smem 超过 48KB，需要显式申请
    if (smem_size >= (48 << 10)) {
      cudaError_t err = cudaFuncSetAttribute(
        Kernel<GemmKernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
      if (err != cudaSuccess) {
        return cutlass::Status::kErrorInternal;
      }
    }

    // 启动 kernel
    Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(
      params_.gemm_kernel_params);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    return cutlass::Status::kSuccess;
  }

  // 支持 operator() 调用风格
  cutlass::Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }
};

} // namespace cutlass