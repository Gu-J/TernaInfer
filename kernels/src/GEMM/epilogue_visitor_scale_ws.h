#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/bfloat16.h"

namespace cutlass {
namespace epilogue {
namespace threadblock {

template <
  typename ThreadblockShape_,
  int      ThreadCount,
  typename OutputTileIterator_,
  typename ElementAccumulator_,
  typename ElementOutput_,
  typename ElementCompute_
>
class EpilogueVisitorScaleWs {
public:

  using ThreadblockShape   = ThreadblockShape_;
  using OutputTileIterator = OutputTileIterator_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementOutput      = ElementOutput_;
  using ElementCompute     = ElementCompute_;
  using LayoutOutput       = cutlass::layout::RowMajor;

  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;
  static int const kIterations        = OutputTileIterator::kIterations;

  // 框架要求必须声明，即使不做 reduction
  using ElementNorm = float;
  using ElementSum  = float;

  using AccumulatorFragment = Array<ElementAccumulator, kElementsPerAccess>;
  using ComputeFragment     = Array<ElementCompute,     kElementsPerAccess>;
  using OutputVector        = Array<ElementOutput,      kElementsPerAccess>;

  struct SharedStorage {};

  // --------------------------------------------------------------------------
  // Arguments / Params
  // scale 和 ws 存储类型是 ElementOutput（即 bf16）
  // 计算时在 device 端转成 ElementCompute（float）
  // --------------------------------------------------------------------------
  struct Arguments {
    ElementOutput const *ptr_scale;  // [M]，bf16
    ElementOutput const *ptr_ws;     // [ws_num]，bf16
    int                  ws_num;
    int                  problem_n;

    Arguments() = default;
    Arguments(
      ElementOutput const *ptr_scale_,
      ElementOutput const *ptr_ws_,
      int ws_num_,
      int problem_n_
    ) : ptr_scale(ptr_scale_), ptr_ws(ptr_ws_),
        ws_num(ws_num_), problem_n(problem_n_) {}
  };

  struct Params {
    ElementOutput const *ptr_scale;
    ElementOutput const *ptr_ws;
    int                  ws_num;
    int                  group_width;   // problem_n / ws_num，host 端预计算

    Params() = default;
    CUTLASS_HOST_DEVICE
    Params(Arguments const &args) :
      ptr_scale(args.ptr_scale),
      ptr_ws(args.ptr_ws),
      ws_num(args.ws_num),
      group_width(args.problem_n / args.ws_num)
    {}
  };

private:

  Params const &     params_;
  SharedStorage &    shared_storage_;
  MatrixCoord        extent_;

  OutputTileIterator iterator_D_;
  typename OutputTileIterator::Fragment fragment_D_;

  MatrixCoord        thread_offset_;

  // --------------------------------------------------------------------------
  // 关键优化：
  //
  // ThreadblockShape::kN = 128，group_width = 1280。
  // 因为 128 < 1280，且 threadblock 的列起始地址是 128 的整数倍，
  // 整个 threadblock 的所有输出列必然落在同一个 ws 分组里。
  // 所以 ws_val_ 在构造函数里算一次，整个 epilogue 不再重算。
  //
  // coeff_ = ws_val_ / scale[i]，每行更新一次（在 visit column_idx==0 时）。
  // 之后同一行内所有 fragment 直接乘 coeff_，无任何额外计算。
  // --------------------------------------------------------------------------
  ElementCompute ws_val_;    // 本 threadblock 对应的 ws 值，构造时确定，不再变化
  ElementCompute coeff_;     // ws_val_ / scale[i]，每行更新一次

public:

  CUTLASS_DEVICE
  EpilogueVisitorScaleWs(
    Params const &        params,
    SharedStorage &       shared_storage,
    MatrixCoord const &   problem_size,
    int                   thread_idx,
    int                   warp_idx,
    int                   lane_idx,
    typename OutputTileIterator::Params params_C,   // 不使用，保留签名兼容性
    typename OutputTileIterator::Params params_D,
    typename OutputTileIterator::Element *ptr_C,    // 不使用
    typename OutputTileIterator::Element *ptr_D,
    ElementNorm * = nullptr,
    ElementSum  * = nullptr,
    MatrixCoord const & threadblock_offset = MatrixCoord(0, 0),
    int column_offset = 0
  ) :
    params_(params),
    shared_storage_(shared_storage),
    extent_(problem_size),
    iterator_D_(params_D, ptr_D, problem_size, thread_idx, threadblock_offset),
    coeff_(ElementCompute(1))
  {
    // 整个 threadblock 只算一次 ws 分组 k
    // threadblock_offset.column() 是本 TB 负责的列起始，是 kN=128 的整数倍
    // group_width=1280，128 整除 1280，所以 TB 内不会跨越分组边界
    int col_ind=threadblock_offset.column();
    int k;

    if (problem_size.column()==3840) {
      if (col_ind < 2560) k=0;
      else if (col_ind < 2560+640) k=1;
      else k=2;
    }
    else 
      k = col_ind / params_.group_width;
    k = (k < params_.ws_num) ? k : (params_.ws_num - 1);

    // 从全局内存读一次 ws[k]，转成 float 存入寄存器
    ws_val_ = ElementCompute(params_.ptr_ws[k]);
  }

  CUTLASS_DEVICE void set_k_partition(int, int) {}
  CUTLASS_DEVICE void set_batch_index(int) {}
  CUTLASS_DEVICE void begin_epilogue() {}

  CUTLASS_DEVICE
  void begin_step(int step_idx) {
    fragment_D_.clear();
  }

  CUTLASS_DEVICE void begin_row(int row_idx) {}

  // --------------------------------------------------------------------------
  // visit()：每个 output fragment 调用一次
  //
  // 热路径上的操作（每次 visit 都执行）：
  //   - 仅一次 FMA：compute_frag[i] *= coeff_
  //   - 两次类型转换（int32->float，float->bf16）
  //
  // 冷路径（每行第一次，column_idx==0）：
  //   - 一次 global load（scale[global_row]）
  //   - 一次除法（ws_val_ / scale）
  //   - 一次 iteration_offset 坐标还原（只为取 global_row）
  // --------------------------------------------------------------------------
  CUTLASS_DEVICE
  void visit(
    int iter_idx,
    int row_idx,
    int column_idx,
    int frag_idx,
    AccumulatorFragment const &accum)
  {
    // 每行只在第一个 fragment 时更新 coeff_
    if (column_idx == 0) {
      thread_offset_ =
        iterator_D_.thread_start() +
        OutputTileIterator::ThreadMap::iteration_offset(frag_idx);

      ElementCompute scale_val =
        ElementCompute(params_.ptr_scale[thread_offset_.row()]);

      // 预计算 coeff = ws / scale，后续同行所有 fragment 直接乘，无额外开销
      coeff_ = (scale_val != ElementCompute(0))
               ? (ws_val_ / scale_val)
               : ElementCompute(0);
    }

    // 热路径：int32 -> float -> 乘系数 -> bf16
    NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>
      to_compute;
    ComputeFragment compute_frag = to_compute(accum);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      compute_frag[i] *= coeff_;
    }

    NumericArrayConverter<ElementOutput, ElementCompute, kElementsPerAccess>
      to_output;
    reinterpret_cast<OutputVector *>(&fragment_D_)[frag_idx] =
      to_output(compute_frag);
  }

  CUTLASS_DEVICE void end_row(int row_idx) {}

  CUTLASS_DEVICE
  void end_step(int step_idx) {
    iterator_D_.store(fragment_D_);
    ++iterator_D_;
  }

  CUTLASS_DEVICE void end_epilogue() {}
};

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass