[How to link to c++ function](https://discourse.llvm.org/t/mlir-how-do-i-link-an-external-c-function-for-an-operation-in-an-mlir-file/1821/2)
[name-mangling](https://zh.wikipedia.org/zh-tw/%E5%90%8D%E5%AD%97%E4%BF%AE%E9%A5%B0)

```
/home/sylvex/Polygeist/build/bin/cgeist /home/sylvex/universal/playground/convert_reduct.cpp -S -function=convert_to_posit -I /home/sylvex/universal/include/ -I /usr/lib/gcc/x86_64-linux-gnu/12/include/ -o convert.mlir
```

```
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
}
```

```
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.mlir.global internal constant @str0("vector::_M_realloc_insert\00") {addr_space = 0 : i32}
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c69_i32 = arith.constant 69 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<1xi32>
    %cast = memref.cast %alloca : memref<1xi32> to memref<?xi32>
    %0 = llvm.mlir.undef : i32
    affine.store %0, %alloca[0] : memref<1xi32>
    %alloca_0 = memref.alloca() : memref<1x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>
    %cast_1 = memref.cast %alloca_0 : memref<1x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>> to memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>
    call @_ZNSt6vectorIiSaIiEEC1Ev(%cast_1) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>) -> ()
    affine.store %c0_i32, %alloca[0] : memref<1xi32>
    scf.while (%arg0 = %c0_i32) : (i32) -> () {
      %1 = arith.cmpi slt, %arg0, %c69_i32 : i32
      scf.condition(%1)
    } do {
      func.call @_ZNSt6vectorIiSaIiEE9push_backERKi(%cast_1, %cast) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>, memref<?xi32>) -> ()
      %1 = affine.load %alloca[0] : memref<1xi32>
      %2 = arith.addi %1, %c1_i32 : i32
      affine.store %2, %alloca[0] : memref<1xi32>
      scf.yield %2 : i32
    }
    return %c0_i32 : i32
  }
  func.func @_ZNSt6vectorIiSaIiEEC1Ev(%arg0: memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>) -> !llvm.ptr
    %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr) -> memref<?x!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>>
    call @_ZNSt12_Vector_baseIiSaIiEEC1Ev(%1) : (memref<?x!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>>) -> ()
    return
  }
  func.func @_ZNSt6vectorIiSaIiEE9push_backERKi(%arg0: memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %c1 = arith.constant 1 : index
    %alloca = memref.alloca() : memref<1x1xmemref<?xi32>>
    %alloca_0 = memref.alloca() : memref<1x1xmemref<?xi32>>
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>) -> !llvm.ptr
    %1 = llvm.getelementptr %0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>
    %2 = llvm.getelementptr %1[1] : (!llvm.ptr) -> !llvm.ptr, memref<?xi32>
    %3 = llvm.load %2 : !llvm.ptr -> memref<?xi32>
    %4 = llvm.getelementptr %1[2] : (!llvm.ptr) -> !llvm.ptr, memref<?xi32>
    %5 = llvm.load %4 : !llvm.ptr -> memref<?xi32>
    %6 = "polygeist.memref2pointer"(%3) : (memref<?xi32>) -> !llvm.ptr
    %7 = "polygeist.memref2pointer"(%5) : (memref<?xi32>) -> !llvm.ptr
    %8 = llvm.icmp "ne" %6, %7 : !llvm.ptr
    scf.if %8 {
      %9 = llvm.mlir.zero : !llvm.ptr
      %10 = llvm.icmp "ne" %0, %9 : !llvm.ptr
      %11 = arith.select %10, %0, %9 : !llvm.ptr
      %12 = llvm.load %2 : !llvm.ptr -> memref<?xi32>
      %13 = "polygeist.pointer2memref"(%11) : (!llvm.ptr) -> memref<?x!llvm.struct<(struct<(i8)>)>>
      func.call @_ZNSt16allocator_traitsISaIiEE9constructIiJRKiEEEvRS0_PT_DpOT0_(%13, %12, %arg1) : (memref<?x!llvm.struct<(struct<(i8)>)>>, memref<?xi32>, memref<?xi32>) -> ()
      %14 = llvm.load %2 : !llvm.ptr -> memref<?xi32>
      %15 = "polygeist.subindex"(%14, %c1) : (memref<?xi32>, index) -> memref<?xi32>
      llvm.store %15, %2 : memref<?xi32>, !llvm.ptr
    } else {
      %cast = memref.cast %alloca_0 : memref<1x1xmemref<?xi32>> to memref<?x1xmemref<?xi32>>
      func.call @_ZNSt6vectorIiSaIiEE3endEv(%arg0, %cast) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>, memref<?x1xmemref<?xi32>>) -> ()
      %9 = affine.load %alloca_0[0, 0] : memref<1x1xmemref<?xi32>>
      affine.store %9, %alloca[0, 0] : memref<1x1xmemref<?xi32>>
      %cast_1 = memref.cast %alloca : memref<1x1xmemref<?xi32>> to memref<?x1xmemref<?xi32>>
      func.call @_ZNSt6vectorIiSaIiEE17_M_realloc_insertIJRKiEEEvN9__gnu_cxx17__normal_iteratorIPiS1_EEDpOT_(%arg0, %cast_1, %arg1) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>, memref<?x1xmemref<?xi32>>, memref<?xi32>) -> ()
    }
    return
  }
  func.func @_ZNSt12_Vector_baseIiSaIiEEC1Ev(%arg0: memref<?x!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>>) -> !llvm.ptr
    %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr) -> memref<?x!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>>
    call @_ZNSt12_Vector_baseIiSaIiEE12_Vector_implC1Ev(%1) : (memref<?x!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>>) -> ()
    return
  }
  func.func @_ZNSt16allocator_traitsISaIiEE9constructIiJRKiEEEvRS0_PT_DpOT0_(%arg0: memref<?x!llvm.struct<(struct<(i8)>)>>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = affine.load %arg2[0] : memref<?xi32>
    affine.store %0, %arg1[0] : memref<?xi32>
    return
  }
  func.func @_ZNSt6vectorIiSaIiEE17_M_realloc_insertIJRKiEEEvN9__gnu_cxx17__normal_iteratorIPiS1_EEDpOT_(%arg0: memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>, %arg1: memref<?x1xmemref<?xi32>>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %true = arith.constant true
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c1 = arith.constant 1 : index
    %alloca = memref.alloca() : memref<1x1xmemref<?xi32>>
    %0 = llvm.mlir.addressof @str0 : !llvm.ptr
    %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr) -> memref<?xi8>
    %2 = llvm.mlir.undef : i64
    %alloca_0 = memref.alloca() : memref<1xi64>
    %cast = memref.cast %alloca_0 : memref<1xi64> to memref<?xi64>
    affine.store %2, %alloca_0[0] : memref<1xi64>
    %alloca_1 = memref.alloca() : memref<1xi64>
    %cast_2 = memref.cast %alloca_1 : memref<1xi64> to memref<?xi64>
    affine.store %c1_i64, %alloca_1[0] : memref<1xi64>
    %3 = call @_ZNKSt6vectorIiSaIiEE8max_sizeEv(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>) -> i64
    %4 = call @_ZNKSt6vectorIiSaIiEE4sizeEv(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>) -> i64
    %5 = arith.subi %3, %4 : i64
    %6 = affine.load %alloca_1[0] : memref<1xi64>
    %7 = arith.cmpi slt, %5, %6 : i64
    scf.if %7 {
      func.call @_ZSt20__throw_length_errorPKc(%1) : (memref<?xi8>) -> ()
    }
    %8 = call @_ZNKSt6vectorIiSaIiEE4sizeEv(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>) -> i64
    %9 = call @_ZNKSt6vectorIiSaIiEE4sizeEv(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>) -> i64
    affine.store %9, %alloca_0[0] : memref<1xi64>
    %alloca_3 = memref.alloca() : memref<memref<?xi64>>
    %10 = affine.load %alloca_1[0] : memref<1xi64>
    %11 = arith.cmpi slt, %9, %10 : i64
    %12 = arith.cmpi sge, %9, %10 : i64
    scf.if %11 {
      affine.store %cast_2, %alloca_3[] : memref<memref<?xi64>>
    }
    scf.if %12 {
      affine.store %cast, %alloca_3[] : memref<memref<?xi64>>
    }
    %13 = affine.load %alloca_3[] : memref<memref<?xi64>>
    %14 = affine.load %13[0] : memref<?xi64>
    %15 = arith.addi %8, %14 : i64
    %16 = call @_ZNKSt6vectorIiSaIiEE4sizeEv(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>) -> i64
    %17 = arith.cmpi slt, %15, %16 : i64
    %18 = scf.if %17 -> (i1) {
      scf.yield %true : i1
    } else {
      %50 = func.call @_ZNKSt6vectorIiSaIiEE8max_sizeEv(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>) -> i64
      %51 = arith.cmpi sgt, %15, %50 : i64
      scf.yield %51 : i1
    }
    %19 = scf.if %18 -> (i64) {
      %50 = func.call @_ZNKSt6vectorIiSaIiEE8max_sizeEv(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>) -> i64
      scf.yield %50 : i64
    } else {
      scf.yield %15 : i64
    }
    %20 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>) -> !llvm.ptr
    %21 = llvm.getelementptr %20[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>
    %22 = llvm.load %21 : !llvm.ptr -> memref<?xi32>
    %23 = llvm.getelementptr %21[1] : (!llvm.ptr) -> !llvm.ptr, memref<?xi32>
    %24 = llvm.load %23 : !llvm.ptr -> memref<?xi32>
    %cast_4 = memref.cast %alloca : memref<1x1xmemref<?xi32>> to memref<?x1xmemref<?xi32>>
    call @_ZNSt6vectorIiSaIiEE5beginEv(%arg0, %cast_4) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>, memref<?x1xmemref<?xi32>>) -> ()
    %25 = call @_ZN9__gnu_cxxmiIPiSt6vectorIiSaIiEEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS8_SB_(%arg1, %cast_4) : (memref<?x1xmemref<?xi32>>, memref<?x1xmemref<?xi32>>) -> i64
    %26 = "polygeist.pointer2memref"(%20) : (!llvm.ptr) -> memref<?x!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>>
    %27 = call @_ZNSt12_Vector_baseIiSaIiEE11_M_allocateEm(%26, %19) : (memref<?x!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>>, i64) -> memref<?xi32>
    %28 = arith.index_cast %25 : i64 to index
    %29 = "polygeist.subindex"(%27, %28) : (memref<?xi32>, index) -> memref<?xi32>
    %30 = affine.load %arg2[0] : memref<?xi32>
    affine.store %30, %29[0] : memref<?xi32>
    %31 = call @_ZNK9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEE4baseEv(%arg1) : (memref<?x1xmemref<?xi32>>) -> memref<?xmemref<?xi32>>
    %32 = affine.load %31[0] : memref<?xmemref<?xi32>>
    %33 = call @_ZNSt12_Vector_baseIiSaIiEE19_M_get_Tp_allocatorEv(%26) : (memref<?x!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>>) -> memref<?x!llvm.struct<(struct<(i8)>)>>
    %34 = call @_ZSt14__relocate_a_1IiiENSt9enable_ifIXsr3std24__is_bitwise_relocatableIT_EE5valueEPS1_E4typeES2_S2_S2_RSaIT0_E(%22, %32, %27, %33) : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?x!llvm.struct<(struct<(i8)>)>>) -> memref<?xi32>
    %35 = "polygeist.subindex"(%34, %c1) : (memref<?xi32>, index) -> memref<?xi32>
    %36 = call @_ZNK9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEE4baseEv(%arg1) : (memref<?x1xmemref<?xi32>>) -> memref<?xmemref<?xi32>>
    %37 = affine.load %36[0] : memref<?xmemref<?xi32>>
    %38 = call @_ZNSt12_Vector_baseIiSaIiEE19_M_get_Tp_allocatorEv(%26) : (memref<?x!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>>) -> memref<?x!llvm.struct<(struct<(i8)>)>>
    %39 = call @_ZSt14__relocate_a_1IiiENSt9enable_ifIXsr3std24__is_bitwise_relocatableIT_EE5valueEPS1_E4typeES2_S2_S2_RSaIT0_E(%37, %24, %35, %38) : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?x!llvm.struct<(struct<(i8)>)>>) -> memref<?xi32>
    %40 = llvm.getelementptr %21[2] : (!llvm.ptr) -> !llvm.ptr, memref<?xi32>
    %41 = llvm.load %40 : !llvm.ptr -> memref<?xi32>
    %42 = "polygeist.memref2pointer"(%41) : (memref<?xi32>) -> !llvm.ptr
    %43 = "polygeist.memref2pointer"(%22) : (memref<?xi32>) -> !llvm.ptr
    %44 = llvm.ptrtoint %43 : !llvm.ptr to i64
    %45 = llvm.ptrtoint %42 : !llvm.ptr to i64
    %46 = arith.subi %45, %44 : i64
    %47 = arith.divsi %46, %c4_i64 : i64
    call @_ZNSt12_Vector_baseIiSaIiEE13_M_deallocateEPim(%26, %22, %47) : (memref<?x!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>>, memref<?xi32>, i64) -> ()
    llvm.store %27, %21 : memref<?xi32>, !llvm.ptr
    llvm.store %39, %23 : memref<?xi32>, !llvm.ptr
    %48 = arith.index_cast %19 : i64 to index
    %49 = "polygeist.subindex"(%27, %48) : (memref<?xi32>, index) -> memref<?xi32>
    llvm.store %49, %40 : memref<?xi32>, !llvm.ptr
    return
  }
  func.func @_ZNSt6vectorIiSaIiEE3endEv(%arg0: memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>, %arg1: memref<?x1xmemref<?xi32>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>) -> !llvm.ptr
    %1 = llvm.getelementptr %0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>
    %2 = llvm.getelementptr %1[1] : (!llvm.ptr) -> !llvm.ptr, memref<?xi32>
    %3 = llvm.load %2 : !llvm.ptr -> memref<?xi32>
    affine.store %3, %arg1[0, 0] : memref<?x1xmemref<?xi32>>
    return
  }
  func.func @_ZNSt12_Vector_baseIiSaIiEE12_Vector_implC1Ev(%arg0: memref<?x!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>>) -> !llvm.ptr
    %1 = llvm.getelementptr %0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>
    %2 = "polygeist.pointer2memref"(%1) : (!llvm.ptr) -> memref<?x3xmemref<?xi32>>
    call @_ZNSt12_Vector_baseIiSaIiEE17_Vector_impl_dataC1Ev(%2) : (memref<?x3xmemref<?xi32>>) -> ()
    return
  }
  func.func @_ZNSt15__new_allocatorIiE9constructIiJRKiEEEvPT_DpOT0_(%arg0: memref<?x!llvm.struct<(i8)>>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = affine.load %arg2[0] : memref<?xi32>
    affine.store %0, %arg1[0] : memref<?xi32>
    return
  }
  func.func @_ZNKSt6vectorIiSaIiEE12_M_check_lenEmPKc(%arg0: memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>, %arg1: i64, %arg2: memref<?xi8>) -> i64 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %true = arith.constant true
    %0 = llvm.mlir.undef : i64
    %alloca = memref.alloca() : memref<1xi64>
    %cast = memref.cast %alloca : memref<1xi64> to memref<?xi64>
    affine.store %0, %alloca[0] : memref<1xi64>
    %alloca_0 = memref.alloca() : memref<1xi64>
    %cast_1 = memref.cast %alloca_0 : memref<1xi64> to memref<?xi64>
    affine.store %arg1, %alloca_0[0] : memref<1xi64>
    %1 = call @_ZNKSt6vectorIiSaIiEE8max_sizeEv(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>) -> i64
    %2 = call @_ZNKSt6vectorIiSaIiEE4sizeEv(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>) -> i64
    %3 = arith.subi %1, %2 : i64
    %4 = affine.load %alloca_0[0] : memref<1xi64>
    %5 = arith.cmpi slt, %3, %4 : i64
    scf.if %5 {
      func.call @_ZSt20__throw_length_errorPKc(%arg2) : (memref<?xi8>) -> ()
    }
    %6 = call @_ZNKSt6vectorIiSaIiEE4sizeEv(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>) -> i64
    %7 = call @_ZNKSt6vectorIiSaIiEE4sizeEv(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>) -> i64
    affine.store %7, %alloca[0] : memref<1xi64>
    %alloca_2 = memref.alloca() : memref<memref<?xi64>>
    %8 = affine.load %alloca_0[0] : memref<1xi64>
    %9 = arith.cmpi slt, %7, %8 : i64
    %10 = arith.cmpi sge, %7, %8 : i64
    scf.if %9 {
      affine.store %cast_1, %alloca_2[] : memref<memref<?xi64>>
    }
    scf.if %10 {
      affine.store %cast, %alloca_2[] : memref<memref<?xi64>>
    }
    %11 = affine.load %alloca_2[] : memref<memref<?xi64>>
    %12 = affine.load %11[0] : memref<?xi64>
    %13 = arith.addi %6, %12 : i64
    %14 = call @_ZNKSt6vectorIiSaIiEE4sizeEv(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>) -> i64
    %15 = arith.cmpi slt, %13, %14 : i64
    %16 = scf.if %15 -> (i1) {
      scf.yield %true : i1
    } else {
      %18 = func.call @_ZNKSt6vectorIiSaIiEE8max_sizeEv(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>) -> i64
      %19 = arith.cmpi sgt, %13, %18 : i64
      scf.yield %19 : i1
    }
    %17 = scf.if %16 -> (i64) {
      %18 = func.call @_ZNKSt6vectorIiSaIiEE8max_sizeEv(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>) -> i64
      scf.yield %18 : i64
    } else {
      scf.yield %13 : i64
    }
    return %17 : i64
  }
  func.func @_ZN9__gnu_cxxmiIPiSt6vectorIiSaIiEEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS8_SB_(%arg0: memref<?x1xmemref<?xi32>>, %arg1: memref<?x1xmemref<?xi32>>) -> i64 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %c4_i64 = arith.constant 4 : i64
    %0 = call @_ZNK9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEE4baseEv(%arg0) : (memref<?x1xmemref<?xi32>>) -> memref<?xmemref<?xi32>>
    %1 = affine.load %0[0] : memref<?xmemref<?xi32>>
    %2 = call @_ZNK9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEE4baseEv(%arg1) : (memref<?x1xmemref<?xi32>>) -> memref<?xmemref<?xi32>>
    %3 = affine.load %2[0] : memref<?xmemref<?xi32>>
    %4 = "polygeist.memref2pointer"(%1) : (memref<?xi32>) -> !llvm.ptr
    %5 = "polygeist.memref2pointer"(%3) : (memref<?xi32>) -> !llvm.ptr
    %6 = llvm.ptrtoint %5 : !llvm.ptr to i64
    %7 = llvm.ptrtoint %4 : !llvm.ptr to i64
    %8 = arith.subi %7, %6 : i64
    %9 = arith.divsi %8, %c4_i64 : i64
    return %9 : i64
  }
  func.func @_ZNSt6vectorIiSaIiEE5beginEv(%arg0: memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>, %arg1: memref<?x1xmemref<?xi32>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>) -> !llvm.ptr
    %1 = llvm.getelementptr %0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>
    %2 = llvm.load %1 : !llvm.ptr -> memref<?xi32>
    affine.store %2, %arg1[0, 0] : memref<?x1xmemref<?xi32>>
    return
  }
  func.func @_ZNSt12_Vector_baseIiSaIiEE11_M_allocateEm(%arg0: memref<?x!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>>, %arg1: i64) -> memref<?xi32> attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %c0_i64 = arith.constant 0 : i64
    %0 = arith.cmpi ne, %arg1, %c0_i64 : i64
    %1 = scf.if %0 -> (memref<?xi32>) {
      %2 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>>) -> !llvm.ptr
      %3 = llvm.mlir.zero : !llvm.ptr
      %4 = llvm.icmp "ne" %2, %3 : !llvm.ptr
      %5 = arith.select %4, %2, %3 : !llvm.ptr
      %6 = "polygeist.pointer2memref"(%5) : (!llvm.ptr) -> memref<?x!llvm.struct<(struct<(i8)>)>>
      %7 = func.call @_ZNSt16allocator_traitsISaIiEE8allocateERS0_m(%6, %arg1) : (memref<?x!llvm.struct<(struct<(i8)>)>>, i64) -> memref<?xi32>
      scf.yield %7 : memref<?xi32>
    } else {
      %2 = llvm.mlir.zero : !llvm.ptr
      %3 = "polygeist.pointer2memref"(%2) : (!llvm.ptr) -> memref<?xi32>
      scf.yield %3 : memref<?xi32>
    }
    return %1 : memref<?xi32>
  }
  func.func @_ZNSt6vectorIiSaIiEE15_S_use_relocateEv() -> i8 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %c1_i8 = arith.constant 1 : i8
    return %c1_i8 : i8
  }
  func.func @_ZNSt6vectorIiSaIiEE11_S_relocateEPiS2_S2_RS0_(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?x!llvm.struct<(struct<(i8)>)>>) -> memref<?xi32> attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = call @_ZSt14__relocate_a_1IiiENSt9enable_ifIXsr3std24__is_bitwise_relocatableIT_EE5valueEPS1_E4typeES2_S2_S2_RSaIT0_E(%arg0, %arg1, %arg2, %arg3) : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?x!llvm.struct<(struct<(i8)>)>>) -> memref<?xi32>
    return %0 : memref<?xi32>
  }
  func.func @_ZNK9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEE4baseEv(%arg0: memref<?x1xmemref<?xi32>>) -> memref<?xmemref<?xi32>> attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %c0 = arith.constant 0 : index
    %0 = "polygeist.subindex"(%arg0, %c0) : (memref<?x1xmemref<?xi32>>, index) -> memref<1xmemref<?xi32>>
    %1 = "polygeist.subindex"(%0, %c0) : (memref<1xmemref<?xi32>>, index) -> memref<?xmemref<?xi32>>
    return %1 : memref<?xmemref<?xi32>>
  }
  func.func @_ZNSt12_Vector_baseIiSaIiEE19_M_get_Tp_allocatorEv(%arg0: memref<?x!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>>) -> memref<?x!llvm.struct<(struct<(i8)>)>> attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>>) -> !llvm.ptr
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.icmp "ne" %0, %1 : !llvm.ptr
    %3 = arith.select %2, %0, %1 : !llvm.ptr
    %4 = "polygeist.pointer2memref"(%3) : (!llvm.ptr) -> memref<?x!llvm.struct<(struct<(i8)>)>>
    return %4 : memref<?x!llvm.struct<(struct<(i8)>)>>
  }
  func.func @_ZNSt12_Vector_baseIiSaIiEE13_M_deallocateEPim(%arg0: memref<?x!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>>, %arg1: memref<?xi32>, %arg2: i64) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = "polygeist.memref2pointer"(%arg1) : (memref<?xi32>) -> !llvm.ptr
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.icmp "ne" %0, %1 : !llvm.ptr
    scf.if %2 {
      %3 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>>) -> !llvm.ptr
      %4 = llvm.icmp "ne" %3, %1 : !llvm.ptr
      %5 = arith.select %4, %3, %1 : !llvm.ptr
      %6 = "polygeist.pointer2memref"(%5) : (!llvm.ptr) -> memref<?x!llvm.struct<(struct<(i8)>)>>
      func.call @_ZNSt16allocator_traitsISaIiEE10deallocateERS0_Pim(%6, %arg1, %arg2) : (memref<?x!llvm.struct<(struct<(i8)>)>>, memref<?xi32>, i64) -> ()
    }
    return
  }
  func.func @_ZN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEC1ERKS1_(%arg0: memref<?x1xmemref<?xi32>>, %arg1: memref<?xmemref<?xi32>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = affine.load %arg1[0] : memref<?xmemref<?xi32>>
    affine.store %0, %arg0[0, 0] : memref<?x1xmemref<?xi32>>
    return
  }
  func.func @_ZNSaIiEC1Ev(%arg0: memref<?x!llvm.struct<(struct<(i8)>)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    return
  }
  func.func @_ZNSt12_Vector_baseIiSaIiEE17_Vector_impl_dataC1Ev(%arg0: memref<?x3xmemref<?xi32>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr) -> memref<?xi32>
    affine.store %1, %arg0[0, 0] : memref<?x3xmemref<?xi32>>
    affine.store %1, %arg0[0, 1] : memref<?x3xmemref<?xi32>>
    affine.store %1, %arg0[0, 2] : memref<?x3xmemref<?xi32>>
    return
  }
  func.func @_ZNKSt6vectorIiSaIiEE8max_sizeEv(%arg0: memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>) -> i64 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %c2305843009213693951_i64 = arith.constant 2305843009213693951 : i64
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>) -> !llvm.ptr
    %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr) -> memref<?x!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>>
    %2 = call @_ZNKSt12_Vector_baseIiSaIiEE19_M_get_Tp_allocatorEv(%1) : (memref<?x!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>>) -> memref<?x!llvm.struct<(struct<(i8)>)>>
    %alloca = memref.alloca() : memref<1xi64>
    %cast = memref.cast %alloca : memref<1xi64> to memref<?xi64>
    %3 = llvm.mlir.undef : i64
    affine.store %3, %alloca[0] : memref<1xi64>
    %alloca_0 = memref.alloca() : memref<1xi64>
    %cast_1 = memref.cast %alloca_0 : memref<1xi64> to memref<?xi64>
    affine.store %c2305843009213693951_i64, %alloca_0[0] : memref<1xi64>
    affine.store %c2305843009213693951_i64, %alloca[0] : memref<1xi64>
    %alloca_2 = memref.alloca() : memref<memref<?xi64>>
    %4 = affine.load %alloca_0[0] : memref<1xi64>
    %5 = arith.cmpi sgt, %4, %c2305843009213693951_i64 : i64
    %6 = arith.cmpi sle, %4, %c2305843009213693951_i64 : i64
    scf.if %5 {
      affine.store %cast, %alloca_2[] : memref<memref<?xi64>>
    }
    scf.if %6 {
      affine.store %cast_1, %alloca_2[] : memref<memref<?xi64>>
    }
    %7 = affine.load %alloca_2[] : memref<memref<?xi64>>
    %8 = affine.load %7[0] : memref<?xi64>
    return %8 : i64
  }
  func.func @_ZNKSt6vectorIiSaIiEE4sizeEv(%arg0: memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>) -> i64 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %c4_i64 = arith.constant 4 : i64
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>)>>) -> !llvm.ptr
    %1 = llvm.getelementptr %0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>
    %2 = llvm.getelementptr %1[1] : (!llvm.ptr) -> !llvm.ptr, memref<?xi32>
    %3 = llvm.load %2 : !llvm.ptr -> memref<?xi32>
    %4 = llvm.load %1 : !llvm.ptr -> memref<?xi32>
    %5 = "polygeist.memref2pointer"(%3) : (memref<?xi32>) -> !llvm.ptr
    %6 = "polygeist.memref2pointer"(%4) : (memref<?xi32>) -> !llvm.ptr
    %7 = llvm.ptrtoint %6 : !llvm.ptr to i64
    %8 = llvm.ptrtoint %5 : !llvm.ptr to i64
    %9 = arith.subi %8, %7 : i64
    %10 = arith.divsi %9, %c4_i64 : i64
    return %10 : i64
  }
  func.func private @_ZSt20__throw_length_errorPKc(memref<?xi8>) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @_ZSt3maxImERKT_S2_S2_(%arg0: memref<?xi64>, %arg1: memref<?xi64>) -> memref<?xi64> attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %alloca = memref.alloca() : memref<memref<?xi64>>
    %0 = affine.load %arg0[0] : memref<?xi64>
    %1 = affine.load %arg1[0] : memref<?xi64>
    %2 = arith.cmpi slt, %0, %1 : i64
    %3 = arith.cmpi sge, %0, %1 : i64
    scf.if %2 {
      affine.store %arg1, %alloca[] : memref<memref<?xi64>>
    }
    scf.if %3 {
      affine.store %arg0, %alloca[] : memref<memref<?xi64>>
    }
    %4 = affine.load %alloca[] : memref<memref<?xi64>>
    return %4 : memref<?xi64>
  }
  func.func @_ZNSt16allocator_traitsISaIiEE8allocateERS0_m(%arg0: memref<?x!llvm.struct<(struct<(i8)>)>>, %arg1: i64) -> memref<?xi32> attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(struct<(i8)>)>>) -> !llvm.ptr
    %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr) -> memref<?x!llvm.struct<(i8)>>
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = "polygeist.pointer2memref"(%2) : (!llvm.ptr) -> memref<?xi8>
    %4 = call @_ZNSt15__new_allocatorIiE8allocateEmPKv(%1, %arg1, %3) : (memref<?x!llvm.struct<(i8)>>, i64, memref<?xi8>) -> memref<?xi32>
    return %4 : memref<?xi32>
  }
  func.func @_ZNSt6vectorIiSaIiEE19_S_nothrow_relocateESt17integral_constantIbLb1EE(%arg0: !llvm.struct<(i8)>) -> i8 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %c1_i8 = arith.constant 1 : i8
    return %c1_i8 : i8
  }
  func.func @_ZNSt17integral_constantIbLb1EEC1EOS0_(%arg0: memref<?x!llvm.struct<(i8)>>, %arg1: memref<?x!llvm.struct<(i8)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    return
  }
  func.func @_ZSt12__relocate_aIPiS0_SaIiEET0_T_S3_S2_RT1_(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?x!llvm.struct<(struct<(i8)>)>>) -> memref<?xi32> attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = call @_ZSt14__relocate_a_1IiiENSt9enable_ifIXsr3std24__is_bitwise_relocatableIT_EE5valueEPS1_E4typeES2_S2_S2_RSaIT0_E(%arg0, %arg1, %arg2, %arg3) : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?x!llvm.struct<(struct<(i8)>)>>) -> memref<?xi32>
    return %0 : memref<?xi32>
  }
  func.func @_ZNSt16allocator_traitsISaIiEE10deallocateERS0_Pim(%arg0: memref<?x!llvm.struct<(struct<(i8)>)>>, %arg1: memref<?xi32>, %arg2: i64) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(struct<(i8)>)>>) -> !llvm.ptr
    %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr) -> memref<?x!llvm.struct<(i8)>>
    call @_ZNSt15__new_allocatorIiE10deallocateEPim(%1, %arg1, %arg2) : (memref<?x!llvm.struct<(i8)>>, memref<?xi32>, i64) -> ()
    return
  }
  func.func @_ZNSt15__new_allocatorIiEC1Ev(%arg0: memref<?x!llvm.struct<(i8)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    return
  }
  func.func @_ZNSt6vectorIiSaIiEE11_S_max_sizeERKS0_(%arg0: memref<?x!llvm.struct<(struct<(i8)>)>>) -> i64 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %c2305843009213693951_i64 = arith.constant 2305843009213693951 : i64
    %alloca = memref.alloca() : memref<1xi64>
    %cast = memref.cast %alloca : memref<1xi64> to memref<?xi64>
    %0 = llvm.mlir.undef : i64
    affine.store %0, %alloca[0] : memref<1xi64>
    %alloca_0 = memref.alloca() : memref<1xi64>
    %cast_1 = memref.cast %alloca_0 : memref<1xi64> to memref<?xi64>
    affine.store %c2305843009213693951_i64, %alloca_0[0] : memref<1xi64>
    affine.store %c2305843009213693951_i64, %alloca[0] : memref<1xi64>
    %alloca_2 = memref.alloca() : memref<memref<?xi64>>
    %1 = affine.load %alloca_0[0] : memref<1xi64>
    %2 = arith.cmpi sgt, %1, %c2305843009213693951_i64 : i64
    %3 = arith.cmpi sle, %1, %c2305843009213693951_i64 : i64
    scf.if %2 {
      affine.store %cast, %alloca_2[] : memref<memref<?xi64>>
    }
    scf.if %3 {
      affine.store %cast_1, %alloca_2[] : memref<memref<?xi64>>
    }
    %4 = affine.load %alloca_2[] : memref<memref<?xi64>>
    %5 = affine.load %4[0] : memref<?xi64>
    return %5 : i64
  }
  func.func @_ZNKSt12_Vector_baseIiSaIiEE19_M_get_Tp_allocatorEv(%arg0: memref<?x!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>>) -> memref<?x!llvm.struct<(struct<(i8)>)>> attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(!llvm.struct<(struct<(struct<(i8)>)>, !llvm.struct<(memref<?xi32>, memref<?xi32>, memref<?xi32>)>)>)>>) -> !llvm.ptr
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.icmp "ne" %0, %1 : !llvm.ptr
    %3 = arith.select %2, %0, %1 : !llvm.ptr
    %4 = "polygeist.pointer2memref"(%3) : (!llvm.ptr) -> memref<?x!llvm.struct<(struct<(i8)>)>>
    return %4 : memref<?x!llvm.struct<(struct<(i8)>)>>
  }
  func.func @_ZNSt15__new_allocatorIiE8allocateEmPKv(%arg0: memref<?x!llvm.struct<(i8)>>, %arg1: i64, %arg2: memref<?xi8>) -> memref<?xi32> attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %c4611686018427387903_i64 = arith.constant 4611686018427387903 : i64
    %c4_i64 = arith.constant 4 : i64
    %c2305843009213693951_i64 = arith.constant 2305843009213693951 : i64
    %0 = arith.cmpi sgt, %arg1, %c2305843009213693951_i64 : i64
    scf.if %0 {
      %5 = arith.cmpi sgt, %arg1, %c4611686018427387903_i64 : i64
      scf.if %5 {
        func.call @_ZSt28__throw_bad_array_new_lengthv() : () -> ()
      }
      func.call @_ZSt17__throw_bad_allocv() : () -> ()
    }
    %1 = arith.muli %arg1, %c4_i64 : i64
    %2 = arith.index_cast %1 : i64 to index
    %alloc = memref.alloc(%2) : memref<?xi8>
    %3 = "polygeist.memref2pointer"(%alloc) : (memref<?xi8>) -> !llvm.ptr
    %4 = "polygeist.pointer2memref"(%3) : (!llvm.ptr) -> memref<?xi32>
    return %4 : memref<?xi32>
  }
  func.func @_ZSt14__relocate_a_1IiiENSt9enable_ifIXsr3std24__is_bitwise_relocatableIT_EE5valueEPS1_E4typeES2_S2_S2_RSaIT0_E(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?x!llvm.struct<(struct<(i8)>)>>) -> memref<?xi32> attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %c4_i64 = arith.constant 4 : i64
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %0 = "polygeist.memref2pointer"(%arg1) : (memref<?xi32>) -> !llvm.ptr
    %1 = "polygeist.memref2pointer"(%arg0) : (memref<?xi32>) -> !llvm.ptr
    %2 = llvm.ptrtoint %1 : !llvm.ptr to i64
    %3 = llvm.ptrtoint %0 : !llvm.ptr to i64
    %4 = arith.subi %3, %2 : i64
    %5 = arith.divsi %4, %c4_i64 : i64
    %6 = arith.cmpi sgt, %5, %c0_i64 : i64
    scf.if %6 {
      %9 = "polygeist.memref2pointer"(%arg2) : (memref<?xi32>) -> !llvm.ptr
      %10 = arith.muli %5, %c4_i64 : i64
      %11 = arith.index_cast %10 : i64 to index
      scf.for %arg4 = %c0 to %11 step %c1 {
        %12 = arith.index_cast %arg4 : index to i32
        %13 = llvm.getelementptr %1[%12] : (!llvm.ptr, i32) -> !llvm.ptr, i8
        %14 = llvm.load %13 : !llvm.ptr -> i8
        %15 = llvm.getelementptr %9[%12] : (!llvm.ptr, i32) -> !llvm.ptr, i8
        llvm.store %14, %15 : i8, !llvm.ptr
      }
    }
    %7 = arith.index_cast %5 : i64 to index
    %8 = "polygeist.subindex"(%arg2, %7) : (memref<?xi32>, index) -> memref<?xi32>
    return %8 : memref<?xi32>
  }
  func.func @_ZSt12__niter_baseIPiET_S1_(%arg0: memref<?xi32>) -> memref<?xi32> attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    return %arg0 : memref<?xi32>
  }
  func.func @_ZNSt15__new_allocatorIiE10deallocateEPim(%arg0: memref<?x!llvm.struct<(i8)>>, %arg1: memref<?xi32>, %arg2: i64) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = "polygeist.memref2pointer"(%arg1) : (memref<?xi32>) -> !llvm.ptr
    %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr) -> memref<?xi8>
    memref.dealloc %1 : memref<?xi8>
    return
  }
  func.func @_ZNSt16allocator_traitsISaIiEE8max_sizeERKS0_(%arg0: memref<?x!llvm.struct<(struct<(i8)>)>>) -> i64 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %c2305843009213693951_i64 = arith.constant 2305843009213693951 : i64
    return %c2305843009213693951_i64 : i64
  }
  func.func @_ZSt3minImERKT_S2_S2_(%arg0: memref<?xi64>, %arg1: memref<?xi64>) -> memref<?xi64> attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %alloca = memref.alloca() : memref<memref<?xi64>>
    %0 = affine.load %arg1[0] : memref<?xi64>
    %1 = affine.load %arg0[0] : memref<?xi64>
    %2 = arith.cmpi slt, %0, %1 : i64
    %3 = arith.cmpi sge, %0, %1 : i64
    scf.if %2 {
      affine.store %arg1, %alloca[] : memref<memref<?xi64>>
    }
    scf.if %3 {
      affine.store %arg0, %alloca[] : memref<memref<?xi64>>
    }
    %4 = affine.load %alloca[] : memref<memref<?xi64>>
    return %4 : memref<?xi64>
  }
  func.func @_ZNKSt15__new_allocatorIiE11_M_max_sizeEv(%arg0: memref<?x!llvm.struct<(i8)>>) -> i64 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %c2305843009213693951_i64 = arith.constant 2305843009213693951 : i64
    return %c2305843009213693951_i64 : i64
  }
  func.func private @_ZSt28__throw_bad_array_new_lengthv() attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @_ZSt17__throw_bad_allocv() attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @_ZNKSt15__new_allocatorIiE8max_sizeEv(%arg0: memref<?x!llvm.struct<(i8)>>) -> i64 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %c2305843009213693951_i64 = arith.constant 2305843009213693951 : i64
    return %c2305843009213693951_i64 : i64
  }
}
```

running self converted model
```
Traceback (most recent call last):
  File "/home/sylvex/mnist_export/model_run.py", line 26, in <module>
    session = ort.InferenceSession(quant_modle_path)
  File "/home/sylvex/.local/lib/python3.10/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 419, in __init__
    self._create_inference_session(providers, provider_options, disabled_optimizers)
  File "/home/sylvex/.local/lib/python3.10/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 472, in _create_inference_session
    sess = C.InferenceSession(session_options, self._model_path, True, self._read_config_from_model)
onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph: [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from mnist_model_int8.onnx failed:This is an invalid model. Type Error: Type 'tensor(int8)' of input parameter (conv1.weight) of operator (Conv) in node (/conv1/Conv) is invalid.
```

```
// onnx/defs/data_type_utils.cc

TypesWrapper::TypesWrapper() {

// DataType strings. These should match the DataTypes defined in onnx.proto
type_str_to_tensor_data_type_["float"] = TensorProto_DataType_FLOAT;
type_str_to_tensor_data_type_["float16"] = TensorProto_DataType_FLOAT16;
type_str_to_tensor_data_type_["bfloat16"] = TensorProto_DataType_BFLOAT16;
type_str_to_tensor_data_type_["double"] = TensorProto_DataType_DOUBLE;
type_str_to_tensor_data_type_["int8"] = TensorProto_DataType_INT8;
type_str_to_tensor_data_type_["int16"] = TensorProto_DataType_INT16;
type_str_to_tensor_data_type_["int32"] = TensorProto_DataType_INT32;
type_str_to_tensor_data_type_["int64"] = TensorProto_DataType_INT64;
type_str_to_tensor_data_type_["uint8"] = TensorProto_DataType_UINT8;
type_str_to_tensor_data_type_["uint16"] = TensorProto_DataType_UINT16;
type_str_to_tensor_data_type_["uint32"] = TensorProto_DataType_UINT32;
type_str_to_tensor_data_type_["uint64"] = TensorProto_DataType_UINT64;
type_str_to_tensor_data_type_["complex64"] = TensorProto_DataType_COMPLEX64;
type_str_to_tensor_data_type_["complex128"] = TensorProto_DataType_COMPLEX128;
type_str_to_tensor_data_type_["string"] = TensorProto_DataType_STRING;
type_str_to_tensor_data_type_["bool"] = TensorProto_DataType_BOOL;
type_str_to_tensor_data_type_["float8e4m3fn"] = TensorProto_DataType_FLOAT8E4M3FN;
type_str_to_tensor_data_type_["float8e4m3fnuz"] = TensorProto_DataType_FLOAT8E4M3FNUZ;
type_str_to_tensor_data_type_["float8e5m2"] = TensorProto_DataType_FLOAT8E5M2;
type_str_to_tensor_data_type_["float8e5m2fnuz"] = TensorProto_DataType_FLOAT8E5M2FNUZ;
type_str_to_tensor_data_type_["uint4"] = TensorProto_DataType_UINT4;
type_str_to_tensor_data_type_["int4"] = TensorProto_DataType_INT4;
```