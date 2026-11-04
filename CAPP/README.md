# Undistortion Standalone Project

高性能图像去畸变处理库，采用AVX2 SIMD优化，支持大尺寸图像处理。

## 项目结构

```
undistortion_standalone/
├── CMakeLists.txt              # CMake构建配置
├── README.md                   # 项目说明文档
├── include/                    # 头文件目录
│   ├── CommonTypes.h          # 通用类型定义
│   ├── Logger.h               # 日志系统
│   └── Undistortion.h         # 去畸变接口
├── src/                       # 源代码目录
│   ├── Logger.cpp             # 日志实现
│   └── Undistortion.cpp       # 去畸变核心实现
├── test/                      # 测试代码目录
│   └── test_undistortion.cpp   # 主测试程序
├── build/                     # 构建输出目录
├── out/                       # 输出目录
└── document/                  # 文档目录
    ├── undistor_prompt.md     # 去畸变提示文档
    ├── system_improvement_plan.md  # 系统改进计划
    └── log_system_design.md   # 日志系统设计文档
```

## 特性

- ✅ **高性能**: AVX2 SIMD优化，8像素并行处理
- ✅ **内存安全**: 完善的边界检查和异常处理
- ✅ **架构简洁**: 高效的流线型设计
- ✅ **易于集成**: 简单的头文件接口
- ✅ **详细日志**: 完整的调试和性能监控

## 性能指标

- **处理速度**: ~70ms/图像 (5120×4096像素)
- **SIMD优化**: AVX2 8像素并行处理
- **内存效率**: 优化的SoA数据布局
- **性能提升**: 相比原实现提升57%

## 快速开始

### 构建项目

```bash
# 创建构建目录
mkdir build && cd build

# 配置CMake
cmake ..

# 编译Release版本
cmake --build . --config Release
```

### 运行测试

```bash
# 运行去畸变测试程序
./bin/test_undistortion.exe
```

## 算法优化

### 数据结构优化
- 使用`std::vector<float>`替代OpenCV Mat
- SoA (Structure of Arrays) 数据布局
- 直接内存访问，避免开销

### SIMD优化
- AVX2 gather指令优化随机内存访问
- 8像素并行处理
- 自动CPU特性检测

### 内存管理
- RAII自动内存管理
- 异常安全的资源清理
- 预分配缓冲区避免运行时分配

## 依赖项

- **C++17**: 现代C++标准
- **OpenCV**: 图像I/O操作
- **AVX2**: SIMD指令集 (自动检测)

## 许可证

本项目的所有修改和优化均基于原始实现进行性能和安全改进。

## 更新日志

### v2.5.1 (2025-11-13)
- ✅ 修复SIMD内存越界问题
- ✅ 添加完善的边界检查
- ✅ 优化矩阵加载性能
- ✅ 移除prefetch以提升性能
- ✅ 优化代码架构，提升57%性能
- ✅ 重组项目结构，提高可维护性