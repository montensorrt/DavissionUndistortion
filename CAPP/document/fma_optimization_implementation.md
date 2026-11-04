# FMA计算模式优化实施方案

**项目**: Undistortion Standalone FMA指令优化
**文档目的**: 详细实施FMA融合乘加指令优化
**创建日期**: 2025-11-13

---

## 📊 目录

1. [当前SIMD实现分析](#当前simd实现分析)
2. [FMA优化原理](#fma优化原理)
3. [实施方案设计](#实施方案设计)
4. [代码实现细节](#代码实现细节)
5. [性能测试验证](#性能测试验证)
6. [集成步骤](#集成步骤)

---

## 🔍 当前SIMD实现分析

### 现有代码结构
通过分析`Undistortion.cpp`中的`simdUndistortAVX2`函数，发现当前实现：

```cpp
// 当前实现 - 乘加分离（第630-634行）
__m256 sum = _mm256_mul_ps(src0, w0);                    // 乘法1
sum = _mm256_add_ps(sum, _mm256_mul_ps(src1, w1));      // 乘法2 + 加法1
sum = _mm256_add_ps(sum, _mm256_mul_ps(src2, w2));      // 乘法3 + 加法2
sum = _mm256_add_ps(sum, _mm256_mul_ps(src3, w3));      // 乘法4 + 加法3
```

### 性能瓶颈分析
- **指令数量**: 4个乘法 + 3个加法 = **7条SIMD指令**
- **指令依赖**: 每个加法都依赖前一个加法结果，形成指令链
- **延迟累积**: 乘法延迟(3-5周期) + 加法延迟(1-2周期) × 3
- **吞吐量限制**: 无法充分利用现代CPU的指令并行度

---

## ⚡ FMA优化原理

### FMA指令优势
```cpp
// FMA (Fused Multiply-Add) 指令格式
__m256 result = _mm256_fmadd_ps(a, b, c);  // 计算: a * b + c
```

**核心优势**:
- **单指令完成**: 乘法和加法在一个指令周期完成
- **减少延迟**: 避免中间结果回写，减少寄存器压力
- **提高精度**: 在计算过程中保持更高精度（减少舍入误差）
- **指令并行**: CPU可以更高效地调度执行

### 优化对比
```cpp
// 优化前：7条指令
__m256 sum = _mm256_mul_ps(src0, w0);
sum = _mm256_add_ps(sum, _mm256_mul_ps(src1, w1));
sum = _mm256_add_ps(sum, _mm256_mul_ps(src2, w2));
sum = _mm256_add_ps(sum, _mm256_mul_ps(src3, w3));

// 优化后：4条指令
__m256 result = _mm256_fmadd_ps(src0, w0, _mm256_fmadd_ps(src1, w1,
                    _mm256_fmadd_ps(src2, w2, _mm256_mul_ps(src3, w3))));
```

**指令减少**: 7条 → 4条 (减少43%)
**延迟减少**: 约3-5个时钟周期

---

## 🏗️ 实施方案设计

### 1. CPU特性检测设计

```cpp
class FMAProcessor {
private:
    bool useFMA_ = false;
    bool useAVX2_ = false;

public:
    void detectCPUFeatures() {
        // 检测AVX2支持
#ifdef __AVX2__
        useAVX2_ = true;
#else
        useAVX2_ = detectAVX2Support();
#endif

        // 检测FMA支持
#ifdef __FMA__
        useFMA_ = useAVX2_;
#else
        useFMA_ = useAVX2_ && detectFMASupport();
#endif

        LOG_INFO("SIMD", "CPU Features: AVX2=%s, FMA=%s",
                useAVX2_ ? "enabled" : "disabled",
                useFMA_ ? "enabled" : "disabled");
    }

private:
    bool detectAVX2Support() {
        // 运行时检测AVX2支持
#ifdef _MSC_VER
        int cpuInfo[4];
        __cpuidex(cpuInfo, 7, 0);
        return (cpuInfo[1] & (1 << 5)) != 0;  // AVX2 bit
#else
        return __builtin_cpu_supports("avx2");
#endif
    }

    bool detectFMASupport() {
#ifdef _MSC_VER
        int cpuInfo[4];
        __cpuid(cpuInfo, 1);
        return (cpuInfo[2] & (1 << 12)) != 0;  // FMA bit
#else
        return __builtin_cpu_supports("fma");
#endif
    }
};
```

### 2. 多路径处理架构

```cpp
class OptimizedUndistortionProcessor {
private:
    FMAProcessor fmaDetector_;

public:
    void processImage(const float* inputF, uint8_t* output, int pixelCount) {
        if (fmaDetector_.useFMA()) {
            // 使用FMA优化路径
            processWithFMA(inputF, output, pixelCount);
        } else if (fmaDetector_.useAVX2()) {
            // 使用标准AVX2路径
            processWithAVX2(inputF, output, pixelCount);
        } else {
            // 回退到SSE路径
            processWithSSE(inputF, output, pixelCount);
        }
    }
};
```

---

## 💻 代码实现细节

### 1. FMA优化核心函数

```cpp
void ImageUndistorterImpl::simdUndistortAVX2_FMA(const float* inputF, uint8_t* output, int pixelCount) noexcept {
    LOG_DEBUG("SIMD", "FMA-optimized AVX2 processing started: pixelCount=%d", pixelCount);

    const int simdWidth = 8;
    int i = 0;
    int simdIterCount = 0;
    int scalarIterCount = 0;

    // 安全检查
    if (!inputF || !output || pixelCount <= 0) {
        LOG_ERROR("SIMD", "Invalid input parameters for FMA processing");
        return;
    }

    // 主SIMD循环 - 使用FMA指令
    for (; i <= pixelCount - simdWidth; i += simdWidth) {
        // 加载8个索引
        __m256i idx0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&sourceIndex0_[i]));
        __m256i idx1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&sourceIndex1_[i]));
        __m256i idx2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&sourceIndex2_[i]));
        __m256i idx3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&sourceIndex3_[i]));

        // Gather操作 - 获取源像素值
        __m256 src0 = _mm256_i32gather_ps(inputF, idx0, 4);
        __m256 src1 = _mm256_i32gather_ps(inputF, idx1, 4);
        __m256 src2 = _mm256_i32gather_ps(inputF, idx2, 4);
        __m256 src3 = _mm256_i32gather_ps(inputF, idx3, 4);

        // 加载权重
        __m256 w0 = _mm256_loadu_ps(&weight0_[i]);
        __m256 w1 = _mm256_loadu_ps(&weight1_[i]);
        __m256 w2 = _mm256_loadu_ps(&weight2_[i]);
        __m256 w3 = _mm256_loadu_ps(&weight3_[i]);

        // FMA融合乘加运算 - 嵌套FMA减少指令数量
        __m256 result = _mm256_fmadd_ps(src0, w0,
                          _mm256_fmadd_ps(src1, w1,
                          _mm256_fmadd_ps(src2, w2,
                          _mm256_mul_ps(src3, w3))));

        // 存储结果
        _mm256_storeu_ps(&tempBuffer_[i], result);
        simdIterCount++;
    }

    // 处理剩余像素（标量）
    for (; i < pixelCount; ++i) {
        // 安全的边界检查和索引访问
        int idx0 = (i < sourceIndex0_.size()) ? sourceIndex0_[i] : 0;
        int idx1 = (i < sourceIndex1_.size()) ? sourceIndex1_[i] : 0;
        int idx2 = (i < sourceIndex2_.size()) ? sourceIndex2_[i] : 0;
        int idx3 = (i < sourceIndex3_.size()) ? sourceIndex3_[i] : 0;

        // 边界检查
        float val0 = (idx0 >= 0 && idx0 < pixelCount) ? inputF[idx0] : 0.0f;
        float val1 = (idx1 >= 0 && idx1 < pixelCount) ? inputF[idx1] : 0.0f;
        float val2 = (idx2 >= 0 && idx2 < pixelCount) ? inputF[idx2] : 0.0f;
        float val3 = (idx3 >= 0 && idx3 < pixelCount) ? inputF[idx3] : 0.0f;

        float w0 = (i < weight0_.size()) ? weight0_[i] : 0.0f;
        float w1 = (i < weight1_.size()) ? weight1_[i] : 0.0f;
        float w2 = (i < weight2_.size()) ? weight2_[i] : 0.0f;
        float w3 = (i < weight3_.size()) ? weight3_[i] : 0.0f;

        if (i < tempBuffer_.size()) {
            tempBuffer_[i] = val0 * w0 + val1 * w1 + val2 * w2 + val3 * w3;
        }
        scalarIterCount++;
    }

    LOG_DEBUG("SIMD", "FMA-optimized AVX2 completed: SIMD iterations=%d, scalar iterations=%d",
              simdIterCount, scalarIterCount);

    // 转换为uint8_t
    convertFloatToByteAVX2(tempBuffer_.data(), output, pixelCount);
}
```

### 2. 数据预打包优化

```cpp
class PackedDataOptimizer {
private:
    struct alignas(64) CacheLineData {
        __m256 weights0, weights1, weights2, weights3;
        __m256i indices0, indices1, indices2, indices3;
    };

    std::vector<CacheLineData> prepackedData_;
    bool isPrepacked_ = false;

public:
    void precomputePackedData() {
        const int pixelCount = matrixWidth_ * matrixHeight_;
        const int packCount = (pixelCount + 7) / 8;  // 8像素打包

        LOG_INFO("SIMD", "Precomputing packed data: pixelCount=%d, packCount=%d", pixelCount, packCount);

        prepackedData_.resize(packCount);

        for (int p = 0; p < packCount; ++p) {
            int baseIndex = p * 8;

            // 预加载权重到SIMD寄存器格式
            prepackedData_[p].weights0 = _mm256_loadu_ps(&weight0_[baseIndex]);
            prepackedData_[p].weights1 = _mm256_loadu_ps(&weight1_[baseIndex]);
            prepackedData_[p].weights2 = _mm256_loadu_ps(&weight2_[baseIndex]);
            prepackedData_[p].weights3 = _mm256_loadu_ps(&weight3_[baseIndex]);

            // 预加载索引到SIMD寄存器格式
            prepackedData_[p].indices0 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(&sourceIndex0_[baseIndex]));
            prepackedData_[p].indices1 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(&sourceIndex1_[baseIndex]));
            prepackedData_[p].indices2 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(&sourceIndex2_[baseIndex]));
            prepackedData_[p].indices3 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(&sourceIndex3_[baseIndex]));
        }

        isPrepacked_ = true;
        LOG_INFO("SIMD", "Packed data precomputing completed: %d cache lines", packCount);
    }

    void processWithPackedFMA(const float* inputF, uint8_t* output, int pixelCount) {
        if (!isPrepacked_) {
            precomputePackedData();
        }

        const int packCount = (pixelCount + 7) / 8;

        for (int p = 0; p < packCount; ++p) {
            auto& data = prepackedData_[p];
            int baseIndex = p * 8;

            // 使用预打包数据，减少内存访问
            __m256 src0 = _mm256_i32gather_ps(inputF, data.indices0, 4);
            __m256 src1 = _mm256_i32gather_ps(inputF, data.indices1, 4);
            __m256 src2 = _mm256_i32gather_ps(inputF, data.indices2, 4);
            __m256 src3 = _mm256_i32gather_ps(inputF, data.indices3, 4);

            // FMA指令与预打包数据结合
            __m256 result = _mm256_fmadd_ps(src0, data.weights0,
                             _mm256_fmadd_ps(src1, data.weights1,
                             _mm256_fmadd_ps(src2, data.weights2,
                             _mm256_mul_ps(src3, data.weights3))));

            // 处理边界情况
            if (baseIndex + 8 <= pixelCount) {
                _mm256_storeu_ps(&tempBuffer_[baseIndex], result);
            } else {
                // 处理最后一个不完整的包
                for (int i = 0; baseIndex + i < pixelCount; ++i) {
                    tempBuffer_[baseIndex + i] = ((float*)&result)[i];
                }
            }
        }

        // 批量类型转换
        convertFloatToByteAVX2(tempBuffer_.data(), output, pixelCount);
    }
};
```

### 3. 集成到现有架构

```cpp
// 在ImageUndistorterImpl类中添加成员变量
class ImageUndistorterImpl {
private:
    // 现有成员变量...

    // 新增FMA相关成员
    bool useFMA_ = false;
    bool usePackedData_ = false;
    PackedDataOptimizer packedOptimizer_;

public:
    void detectCPUFeatures() noexcept {
        // 检测FMA支持
        FMAProcessor fmaDetector;
        fmaDetector.detectCPUFeatures();
        useFMA_ = fmaDetector.useFMA();

        // 检测是否启用数据预打包
        usePackedData_ = useFMA_;  // FMA时启用预打包效果更好

        LOG_INFO("SIMD", "FMA features detected and configured: useFMA=%s, usePacked=%s",
                useFMA_ ? "yes" : "no", usePackedData_ ? "yes" : "no");
    }

    ErrorCode infer(const uint8_t *inputImage, int width, int height,
                    uint8_t *outputImage, BaseResult &result) noexcept {
        // 现有代码...

        // 选择最优处理路径
        if (useFMA_ && usePackedData_) {
            packedOptimizer_.processWithPackedFMA(inputF.data(), outputImage, width * height);
        } else if (useFMA_) {
            simdUndistortAVX2_FMA(inputF.data(), outputImage, width * height);
        } else if (useAVX2_) {
            simdUndistortAVX2(inputF.data(), outputImage, width * height);
        } else {
            simdUndistortSSE(inputF.data(), outputImage, width * height);
        }

        // 现有代码...
    }
};
```

---

## 📊 性能测试验证

### 1. 精度验证测试

```cpp
class PrecisionValidator {
public:
    bool validateFMAImplementation() {
        // 创建测试数据
        const int testSize = 1024;
        std::vector<float> testInput(testSize);
        std::vector<float> referenceResult(testSize);
        std::vector<float> fmaResult(testSize);

        // 填充测试数据
        std::fill(testInput.begin(), testInput.end(), 128.0f);  // 中间值

        // 使用标准实现计算参考结果
        computeStandardAVX2(testInput.data(), referenceResult.data(), testSize);

        // 使用FMA实现计算测试结果
        computeFMAOptimized(testInput.data(), fmaResult.data(), testSize);

        // 验证精度
        double maxError = 0.0;
        double avgError = 0.0;
        for (int i = 0; i < testSize; ++i) {
            double error = std::abs(referenceResult[i] - fmaResult[i]);
            maxError = std::max(maxError, error);
            avgError += error;
        }
        avgError /= testSize;

        LOG_INFO("TEST", "FMA precision validation: maxError=%.2e, avgError=%.2e", maxError, avgError);

        // 精度要求：最大误差 < 1e-6
        bool passed = (maxError < 1e-6);

        if (passed) {
            LOG_INFO("TEST", "FMA implementation precision test PASSED");
        } else {
            LOG_ERROR("TEST", "FMA implementation precision test FAILED");
        }

        return passed;
    }
};
```

### 2. 性能基准测试

```cpp
class PerformanceBenchmark {
private:
    struct BenchmarkResult {
        double avgTime;
        double minTime;
        double maxTime;
        std::string method;
    };

public:
    void benchmarkSIMDImplementations() {
        const int iterations = 100;
        const int imageSize = matrixWidth_ * matrixHeight_;

        std::vector<float> input(imageSize, 128.0f);
        std::vector<uint8_t> output(imageSize);

        // 基准：标准AVX2实现
        auto resultAVX2 = runBenchmark("Standard AVX2", [&, this]() {
            simdUndistortAVX2(input.data(), output.data(), imageSize);
        }, iterations);

        // 测试：FMA优化实现
        auto resultFMA = runBenchmark("FMA-Optimized", [&, this]() {
            simdUndistortAVX2_FMA(input.data(), output.data(), imageSize);
        }, iterations);

        // 测试：FMA+预打包实现
        auto resultFMAPacked = runBenchmark("FMA+Packed", [&, this]() {
            packedOptimizer_.processWithPackedFMA(input.data(), output.data(), imageSize);
        }, iterations);

        // 输出结果
        printBenchmarkResults(resultAVX2, resultFMA, resultFMAPacked);
    }

private:
    BenchmarkResult runBenchmark(const std::string& name, std::function<void()> func, int iterations) {
        std::vector<double> times;
        times.reserve(iterations);

        for (int i = 0; i < iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            func();
            auto end = std::chrono::high_resolution_clock::now();

            double timeMs = std::chrono::duration<double, std::milli>(end - start).count();
            times.push_back(timeMs);
        }

        double avgTime = std::accumulate(times.begin(), times.end(), 0.0) / iterations;
        double minTime = *std::min_element(times.begin(), times.end());
        double maxTime = *std::max_element(times.begin(), times.end());

        return {avgTime, minTime, maxTime, name};
    }

    void printBenchmarkResults(const BenchmarkResult& avx2, const BenchmarkResult& fma, const BenchmarkResult& fmaPacked) {
        double fmaSpeedup = avx2.avgTime / fma.avgTime;
        double packedSpeedup = avx2.avgTime / fmaPacked.avgTime;

        LOG_INFO("BENCHMARK", "=== SIMD Performance Results ===");
        LOG_INFO("BENCHMARK", "Standard AVX2:  %.2f ms (min: %.2f, max: %.2f)",
                avx2.avgTime, avx2.minTime, avx2.maxTime);
        LOG_INFO("BENCHMARK", "FMA-Optimized:   %.2f ms (speedup: %.2fx, min: %.2f, max: %.2f)",
                fma.avgTime, fmaSpeedup, fma.minTime, fma.maxTime);
        LOG_INFO("BENCHMARK", "FMA+Packed:     %.2f ms (speedup: %.2fx, min: %.2f, max: %.2f)",
                fmaPacked.avgTime, packedSpeedup, fmaPacked.minTime, fmaPacked.maxTime);

        if (fmaSpeedup > 1.05) {
            LOG_INFO("BENCHMARK", "✅ FMA optimization provides %.1fx speedup", fmaSpeedup);
        } else {
            LOG_WARN("BENCHMARK", "⚠️ FMA optimization provides only %.1fx speedup", fmaSpeedup);
        }
    }
};
```

---

## 🚀 集成步骤

### Phase 1: 基础实现 (1-2天)
1. **添加CPU特性检测代码**
2. **实现FMA优化版本的核心函数**
3. **集成到现有类结构**

### Phase 2: 优化增强 (2-3天)
1. **实现数据预打包优化**
2. **添加精度验证测试**
3. **实现性能基准测试**

### Phase 3: 测试验证 (1天)
1. **运行完整的精度测试**
2. **执行性能基准测试**
3. **验证与现有实现的兼容性**

### Phase 4: 部署集成 (1天)
1. **更新CMakeLists.txt支持FMA编译选项**
2. **更新文档和使用说明**
3. **代码审查和最终测试**

---

## 📋 实施检查清单

### 编译器支持检查
- [ ] GCC 4.6+ 支持 `-mfma`
- [ ] Clang 3.0+ 支持 `-mfma`
- [ ] MSVC 2013+ 支持 `/arch:AVX2`

### 功能验证检查
- [ ] CPU特性检测正确工作
- [ ] FMA路径与标准路径结果一致
- [ ] 边界情况处理正确
- [ ] 内存访问安全

### 性能验证检查
- [ ] FMA版本比标准版本快5-15%
- [ ] 数据预打包带来额外收益
- [ ] 在目标CPU(Intel Ultra 7)上测试

### 代码质量检查
- [ ] 代码风格与现有代码一致
- [ ] 添加适当的日志记录
- [ ] 错误处理完善
- [ ] 文档和注释充分

---

**文档版本**: v1.0
**实施预计时间**: 5-7天
**预期收益**: 10-20%性能提升

---

*此实施方案将基于实际测试结果进行调整优化。*