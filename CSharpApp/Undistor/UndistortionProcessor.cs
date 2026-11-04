using System;
using System.Runtime.InteropServices;
using OpenCvSharp;

namespace Undistor.NativeAPI
{
    public class UndistortionProcessor : IDisposable
    {
        private IntPtr _undistorter;
        private bool _isInitialized;
        private bool _disposed;
        private UndistortionConfig_NativeAPI _config;
        private int _outputWidth;
        private int _outputHeight;
        private int _outputChannels;

        public bool IsInitialized => _isInitialized;
        public int OutputWidth => _outputWidth;
        public int OutputHeight => _outputHeight;
        public int OutputChannels => _outputChannels;

        public UndistortionProcessor()
        {
            _undistorter = IntPtr.Zero;
            _isInitialized = false;
            _disposed = false;
        }

        public bool Initialize(string matrixPath, int matrixWidth, int matrixHeight, int outputChannels)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(UndistortionProcessor));
            }

            if (_isInitialized)
            {
                Console.WriteLine("警告：去畸变处理器已经初始化，将重新初始化");
                Cleanup();
            }

            try
            {
                _undistorter = Undistortion_NativeAPI.AOI_Undistortion_Create();
                if (_undistorter == IntPtr.Zero)
                {
                    Console.WriteLine("✗ 创建Undistortion实例失败");
                    return false;
                }

                _config = new UndistortionConfig_NativeAPI
                {
                    matrixPath = new byte[512],
                    matrixWidth = matrixWidth,
                    matrixHeight = matrixHeight,
                    outputChannels = outputChannels
                };

                byte[] pathBytes = System.Text.Encoding.UTF8.GetBytes(matrixPath);
                int copyLen = Math.Min(pathBytes.Length, 511);
                Array.Copy(pathBytes, _config.matrixPath, copyLen);

                int result = Undistortion_NativeAPI.AOI_Undistortion_SetConfig(_undistorter, ref _config);
                if (result != 0)
                {
                    Console.WriteLine($"✗ 设置配置失败，错误代码: {result}");
                    Cleanup();
                    return false;
                }

                if (Undistortion_NativeAPI.AOI_Undistortion_IsConfigured(_undistorter) == 0)
                {
                    Console.WriteLine("✗ Undistorter未完全配置");
                    Cleanup();
                    return false;
                }

                _outputWidth = matrixWidth;
                _outputHeight = matrixHeight;
                _outputChannels = outputChannels;

                _isInitialized = true;
                Console.WriteLine($"✓ 去畸变处理器初始化成功");
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"✗ 初始化异常: {ex.Message}");
                Cleanup();
                return false;
            }
        }

        public Mat ProcessImage(Mat inputMat)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(UndistortionProcessor));
            }

            if (!_isInitialized)
            {
                Console.WriteLine("✗ 去畸变处理器未初始化，请先调用Initialize方法");
                return null;
            }

            if (inputMat == null || inputMat.Empty())
            {
                Console.WriteLine("✗ 输入图像为空");
                return null;
            }

            try
            {
                int inputWidth = inputMat.Width;
                int inputHeight = inputMat.Height;
                int inputChannels = inputMat.Channels();

                // 统一使用 GetArray 读取数据
                int inputSize = (int)(inputMat.Total() * inputMat.ElemSize());
                byte[] inputData = new byte[inputSize];
                inputMat.GetArray(out inputData);

                int outputSize = _outputWidth * _outputHeight * _outputChannels;
                byte[] outputData = new byte[outputSize];

                var stopwatch = System.Diagnostics.Stopwatch.StartNew();

                // 固定内存并获取IntPtr
                var inputHandle = GCHandle.Alloc(inputData, GCHandleType.Pinned);
                var outputHandle = GCHandle.Alloc(outputData, GCHandleType.Pinned);

                try
                {
                    IntPtr inputPtr = inputHandle.AddrOfPinnedObject();
                    IntPtr outputPtr = outputHandle.AddrOfPinnedObject();

                    int errorCode = Undistortion_NativeAPI.AOI_Undistortion_Infer(
                        _undistorter,
                        inputPtr,
                        inputWidth,
                        inputHeight,
                        outputPtr,
                        out UndistortionResult_NativeAPI result);

                    stopwatch.Stop();

                    if (errorCode != 0)
                    {
                        Console.WriteLine($"✗ 去畸变执行失败，错误代码: {errorCode}");
                        return null;
                    }

                    Mat outputMat;
                    if (_outputChannels == 1)
                    {
                        outputMat = new Mat(_outputHeight, _outputWidth, MatType.CV_8UC1);
                    }
                    else if (_outputChannels == 3)
                    {
                        outputMat = new Mat(_outputHeight, _outputWidth, MatType.CV_8UC3);
                    }
                    else
                    {
                        Console.WriteLine($"✗ 不支持的输出通道数: {_outputChannels}");
                        return null;
                    }

                    outputMat.SetArray(outputData);

                    Console.WriteLine($"✓ 去畸变完成，耗时: {stopwatch.ElapsedMilliseconds}ms");
                    return outputMat;
                }
                finally
                {
                    // 释放固定的内存句柄
                    if (inputHandle.IsAllocated)
                        inputHandle.Free();
                    if (outputHandle.IsAllocated)
                        outputHandle.Free();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"✗ 处理图像异常: {ex.Message}");
                return null;
            }
        }

        public string GetStatus()
        {
            if (_disposed)
            {
                return "已释放";
            }

            if (!_isInitialized)
            {
                return "未初始化";
            }

            return $"已初始化 - 输出尺寸: {_outputWidth}x{_outputHeight}, 通道数: {_outputChannels}";
        }

        private void Cleanup()
        {
            if (_undistorter != IntPtr.Zero)
            {
                Undistortion_NativeAPI.AOI_Undistortion_Destroy(_undistorter);
                _undistorter = IntPtr.Zero;
            }

            _isInitialized = false;
            _outputWidth = 0;
            _outputHeight = 0;
            _outputChannels = 0;
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                Cleanup();
                _disposed = true;
            }
        }

        ~UndistortionProcessor()
        {
            Dispose(false);
        }
    }
}