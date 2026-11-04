using System;
using Microsoft.Win32.SafeHandles;

namespace Undistor.NativeAPI;

/// <summary>
/// 通用的原生资源句柄，提供安全的资源管理
/// 基于 SafeHandleZeroOrMinusOneIsInvalid，确保原生资源在 .NET GC 时正确释放
/// </summary>
public sealed class GenericNativeHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    private readonly Action<IntPtr> _releaseAction;

    /// <summary>
    /// 私有构造函数，通过静态工厂方法创建实例
    /// </summary>
    /// <param name="handle">原生句柄</param>
    /// <param name="releaseAction">释放句柄的委托</param>
    private GenericNativeHandle(IntPtr handle, Action<IntPtr> releaseAction) : base(true)
    {
        _releaseAction = releaseAction ?? throw new ArgumentNullException(nameof(releaseAction));
        SetHandle(handle);
    }

    /// <summary>
    /// 创建 GenericNativeHandle 实例
    /// </summary>
    /// <param name="handle">原生句柄指针</param>
    /// <param name="releaseAction">释放句柄的委托函数</param>
    /// <returns>GenericNativeHandle 实例</returns>
    public static GenericNativeHandle Create(IntPtr handle, Action<IntPtr> releaseAction)
    {
        if (handle == IntPtr.Zero)
            throw new ArgumentException("Handle cannot be zero", nameof(handle));
        
        return new GenericNativeHandle(handle, releaseAction);
    }

    /// <summary>
    /// 获取原生句柄指针（用于传递给原生API）
    /// </summary>
    public IntPtr RawHandle => handle;

    /// <summary>
    /// 释放原生资源
    /// </summary>
    /// <returns>释放是否成功</returns>
    protected override bool ReleaseHandle()
    {
        try
        {
            if (handle != IntPtr.Zero)
            {
                _releaseAction(handle);
                return true;
            }
            return false;
        }
        catch
        {
            // 释放过程中的异常不能传播到外部
            return false;
        }
    }
}