namespace Undistortion.Inference;

/// <summary>C-API 调用失败时抛出的异常（携带原生错误码与错误串）</summary>
public sealed class UndistortionException : Exception
{
    public UndistortionException(UndistErrorCode code, string? nativeMessage)
        : base($"[{code}({(int)code})] {(string.IsNullOrWhiteSpace(nativeMessage) ? "去畸变调用失败" : nativeMessage)}")
    {
        Code = code;
    }

    /// <summary>原生错误码</summary>
    public UndistErrorCode Code { get; }
}
