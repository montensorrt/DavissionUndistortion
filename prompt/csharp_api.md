
第一步；参考以下实例文件，帮我总结接口函数封装风格和逻辑，帮我总结出一个封装风格和逻辑的模板；
c接口头文件：CSharpApp/demo/GridStitch_C_API.h
p/invok封装文件：CSharpApp/Demo/GridStitch_NativeAPI.cs文件
csharp封装文件：CSharpApp/Demo/GridStitchProcessor.cs文件

第二步；基于CAPP文件include路径下的所有*_C_API.h头文件，帮我生成对应的p/invok封装文件，文件路径为CSharpApp/Undistor/，文件命名规则为去掉_C_API.h后缀，添加_NativeAPI.cs后缀，要求风格和逻辑和GridStitch_NativeAPI.cs总结出的模板保持一致，

第三步；参考总结的模板和csharp封装文件：CSharpApp/Demo/GridStitchProcessor.cs文件逻辑，帮我实现对应的csharp封装文件，文件路径为CSharpApp/Undistor/，文件命名规则为去掉_C_API.h后缀，添加Processor.cs后缀，同路径下生成c#的调用主程序文件，文件命名规则为去掉_C_API.h后缀，添加Runner.cs后缀，要求风格和逻辑和GridStitchProcessor.cs总结出的模板保持一致，
要求：风格要保持一致，尽可能精简，防止冗余，全程使用中文回答



针对刚才的实现需要实现以下几点需求的优化：
1、*Processor.cs文件，只需要保留UndistortionProcessor.cs一个即可，删除其他多余的Processor.cs文件
2、所有Undistor路径下新生成的文件namespace PjktAuto.AOI.Hawk8000V3.NativeAPI全部改为Undistor.NativeAPI

我如何使用vs打开或者管理开发Undistor这个c#项目，帮我提出建议，需要打开vs创建项目将他们全部添加进去吗？
要求：风格要保持一致，尽可能精简，防止冗余，全程使用中文回答


执行UndistortionRunner.cs文件的测试参数，参考CAPP/test/test_undistortion.cpp文件进行填充和设置，同时处理流程也要参考test_undistortion.cpp文件的逻辑进行实现，比如图像去畸变之前先旋转90度，去畸变过后，再旋转回去，帮我实现UndistortionRunner.cs文件的测试主程序代码，要求：风格要保持一致，尽可能精简，防止冗余，全程使用中文回答
cmake list 生成vs的工程文件，
# 生成 VS2022 x64 项目文件
cmake .. -G "Visual Studio 17 2022" -A x64


问题分析
基于目录结构检查，我发现了配置的问题：

当前配置问题：
实际DLL位置：
DLL文件实际位于：Undistortion.dll
而不是：CAPP\build\$(Configuration)\Undistortion*.dll
路径解析分析：
$(Configuration)变量展开：

Debug配置：..\..\CAPP\build\Debug\Undistortion*.dll
Release配置：..\..\CAPP\build\Release\Undistortion*.dll
实际情况：

❌ CAPP\build\Debug\ 目录不存在
❌ CAPP\build\Release\ 目录不存在
✅ bin 目录存在且包含 Undistortion.dll







当前文件UndistortionProcessor.cs时我新拷贝过来的，帮我修改其内部的一些变量名字和Undistor项目内的一致，
要求：风格要保持一致，尽可能精简，防止冗余，全程使用中文回答
