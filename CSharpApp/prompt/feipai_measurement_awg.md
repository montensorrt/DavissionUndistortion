第一步先帮我梳理和理解当前c接口和p/invok接口的封装逻辑和风格，
文件数据如下：
c接口头文件：CAPP/include/EdgeMeasurementDetector_C_API.h
c接口源文件：CAPP/src/EdgeMeasurementDetector_C_API.cpp
p/invok封装文件：CSharpApp/ClassLibrary/EdgeMeasurementDetector.cs

第二步：参考第一步的梳理内容基于c接口文件帮我实现新增c接口的p/invok封装文件，文件名为：AWGEdgeMeasurementDetector.cs，文件路径为：CSharpApp/ClassLibrary/AWGEdgeMeasurementDetector.cs，函数参数和返回值类型请参考现有的c接口函数封装风格和逻辑进行实现
c接口头文件：CAPP/include/AWGEdgeMeasurementDetector_C_API.h
c接口源文件：CAPP/src/AWGEdgeMeasurementDetector_C_API.cpp

要求：风格要保持一致，尽可能精简，防止冗余，全程使用中文回答


参考以下实例文件，帮我总结接口函数封装风格和逻辑，
c接口头文件：CAPP/include/GridStitch_C_API.h
p/invok封装文件：CSharpApp/Demo/GridStitch_NativeAPI.cs文件

帮我分析他和当前的p/invok文件名为：AWGEdgeMeasurementDetector.cs，的逻辑和风格有哪些区别和相似点，帮我总结出一个封装风格和逻辑的模板，方便我后续进行c接口的p/invok封装工作，特别是        IntPtr images /* ָ�������׵�ַ */,
        IntPtr outData,数据类型转为了IntPtr的封装方式，当前使用的byte[] data类型封装方式的区别和联系，他们的本质区别是否和当前的c和cpp接口直接适配，
要求：风格要保持一致，尽可能精简，防止冗余，全程使用中文回答


先帮我总结以下Undistortion和UndistortionProcessor两个封装风格和逻辑：
p/invok封装文件:CSharpApp/ClassLibrary/Undistortion.cs
c#处理封装文件:CSharpApp/ClassLibrary/UndistortionProcessor.cs
基于当前的p/invok文件名为：AWGEdgeMeasurementDetector.cs，帮我实现其c#处理器封装文件，文件名为：AWGEdgeMeasurementDetectorProcessor.cs，文件路径为：CSharpApp/ClassLibrary/AWGEdgeMeasurementDetectorProcessor.cs，要求风格和逻辑和UndistortionProcessor.cs文件保持一致，尽可能精简，防止冗余，全程使用中文回答

前提知识：
ClassLibrary项目下的文件是p/invok接口和初步的功能封装文件，Test项目下的文件是具体实现方法的接口和内容，TestRunner项目是总的调用流程入口和逻辑控制，

在Test项目路径下，参照UndistortionBatchRunner.cs文件，帮我实现AWGEdgeMeasurementDetectorBatchRunner.cs文件，文件路径为：CSharpApp/Test/AWGEdgeMeasurementDetectorBatchRunner.cs，要求风格和逻辑和UndistortionBatchRunner.cs文件保持一致，并在TestRunner项目中实现调用AWGEdgeMeasurementDetectorBatchRunner.cs，调用CSharpApp/ClassLibrary/AWGEdgeMeasurementDetectorProcessor.cs接口实现最终的测试，具体的参数和逻辑参考CPPAPP/test/test_awg_coc_measurement.cpp的c++测试文件，其内部的一些默认参数可以直接从AWGEdgeMeasurementDetector.h文件中获取，帮我实现上述需求。
要求：风格要保持一致，尽可能精简，防止冗余，全程使用中文回答


梳理梳理AOITestConfigManager.cs文件内容，参考EdgeMeasurementDetectorTestHigh.cs脚本的配置文件创建的默认值初始化，为public static AOI_EdgeMeasurementConfig CreateEdgeMeasurementConfig()，帮我在AOITestConfigManager.cs文件中也创建public static AOI_EdgeMeasurementConfig CreateAWGEdgeMeasurementConfig()，其默认值和参数都从参考CPPAPP/test/test_awg_coc_measurement.cpp的c++测试文件和其对应的头文件中查找填充，
要求：风格要保持一致，尽可能精简，防止冗余，全程使用中文回答

cmake list 生成vs的工程文件，
# 生成 VS2022 x64 项目文件
cmake ..\CAPP -G "Visual Studio 17 2022" -A x64


参考CPPAPP/test/test_awg_coc_measurement.cpp的c++测试文件中的结果获取和绘制逻辑以及数据保存逻辑，特别是将结果单独保存为另一个文件，带时间戳，帮我将这些内容在AWGEdgeMeasurementDetectorBatchRunner.cs测试文件中实现。
要求：风格要保持一致，尽可能精简，防止冗余，全程使用中文回答



帮我梳理AOI_SetNumThreads的p/invok接口实现和调用逻辑的测试，
c接口头文件：CAPP/include/CommonTypes_C_API.h的AOI_SetNumThreads函数用来控制多线程数量
在p/invok封装文件：CSharpApp/ClassLibrary/CommonTypes.cs的AOI_SetNumThreads函数的接口封装，
在Test项目路径下，AWGEdgeMeasurementDetectorBatchRunner.cs文件中帮我添加调用AOI_SetNumThreads接口的逻辑，
要求：风格要保持一致，尽可能精简，防止冗余，全程使用中文回答