#include "CommonTypes_C_API.h"

extern "C" {

// ===================================================================
// 版本信息接口实现
// ===================================================================

AOI_API int AOI_CALL AOI_GetVersion(int* major, int* minor, int* patch) {
    if (!major || !minor || !patch) {
        return 0;
    }
    
    *major = AOI_C_API_VERSION_MAJOR;
    *minor = AOI_C_API_VERSION_MINOR;
    *patch = AOI_C_API_VERSION_PATCH;
    
    return 1;
}

} // extern "C"