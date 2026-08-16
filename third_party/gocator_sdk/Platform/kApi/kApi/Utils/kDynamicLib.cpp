/** 
 * @file    kDynamicLib.cpp
 *
 * @internal
 * Copyright (C) 2013-2024 by LMI Technologies Inc.
 * Licensed under the MIT License.
 * Redistributed files must retain the above copyright notice.
 */
#define K_PLATFORM
#include <kApi/Utils/kDynamicLib.h>
#include <kApi/Utils/kSymbolInfo.h>
#include <kApi/Io/kPath.h>
#include <stdio.h>

kBeginValueEx(k, kDynamicLibOption)
    kAddEnumerator(kDynamicLibOption, kDYNAMIC_LIB_OPTION_NONE)
    kAddEnumerator(kDynamicLibOption, kDYNAMIC_LIB_ALTERNATE_SEARCH_PATH)
kEndValueEx()

kBeginClassEx(k, kDynamicLib)
    kAddPrivateVMethod(kDynamicLib, kObject, VRelease)
kEndClassEx()

kFx(kStatus) kDynamicLib_Construct(kDynamicLib* library, const kChar* path, kAlloc allocator)
{
    return kDynamicLib_ConstructEx(library, path, kDYNAMIC_LIB_OPTION_NONE, allocator);
} 

kFx(kStatus) kDynamicLib_ConstructEx(kDynamicLib* library, const kChar* path, kDynamicLibOption options, kAlloc allocator)
{
    kAlloc alloc = kAlloc_Fallback(allocator);
    kType type = kTypeOf(kDynamicLib); 
    kStatus status; 

    kCheck(kAlloc_GetObject(alloc, type, library)); 

    if (!kSuccess(status = xkDynamicLib_Init(*library, type, path, options, alloc)))
    {
        kAlloc_FreeRef(alloc, library); 
    }

    return status; 
} 

kFx(kStatus) kDynamicLib_ConstructFromHandle(kDynamicLib* library, kPointer handle, kAlloc allocator)
{
    kAlloc alloc = kAlloc_Fallback(allocator);
    kType type = kTypeOf(kDynamicLib); 
    kStatus status; 

    kCheck(kAlloc_GetObject(alloc, type, library)); 

    if (!kSuccess(status = xkDynamicLib_InitFromHandle(*library, type, handle, alloc)))
    {
        kAlloc_FreeRef(alloc, library); 
    }

    return status; 
} 

kFx(kStatus) xkDynamicLib_Init(kDynamicLib library, kType type, const kChar* path, kDynamicLibOption options, kAlloc allocator)
{
    kObjR(kDynamicLib, library);  
    kStatus status = kOK;

    kCheck(kObject_Init(library, type, allocator)); 

    obj->handle = kNULL;
    obj->isOwned = kTRUE;

    xkSymbolInfo_BeginLoad();
    {
        status = xkDynamicLib_OpenHandle(path, &obj->handle, options);
    }
    xkSymbolInfo_EndLoad();

    if (!kSuccess(status))
    {
        xkDynamicLib_VRelease(library);
    }

    return status; 
}

kFx(kStatus) xkDynamicLib_InitFromHandle(kDynamicLib library, kType type, kPointer handle, kAlloc allocator)
{
    kObjR(kDynamicLib, library);  

    kCheck(kObject_Init(library, type, allocator)); 

    obj->handle = handle;
    obj->isOwned = kFALSE;

    return kOK; 
}

kFx(kStatus) xkDynamicLib_VRelease(kDynamicLib library)
{
    kObj(kDynamicLib, library); 

    if (!kIsNull(obj->handle) && obj->isOwned)
    {
        kCheck(xkDynamicLib_CloseHandle(obj->handle));
    }

    kCheck(kObject_VRelease(library)); 

    return kOK;
}

kFx(kStatus) kDynamicLib_FindFunction(kDynamicLib library, const kChar* name, kFunction* function)
{
    kObj(kDynamicLib, library); 

    return xkDynamicLib_Resolve(obj->handle, name, function);
}

#if defined(K_WINDOWS)

kFx(kStatus) xkDynamicLib_OpenHandle(const kChar* path, kPointer* handle, kDynamicLibOption options)
{
    WCHAR wpath[MAX_PATH]; 
    DWORD flags = (options & kDYNAMIC_LIB_ALTERNATE_SEARCH_PATH) != 0 ? LOAD_WITH_ALTERED_SEARCH_PATH : 0;

    kCheck(xkPath_NormalizedToNativeWideWin(path, wpath, kCountOf(wpath))); 

    if (kIsNull(*handle = (kPointer) LoadLibraryExW(wpath, kNULL, flags)))
    {
        return kERROR_NOT_FOUND; 
    }

    return kOK;
}

kFx(kStatus) xkDynamicLib_Resolve(kPointer handle, const kChar* name, kFunction* function)
{
    FARPROC address = GetProcAddress((HMODULE)handle, name); 

    if (kIsNull(address))
    {
        return kERROR_NOT_FOUND; 
    }

    *(FARPROC*)function = address; 

    return kOK;
}

kFx(kStatus) xkDynamicLib_CloseHandle(kPointer handle)
{
    if (!FreeLibrary((HMODULE)handle))
    {
        return kERROR_OS; 
    }

    return kOK;
}

#elif defined (K_POSIX)

kFx(kStatus) xkDynamicLib_OpenHandle(const kChar* path, kPointer* handle, kDynamicLibOption options)
{
    kChar nativePath[kPATH_MAX];

    kCheck(xkPath_FromVirtual(path, nativePath, kCountOf(nativePath)));

    if (kIsNull(*handle = (kPointer)dlopen(nativePath, RTLD_NOW)))
    {
        return kERROR_NOT_FOUND; 
    }

    return kOK;
}

kFx(kStatus) xkDynamicLib_Resolve(kPointer handle, const kChar* name, kFunction* function)
{
    void* address = dlsym(handle, name); 

    if (kIsNull(address))
    {
        return kERROR_NOT_FOUND; 
    }

    *(void**)function = address;

    return kOK;
}

kFx(kStatus) xkDynamicLib_CloseHandle(kPointer handle)
{
    if (dlclose(handle) != 0)
    {
        return kERROR_OS; 
    }

    return kOK;
}

#endif
