/** 
 * @file    kLock.cpp
 * @brief   Declares the kLock class. 
 *
 * @internal
 * Copyright (C) 2005-2024 by LMI Technologies Inc.
 * Licensed under the MIT License.
 * Redistributed files must retain the above copyright notice.
 */
#define K_PLATFORM
#include <kApi/Threads/kLock.h>

kBeginValueEx(k, kLockOption)
    kAddEnumerator(kLockOption, kLOCK_OPTION_TIMEOUT)
    kAddEnumerator(kLockOption, xkLOCK_OPTION_PRIORITY_INHERITANCE)
kEndValueEx()

kBeginClassEx(k, kLock)
    kAddPrivateVMethod(kLock, kObject, VRelease)
kEndClassEx()

kFx(kStatus) kLock_Construct(kLock* lock, kAlloc allocator)
{
    return kLock_ConstructEx(lock, kFALSE, allocator);
} 

kFx(kStatus) kLock_ConstructEx(kLock* lock, kLockOption options, kAlloc allocator)
{
    kAlloc alloc = kAlloc_Fallback(allocator);
    kType type = kTypeOf(kLock);
    kStatus status;

    kCheck(kAlloc_GetObject(alloc, type, lock));

    if (!kSuccess(status = xkLock_Init(*lock, options, type, alloc)))
    {
        kAlloc_FreeRef(alloc, lock);
    }

    return status;
}

#if defined(K_WINDOWS)

kFx(kStatus) xkLock_Init(kLock lock, kLockOption options, kType type, kAlloc allocator)
{
    kObjR(kLock, lock);

    kCheck(kObject_Init(lock, type, allocator)); 

    obj->options = options;
    kZero(obj->criticalSection); 
    kZero(obj->mutex);

    //on Windows, critical sections are faster than mutexes; accordingly, we'll 
    //only use a mutex when the lock needs to support timeouts.
    //priority inheritance isn't supported, but Windows supports random boosting by default
    if ((obj->options & kLOCK_OPTION_TIMEOUT) != 0)
    {
        if (kIsNull(obj->mutex = CreateMutex(NULL, FALSE, NULL)))
        {
            kObject_VRelease(lock);
            return kERROR_OS;
        }
    }
    else 
    {
        InitializeCriticalSection(&obj->criticalSection);
    }

    return kOK; 
}

kFx(kStatus) xkLock_VRelease(kLock lock)
{
    kObj(kLock, lock); 
    
    if ((obj->options & kLOCK_OPTION_TIMEOUT) != 0)
    {
        if (!kIsNull(obj->mutex))
        {
            CloseHandle(obj->mutex);
        }

        obj->mutex = kNULL;
    }
    else
    {
        DeleteCriticalSection(&obj->criticalSection); 
    }

    kCheck(kObject_VRelease(lock)); 

    return kOK; 
}

kFx(kStatus) kLock_Enter(kLock lock)
{
    kObj(kLock, lock);

    if ((obj->options & kLOCK_OPTION_TIMEOUT) != 0)
    {
        DWORD waitResult = WaitForSingleObject(obj->mutex, kOS_INFINITE);

        if (waitResult == WAIT_TIMEOUT)
        {
            return kERROR_TIMEOUT;
        }
        else if (waitResult != WAIT_OBJECT_0)
        {
            return kERROR_OS;
        }
    }
    else
    {
        EnterCriticalSection(&obj->criticalSection);
    }

    return kOK;
}

kFx(kStatus) kLock_EnterEx(kLock lock, k64u timeout)
{
    kObj(kLock, lock);

    kAssert((obj->options & kLOCK_OPTION_TIMEOUT) != 0 || (timeout == kINFINITE)); 

    if ((obj->options & kLOCK_OPTION_TIMEOUT) != 0)
    {
        DWORD osTimeout = (DWORD)xkTimeToKernelTime(timeout);
        DWORD waitResult = WaitForSingleObject(obj->mutex, osTimeout);

        if (waitResult == WAIT_TIMEOUT)
        {
            return kERROR_TIMEOUT;
        }
        else if (waitResult != WAIT_OBJECT_0)
        {
            return kERROR_OS;
        }
    }
    else
    {
        EnterCriticalSection(&obj->criticalSection);
    }

    return kOK;
}

kFx(kStatus) kLock_Exit(kLock lock)
{
    kObj(kLock, lock); 

    if ((obj->options & kLOCK_OPTION_TIMEOUT) != 0)
    {
        if (!ReleaseMutex(obj->mutex))
        {
            return kERROR_OS;
        }
    }
    else
    {
        LeaveCriticalSection(&obj->criticalSection);
    }

    return kOK;
}

#elif defined(K_POSIX)

kFx(kStatus) xkLock_Init(kLock lock, kLockOption options, kType type, kAlloc allocator)
{
    kObjR(kLock, lock); 
    pthread_mutexattr_t mutexattr; 
    kBool attrInit = kFALSE; 
    kBool mutexInit = kFALSE; 
    kStatus status; 
    
    kCheck(kObject_Init(lock, type, allocator)); 

    kZero(obj->mutex);
    obj->options = options;

    if (((obj->options & kLOCK_OPTION_TIMEOUT) != 0) &&
        ((obj->options & xkLOCK_OPTION_PRIORITY_INHERITANCE) != 0))
    {
        /*
         * This combination of options is only supported on sufficiently recent versions of Glibc/Linux. 
         * Priority inheritance in combination with waitable locks against CLOCK_MONOTONIC require Glibc 
         * 2.35 and Linux 5.14 (FUTEX_LOCK_PI2). On unsupported kernels, the lock should outright fail. 
         * However, there were bugs in Glibc all the way up to version 2.32 to properly raise an adequate 
         * error. Besides, code that is written against kLock generally skips error checking entirely. 
         * As such, allowing this combination of options is (although possible) quite risky. Therefore, 
         * we prohibit it for now. 
         */
        return kERROR_PARAMETER;
    }

    kTry
    {
        kTest(pthread_mutexattr_init(&mutexattr) == 0); 
        attrInit = kTRUE; 

        kTest(pthread_mutexattr_settype(&mutexattr, PTHREAD_MUTEX_RECURSIVE) == 0);

        if ((obj->options & xkLOCK_OPTION_PRIORITY_INHERITANCE) != 0)
        {
            kTest(pthread_mutexattr_setprotocol(&mutexattr, PTHREAD_PRIO_INHERIT) == 0);
        }

        kTest(pthread_mutex_init(&obj->mutex, &mutexattr) == 0);
        mutexInit = kTRUE; 
    }
    kCatchEx(&status)
    {
        if (mutexInit) pthread_mutex_destroy(&obj->mutex);
        
        kObject_VRelease(lock); 

        kEndCatchEx(status); 
    }
    kFinallyEx
    {
        if (attrInit) pthread_mutexattr_destroy(&mutexattr); 

        kEndFinallyEx();
    }

    return kOK; 
}

kFx(kStatus) xkLock_VRelease(kLock lock)
{
    kObj(kLock, lock); 
    
    kCheck(pthread_mutex_destroy(&obj->mutex) == 0);

    kCheck(kObject_VRelease(lock)); 

    return kOK; 
}

kFx(kStatus) kLock_Enter(kLock lock)
{
    kObj(kLock, lock);

    kCheckTrue(pthread_mutex_lock(&obj->mutex) == 0, kERROR_OS);

    return kOK; 
}

kFx(kStatus) kLock_EnterEx(kLock lock, k64u timeout)
{
    kObj(kLock, lock);
    struct timespec tm;
    int result = 0;

    kAssert(((obj->options & kLOCK_OPTION_TIMEOUT) != 0) || (timeout == kINFINITE)); 

    if (timeout == kINFINITE)
    {
        kCheckTrue(pthread_mutex_lock(&obj->mutex) == 0, kERROR_OS);
    }
    else
    {
        kCheck(xkFormatTimeout(timeout, &tm));

        result = pthread_mutex_timedlock_(&obj->mutex, &tm);

        if (result == ETIMEDOUT)
        {
            return kERROR_TIMEOUT;
        }
        else if (result != 0)
        {
            return kERROR_OS; 
        }
    }

    return kOK; 
}

kFx(kStatus) kLock_Exit(kLock lock)
{
    kObj(kLock, lock); 

    kCheck(pthread_mutex_unlock(&obj->mutex) == 0);

    return kOK;
}

#endif
