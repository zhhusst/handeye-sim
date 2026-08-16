/** 
 * @file    kAtomic.cpp
 *
 * @internal
 * Copyright (C) 2012-2024 by LMI Technologies Inc.
 * Licensed under the MIT License.
 * Redistributed files must retain the above copyright notice.
 */
#define K_PLATFORM
#include <kApi/Threads/kAtomic.h>

kBeginStaticClassEx(k, kAtomic)
kEndStaticClassEx()

kFx(kStatus) xkAtomic_InitStatic()
{
    return kOK;
}

kFx(kStatus) xkAtomic_ReleaseStatic()
{
    return kOK; 
}
