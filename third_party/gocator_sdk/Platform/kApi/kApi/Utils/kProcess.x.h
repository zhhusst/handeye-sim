/** 
 * @file    kProcess.x.h
 *
 * @internal
 * Copyright (C) 2018-2024 by LMI Technologies Inc.
 * Licensed under the MIT License.
 * Redistributed files must retain the above copyright notice.
 */

#ifndef K_API_PROCESS_X_H
#define K_API_PROCESS_X_H

kDeclareFullClassEx(k, kProcess, kObject)

#if defined(K_PLATFORM) 

kFx(kStatus) kProcess_InitPlatformFields(kProcess process);
kFx(kStatus) kProcess_ReleasePlatformFields(kProcess process);

#if defined(K_WINDOWS)

kFx(kStatus) kProcess_ConstructArgument(kProcess process, WCHAR** wargument);

#   define kProcessPlatformFields()             \
        HANDLE process;                         \
        HANDLE stdOutHandle;                   \
        HANDLE stdErrHandle;                   \
        HANDLE stdInHandle;

#   define kProcessPlatformStaticFields()       \
        HANDLE job;                             

#elif defined(K_LINUX)

#   define kPROCESS_PIPE_READ      (0) 
#   define kPROCESS_PIPE_WRITE     (1) 

#   define kPROCESS_DUP2_ACTIONS_MAX    (3)
#   define kPROCESS_CLOSE_ACTIONS_MAX   (3)
#   define kPROCESS_PATH_MAX            (512)
#   define kPROCESS_ARG_MAX             (4096)

typedef struct kProcessChildArgs
{
    struct
    {
        int fileDes;
        int newFileDes;
    } dup2Actions[kPROCESS_DUP2_ACTIONS_MAX];
    kSize dup2ActionCount;

    struct
    {
        int fileDes;
    } closeActions[kPROCESS_CLOSE_ACTIONS_MAX];
    kSize closeActionCount;

    char execPath[kPROCESS_PATH_MAX];
    const char* execArgs[kPROCESS_ARG_MAX];
} kProcessChildArgs;

kInlineFx(void) kProcessChildArgs_AddDup2(kProcessChildArgs* args, int fileDes, int newFileDes)
{
    args->dup2Actions[args->dup2ActionCount].fileDes = fileDes;
    args->dup2Actions[args->dup2ActionCount].newFileDes = newFileDes;
    args->dup2ActionCount++;
}

kInlineFx(void) kProcessChildArgs_AddClose(kProcessChildArgs* args, int fileDes)
{
    args->closeActions[args->closeActionCount++].fileDes = fileDes;
}

#   define kProcessPlatformFields()         \
        k32s process;                       \
        k32s stdinPipe[2];                  \
        k32s stdoutPipe[2];                 \
        k32s stderrPipe[2];                 \
        kSemaphore startSem;                \
        kThread forkThread;                 \
        k64s exitStatus;                    \
        kProcessChildArgs childArgs;

#   define kProcessPlatformStaticFields()   \

kFx(kStatus) kProcess_SetupParent(kProcess process);

/** 
 * Runs the child after (v)forking. This is *not* a member of kProcess, because it can be unsafe to peek/poke into the object
 * if the kProcess "obj" memory is from a shared pool (mmap).
 *
 * @param   args        Post-(v)fork arguments (pre-exec).
 * @param   parentPid   Parent pid.
 */
kFx(void) kProcess_RunChild(const kProcessChildArgs* args, pid_t parentPid);

kFx(kStatus) kProcess_CloseHandles(kProcess process);

kFx(kStatus) kProcess_ForkThreadEntry(kProcess process);

#else

#   define kProcessPlatformFields()
#   define kProcessPlatformStaticFields() 

#endif

typedef struct kProcessClass
{
    kObjectClass base;

    kAtomic32s isRunning;
    kBool isTerminated;
    k64s exitCode;

    kString path;
    kArrayList arguments;

    kStream processStdIn;
    kStream processStdOut;
    kStream processStdErr;
    kBool standardHandlesEnabled;

    kProcessPlatformFields()
} kProcessClass;

typedef struct kProcessVTable
{
    kObjectVTable base;
} kProcessVTable;

typedef struct kProcessStatic
{
    kStream appStdIn;
    kStream appStdOut;
    kStream appStdErr;

    kProcessPlatformStaticFields()
} kProcessStatic;

#endif

#if defined(K_PLATFORM)
#if defined(K_WINDOWS)

kInlineFx(kStatus) kProcess_IsValid(kProcess process)
{
    kObj(kProcess, process);
    return obj->process != NULL;
}

#elif defined(K_LINUX)

kInlineFx(kStatus) kProcess_IsValid(kProcess process)
{
    kObj(kProcess, process);
    return obj->process != -1;
}

#else

kInlineFx(kStatus) kProcess_IsValid(kProcess process)
{
    return kTRUE;
}

#endif
#endif

kFx(kStatus) xkProcess_InitStatic();
kFx(kStatus) xkProcess_ReleaseStatic();

kFx(kStatus) kProcess_WaitImpl(kProcess process, k64u timeout);

kFx(kStatus) kProcess_Init(kProcess process, const kChar* path, kType type, kAlloc alloc);
kFx(kStatus) kProcess_VRelease(kProcess process);
kFx(kStatus) kProcess_Reap(kProcess process);

kFx(kStream) kProcess_AppStdIn();
kFx(kStream) kProcess_AppStdOut();
kFx(kStream) kProcess_AppStdErr();

#endif
