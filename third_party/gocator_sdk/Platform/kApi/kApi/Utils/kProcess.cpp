/** 
 * @file    kProcess.cpp
 *
 * @internal
 * Copyright (C) 2018-2024 by LMI Technologies Inc.
 * Licensed under the MIT License.
 * Redistributed files must retain the above copyright notice.
 */

#define K_PLATFORM
#include <kApi/Utils/kProcess.h>

#include <kApi/Data/kArrayList.h>
#include <kApi/Data/kString.h>
#include <kApi/Io/kPath.h>
#include <kApi/Threads/kSemaphore.h>
#include <kApi/Threads/kThread.h>
#include <kApi/Io/kPipeStream.x.h>

kBeginFullClassEx(k, kProcess)
    kAddVMethod(kProcess, kObject, VRelease)
kEndFullClassEx()

#if defined(K_WINDOWS) || defined(K_LINUX)

kFx(kStatus) kProcess_Construct(kProcess* process, const kChar* path, kAlloc allocator)
{
    kAlloc alloc = kAlloc_Fallback(allocator);
    kStatus status = kOK;

    kCheck(kAlloc_GetObject(alloc, kTypeOf(kProcess), process));

    if (!kSuccess(status = kProcess_Init(*process, path, kTypeOf(kProcess), alloc)))
    {
        kAlloc_FreeRef(alloc, process);
    }

    return status;
}

kFx(kStatus) kProcess_Init(kProcess process, const kChar* path, kType type, kAlloc alloc)
{
    kObjR(kProcess, process);
    kStatus status = kOK;  

    kCheck(kObject_Init(process, type, alloc));

    kAtomic32s_Init(&obj->isRunning, kFALSE);
    obj->isTerminated = kFALSE;
    obj->exitCode = k64S_MAX;
    obj->path = kNULL;
    obj->arguments = kNULL;
    obj->processStdIn = kNULL; 
    obj->processStdOut = kNULL; 
    obj->processStdErr = kNULL; 
    obj->standardHandlesEnabled = kTRUE;

    kTry
    {
        kTest(kString_Construct(&obj->path, path, alloc));
        kTest(kArrayList_Construct(&obj->arguments, kTypeOf(kString), 0, alloc));

        kTest(kProcess_InitPlatformFields(process));
    }
    kCatch(&status)
    {
        kProcess_VRelease(process);
        kEndCatch(status);
    }

    return kOK;
}

kFx(kStatus) kProcess_VRelease(kProcess process)
{
    kObj(kProcess, process);

    kCheck(kProcess_WaitImpl(process, kINFINITE));
    kCheck(kProcess_Reap(process));

    kCheck(kProcess_ReleasePlatformFields(process));

    kCheck(kObject_Destroy(obj->path));
    kCheck(kObject_Dispose(obj->arguments));

    return kObject_VRelease(process);
}

kFx(kStatus) kProcess_AddArgument(kProcess process, const kChar* argument)
{
    kObj(kProcess, process);
    kString arg = kNULL;
    kStatus status = kOK;

    kCheckState(!kAtomic32s_Get(&obj->isRunning));
    
    kCheck(kString_Construct(&arg, argument, kNULL));

    if (!kSuccess(status = kArrayList_AddT(obj->arguments, &arg)))
    {
        kObject_Destroy(arg);
        return status;
    }
        
    return status;    
}

kFx(kStatus) kProcess_AddArguments(kProcess process, const kChar* arguments[], kSize argCount)
{
    kObj(kProcess, process);
    kSize i = 0;

    for (i = 0; i < argCount; i++)
    {
        kCheck(kProcess_AddArgument(process, arguments[i]));
    }

    return kOK;
}

kFx(kStatus) kProcess_ClearArguments(kProcess process)
{
    kObj(kProcess, process);

    kCheckState(!kAtomic32s_Get(&obj->isRunning));

    return kArrayList_Purge(obj->arguments);
}

kFx(kStream) kProcess_AppStdIn()
{
    kStaticObj(kProcess);

    return sobj->appStdIn;
}

kFx(kStream) kProcess_AppStdOut()
{
    kStaticObj(kProcess);

    return sobj->appStdOut;
}

kFx(kStream) kProcess_AppStdErr()
{
    kStaticObj(kProcess);

    return sobj->appStdErr;
}

kFx(k64s) kProcess_ExitCode(kProcess process)
{
    kObj(kProcess, process);
    return obj->exitCode;
}

kFx(kStatus) kProcess_EnableStandardHandles(kProcess process, kBool enabled)
{
    kObj(kProcess, process);

    obj->standardHandlesEnabled = enabled;

    return kOK;
}

kFx(kBool) kProcess_StandardHandlesEnabled(kProcess process)
{
    kObj(kProcess, process);

    return obj->standardHandlesEnabled;
}

#endif

#if defined(K_WINDOWS)

kFx(kStatus) xkProcess_InitStatic()
{
    kStaticObj(kProcess);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION jeli = { 0 };

    sobj->appStdIn = kNULL;
    sobj->appStdOut = kNULL;
    sobj->appStdErr = kNULL;

    // Create a job that when closed, kills any processes attached to it.
    sobj->job = CreateJobObject(NULL, NULL);

    if (sobj->job)
    {
        jeli.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_BREAKAWAY_OK;

        if (SetInformationJobObject(sobj->job, JobObjectExtendedLimitInformation, &jeli, sizeof(jeli)) == 0)
        {
            CloseHandle(sobj->job);
            sobj->job = NULL;
        }
    }

    if (!kSuccess(kPipeStream_ConstructFromHandle(&sobj->appStdIn, GetStdHandle(STD_INPUT_HANDLE), kNULL)))
    {
        sobj->appStdIn = kNULL;
    }

    if (!kSuccess(kPipeStream_ConstructFromHandle(&sobj->appStdOut, GetStdHandle(STD_OUTPUT_HANDLE), kNULL)))
    {
        sobj->appStdOut = kNULL;
    }

    if (!kSuccess(kPipeStream_ConstructFromHandle(&sobj->appStdErr, GetStdHandle(STD_ERROR_HANDLE), kNULL)))
    {
        sobj->appStdErr = kNULL;
    }

    return kOK;
}

kFx(kStatus) xkProcess_ReleaseStatic()
{
    kStaticObj(kProcess);

    kCheck(kObject_Destroy(sobj->appStdIn));
    kCheck(kObject_Destroy(sobj->appStdOut));
    kCheck(kObject_Destroy(sobj->appStdErr));

    if (sobj->job)
    {
        CloseHandle(sobj->job);
        sobj->job = NULL;
    }

    return kOK;
}

kFx(kStatus) kProcess_InitPlatformFields(kProcess process)
{
    kObj(kProcess, process);

    obj->process = kNULL;
    obj->stdOutHandle = kNULL;
    obj->stdErrHandle = kNULL; 
    obj->stdInHandle = kNULL;

    return kOK;
}

kFx(kStatus) kProcess_ReleasePlatformFields(kProcess process)
{
    return kOK;
}

kFx(kStatus) kProcess_ConstructArgument(kProcess process, WCHAR** wargument)
{
    kObj(kProcess, process);
    kString argument = kNULL;
    k32s arglength  = 0;
    kSize i = 0;
    kChar fileName[kPATH_MAX];
    kStatus status = kOK;

    kCheck(kPath_FileName(kString_Chars(obj->path), fileName, kCountOf(fileName)));

    kTry
    {
        kTest(kString_Construct(&argument, fileName, kNULL));

        for (i = 0; i < kArrayList_Count(obj->arguments); i++)
        {
            kString a = kArrayList_AsT(obj->arguments, i, kString);

            kTest(kString_Addf(argument, " \"%s\"", kString_Chars(a)));
        }

        if (!(arglength = MultiByteToWideChar(CP_UTF8, 0, kString_Chars(argument), -1, kNULL, 0)))
        {
            kThrow(kERROR_OS);
        }

        kTest(kMemAlloc(arglength * sizeof(WCHAR) + 2, wargument));

        if (MultiByteToWideChar(CP_UTF8, 0, kString_Chars(argument), -1, *wargument, arglength) == 0)
        {
            kThrow(kERROR_PARAMETER);
        }
    }
    kCatchEx(&status)
    {
        kMemFree(&wargument);
        *wargument = kNULL;

        kEndCatchEx(status);
    }
    kFinallyEx
    {
        kObject_Destroy(argument);

        kEndFinally();
    }

    return kOK;
}

kFx(kStatus) kProcess_Start(kProcess process)
{
    kObj(kProcess, process);
    kStaticObj(kProcess);
    PROCESS_INFORMATION procInfo = { 0 };
    STARTUPINFO startupInfo = { 0 };
    HANDLE outputWriteOut = NULL;
    HANDLE outputWriteErr = NULL;
    kStatus exception = kOK;
    HANDLE inputRead = kNULL;
    WCHAR wpath[MAX_PATH];
    WCHAR* wargument = kNULL;
    
    kCheckState(!kAtomic32s_Get(&obj->isRunning));

    obj->exitCode = k64S_MAX;

    if (MultiByteToWideChar(CP_UTF8, 0, kString_Chars(obj->path), -1, wpath, kCountOf(wpath)) == 0)
    {
        return kERROR_PARAMETER;
    }    
    
    kTry
    {   
        kTest(kProcess_ConstructArgument(process, &wargument));

        startupInfo.cb = sizeof(startupInfo);

        if (obj->standardHandlesEnabled)
        {
            SECURITY_ATTRIBUTES securityAttrs = { 0 };

            securityAttrs.nLength = sizeof(securityAttrs);
            securityAttrs.lpSecurityDescriptor = NULL;
            securityAttrs.bInheritHandle = kTRUE;

            if (!CreatePipe(&obj->stdOutHandle, &outputWriteOut, &securityAttrs, 0))
            {
                kThrow(kERROR_OS);
            }

            if (!CreatePipe(&obj->stdErrHandle, &outputWriteErr, &securityAttrs, 0))
            {
                kThrow(kERROR_OS);
            }

            // Don't inherit the read handle - otherwise child cannot detect read handle closing.
            if (!SetHandleInformation(obj->stdOutHandle, HANDLE_FLAG_INHERIT, 0))
            {
                kThrow(kERROR_OS);
            }

            if (!SetHandleInformation(obj->stdErrHandle, HANDLE_FLAG_INHERIT, 0))
            {
                kThrow(kERROR_OS);
            }

            if (!CreatePipe(&inputRead, &obj->stdInHandle, &securityAttrs, 0))
            {
                kThrow(kERROR_OS);
            }

            if (!SetHandleInformation(obj->stdInHandle, HANDLE_FLAG_INHERIT, 0))
            {
                kThrow(kERROR_OS);
            }

            startupInfo.hStdOutput = outputWriteOut;
            startupInfo.hStdError = outputWriteErr;
            startupInfo.hStdInput = inputRead;
            startupInfo.dwFlags = STARTF_USESTDHANDLES;
        }

        if (!CreateProcess(wpath, wargument, NULL, NULL, TRUE, CREATE_NO_WINDOW | CREATE_BREAKAWAY_FROM_JOB, NULL, NULL, &startupInfo, &procInfo))
        {
            kThrow(kERROR_OS);
        }

        if (!CloseHandle(procInfo.hThread))
        {
            kThrow(kERROR_OS);
        }
        obj->process = procInfo.hProcess;

        // Add the process to the job to guarantee clean-up.

        if (sobj->job)
        {
            if (AssignProcessToJobObject(sobj->job, procInfo.hProcess) == 0)
            {
                kThrow(kERROR_OS);
            }
        }

        // Don't need to write. Open write handle prevents detection of write handle closing by child.
        if (!kIsNull(outputWriteOut) && !CloseHandle(outputWriteOut))
        {
            kThrow(kERROR_OS);
        }
        outputWriteOut = NULL;

        if (!kIsNull(outputWriteErr) && !CloseHandle(outputWriteErr))
        {
            kThrow(kERROR_OS);
        }
        outputWriteErr = NULL;

        if (!kIsNull(inputRead) && !CloseHandle(inputRead))
        {
            kThrow(kERROR_OS);
        }
        inputRead = NULL;

        kAtomic32s_Exchange(&obj->isRunning, kTRUE);
    }
    kCatchEx(&exception)
    {
        if (outputWriteOut)
        {
            CloseHandle(outputWriteOut);
        }

        if (outputWriteErr)
        {
            CloseHandle(outputWriteErr);
        }

        if (inputRead)
        {
            CloseHandle(inputRead);
        }

        if (obj->stdOutHandle)
        {
            CloseHandle(obj->stdOutHandle);
            obj->stdOutHandle = NULL;
        }

        if (obj->stdErrHandle)
        {
            CloseHandle(obj->stdErrHandle);
            obj->stdErrHandle = NULL;
        }

        if (obj->stdInHandle)
        {
            CloseHandle(obj->stdInHandle);
            obj->stdInHandle = NULL;
        }

        if (obj->process)
        {
            TerminateProcess(obj->process, 1);
            CloseHandle(obj->process);
            obj->process = NULL;
        }

        kEndCatchEx(exception);
    }
    kFinallyEx
    {
        kMemFree(wargument);
        kEndFinallyEx();
    }

    return kOK;
}

kFx(kBool) kProcess_IsAlive(kProcess process)
{
    kObj(kProcess, process);
    DWORD exitCode = 0;

    if (!kAtomic32s_Get(&obj->isRunning))
    {
        return kFALSE;
    }

    if (obj->isTerminated)
    {
        return kFALSE;
    }

    if (!GetExitCodeProcess(obj->process, &exitCode))
    {
        return kFALSE;
    }

    return exitCode == STILL_ACTIVE;
}

kFx(kStatus) kProcess_Terminate(kProcess process)
{
    kObj(kProcess, process);

    if (obj->process != NULL)
    {
        TerminateProcess(obj->process, 0);

        obj->isTerminated = kTRUE;
    }

    return kOK;
}

kFx(kStatus) kProcess_Reap(kProcess process)
{
    kObj(kProcess, process);
    DWORD ec = 0;

    if (obj->process != kNULL)
    {
        if (!obj->isTerminated)
        {
            if (GetExitCodeProcess(obj->process, &ec))
            {
                obj->exitCode = ec;
            }
        }

        CloseHandle(obj->process);
        obj->process = kNULL;
    }

    if (obj->stdInHandle != kNULL)
    {
        CloseHandle(obj->stdInHandle);
        obj->stdInHandle = kNULL;
    }

    if (obj->stdOutHandle != kNULL)
    {
        CloseHandle(obj->stdOutHandle);
        obj->stdOutHandle = kNULL;
    }

    if (obj->stdErrHandle != kNULL)
    {
        CloseHandle(obj->stdErrHandle);
        obj->stdErrHandle = kNULL;
    }

    kCheck(kDestroyRef(&obj->processStdIn));
    kCheck(kDestroyRef(&obj->processStdOut));
    kCheck(kDestroyRef(&obj->processStdErr));

    kAtomic32s_Exchange(&obj->isRunning, kFALSE);

    return kOK;
}

kFx(kStatus) kProcess_WaitImpl(kProcess process, k64u timeout)
{
    kObj(kProcess, process);
    DWORD osTimeout, waitResult;

    if (!kAtomic32s_Get(&obj->isRunning))
    {
        return kOK;
    }

    osTimeout = (DWORD) xkTimeToKernelTime(timeout);
    waitResult = WaitForSingleObject(obj->process, osTimeout);

    if (waitResult == WAIT_TIMEOUT)
    {
        return kERROR_TIMEOUT;
    }
    else if (waitResult == WAIT_OBJECT_0)
    {
        /* good */
    }
    else
    {
        return kERROR_OS;
    }

    return kProcess_Reap(process);
}

kFx(kStream) kProcess_StdIn(kProcess process)
{
    kObj(kProcess, process);

    if (!obj->standardHandlesEnabled)
    {
        return kNULL;
    }
    
    if (kIsNull(obj->processStdIn) && kAtomic32s_Get(&obj->isRunning))
    {
        kPipeStream_ConstructFromHandle(&obj->processStdIn, obj->stdInHandle, kNULL);
    }
    
    return obj->processStdIn;
 }

kFx(kStream) kProcess_StdOut(kProcess process)
{
    kObj(kProcess, process);

    if (!obj->standardHandlesEnabled)
    {
        return kNULL;
    }

    if (kIsNull(obj->processStdOut) && kAtomic32s_Get(&obj->isRunning))
    {
        kPipeStream_ConstructFromHandle(&obj->processStdOut, obj->stdOutHandle, kNULL);
    }

    return obj->processStdOut;
}

kFx(kStream) kProcess_StdErr(kProcess process)
{
    kObj(kProcess, process);

    if (!obj->standardHandlesEnabled)
    {
        return kNULL;
    }

    if (kIsNull(obj->processStdErr) && kAtomic32s_Get(&obj->isRunning))
    {
        kPipeStream_ConstructFromHandle(&obj->processStdErr, obj->stdErrHandle, kNULL);
    }

    return obj->processStdErr;
}

kFx(k32s) kProcess_Id()
{
    return _getpid();
}

#elif defined(K_LINUX)

kFx(kStatus) xkProcess_InitStatic()
{
    kStaticObj(kProcess);
    kStatus status = kOK;

    sobj->appStdIn = kNULL;
    sobj->appStdOut = kNULL;
    sobj->appStdErr = kNULL;
    
    kTry
    {
        kTest(kPipeStream_ConstructFromHandle(&sobj->appStdIn, STDIN_FILENO, kNULL));
        kTest(kPipeStream_ConstructFromHandle(&sobj->appStdOut, STDOUT_FILENO, kNULL));
        kTest(kPipeStream_ConstructFromHandle(&sobj->appStdErr, STDERR_FILENO, kNULL));
    }
    kCatch(&status)
    {
        kObject_Destroy(sobj->appStdIn);
        kObject_Destroy(sobj->appStdOut);
        kObject_Destroy(sobj->appStdErr);
        
        sobj->appStdIn = kNULL;
        sobj->appStdOut = kNULL;
        sobj->appStdErr = kNULL;
        
        kEndCatch(status);
    }

    return kOK;
}

kFx(kStatus) xkProcess_ReleaseStatic()
{
    kStaticObj(kProcess);

    kCheck(kObject_Destroy(sobj->appStdIn));
    kCheck(kObject_Destroy(sobj->appStdOut));
    kCheck(kObject_Destroy(sobj->appStdErr));

    return kOK;
}

kFx(kStatus) kProcess_InitPlatformFields(kProcess process)
{
    kObj(kProcess, process);

    obj->process = -1;
    obj->stdinPipe[kPROCESS_PIPE_READ] = -1;
    obj->stdinPipe[kPROCESS_PIPE_WRITE] = -1;
    obj->stdoutPipe[kPROCESS_PIPE_READ] = -1;
    obj->stdoutPipe[kPROCESS_PIPE_WRITE] = -1;
    obj->stderrPipe[kPROCESS_PIPE_READ] = -1;
    obj->stderrPipe[kPROCESS_PIPE_WRITE] = -1;
    obj->startSem = kNULL; 
    obj->forkThread = kNULL; 
    obj->exitStatus = k64S_MAX;

    kZero(obj->childArgs);

    kCheck(kSemaphore_Construct(&obj->startSem, 0, kNULL));
    kCheck(kThread_Construct(&obj->forkThread, kNULL));

    return kOK;
}

kFx(kStatus) kProcess_ReleasePlatformFields(kProcess process)
{
    kObj(kProcess, process);

    kCheck(kObject_Destroy(obj->startSem));
    kCheck(kObject_Destroy(obj->forkThread));

    return kOK;
}

kFx(kStatus) kProcess_Start(kProcess process)
{
    kObj(kProcess, process);

    kCheckState(!kAtomic32s_Get(&obj->isRunning));

    obj->exitCode = obj->exitStatus = k64S_MAX;

    kCheck(kThread_Start(obj->forkThread, kProcess_ForkThreadEntry, process));

    kCheck(kSemaphore_Wait(obj->startSem, kINFINITE));
    
    if (!kAtomic32s_Get(&obj->isRunning))
    {
        kCheck(kThread_Join(obj->forkThread, kINFINITE, kNULL));
        return kERROR_OS;
    }

    return kOK;
}

kFx(kStatus) kProcess_ForkThreadEntry(kProcess process)
{
    kObj(kProcess, process);
    pid_t parent = 0;
    k32s status = 0;
    kSize i = 0;
    kSize argumentCount = 0;
    const kChar* fileNameStart = kNULL;

    kTry
    {
        if (pipe(obj->stdinPipe) < 0 ||
            pipe(obj->stdoutPipe) < 0 ||
            pipe(obj->stderrPipe) < 0) 
        {
            kProcess_CloseHandles(process);
            kThrow(kERROR_OS); 
        }

        kZero(obj->childArgs);

        if (obj->standardHandlesEnabled)
        {
            kProcessChildArgs_AddDup2(&obj->childArgs, obj->stdinPipe[kPROCESS_PIPE_READ], STDIN_FILENO);
            kProcessChildArgs_AddDup2(&obj->childArgs, obj->stdoutPipe[kPROCESS_PIPE_WRITE], STDOUT_FILENO);
            kProcessChildArgs_AddDup2(&obj->childArgs, obj->stderrPipe[kPROCESS_PIPE_WRITE], STDERR_FILENO);

            kProcessChildArgs_AddClose(&obj->childArgs, obj->stdinPipe[kPROCESS_PIPE_WRITE]);
            kProcessChildArgs_AddClose(&obj->childArgs, obj->stdoutPipe[kPROCESS_PIPE_READ]);
            kProcessChildArgs_AddClose(&obj->childArgs, obj->stderrPipe[kPROCESS_PIPE_READ]);
        }

        kTest(kStrCopy(obj->childArgs.execPath, kCountOf(obj->childArgs.execPath), kString_Chars(obj->path)));
        kTest(!kIsNull(fileNameStart = kStrFindLast(kString_Chars(obj->path), "/")));

        argumentCount = kArrayList_Count(obj->arguments);
        obj->childArgs.execArgs[0] = &fileNameStart[1];
        obj->childArgs.execArgs[argumentCount + 1] = kNULL;

        for (i = 0; i < argumentCount; i++)
        {
            kString item = kArrayList_AsT(obj->arguments, i, kString); 
            obj->childArgs.execArgs[i + 1] = kString_Chars(item);
        }

        parent = getpid();

        // This creates a new process by duplicating the current one.
        // Return code is used to distinguish between child and parent:
        //    0 --> child process
        //   >0 --> this is the parent process
        obj->process = vfork();

        if (obj->process == -1) // Failure
        {
            kProcess_CloseHandles(process);
            kThrow(kERROR_OS); 
        }     
    
        // From here we have two different processes - parent and child.

        if (obj->process > 0) // We are in the parent process.
        {
            kAtomic32s_Exchange(&obj->isRunning, kTRUE);
            kTest(kProcess_SetupParent(process));
        }
        else // We are in the child process. Will never return as exit() is called inside the function
        {
            kProcess_RunChild(&obj->childArgs, parent);
        }
    }
    kFinally
    {
        kSemaphore_Post(obj->startSem);
        kEndFinally();
    }

    // wait for the child to exit
    do 
    {
        if (waitpid(obj->process, &status, WUNTRACED | WCONTINUED) == -1)
        {
            return kOK;
        }

        if (WIFEXITED(status)) 
        {
            obj->exitStatus =  WEXITSTATUS(status);
        }
    } while (!WIFEXITED(status) && !WIFSIGNALED(status));

    return kOK;
}

kFx(kStatus) kProcess_SetupParent(kProcess process)
{
    kObj(kProcess, process);
    kStatus status = kOK;

    if (close(obj->stdinPipe[kPROCESS_PIPE_READ]))
    {
        status = kERROR_OS;
    }

    if (close(obj->stdoutPipe[kPROCESS_PIPE_WRITE]))
    {
        status = kERROR_OS;
    }

    if (close(obj->stderrPipe[kPROCESS_PIPE_WRITE]))
    {
        status = kERROR_OS;
    }
    
    obj->stdinPipe[kPROCESS_PIPE_READ] = -1;
    obj->stdoutPipe[kPROCESS_PIPE_WRITE] = -1;
    obj->stderrPipe[kPROCESS_PIPE_WRITE] = -1;

    return status;
}

kFx(void) kProcess_RunChild(const kProcessChildArgs* args, pid_t parentPid)
{
    k32s returnCode = 0;
    kSize i;

    // SIGTERM is the signal that child will get when _THREAD_ which forked ends.
    // That's why we use a separate thread for kProcess_Start on Linux.
    // PR_SET_PDEATHSIG is not supported on WSL so do not care if it's EINVAL.
    if (prctl(PR_SET_PDEATHSIG, SIGTERM) == -1 && errno != EINVAL) 
    { 
        exit(-1);
    }

    // Test in case the original parent exited just before the prctl() call.
    if (getppid() != parentPid)
    {
        exit(0);
    }

    for (i = 0; i < args->dup2ActionCount; i++)
    {
        dup2(args->dup2Actions[i].fileDes, args->dup2Actions[i].newFileDes);
    }

    for (i = 0; i < args->closeActionCount; i++)
    {
        close(args->closeActions[i].fileDes);
    }

    // This replaces the current process image with a new process image.
    // Only returns if an error has occurred.
    execv(args->execPath, (char* const*) args->execArgs);

    exit(returnCode);
}

kFx(kBool) kProcess_IsAlive(kProcess process)
{
    kObj(kProcess, process);

    if (!kAtomic32s_Get(&obj->isRunning))
    {
        return kFALSE;
    }

    if (obj->isTerminated)
    {
        return kFALSE;
    }

    if (kThread_Join(obj->forkThread, 0, kNULL) == kERROR_TIMEOUT)
    {
        return kTRUE;
    }

    return kFALSE;
}

kFx(kStatus) kProcess_Terminate(kProcess process)
{
    kObj(kProcess, process);
    kStatus status = kOK;

    if (obj->process != -1)
    {
        if ((kill(obj->process, SIGKILL) < 0) && (errno != ESRCH))
        {
            status = kERROR_OS;
        }

        obj->isTerminated = kTRUE;
    }

    return status;
}

kFx(kStatus) kProcess_Reap(kProcess process)
{
    kObj(kProcess, process);
    kStatus status = kOK;

    if (!obj->isTerminated)
    {
        obj->exitCode = obj->exitStatus;
    }

    if ((obj->stdinPipe[kPROCESS_PIPE_WRITE] >= 0) && close(obj->stdinPipe[kPROCESS_PIPE_WRITE]))
    {
        status = kERROR_OS;
    }

    if ((obj->stdoutPipe[kPROCESS_PIPE_READ] >= 0) && close(obj->stdoutPipe[kPROCESS_PIPE_READ]))
    {
        status = kERROR_OS;
    }

    if ((obj->stderrPipe[kPROCESS_PIPE_READ] >= 0) && close(obj->stderrPipe[kPROCESS_PIPE_READ]))
    {
        status = kERROR_OS;
    }

    obj->stdinPipe[kPROCESS_PIPE_WRITE] = -1;
    obj->stdoutPipe[kPROCESS_PIPE_READ] = -1;
    obj->stderrPipe[kPROCESS_PIPE_READ] = -1;

    obj->process = -1;

    kCheck(kDestroyRef(&obj->processStdIn));
    kCheck(kDestroyRef(&obj->processStdOut));
    kCheck(kDestroyRef(&obj->processStdErr));

    kAtomic32s_Exchange(&obj->isRunning, kFALSE);

    return status;
}

kFx(kStatus) kProcess_WaitImpl(kProcess process, k64u timeout)
{
    kObj(kProcess, process);

    if (!kAtomic32s_Get(&obj->isRunning))
    {
        return kOK;
    }

    kCheck(kThread_Join(obj->forkThread, timeout, kNULL));

    return kProcess_Reap(process);
}

kFx(kStream) kProcess_StdIn(kProcess process)
{
    kObj(kProcess, process);

    if (!obj->standardHandlesEnabled)
    {
        return kNULL;
    }

    if (kIsNull(obj->processStdIn) && kAtomic32s_Get(&obj->isRunning))
    {
        kPipeStream_ConstructFromHandle(&obj->processStdIn, obj->stdinPipe[kPROCESS_PIPE_WRITE], kNULL);
    }

    return obj->processStdIn;
}

kFx(kStream) kProcess_StdOut(kProcess process)
{
    kObj(kProcess, process);

    if (!obj->standardHandlesEnabled)
    {
        return kNULL;
    }

    if (kIsNull(obj->processStdOut) && kAtomic32s_Get(&obj->isRunning))
    {
        kPipeStream_ConstructFromHandle(&obj->processStdOut, obj->stdoutPipe[kPROCESS_PIPE_READ], kNULL);
    }

    return obj->processStdOut;
}

kFx(kStream) kProcess_StdErr(kProcess process)
{
    kObj(kProcess, process);

    if (!obj->standardHandlesEnabled)
    {
        return kNULL;
    }

    if (kIsNull(obj->processStdErr) && kAtomic32s_Get(&obj->isRunning))
    {
        kPipeStream_ConstructFromHandle(&obj->processStdErr, obj->stderrPipe[kPROCESS_PIPE_READ], kNULL);
    }

    return obj->processStdErr;
}

kFx(kStatus) kProcess_CloseHandles(kProcess process)
{
    kObj(kProcess, process);
    kStatus status = kOK;

    if ((obj->stdinPipe[kPROCESS_PIPE_READ] >= 0) && (close(obj->stdinPipe[kPROCESS_PIPE_READ]))) status = kERROR_OS;
    if ((obj->stdinPipe[kPROCESS_PIPE_WRITE] >= 0) && (close(obj->stdinPipe[kPROCESS_PIPE_WRITE]))) status = kERROR_OS;
    if ((obj->stdoutPipe[kPROCESS_PIPE_READ] >= 0) && (close(obj->stdoutPipe[kPROCESS_PIPE_READ]))) status = kERROR_OS;
    if ((obj->stdoutPipe[kPROCESS_PIPE_WRITE] >= 0) && (close(obj->stdoutPipe[kPROCESS_PIPE_WRITE]))) status = kERROR_OS;
    if ((obj->stderrPipe[kPROCESS_PIPE_READ] >= 0) && (close(obj->stderrPipe[kPROCESS_PIPE_READ]))) status = kERROR_OS;
    if ((obj->stderrPipe[kPROCESS_PIPE_WRITE] >= 0) && (close(obj->stderrPipe[kPROCESS_PIPE_WRITE]))) status = kERROR_OS;


    obj->stdinPipe[kPROCESS_PIPE_READ] = -1;
    obj->stdinPipe[kPROCESS_PIPE_WRITE] = -1;
    obj->stdoutPipe[kPROCESS_PIPE_READ] = -1;
    obj->stdoutPipe[kPROCESS_PIPE_WRITE] = -1;
    obj->stderrPipe[kPROCESS_PIPE_READ] = -1;
    obj->stderrPipe[kPROCESS_PIPE_WRITE] = -1;

    return status;
}

kFx(k32s) kProcess_Id()
{
    return getpid();
}

#else

kFx(kStatus) xkProcess_InitStatic()
{
    return kOK;
}

kFx(kStatus) xkProcess_ReleaseStatic()
{
    return kOK;
}

kFx(kStatus) kProcess_Construct(kProcess* process, const kChar* path, kAlloc allocator)
{
    return kERROR_UNIMPLEMENTED;
}

kFx(kStatus) kProcess_Init(kProcess process, const kChar* path, kType type, kAlloc alloc)
{
    return kERROR_UNIMPLEMENTED;
}

kFx(kStatus) kProcess_VRelease(kProcess process)
{
    return kERROR_UNIMPLEMENTED;
}

kFx(kStatus) kProcess_AddArgument(kProcess process, const kChar* argument)
{
    return kERROR_UNIMPLEMENTED;
}

kFx(kStatus) kProcess_AddArguments(kProcess process, const kChar* arguments[], kSize argCount)
{
    return kERROR_UNIMPLEMENTED;
}

kFx(kStatus) kProcess_ClearArguments(kProcess process)
{
    return kERROR_UNIMPLEMENTED;
}

kFx(kStatus) kProcess_Start(kProcess process)
{
    return kERROR_UNIMPLEMENTED;
}

kFx(kBool) kProcess_IsAlive(kProcess process)
{
    return kFALSE;
}

kFx(kStatus) kProcess_Terminate(kProcess process)
{
    return kERROR_UNIMPLEMENTED;
}

kFx(kStatus) kProcess_Reap(kProcess process)
{
    return kERROR_UNIMPLEMENTED;
}

kFx(kStatus) kProcess_WaitImpl(kProcess process, k64u timeout)
{
    return kERROR_UNIMPLEMENTED;
}

kFx(kStream) kProcess_StdIn(kProcess process)
{
    return kNULL;
}

kFx(kStream) kProcess_StdOut(kProcess process)
{
    return kNULL;
}

kFx(kStream) kProcess_StdErr(kProcess process)
{
    return kNULL;
}

kFx(kStream) kProcess_AppStdIn()
{
    return kNULL;
}

kFx(kStream) kProcess_AppStdOut()
{
    return kNULL;
}

kFx(kStream) kProcess_AppStdErr()
{
    return kNULL;
}

kFx(k32s) kProcess_Id()
{
    return 0;
}

#endif


