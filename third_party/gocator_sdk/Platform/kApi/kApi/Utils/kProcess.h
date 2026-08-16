/** 
 * @file    kProcess.h
 * @brief   Declares the kProcess class. 
 *
 * @internal
 * Copyright (C) 2018-2024 by LMI Technologies Inc.
 * Licensed under the MIT License.
 * Redistributed files must retain the above copyright notice.
 */

#ifndef K_API_PROCESS_H
#define K_API_PROCESS_H

#include <kApi/kApiDef.h>
#include <kApi/Utils/kProcess.x.h>

/**
 * @class   kProcess
 * @extends kObject
 * @ingroup kApi-Utils
 * @brief   Represents a process. 
 
 * kProcess class allows to start a process and offers the possibility to define multiple arguments.
 * For Input and output kPipeStream class can be used. For a kPipeStream of stdin, stdout or stderr 
 * for the current process see kStdIn(), kStdOut() and kStdErr() in kPipeStream.h.
 * 
 */
//typedef kObject kProcess;        --forward-declared in kApiDef.x.h


/** 
 * Constructs a kProcess object.
 *
 * Allows the creation and start of a child process. The search path will not be used when looking for the executable.
 * This class is not thread safe. The child process gets terminated when the calling process ends.
 *
 * Windows specific:
 * The path string can specify the full path and file name of the module to execute or it can specify a partial name. 
 * In the case of a partial name, the function uses the current drive and current directory to complete the specification. 
 *
 * @public              @memberof kProcess
 * @param   process     Destination for the constructed object handle.
 * @param   path        Path of the new process. 
 * @param   allocator   Memory allocator (or kNULL for default). 
 * @return              Operation status. 
 */
kFx(kStatus) kProcess_Construct(kProcess* process, const kChar* path, kAlloc allocator); 

/** 
 * Adds an argument. Can only be used when process is not running.
 *
 * @public              @memberof kProcess
 * @param   process     kProcess object.
 * @param   argument    New argument. 
 * @return              Operation status. 
 */
kFx(kStatus) kProcess_AddArgument(kProcess process, const kChar* argument);

/** 
 * Adds an array of arguments. Can only be used when process is not running.
 *
 * @public              @memberof kProcess
 * @param   process     kProcess object.
 * @param   arguments   Array with the arguments.
 * @param   argCount    Number of arguments.
 * @return              Operation status. 
 */
kFx(kStatus) kProcess_AddArguments(kProcess process, const kChar* arguments[], kSize argCount);

/** 
 * Deletes all arguments. Can only be used when process is not running.
 *
 * @public              @memberof kProcess
 * @param   process     kProcess object.
 * @return              Operation status. 
 */
kFx(kStatus) kProcess_ClearArguments(kProcess process);

/** 
 * Starts the new process. 
 *
 * Starts the new process given the path and arguments. You have to call kProcess_Wait() between 
 * consecutive calls of kProcess_Start. There is no guaranty when exactly after this call the child process 
 * has really started. 
 *
 * @public          @memberof kProcess
 * @param   process kProcess object. 
 * @return          Operation status.  
 */
kFx(kStatus) kProcess_Start(kProcess process);

/** 
 * Determines if the process is still alive.
 *
 * @public          @memberof kProcess
 * @param   process kProcess object. 
 * @return          kTRUE is process is still alive, otherwise kFALSE.  
 */
kFx(kBool) kProcess_IsAlive(kProcess process);

/** 
 * Initiates process termination.
 * 
 * This function can be called to terminate the process and will return immediately.
 * 
 * Note that a terminated process does not have a meaningful exit code.
 *
 * @public              @memberof kProcess
 * @param   process     kProcess object.
 * @return              Operation status. 
 */
kFx(kStatus) kProcess_Terminate(kProcess process);

/** 
 * Waits for child process completion.
 * 
 * @public            @memberof kProcess
 * @param   process   kProcess object. 
 * @param   timeout   Timeout (microseconds).  
 * @return            Operation status.  
 */
kInlineFx(kStatus) kProcess_Wait(kProcess process, k64u timeout)
{
    return kProcess_WaitImpl(process, timeout);
}

/** 
 * Gets the exit code of a process. 
 *
 * Make sure to wait for process completion via kProcess_Wait() before calling this function.
 * 
 * Note that a terminated process (via kProcess_Terminate()) will not have a meaningful exit code.
 * Note that in process land, zero means success, while non-zero means failure.
 *
 * @public              @memberof kProcess
 * @param   process     kProcess object.
 * @return              Exit code. 
 */
kFx(k64s) kProcess_ExitCode(kProcess process);

/** 
 * Returns the stdin of the process as kStream.
 *
 * It is only possible to write to this stream. Process must be alive. kStream is only valid until
 * the next call of kProcess_Wait().
 *
 * @public           @memberof kProcess
 * @param   process  kProcess object. 
 * @return           kStream object.
 */
kFx(kStream) kProcess_StdIn(kProcess process);

/** 
 * Returns the stdout of the process as kStream.
 *
 * It is only possible to read from this stream. Process must be alive. kStream is only valid until
 * the next call of kProcess_Wait().
 *
 * @public           @memberof kProcess
 * @param   process  kProcess object. 
 * @return           kStream object.  
 */
kFx(kStream) kProcess_StdOut(kProcess process);

/** 
 * Returns the stderr of the process as kStream.
 *
 * It is only possible to read from this stream. Process must be alive. kStream is only valid until
 * the next call of kProcess_Wait().
 *
 * @public           @memberof kProcess
 * @param   process  kProcess object.
 * @return           kStream object.  
 */
kFx(kStream) kProcess_StdErr(kProcess process);

/** 
 * Returns the process id of the current process.
 *
 * On unsupported platforms 0 is returned.
 *
 * @public           @memberof kProcess
 * @return           Process id.  
 */
kFx(k32s) kProcess_Id();

/**
 * Enable read or write from/to the standard handles (stdin/stdout/stderr).
 * 
 * If the standard handles are enabled, kProcess_StdIn(), kProcess_StdOut() and
 * kProcess_StdErr() will return valid handles. Otherwise, they will return kNULL.
 *
 * By default, the handles are enabled.
 *
 * @public           @memberof kProcess
 * @return           Process id.
 */
kFx(kStatus) kProcess_EnableStandardHandles(kProcess process, kBool enabled);

/**
 * Reports whether standard handles are currently enabled.
 *
 * @public           @memberof kProcess
 * @return           Process id.
 */
kFx(kBool) kProcess_StandardHandlesEnabled(kProcess process);

#endif
