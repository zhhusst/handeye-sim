/** 
 * @file    kTimer.h
 * @brief   Declares the kTimer class. 
 *
 * @internal
 * Copyright (C) 2005-2024 by LMI Technologies Inc.
 * Licensed under the MIT License.
 * Redistributed files must retain the above copyright notice.
 */
#ifndef K_API_TIMER_H
#define K_API_TIMER_H

#include <kApi/kApiDef.h>
#include <kApi/Threads/kTimer.x.h>

/**
 * @class   kTimer
 * @extends kObject
 * @ingroup kApi-Threads
 * @brief   Represents an interval timer. 
 */
//typedef kObject kTimer;     --forward-declared in kApiDef.x.h 

/** 
 * Provides the current time in underlying clock ticks. 
 *
 * This function provides a time value in ticks; successive calls to this function can be 
 * used to measure a time interval. 
 *
 * @public      @memberof kTimer
 * @return      Current time, in clock ticks. 
 * @see         kTimer_FromTicks, kTimer_ToTicks
 */
kInlineFx(k64u) kTimer_Ticks() 
{ 
    return kApiLib_TimerQueryHandler()(); 
}

/** 
 * Converts the specified number of clock ticks to microseconds.
 *
 * @public              @memberof kTimer
 * @param   ticks       Time in clock ticks.
 * @return              Time in microseconds. 
 */
kInlineFx(k64u) kTimer_FromTicks(k64u ticks) 
{ 
    return ticks * kApiLib_TimerMultiplier() / kApiLib_TimerDivider(); 
}

/** 
 * Converts the specified number of microseconds to clock ticks.
 *
 * @public              @memberof kTimer
 * @param   time        Time in microseconds.
 * @return              Time in clock ticks.
 */
kInlineFx(k64u) kTimer_ToTicks(k64u time) 
{ 
    return kApiLib_TimerDivider() * time / kApiLib_TimerMultiplier(); 
}

/** 
 * Provides the current time in microseconds. 
 *
 * This function provides a time value in microseconds; successive calls to this function can be 
 * used to measure a time interval. 
 *
 * @public              @memberof kTimer
 * @return              Current time, in microseconds. 
 */
kInlineFx(k64u) kTimer_Now() 
{ 
    return kTimer_FromTicks(kTimer_Ticks()); 
}

/** 
 * Performs CPU busywork for at least the specified duration.
 *
 * Thread sleep functions (e.g., kThread_Sleep kThread_SleepAtLeast) should usually be used to 
 * implement delays because they avoid wasting CPU cycles. However, if a very short 
 * (e.g., << 1 ms) delay is desired, thread sleep functions will likely result in longer delays 
 * than requested due to kernel timer resolutions. In most cases, this will not be problematic; 
 * however, if the excess delay incurred by thread sleeps is signficantly disadvantageous, 
 * kTimer_Spin can be used instead, with the caveat that it will waste CPU cycles for the duration
 * of the specified delay.
 *
 * @public              @memberof kTimer
 * @return              Delay time, in microseconds. 
 */
kInlineFx(k64u) kTimer_Spin(k64u duration) 
{ 
    k64u startTime = kTimer_Now(); 

    while (kTimer_Now() < (startTime + duration)); 

    return kOK; 
}

/** 
 * Constructs a timer object.
 *
 * @public              @memberof kTimer
 * @param   timer       Destination for the constructed object handle.
 * @param   allocator   Memory allocator (or kNULL for default). 
 * @return              Operation status. 
 */
kFx(kStatus) kTimer_Construct(kTimer* timer, kAlloc allocator); 

/** 
 * Starts the timer.
 *
 * If the timer will be used to count up, rather than count down, totalTime can be zero.
 *
 * @public              @memberof kTimer
 * @param   timer       Timer object. 
 * @param   totalTime   Total time to count down, in microseconds. 
 * @return              Operation status. 
 */
kFx(kStatus) kTimer_Start(kTimer timer, k64u totalTime); 

/** 
 * Stops the timer.
 *
 * kTimer_Elapsed can be used to report the time between start and stop. 
 * 
 * The use of kTimer_Stop is strictly optional; it is not necessary to call kTimer_Stop for each 
 * invocation of kTimer_Start. 
 *
 * @public              @memberof kTimer
 * @param   timer       Timer object. 
 * @return              Operation status. 
 */
kFx(kStatus) kTimer_Stop(kTimer timer); 

/** 
 * Reports whether a timer has been started.
 *
 * @public              @memberof kTimer
 * @param   timer       Timer object. 
 * @return              kTRUE if the timer was started.
 */
kFx(kBool) kTimer_IsStarted(kTimer timer);

/** 
 * Reports whether a count-down timer has expired.
 *
 * @public              @memberof kTimer
 * @param   timer       Timer object. 
 * @return              kTRUE if the timer has expired. 
 */
kFx(kBool) kTimer_IsExpired(kTimer timer);

/** 
 * Reports the duration, in microseconds, for which the timer has been running.
 *
 * @public              @memberof kTimer
 * @param   timer       Timer object. 
 * @return              Elapsed time, in microseconds. 
 */
kFx(k64u) kTimer_Elapsed(kTimer timer); 

/** 
 * Reports the remaining time, in microseconds, for a countdown timer.
 *
 * @public              @memberof kTimer
 * @param   timer       Timer object. 
 * @return              Remaining time, in microseconds. 
 */
kFx(k64u) kTimer_Remaining(kTimer timer); 

#endif
