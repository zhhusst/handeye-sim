/** 
 * @file    kHeapAlloc.h
 * @brief   Declares the kHeapAlloc class. 
 *
 * @internal
 * Copyright (C) 2020-2024 by LMI Technologies Inc.
 */
#ifndef K_API_HEAP_ALLOC_H
#define K_API_HEAP_ALLOC_H

#include <kApi/kApiDef.h>

/**
 * Signature of callback to notify about heap exhaustion.
 *
 * @public                     @memberof kHeapAlloc
 * @param   heapAlloc          kHeapAlloc object.
 * @param   receiver           Pointer to the receiver specified in kHeapAlloc_AddExhaustionHandler function.
 * @param   minimumRegionSize  The smallest region that could be added to satisfy the memory request.
 * @return                     If callback returns kOK, kHeapAlloc will re-attempt allocation; otherwise,
 *                             the allocation operation will be allowed to fail.
 */
typedef kStatus(kCall* kHeapAllocExhaustionFx)(kHeapAlloc heapAlloc, kPointer receiver, kSize minimumRegionSize);

/**
 * @public  
 * @struct  kHeapAllocStats
 * @ingroup kApi-Utils
 * @brief   Structure for statistics calculated by the kHeapAlloc_Stats function.
 */
struct kHeapAllocStats
{
    kSize memoryRegionsCount;   ///< Number of memory regions added.  

    kSize allocatedBlocksCount; ///< Number of memory blocks allocated.
    kSize allocatedBytesCount;  ///< Number of bytes in all allocated memory blocks.

    kSize freeBlocksCount;      ///< Number of free blocks available for allocations.
    kSize freeBytesCount;       ///< Number of bytes in all free blocks.

    kSize maxFreeBlockSize;     ///< The largest free block available for allocation.
};

/**
 * @class   kHeapAlloc
 * @extends kAlloc
 * @ingroup kApi-Utils
 * @brief   Allocates memory from user-defined memory regions.
 *
 * The kHeapAlloc class establishes a memory heap within one or more pre-allocated regions of memory. 
 * After construction, the inherited kAlloc_Get/kAlloc_Free functions can be used to allocate or free memory from 
 * within the heap. 
 *
 * Heap management information is stored in the preallocated memory regions that comprise the memory heap. Accordingly, 
 * the effective capacity of the heap allocator will be somewhat less than the sum of the preallocated memory regions.
 * The amount of memory consumed by heap management information increases with the number free blocks that must be managed 
 * by the heap, which in turn increases with heap fragmentation.
 * 
 * It is valid to destroy a kHeapAlloc instance without first deallocating all of the memory allocated by it. 
 * This approach may sometimes be advantageous for performance or convenience. However, it is invalid to access  
 * memory that was allocated from a kHeapAlloc instance after the kHeapAlloc instance has been destroyed.
 *
 * All public functions are thread-safe. kAlloc_Get/kAlloc_Free functions have O(log n) time complexity,
 * where n is number of free blocks managed by the heap.
 */
// typedef kObject kHeapAlloc; --forward - declared in kFsDef.x.h


/** 
 * Constructs a new kHeapAlloc instance. 
 *
 * @public              @memberof kHeapAlloc
 * @param   heapAlloc   Receives the constructed kHeapAlloc instance. 
 * @param   allocator   Memory allocator for this object instance (or kNULL for default). 
 * @return              Operation status. 
 */
kFx(kStatus) kHeapAlloc_Construct(kHeapAlloc* heapAlloc, kAlloc allocator);

/**
 * Adds a pre-allocated memory region for use of the kHeapAlloc object.
 *
 * At least one memory block must be added to a kHeapAlloc instance before any allocation requests.
 * Additional memory regions can be added any time.
 *
 * @public              @memberof kHeapAlloc
 * @param   heapAlloc   kHeapAlloc object.
 * @param   address     Start address of the preallocated memory region.
 * @param   size        Block size, in bytes.
 * @return              Operation status.
 */
kFx(kStatus) kHeapAlloc_AddMemoryRegion(kHeapAlloc heapAlloc, kPointer address, kSize size);

/**
 * Removes all added memory regions.
 *
 * This function is called automatically when a kHeapAlloc object is destroyed. Any memory buffers 
 * previously allocated from the kHeapAlloc instance are invalidated.
 *
 * @public              @memberof kHeapAlloc
 * @param   heapAlloc   kHeapAlloc object.
 * @return              Operation status.
 */
kFx(kStatus) kHeapAlloc_RemoveMemoryRegions(kHeapAlloc heapAlloc);

/**
 * Frees all memory allocations that have been made from this kHeapAlloc instance. 
 *
 * This function is called automatically when a kHeapAlloc object is destroyed.
 *
 * @public              @memberof kHeapAlloc
 * @param   heapAlloc   kHeapAlloc object.
 * @return              Operation status.
 */
kFx(kStatus) kHeapAlloc_Clear(kHeapAlloc heapAlloc);

/**
 * Reports used and free blocks from all added memory regions.
 *
 * The function call has O(log n) time complexity, where n is the number of free blocks.
 *
 * @public              @memberof kHeapAlloc
 * @param   heapAlloc   kHeapAlloc object.
 * @param   stats       Calculated statistics.
 * @return              Operation status.
 */
kFx(kStatus) kHeapAlloc_Stats(kHeapAlloc heapAlloc, kHeapAllocStats* stats);

/**
 * Adds a callback that will be invoked upon memory exhaustion.
 *
 * Various internal locks may be held by the thread that invokes the callback; accordingly, the implementation of the callback
 * should avoid acquiring any additional locks that might result in deadlock.
 *
 * @public              @memberof kHeapAlloc
 * @param   heapAlloc   kHeapAlloc object.
 * @param   callback    Callback to notify about heap exhaustion.
 * @param   receiver    Pointer to the received which is passed to the callback.
 * @return              Operation status.
 */
kFx(kStatus) kHeapAlloc_AddExhaustionHandler(kHeapAlloc heapAlloc, kHeapAllocExhaustionFx callback, kPointer receiver);

#endif
