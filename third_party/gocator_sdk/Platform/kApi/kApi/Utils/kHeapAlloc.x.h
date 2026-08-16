/** 
 * @file    kHeapAlloc.x.h
 *
 * @internal
 * Copyright (C) 2020-2024 by LMI Technologies Inc.
 */
#ifndef K_API_HEAP_ALLOC_X_H
#define K_API_HEAP_ALLOC_X_H

#include <kApi.extern/rbtree/rbtree.h>

// Mask used for xkHeapAllocBlockHeader prev attribute
#define xkHEAP_ALLOC_BLOCK_HEADER_USED_MASK (1) // Used to distinguish between used and free block in the linked list.

// A header for the two way linked list containing all used and free blocks in a region.
struct xkHeapAllocBlockHeader
{
    xkHeapAllocBlockHeader* prev; // Pointer to the previous block. kNULL if the first in region. 
                                  // Do not use as variable since it has masked information included.
                                  // Use inline functions like xkHeapAllocBlockHeader_Prev()
    xkHeapAllocBlockHeader* next; // Pointer to the next block. 
};

// A header for the two way linked list in the case that the block is the last block in the region.
struct xkHeapAllocEndBlock
{
    xkHeapAllocBlockHeader header;  // header.next is kNULL to indicate that this is xkHeapAllocEndBlock.
    xkHeapAllocBlockHeader* nextRegionStart; // Start for the next defined region. kNULL indicates the last region.
};

typedef struct kHeapAllocClass
{
    kAllocClass base;

    rbtree_t rbtree;                    // Red-black tree for free blocks in all regions.
    xkHeapAllocBlockHeader* firstBlock; // Pointer to the first block in the first region.
    kSize minBlockSize;                 // Minimum block size which is allocated.
    kLock lock;
    kHeapAllocStats stats;
    kHeapAllocExhaustionFx onFull;      // Callback function which is called when allocation request is failed due to out-of-memory.
    kPointer onFullReceiver;            // Receiver for the callback function
} kHeapAllocClass;

// A data structure for node in the tree. 
struct xkHeapAllocFreeBlock
{
    rbnode_t node;
    kSize size;      // Total size of the block including block headers.
};

kDeclareClassEx(k, kHeapAlloc, kAlloc)

/*
* Private methods.
*/

kInlineFx(kSize) xkHeapAllocBlockHeader_BlockHeaderSpace()
{
    return kSize_Align(sizeof(xkHeapAllocBlockHeader), kALIGN_ANY);
}


kInlineFx(xkHeapAllocBlockHeader*) xkHeapAllocBlockHeader_Next(const xkHeapAllocBlockHeader* block)
{
    return block->next;
}

kInlineFx(xkHeapAllocBlockHeader*) xkHeapAllocBlockHeader_Prev(const xkHeapAllocBlockHeader* block)
{
    return (xkHeapAllocBlockHeader*)((kSize)block->prev & ~xkHEAP_ALLOC_BLOCK_HEADER_USED_MASK);
}

kInlineFx(kBool) xkHeapAllocBlockHeader_Used(const xkHeapAllocBlockHeader* block)
{
    return (kBool)((kSize)block->prev & xkHEAP_ALLOC_BLOCK_HEADER_USED_MASK);
}

kInlineFx(void) xkHeapAllocBlockHeader_SetPrev(xkHeapAllocBlockHeader* block, xkHeapAllocBlockHeader* prev)
{
    if (xkHeapAllocBlockHeader_Used(block))
    {
        block->prev = (xkHeapAllocBlockHeader*)((kSize)prev | xkHEAP_ALLOC_BLOCK_HEADER_USED_MASK);
        return;
    }

    block->prev = prev;  
}

kInlineFx(void) xkHeapAllocBlockHeader_SetNext(xkHeapAllocBlockHeader* block, xkHeapAllocBlockHeader* next)
{
    block->next = next;
}

kInlineFx(void) xkHeapAllocBlockHeader_SetUsed(xkHeapAllocBlockHeader* block)
{
    block->prev = (xkHeapAllocBlockHeader*)((kSize)block->prev | xkHEAP_ALLOC_BLOCK_HEADER_USED_MASK);
}

kInlineFx(void) xkHeapAllocBlockHeader_SetFree(xkHeapAllocBlockHeader* block)
{
    block->prev = (xkHeapAllocBlockHeader*)((kSize)block->prev & ~xkHEAP_ALLOC_BLOCK_HEADER_USED_MASK);
}

kInlineFx(kSize) xkHeapAllocBlockHeader_Size(const xkHeapAllocBlockHeader* block)
{
    if (kIsNull(block) || xkHeapAllocBlockHeader_Next(block) == kNULL)
        return 0;

    return kPointer_Diff(xkHeapAllocBlockHeader_Next(block), (kPointer*)block);
}

kInlineFx(void) xkHeapAllocBlockHeader_Init(xkHeapAllocBlockHeader* block, xkHeapAllocBlockHeader* prev, xkHeapAllocBlockHeader* next, kBool used)
{
    xkHeapAllocBlockHeader_SetPrev(block, prev);
    xkHeapAllocBlockHeader_SetNext(block, next);
    if (used)
        xkHeapAllocBlockHeader_SetUsed(block);
    else
        xkHeapAllocBlockHeader_SetFree(block);
}

kFx(kStatus) xkHeapAlloc_Init(kHeapAlloc heapAlloc, kType type, kAlloc allocator);

kFx(kStatus) xkHeapAlloc_VRelease(kHeapAlloc heapAlloc);

kFx(kStatus) xkHeapAlloc_VGet(kHeapAlloc heapAlloc, kSize size, void* mem, kMemoryAlignment alignment);
kFx(kStatus) xkHeapAlloc_VFree(kHeapAlloc heapAlloc, void* mem);

/**
 * Reallocates the given memory block. It must be previously allocated by this kHeapAlloc object.
 *
 * The function reuses allocated block if possible. If reusing is not possible a new memory block is allocated and
 * the old data is copied to the new block. Reusing is done if following criterias are met:
 *  - a big enough free block is found right after the current block, or a new size is smaller than allocated size
 *  - no memory address change is forced because of alignment change
 *
 * @public              @memberof kHeapAlloc
 * @param   heapAlloc   kHeapAlloc object.
 * @param   size        New block size, in bytes.
 * @param   mem         Pointer to the block to be reallocated. After successful reallocation pointer to a new block is returned.
 * @param   alignment   Memory alignment. Default is kALIGN_ANY (3).
 * @return              Operation status.
 */
kFx(kStatus) xkHeapAlloc_Realloc(kHeapAlloc heapAlloc, kSize size, void* mem, kMemoryAlignment alignment = kALIGN_ANY);

kFx(xkHeapAllocFreeBlock*) xkHeapAlloc_FindFreeBlock(kHeapAlloc heapAlloc, kSize size);

kFx(xkHeapAllocBlockHeader*) xkHeapAlloc_BlockIteratorBegin(kHeapAlloc heapAlloc);
kFx(xkHeapAllocBlockHeader*) xkHeapAlloc_BlockIteratorNext(const xkHeapAllocBlockHeader* block);

kFx(kBool) xkHeapAlloc_IsNullNode(const rbnode_t* node);

#endif
