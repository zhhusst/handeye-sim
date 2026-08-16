/** 
 * @file    kHeapAlloc.cpp
 *
 * @internal
 * Copyright (C) 2020-2024 by LMI Technologies Inc.
 */

#include <kApi/Utils/kHeapAlloc.h>
#include <kApi/Utils/kHeapAlloc.x.h>

#include <kApi/Threads/kLock.h>
#include <kApi.extern/rbtree/rbtree.h>

kBeginClassEx(k, kHeapAlloc)
    kAddPrivateVMethod(kHeapAlloc, kObject, VRelease)
    kAddPrivateVMethod(kHeapAlloc, kAlloc, VGet)
    kAddPrivateVMethod(kHeapAlloc, kAlloc, VFree)
kEndClassEx()

class xkHeapAlloc_Lock
{
    kLock m_lock = kNULL;
public:
    xkHeapAlloc_Lock(kHeapAlloc heapAlloc)
    {
        kObj(kHeapAlloc, heapAlloc);

        m_lock = obj->lock;
        kLock_Enter(m_lock);
    }

    ~xkHeapAlloc_Lock()
    {
        kLock_Exit(m_lock);
    }
};

kFx(kStatus) kHeapAlloc_Construct(kHeapAlloc* heapAlloc, kAlloc allocator)
{
    const auto alloc = kAlloc_Fallback(allocator);
    kStatus status;

    kCheck(kAlloc_GetObject(alloc, kTypeOf(kHeapAlloc), heapAlloc));

    if (!kSuccess(status = xkHeapAlloc_Init(*heapAlloc, kTypeOf(kHeapAlloc), alloc)))
    {
        kAlloc_FreeRef(alloc, heapAlloc);
        return status;
    }

    return status;
}

static kStatus xkHeapAlloc_AddFreeBlock(kHeapAlloc heapAlloc, xkHeapAllocBlockHeader* block)
{
    kObj(kHeapAlloc, heapAlloc);

    xkHeapAllocBlockHeader_SetFree(block);

    const auto freeBlock = (xkHeapAllocFreeBlock*)kPointer_ByteOffset(block, xkHeapAllocBlockHeader_BlockHeaderSpace());
    freeBlock->node.key = &freeBlock->size;
    freeBlock->size = xkHeapAllocBlockHeader_Size(block);

    rbtree_insert(&obj->rbtree, &freeBlock->node);

    obj->stats.freeBlocksCount++;
    obj->stats.freeBytesCount += freeBlock->size;

    return kOK;
}

static kStatus xkHeapAlloc_DeleteFromTree(kHeapAlloc heapAlloc, const xkHeapAllocBlockHeader* block)
{
    kObj(kHeapAlloc, heapAlloc);

    const auto freeBlockInTree = (xkHeapAllocFreeBlock*)kPointer_ByteOffset(block, xkHeapAllocBlockHeader_BlockHeaderSpace());

    rbtree_delete(&obj->rbtree, freeBlockInTree->node.key);

    obj->stats.freeBlocksCount--;
    obj->stats.freeBytesCount -= xkHeapAllocBlockHeader_Size(block);
    
    return kOK;
}

static void xkHeapAlloc_MergeUsedBlockWithNextFree(kHeapAlloc heapAlloc, xkHeapAllocBlockHeader* block)
{
    kObj(kHeapAlloc, heapAlloc);

    const auto next = xkHeapAllocBlockHeader_Next(block);
    const auto nextSize = xkHeapAllocBlockHeader_Size(next);
#if K_ASSERT_ENABLED
    const auto currentSize = xkHeapAllocBlockHeader_Size(block);
#endif

    const auto nextNext = xkHeapAllocBlockHeader_Next(next);
    if (kIsNull(nextNext))
    {
        return;
    }

    xkHeapAlloc_DeleteFromTree(heapAlloc, next);

    xkHeapAllocBlockHeader_SetNext(block, nextNext);
    xkHeapAllocBlockHeader_SetPrev(nextNext, block);

    kAssert((currentSize + nextSize) == xkHeapAllocBlockHeader_Size(block));

    obj->stats.allocatedBytesCount += nextSize;
}

static void xkHeapAlloc_MergeFreeBlockWithNextFree(kHeapAlloc heapAlloc, xkHeapAllocBlockHeader* block)
{
    kObj(kHeapAlloc, heapAlloc);

    const auto next = xkHeapAllocBlockHeader_Next(block);
#if K_ASSERT_ENABLED
    const auto nextSize = xkHeapAllocBlockHeader_Size(next);
    const auto currentSize = xkHeapAllocBlockHeader_Size(block);
#endif

    const auto nextNext = xkHeapAllocBlockHeader_Next(next);
    if (kIsNull(nextNext))
    {
        return;
    }

    xkHeapAlloc_DeleteFromTree(heapAlloc, next);
    xkHeapAlloc_DeleteFromTree(heapAlloc, block);

    xkHeapAllocBlockHeader_SetNext(block, nextNext);
    xkHeapAllocBlockHeader_SetPrev(nextNext, block);

    kAssert((currentSize + nextSize) == xkHeapAllocBlockHeader_Size(block));

    xkHeapAlloc_AddFreeBlock(heapAlloc, block);
}

static xkHeapAllocBlockHeader* xkHeapAlloc_SplitFreeBlock(kHeapAlloc heapAlloc, xkHeapAllocBlockHeader* block, kSize firstPartSize)
{
    kObj(kHeapAlloc, heapAlloc);

    const auto currentBlockSize = xkHeapAllocBlockHeader_Size(block);

    // Split the block into two parts if it is large enough
    if (currentBlockSize > firstPartSize && (currentBlockSize - firstPartSize) > obj->minBlockSize)
    {
        const auto newBlock = (xkHeapAllocBlockHeader*)kPointer_ByteOffset(block, firstPartSize);
        const auto oldNext = xkHeapAllocBlockHeader_Next(block);

        xkHeapAllocBlockHeader_SetNext(block, newBlock);
        xkHeapAllocBlockHeader_SetPrev(oldNext, newBlock);
        xkHeapAllocBlockHeader_Init(newBlock, block, oldNext, kFALSE);

        // Remaining part is a free block
        xkHeapAlloc_AddFreeBlock(heapAlloc, newBlock);

        return newBlock;
    }

    return kNULL;
}

static kBool xkHeapAlloc_TryReuseBlock(kHeapAlloc heapAlloc, xkHeapAllocBlockHeader* block, kSize sizeRequired, kSize oldSize)
{
    kObj(kHeapAlloc, heapAlloc);

    // We can shrink the block without copy if a new size is smaller 
    if ((kSize)oldSize >= sizeRequired)
    {
        // Split and add a new free block if remaining size is big enough
        obj->stats.allocatedBytesCount -= xkHeapAllocBlockHeader_Size(block);
        const auto newFreeBlock = xkHeapAlloc_SplitFreeBlock(heapAlloc, block, sizeRequired);
        obj->stats.allocatedBytesCount += xkHeapAllocBlockHeader_Size(block);

        // Return without merging if block is not split or the block after new block is used
        if (kIsNull(newFreeBlock) || xkHeapAllocBlockHeader_Used(xkHeapAllocBlockHeader_Next(newFreeBlock)))
        {
            return true;
        }

        // Concatenate two free blocks 
        xkHeapAlloc_MergeFreeBlockWithNextFree(heapAlloc, newFreeBlock);
        return true;
    }

    // Merge if the next block is free and big enough
    const auto nextBlock = xkHeapAllocBlockHeader_Next(block);
    if (!xkHeapAllocBlockHeader_Used(nextBlock) &&
        (xkHeapAllocBlockHeader_Size(nextBlock) + (kSize)oldSize) >= sizeRequired)
    {
        xkHeapAlloc_MergeUsedBlockWithNextFree(heapAlloc, block);

        // Split if remaining part is big 
        obj->stats.allocatedBytesCount -= xkHeapAllocBlockHeader_Size(block);
        xkHeapAlloc_SplitFreeBlock(heapAlloc, block, sizeRequired);
        obj->stats.allocatedBytesCount += xkHeapAllocBlockHeader_Size(block);;

        return true;
    }

    return false;
}
static kStatus xkHeapAlloc_AddFreeAndMerge(kHeapAlloc heapAlloc, xkHeapAllocBlockHeader* block)
{
    kObj(kHeapAlloc, heapAlloc);

    const auto prev = xkHeapAllocBlockHeader_Prev(block);
    const auto next = xkHeapAllocBlockHeader_Next(block);
#if K_ASSERT_ENABLED
    const auto prevSize = xkHeapAllocBlockHeader_Size(prev);
    const auto currentSize = xkHeapAllocBlockHeader_Size(block);
    const auto nextSize = xkHeapAllocBlockHeader_Size(next);
#endif

    // Merge the block with both neighbors if those are empty
    if (!kIsNull(prev) && !xkHeapAllocBlockHeader_Used(prev) &&
        !kIsNull(next) && !xkHeapAllocBlockHeader_Used(next))
    {
        const auto nextNext = xkHeapAllocBlockHeader_Next(next);

        kAssert(!kIsNull(nextNext));
        kCheckTrue(!kIsNull(nextNext), kERROR);

        xkHeapAlloc_DeleteFromTree(heapAlloc, prev);
        xkHeapAlloc_DeleteFromTree(heapAlloc, next);

        xkHeapAllocBlockHeader_SetNext(prev, nextNext);
        xkHeapAllocBlockHeader_SetPrev(nextNext, prev);

        kAssert((prevSize + nextSize + currentSize) == xkHeapAllocBlockHeader_Size(prev));

        return xkHeapAlloc_AddFreeBlock(heapAlloc, prev);
    }

    // Merge the block with the previous if it is empty
    if (!kIsNull(prev) && !xkHeapAllocBlockHeader_Used(prev))
    {
        xkHeapAlloc_DeleteFromTree(heapAlloc, prev);

        xkHeapAllocBlockHeader_SetNext(prev, next);
        xkHeapAllocBlockHeader_SetPrev(next, prev);

        kAssert((prevSize + currentSize) == xkHeapAllocBlockHeader_Size(prev));

        return xkHeapAlloc_AddFreeBlock(heapAlloc, prev);
    }
    
    // Merge the block with the next block if it is empty
    if (!kIsNull(next) && !xkHeapAllocBlockHeader_Used(next))
    {
        const auto nextNext = xkHeapAllocBlockHeader_Next(next);

        kAssert(!kIsNull(nextNext));
        kCheckTrue(!kIsNull(nextNext), kERROR);

        xkHeapAlloc_DeleteFromTree(heapAlloc, next);

        xkHeapAllocBlockHeader_SetNext(block, nextNext);
        xkHeapAllocBlockHeader_SetPrev(nextNext, block);

        kAssert((currentSize + nextSize) == xkHeapAllocBlockHeader_Size(block));

        return xkHeapAlloc_AddFreeBlock(heapAlloc, block);
    }
    
    // A single free block without merging
    return xkHeapAlloc_AddFreeBlock(heapAlloc, block);
}

static int xkHeapAlloc_CompareSizes(const kSize sizeA, const kSize sizeB)
{
    if (sizeA < sizeB)
        return -1;

    if (sizeA > sizeB)
        return 1;

    return 0;
}

static int xkHeapAlloc_Compare(const void* a, const void* b)
{
    if (kIsNull(a) || kIsNull(b))
    {
        return 0;
    }

    const auto sizeCompare = xkHeapAlloc_CompareSizes(*(kSize*)a, *(kSize*)b);
    if (sizeCompare != 0)
        return sizeCompare;

    // Use address as a secondary key since all nodes must be ordered in the tree. 
    if (a < b)
        return -1;

    if (a > b)
        return 1;

    return 0;
}

static void xkHeapAlloc_InitWithoutLock(kHeapAlloc heapAlloc)
{
    kObj(kHeapAlloc, heapAlloc);

    rbtree_init(&obj->rbtree, xkHeapAlloc_Compare);

    obj->firstBlock = kNULL;
    obj->minBlockSize = kSize_Align(xkHeapAllocBlockHeader_BlockHeaderSpace()+ sizeof(xkHeapAllocFreeBlock), kALIGN_ANY);

    obj->stats.memoryRegionsCount = 0;
    obj->stats.allocatedBlocksCount = 0;
    obj->stats.allocatedBytesCount = 0;
    obj->stats.freeBlocksCount = 0;
    obj->stats.freeBytesCount = 0;
    obj->stats.maxFreeBlockSize = 0;
}

kFx(kStatus) kHeapAlloc_RemoveMemoryRegions(kHeapAlloc heapAlloc)
{
    kObj(kHeapAlloc, heapAlloc);

    xkHeapAlloc_Lock lock(heapAlloc);
    xkHeapAlloc_InitWithoutLock(heapAlloc);

    return kOK;
}

kFx(kStatus) kHeapAlloc_Clear(kHeapAlloc heapAlloc)
{
    kObj(kHeapAlloc, heapAlloc);

    xkHeapAlloc_Lock lock(heapAlloc);

    if (kIsNull(obj->firstBlock))
    {
        return kOK;
    }

    auto* it = obj->firstBlock;
    xkHeapAlloc_InitWithoutLock(heapAlloc);
    obj->firstBlock = it;

    // go through all memory regions 
    while (it)
    {
        auto* startBlockInRegion = it;

        while (xkHeapAllocBlockHeader_Next(it))
        {
            it = xkHeapAllocBlockHeader_Next(it);
        }

        auto* foundEndBlock = ((xkHeapAllocEndBlock*)it);

        // clear list of allocated and free blocks
        xkHeapAllocBlockHeader_SetPrev(&foundEndBlock->header, startBlockInRegion);
        xkHeapAllocBlockHeader_SetNext(startBlockInRegion, &foundEndBlock->header);

        // add first block as a free block
        xkHeapAlloc_AddFreeBlock(heapAlloc, startBlockInRegion);
        obj->stats.memoryRegionsCount++;

        it = foundEndBlock->nextRegionStart;
    }

    return kOK;
}

kFx(kStatus) xkHeapAlloc_Init(kHeapAlloc heapAlloc, kType type, kAlloc allocator)
{
    kObjR(kHeapAlloc, heapAlloc);
    kStatus status;

    kCheck(kAlloc_Init(heapAlloc, type, allocator));
     
    // Member variable initialization
    obj->lock = kNULL;
    obj->onFull = kNULL;
    obj->onFullReceiver = kNULL;             

    xkHeapAlloc_InitWithoutLock(heapAlloc);

    kTry
    {
        kTest(kLock_ConstructEx(&obj->lock, xkLOCK_OPTION_PRIORITY_INHERITANCE, allocator));
    }
    kCatch(&status)
    {
        xkHeapAlloc_VRelease(heapAlloc);
        kEndCatch(status);
    }

    return kOK;
}

kFx(kStatus) xkHeapAlloc_VRelease(kHeapAlloc heapAlloc)
{
    kObj(kHeapAlloc, heapAlloc);

    kCheck(kAlloc_VRelease(heapAlloc));

    return kObject_Destroy(obj->lock);
}

kFx(xkHeapAllocFreeBlock*) xkHeapAlloc_FindFreeBlock(kHeapAlloc heapAlloc, kSize size)
{
    kObj(kHeapAlloc, heapAlloc);

    auto node = obj->rbtree.root;
    xkHeapAllocFreeBlock* result = kNULL;

    while (node != RBTREE_NULL)
    {
        if (kIsNull(node->key))
        {
            return kNULL;
        }

        const auto comparison = xkHeapAlloc_CompareSizes(size, *(kSize*)node->key);

        if (comparison == 0)
        {
            // Exact match 
            return (xkHeapAllocFreeBlock*)node;
        }

        if (comparison < 0)
        {
            // last node which has enough space
            result = (xkHeapAllocFreeBlock*)node;
            node = node->left;
        }
        else
        {
            node = node->right;
        }
    }
    return result;
}

static xkHeapAllocBlockHeader* xkHeapAlloc_BlockHeaderFromFreeHeader(const xkHeapAllocFreeBlock* freeBlock)
{
    if (kIsNull(freeBlock))
        return kNULL;

    return (xkHeapAllocBlockHeader*)((kByte*)freeBlock - xkHeapAllocBlockHeader_BlockHeaderSpace());
}

kFx(xkHeapAllocBlockHeader*) xkHeapAlloc_BlockIteratorBegin(kHeapAlloc heapAlloc)
{
    kObj(kHeapAlloc, heapAlloc);
    return obj->firstBlock;
}

kFx(xkHeapAllocBlockHeader*) xkHeapAlloc_BlockIteratorNext(const xkHeapAllocBlockHeader* block)
{
    if (kIsNull(block))
        return kNULL;

    auto const next = xkHeapAllocBlockHeader_Next(block);
    if (kIsNull(next))
    {
        return next;
    }

    if (kIsNull(xkHeapAllocBlockHeader_Next(next)))
    {
        const auto foundEndBlock = (xkHeapAllocEndBlock*)next;
        if (kIsNull(foundEndBlock->nextRegionStart))
        {
            return kNULL;
        }

        return foundEndBlock->nextRegionStart;
    }

    return next;
}

// Aligns start address of the allocation.

kFx(void *) xkHeapAllocBlock_AllocatedAlignedBlock(const xkHeapAllocBlockHeader* block, kSize alignment)
{
    if (kIsNull(block))
        return kNULL;

    const auto freeBlock = kPointer_ByteOffset(block, xkHeapAllocBlockHeader_BlockHeaderSpace());
    const auto freeBlockAligned = (kSize*)kSize_Align((kSize)freeBlock, alignment);

    // In the default alignment case, allocated memory is just after the block header.
    // In other alignment cases extra gap is added only if needed.
    // | Block prev | | Block next | | allocation |    
    if (freeBlockAligned > (kSize*)freeBlock)
    {
        // An extra gap is used after the block header.
        // Start address of the block is written just before allocation.
        // | Block prev | | Block next | | xx |...| xx | | Block start | | allocation |
        // <----------------------------------------------/

        const auto blockRedirectHeader = xkHeapAlloc_BlockHeaderFromFreeHeader((const xkHeapAllocFreeBlock*)freeBlockAligned);
        blockRedirectHeader->next = (xkHeapAllocBlockHeader*) block;
    }
    return freeBlockAligned;
}

// Finds a linked list header associated to the allocation pointer which was returned to the user.

static xkHeapAllocBlockHeader* xkHeapAlloc_BlockFromAllocatedPointer(void* allocation)
{
    // Get a candidate for a block header.  
    auto block = xkHeapAlloc_BlockHeaderFromFreeHeader((const xkHeapAllocFreeBlock*)allocation);

    // Blocks are ordered in a two way linked list always in ascending order. I.e. Address of the
    // next block is always bigger than the address of the block itself.
    // This is used to detect situation when the extra gap is added.

    // In the default alignment case the block candidate is correct. In other alignment cases
    // extra gap was possibly added and the candidate is incorrect.
    // | Block prev | | Block next | | allocation |
    // This is determined by checking the order of next and allocation addresses.
    if (xkHeapAllocBlockHeader_Next(block) < allocation)
    {
        // If the extra gap was added the start of the block is populated just before allocation.
        // | Block prev | | Block next | | xx |...| xx | | Block start | | allocation |
        
        block = xkHeapAllocBlockHeader_Next(block);
    }

    return block;
}

kFx(kStatus) kHeapAlloc_Stats(kHeapAlloc heapAlloc, kHeapAllocStats* stats)
{
    kObj(kHeapAlloc, heapAlloc);

    kCheckArgs(!kIsNull(stats));

    xkHeapAlloc_Lock lock(heapAlloc);
    *stats = obj->stats;

    const auto lastNode = rbtree_last(&obj->rbtree);
    if (kIsNull(lastNode) || kIsNull(lastNode->key))
    {
        stats->maxFreeBlockSize = 0;
    }
    else
    {
        stats->maxFreeBlockSize = *(kSize*)lastNode->key;

        // Make sure that user can allocate max block with default align
        stats->maxFreeBlockSize -= xkHeapAllocBlockHeader_BlockHeaderSpace();
    }

    return kOK;
}

kFx(kStatus) kHeapAlloc_AddMemoryRegion(kHeapAlloc heapAlloc, kPointer startAddress, kSize size)
{
    kObj(kHeapAlloc, heapAlloc);

    const auto endBlockSize = kSize_Align(sizeof(xkHeapAllocEndBlock),kALIGN_ANY);

    // Make sure that both address and size are in minimum alignment.
    const auto address = (kPointer)kSize_Align((kSize)startAddress, kALIGN_ANY);
    size -= kPointer_Diff(address, startAddress);
    size -= (size % kALIGN_ANY_SIZE);

    // Space for two blocks are needed at a minimum.
    if (size < (xkHeapAllocBlockHeader_BlockHeaderSpace()+ sizeof(xkHeapAllocFreeBlock) + endBlockSize))
    {
        // We do not return error code in situation when the region is ignored due to the size.
        return kOK;
    }

    xkHeapAlloc_Lock lock(heapAlloc);

    // Linked list for the block
    const auto block = (xkHeapAllocBlockHeader*)address;
    xkHeapAllocBlockHeader_Init(block,
        kNULL,
        (xkHeapAllocBlockHeader*)kPointer_ByteOffset(address, size - endBlockSize),
        kFALSE);

    // Block is started as a free block
    xkHeapAlloc_AddFreeBlock(heapAlloc, block);

    // Special block for the end of region
    const auto endBlock = (xkHeapAllocEndBlock*)xkHeapAllocBlockHeader_Next(block);
    xkHeapAllocBlockHeader_Init(&endBlock->header, block, kNULL, kTRUE);
    endBlock->nextRegionStart = kNULL;

    if (kIsNull(obj->firstBlock))
    {
        obj->firstBlock = block;
    }
    else
    {
        // Add link to the previous region
        xkHeapAllocBlockHeader* it = obj->firstBlock;
        do
        {
            while (xkHeapAllocBlockHeader_Next(it))
            {
                it = xkHeapAllocBlockHeader_Next(it);
            }

            const auto foundEndBlock = ((xkHeapAllocEndBlock*)it);

            if (kIsNull(foundEndBlock->nextRegionStart))
            {
                foundEndBlock->nextRegionStart = block;
                break;
            }

            it = foundEndBlock->nextRegionStart;
        } while (true);
    }

    obj->stats.memoryRegionsCount++;

    return kOK;
}

static kSize xkHeapAlloc_MinimumAllocation(kHeapAlloc heapAlloc, kSize size, kMemoryAlignment alignment)
{
    kObj(kHeapAlloc, heapAlloc);

    auto sizeRequired = size + xkHeapAllocBlockHeader_BlockHeaderSpace();
    sizeRequired = kSize_Align(sizeRequired, alignment);
    if (alignment > kALIGN_ANY)
    {
        sizeRequired += kMemoryAlignment_Size(alignment);
    }

    sizeRequired = kMax(sizeRequired, obj->minBlockSize);

    return sizeRequired;
}

kFx(kBool) xkHeapAlloc_IsNullNode(const rbnode_t* node)
{
    return (!node || node == RBTREE_NULL);
}

kFx(kStatus) kHeapAlloc_AddExhaustionHandler(kHeapAlloc heapAlloc, kHeapAllocExhaustionFx callback, kPointer receiver)
{
    kObj(kHeapAlloc, heapAlloc);

    kCheckArgs(!kIsNull(callback));
    xkHeapAlloc_Lock lock(heapAlloc);

    obj->onFull = callback;
    obj->onFullReceiver = receiver;

    return kOK;
}

kFx(kStatus) xkHeapAlloc_Realloc(kHeapAlloc heapAlloc, kSize size, void* mem, kMemoryAlignment alignment)
{
    kObj(kHeapAlloc, heapAlloc);

    if (kIsNull(mem))
    {
        return kERROR_MEMORY;
    }

    const auto oldMemory = *(void**)mem;
    if (kIsNull(oldMemory))
    {
        return kAlloc_Get(heapAlloc, size, mem, alignment);
    }

    xkHeapAlloc_Lock lock(heapAlloc);
    void* newAlloc = kNULL;
    const auto block = xkHeapAlloc_BlockFromAllocatedPointer(oldMemory);
    const auto oldPayloadSize = kPointer_Diff(xkHeapAllocBlockHeader_Next(block), (kPointer*)oldMemory);

    const auto sizeRequired = xkHeapAlloc_MinimumAllocation(heapAlloc, size, alignment);
    const auto startAddressMatch = (kSize_Align((kSize)oldMemory, alignment) == (kSize)oldMemory);

    // Try to optimize memcpy if the start address is not going to be changed
    if (startAddressMatch && xkHeapAlloc_TryReuseBlock(heapAlloc, block, sizeRequired, xkHeapAllocBlockHeader_Size(block)))
    {
        return kOK;
    }

    if (kAlloc_Get(heapAlloc, size, &newAlloc, alignment)!= kOK)
    {
        return kERROR_MEMORY;
    }

    kMemCopy(newAlloc, oldMemory, kMin_((kSize)oldPayloadSize, size));

    xkHeapAlloc_VFree(heapAlloc, oldMemory);
    kPointer_WriteAs(mem, newAlloc, kPointer);
    
    return kOK;
}

kFx(kStatus) xkHeapAlloc_VGet(kHeapAlloc heapAlloc, kSize size, void* mem, kMemoryAlignment alignment)
{
    kObj(kHeapAlloc, heapAlloc);

    xkHeapAllocBlockHeader* currentBlock = kNULL;
    const auto sizeRequired = xkHeapAlloc_MinimumAllocation(heapAlloc, size, alignment);
    xkHeapAlloc_Lock lock(heapAlloc);

    // Try once if exhaustion callback is not set.
    // If the callback is set, try until succeed or the callback returns other than kOK.
    while (kIsNull(currentBlock))
    {
        const auto freeBlock = xkHeapAlloc_FindFreeBlock(heapAlloc, sizeRequired);
        currentBlock = xkHeapAlloc_BlockHeaderFromFreeHeader(freeBlock);

        if (kIsNull(freeBlock))
        {
            const auto endBlockSize = kSize_Align(sizeof(xkHeapAllocEndBlock), kALIGN_ANY);

            if (kIsNull(obj->onFull) ||
                kIsError(obj->onFull(heapAlloc, obj->onFullReceiver, sizeRequired + xkHeapAllocBlockHeader_BlockHeaderSpace() + endBlockSize)))
            {
                break;
            }
        }
        else
        {
            rbtree_delete_node(&obj->rbtree, &freeBlock->node);

            const auto currentBlockSize = xkHeapAllocBlockHeader_Size(currentBlock);
            xkHeapAllocBlockHeader_SetUsed(currentBlock);
            obj->stats.freeBlocksCount--;
            obj->stats.freeBytesCount-= currentBlockSize;

            xkHeapAlloc_SplitFreeBlock(heapAlloc, currentBlock, sizeRequired);
        }
    }

    const auto freeBlockAligned = xkHeapAllocBlock_AllocatedAlignedBlock(currentBlock, alignment);

    kPointer_WriteAs(mem, freeBlockAligned, kPointer);

    if (kIsNull(freeBlockAligned))
    {
        return kERROR_MEMORY;
    }
    
    obj->stats.allocatedBlocksCount++;
    obj->stats.allocatedBytesCount+= xkHeapAllocBlockHeader_Size(currentBlock);

    return kOK;    
}

kFx(kStatus) xkHeapAlloc_VFree(kHeapAlloc heapAlloc, void* mem)
{
    kObj(kHeapAlloc, heapAlloc);

    if (kIsNull(mem))
    {
        return kOK;
    }

    xkHeapAlloc_Lock lock(heapAlloc);

    const auto block = xkHeapAlloc_BlockFromAllocatedPointer(mem);

    obj->stats.allocatedBlocksCount--;
    obj->stats.allocatedBytesCount -= xkHeapAllocBlockHeader_Size(block);

    return xkHeapAlloc_AddFreeAndMerge(heapAlloc, block);
}
