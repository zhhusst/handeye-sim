/** 
 * @file    kDataTreeItem.cpp
 *
 * @internal
 * Copyright (C) 2008-2024 by LMI Technologies Inc.
 * Licensed under the MIT License.
 * Redistributed files must retain the above copyright notice.
 */
#include <kApi/Data/kDataTree.h>
#include <kApi/Data/kBox.h>
#include <kApi/Io/kSerializer.x.h>

kBeginClassEx(k, kDataTreeItem) 

    kAddPrivateVersionEx(kDataTreeItem, "kdat5", "5.0.0.0", "129-0", WriteDat5V0, ReadDat5V0)
    kAddPrivateVersionEx(kDataTreeItem, "kdat6", "5.7.1.0", "kDataTreeItem-0", WriteDat6V0, ReadDat6V0)

    kAddPrivateFrameworkConstructor(kDataTreeItem, ConstructFramework);

    kAddVMethod(kDataTreeItem, kObject, VRelease)
    kAddVMethod(kDataTreeItem, kObject, VClone)
    kAddVMethod(kDataTreeItem, kObject, VHasShared)
    kAddVMethod(kDataTreeItem, kObject, VSize)
    kAddVMethod(kDataTreeItem, kObject, VAllocTraits)

kEndClassEx() 

kFx(kStatus) kDataTreeItem_Construct(kDataTreeItem* item, const kChar* name, kObject allocator)
{
    kAlloc alloc = kAlloc_Fallback(allocator);
    kStatus status; 

    kCheck(kAlloc_GetObject(alloc, kTypeOf(kDataTreeItem), item)); 

    if (!kSuccess(status = kDataTreeItem_Init(*item, kTypeOf(kDataTreeItem), name, alloc)))
    {
        kAlloc_FreeRef(alloc, item); 
    }

    return status; 
}

kFx(kStatus) xkDataTreeItem_ConstructFramework(kDataTreeItem* item, kAlloc allocator)
{
    return kDataTreeItem_Construct(item, "", allocator);
}

kFx(kStatus) kDataTreeItem_Init(kDataTreeItem item, kType classType, const kChar* name, kAlloc allocator)
{
    kObjR(kDataTreeItem, item); 
    kStatus status;

    kCheck(kObject_Init(item, classType, allocator)); 

    obj->value = kNULL; 
    obj->parent = kNULL;
    obj->firstChild = kNULL; 
    obj->previousSibling = kNULL; 
    obj->nextSibling = kNULL; 
    obj->name.text[0] = 0;  
    obj->name.buffer = 0; 
    obj->name.capacity = 0; 

    if (!kSuccess(status = kDataTreeItem_SetName(item, name)))
    {
        kDataTreeItem_VRelease(item); 
        return status;
    }

    return kOK; 
}

kFx(kStatus) kDataTreeItem_VClone(kDataTreeItem item, kDataTreeItem source, kAlloc valueAlloc, kObject context)
{
    kObj(kDataTreeItem, item); 
    kObjN(kDataTreeItem, srcObj, source); 
    kAlloc objectAlloc = kObject_Alloc(item);
    kDataTreeItem childIt = srcObj->firstChild; 
    kDataTreeItem childCopy = kNULL; 
    kStatus status;
 
    kTry
    {
        kTest(kDataTreeItem_SetName(item, kDataTreeItem_Name(source)));

        kTest(kObject_Clone(&obj->value, srcObj->value, objectAlloc, valueAlloc, context)); 

        while (!kIsNull(childIt))
        {
            kTest(kObject_Clone(&childCopy, childIt, objectAlloc, valueAlloc, context));
            
            kTest(kDataTreeItem_Insert(item, kNULL, childCopy)); 
            childCopy = kNULL;

            childIt = kDataTreeItem_NextSibling(childIt); 
        }
    }
    kCatch(&status)
    {
        kObject_Dispose(childCopy); 
        kEndCatch(status);
    }

    return kOK; 
}

kFx(kBool) kDataTreeItem_VHasShared(kDataTreeItem item)
{
    kObj(kDataTreeItem, item); 

    if (kObject_IsShared(item) || (!kIsNull(obj->value) && kObject_HasShared(obj->value)))
    {
        return kTRUE; 
    }

    kDataTreeItem childIt = kDataTreeItem_FirstChild(item); 

    while (!kIsNull(childIt))
    {
        if (kObject_HasShared(childIt))
        {
            return kTRUE; 
        }

        childIt = kDataTreeItem_NextSibling(childIt); 
    }

    return kFALSE; 
}

kFx(kSize) kDataTreeItem_VSize(kDataTreeItem item)
{
    kObj(kDataTreeItem, item); 
    kSize size = sizeof(kDataTreeItemClass); 
    kDataTreeItem childIt = kDataTreeItem_FirstChild(item); 

    if (!kIsNull(obj->name.buffer))     size += obj->name.capacity;
    if (!kIsNull(obj->value))           size += kObject_Size(obj->value);

    while (!kIsNull(childIt))
    {
        size += kObject_Size(childIt); 
        childIt = kDataTreeItem_NextSibling(childIt); 
    }

    return size; 
}

kFx(kAllocTrait) kDataTreeItem_VAllocTraits(kDataTreeItem item)
{
    kObj(kDataTreeItem, item); 
    kAllocTrait traits = kAlloc_Traits(kObject_Alloc(item));
    kDataTreeItem childIt = kDataTreeItem_FirstChild(item); 

    if (!kIsNull(obj->value))
    {
        traits |= kObject_AllocTraits(obj->value); 
    }

    while (!kIsNull(childIt))
    {
        traits |= kObject_AllocTraits(childIt);
        childIt = kDataTreeItem_NextSibling(childIt); 
    }

    return traits; 
}

kFx(kStatus) xkDataTreeItem_WriteDat5V0(kDataTreeItem item, kSerializer serializer)
{
    kObj(kDataTreeItem, item); 
    kDataTreeItem childIt = kDataTreeItem_FirstChild(item); 
    kString name = kNULL;         // NOTE: FS5 kDataTreeItem serialization scheme contains a string object

    kTry
    {
        kTest(kString_Construct(&name, kDataTreeItem_Name(item), kNULL));
        kTest(kSerializer_WriteObject(serializer, name)); 

        kTest(kSerializer_WriteObject(serializer, obj->value)); 

        while (!kIsNull(childIt))
        {
            kTest(kSerializer_WriteObject(serializer, childIt)); 
            childIt = kDataTreeItem_NextSibling(childIt); 
        }

        kTest(kSerializer_WriteObject(serializer, kNULL)); 
    }
    kFinally
    {
        kDestroyRef(&name);
        kEndFinally();
    }
    return kOK; 
}

kFx(kStatus) xkDataTreeItem_ReadDat5V0(kDataTreeItem item, kSerializer serializer)
{
    kObj(kDataTreeItem, item);
    kAlloc allocator = kObject_Alloc(item);
    kDataTreeItem childIt;
    kString name = kNULL;         // NOTE: FS5 kDataTreeItem serialization scheme contains a string object

    kTry
    {
        kTest(kSerializer_ReadObject(serializer, &name, allocator));
        kTest(kDataTreeItem_SetName(item, kString_Chars(name)));

        kTest(kSerializer_ReadObject(serializer, &obj->value, allocator)); 

        kTest(kSerializer_ReadObject(serializer, &childIt, allocator));

        while (!kIsNull(childIt))
        {
            kTest(kDataTreeItem_Insert(item, kNULL, childIt)); 
            kTest(kSerializer_ReadObject(serializer, &childIt, allocator));
        }
    }
    kFinally
    {
        kDestroyRef(&name);
        kEndFinally();
    }

    return kOK; 
}

kFx(kStatus) xkDataTreeItem_WriteDat6V0(kDataTreeItem item, kSerializer serializer)
{
    kObj(kDataTreeItem, item); 
    kDataTreeItem childIt = kDataTreeItem_FirstChild(item); 
    const kChar* name = kDataTreeItem_Name(item);
    kSize nameLength = kStrLength(name); 

    kCheck(kSerializer_WriteSize(serializer, nameLength)); 
    kCheck(kSerializer_WriteCharArray(serializer, name, nameLength)); 

    kCheck(kSerializer_WriteObject(serializer, obj->value)); 

    while (!kIsNull(childIt))
    {
        kCheck(kSerializer_WriteObject(serializer, childIt)); 
        childIt = kDataTreeItem_NextSibling(childIt); 
    }

    kCheck(kSerializer_WriteObject(serializer, kNULL)); 

    return kOK; 
}

kFx(kStatus) xkDataTreeItem_ReadDat6V0(kDataTreeItem item, kSerializer serializer)
{
    kObjR(kDataTreeItem, item); 
    kAlloc allocator = kObject_Alloc(item);
    kDataTreeItem childIt;
    kSize nameLength = 0;
    kChar* name = kNULL;

    kCheck(kSerializer_ReadSize(serializer, &nameLength)); 

    kCheck(kDataTreeItem_Reserve(item, nameLength));
    name = kDataTreeItem_Name(item); 

    kCheck(kSerializer_ReadCharArray(serializer, name, nameLength)); 
    name[nameLength] = 0;

    kCheck(kSerializer_ReadObject(serializer, &obj->value, allocator)); 

    kCheck(kSerializer_ReadObject(serializer, &childIt, allocator));

    while (!kIsNull(childIt))
    {
        kCheck(kDataTreeItem_Insert(item, kNULL, childIt)); 
        kCheck(kSerializer_ReadObject(serializer, &childIt, allocator));
    }

    return kOK; 
}

kFx(kStatus) kDataTreeItem_VRelease(kDataTreeItem item)
{
    kObj(kDataTreeItem, item);
    kDataTreeItem childIt = kDataTreeItem_FirstChild(item); 
    kDataTreeItem nextChild = kNULL; 

    kCheck(kAlloc_Free(obj->base.alloc, obj->name.buffer));
    kCheck(kDisposeRef(&obj->value)); 

    while (!kIsNull(childIt))
    {
        nextChild = kDataTreeItem_NextSibling(childIt); 
        kCheck(kObject_Destroy(childIt)); 
        childIt = nextChild;  
    }

    kCheck(kObject_VRelease(item));

    return kOK; 
}

kFx(kStatus) kDataTreeItem_SetValue(kDataTreeItem item, kType type, const void* value)
{
    kObj(kDataTreeItem, item);

    if (!kIsNull(value) && !kIsNull(obj->value) && kObject_Is(obj->value, kTypeOf(kBox)) && (kBox_ItemType(obj->value) == type))
    {
        kMemCopy(kBox_Data(obj->value), value, kType_Size(type)); 
    }
    else
    {
        kCheck(kDisposeRef(&obj->value)); 

        if (!kIsNull(value))
        {
            kCheck(kBox_Construct(&obj->value, type, obj->base.alloc));
            kMemCopy(kBox_Data(obj->value), value, kType_Size(type)); 
        }
    }

    return kOK;
}

kFx(kStatus) kDataTreeItem_SetText(kDataTreeItem item, const kChar* value)
{
    kObj(kDataTreeItem, item);

    if (!kIsNull(value) && !kIsNull(obj->value) && kObject_Is(obj->value, kTypeOf(kString)))
    {
        kCheck(kString_Import(obj->value, value, kFALSE));
    }
    else
    {
        kCheck(kDisposeRef(&obj->value)); 

        if (!kIsNull(value))
        {
            kCheck(kString_Construct(&obj->value, value, obj->base.alloc)); 
        }
    }

    return kOK;
}

kFx(kStatus) kDataTreeItem_SetData(kDataTreeItem item, kObject value, kBool clone)
{
    kObj(kDataTreeItem, item);

    kCheck(kDisposeRef(&obj->value)); 

    if (!kIsNull(value))
    {
        if (clone)
        {
            kCheck(kObject_Clone(&obj->value, value, obj->base.alloc)); 
        }
        else
        {
            obj->value = value; 
        }
    }

    return kOK;
}

kFx(kStatus) kDataTreeItem_Value(kDataTreeItem item, kType type, void* value)
{
    kObj(kDataTreeItem, item);
    
    if (kIsNull(value) || kIsNull(obj->value) || !kObject_Is(obj->value, kTypeOf(kBox)) || (kBox_ItemType(obj->value) != type))
    {
        return kERROR_PARAMETER; 
    }

    kMemCopy(value, kBox_Data(obj->value), kBox_ItemSize(obj->value)); 

    return kOK; 
}

kFx(kStatus) kDataTreeItem_Text(kDataTreeItem item, kChar* value, k32u capacity)
{
    kObj(kDataTreeItem, item);
    
    if (kIsNull(value) || (capacity == 0) || kIsNull(obj->value) || !kObject_Is(obj->value, kTypeOf(kString)))
    {
        return kERROR_PARAMETER; 
    }

    kCheck(kStrCopy(value, capacity, kString_Chars(obj->value)));

    return kOK; 
}

kFx(kStatus) kDataTreeItem_Data(kDataTreeItem item, kObject* value, kAlloc allocator)
{
    kObj(kDataTreeItem, item);
    
    if (kIsNull(value) || kIsNull(obj->value))
    {
        return kERROR_PARAMETER; 
    }

    kCheck(kObject_Clone(value, obj->value, allocator)); 

    return kOK; 
}

kFx(kStatus) kDataTreeItem_Child(kDataTreeItem item, const kChar* childName, kDataTreeItem* child)
{
    kObj(kDataTreeItem, item);
    kDataTreeItem childIt = obj->firstChild; 

    while (!kIsNull(childIt))
    {
        if (kStrEquals(kDataTreeItem_Name(childIt), childName))
        {
            *child = childIt; 
            return kOK; 
        }

        childIt = xkDataTreeItem_Cast(childIt)->nextSibling; 
    }

    return kERROR;
}

kFx(kStatus) kDataTreeItem_Insert(kDataTreeItem item, kDataTreeItem before, kDataTreeItem child)
{
    kObj(kDataTreeItem, item);
    kObjN(kDataTreeItem, childObj, child); 

    if (kIsNull(obj->firstChild))
    {        
        childObj->parent = item;   
        childObj->previousSibling = kNULL; 
        childObj->nextSibling = kNULL; 

        obj->firstChild = child;     
    }
    else if (kIsNull(before))
    {
        kDataTreeItem previous = kDataTreeItem_LastChild(item); 
        kObjN(kDataTreeItem, previousObj, previous); 

        childObj->parent = item; 
        childObj->previousSibling = previous; 
        childObj->nextSibling = kNULL; 

        previousObj->nextSibling = child; 
    }
    else if (before == obj->firstChild)
    {
        kDataTreeItem next = before; 
        kObjN(kDataTreeItem, nextObj, next);

        childObj->parent = item; 
        childObj->previousSibling = kNULL; 
        childObj->nextSibling = next; 

        nextObj->previousSibling = child; 
        obj->firstChild = child; 
    }
    else 
    {
        kDataTreeItem next = before; 
        kObjN(kDataTreeItem, nextObj, next);
        kDataTreeItem previous = nextObj->previousSibling; 
        kObjN(kDataTreeItem, previousObj, previous);

        childObj->parent = item; 
        childObj->previousSibling = previous; 
        childObj->nextSibling = next; 

        previousObj->nextSibling = child;        
        nextObj->previousSibling = child;        
    }

    return kOK; 
}

kFx(kStatus) kDataTreeItem_Remove(kDataTreeItem item)
{
    kObj(kDataTreeItem, item); 
    kObjN(kDataTreeItem, parentObj, obj->parent); 

    if (parentObj->firstChild == item)
    {
        parentObj->firstChild = obj->nextSibling; 
    }

    if (!kIsNull(obj->previousSibling))
    {
        xkDataTreeItem_Cast(obj->previousSibling)->nextSibling = obj->nextSibling;        
    }

    if (!kIsNull(obj->nextSibling))
    {
        xkDataTreeItem_Cast(obj->nextSibling)->previousSibling = obj->previousSibling; 
    }

    obj->parent = kNULL; 
    obj->previousSibling = kNULL; 
    obj->nextSibling = kNULL; 

    return kOK; 
}

kFx(kDataTreeItem) kDataTreeItem_Parent(kDataTreeItem item)
{
    kObj(kDataTreeItem, item); 
    return obj->parent; 
}

kFx(kDataTreeItem) kDataTreeItem_FirstChild(kDataTreeItem item)
{
    kObj(kDataTreeItem, item); 
    return obj->firstChild; 
}

kFx(kDataTreeItem) kDataTreeItem_LastChild(kDataTreeItem item)
{
    kObj(kDataTreeItem, item); 
    kDataTreeItem child = obj->firstChild; 

    if (!kIsNull(child))
    {
        while (!kIsNull(xkDataTreeItem_Cast(child)->nextSibling))
        {
            child = xkDataTreeItem_Cast(child)->nextSibling; 
        }
    }
    
    return child;
}

kFx(kDataTreeItem) kDataTreeItem_PreviousSibling(kDataTreeItem item)
{
    kObj(kDataTreeItem, item); 
    return obj->previousSibling; 
}

kFx(kDataTreeItem) kDataTreeItem_NextSibling(kDataTreeItem item)
{
    kObj(kDataTreeItem, item); 
    return obj->nextSibling; 
}

kFx(kObject) kDataTreeItem_GetData(kDataTreeItem item)
{
    kObj(kDataTreeItem, item); 
    return obj->value; 
}

kFx(kStatus) kDataTreeItem_Reserve(kDataTreeItem item, kSize size)
{
    kObj(kDataTreeItem, item);
    kDataTreeItemName* field = &obj->name;
    kSize bufferSize = size + 1;

    if (bufferSize <= kCountOf(field->text))
    {
        if (!kIsNull(field->buffer))
        {
            kAlloc_Free(obj->base.alloc, field->buffer);
            field->buffer = kNULL;
            field->capacity = 0;
        }
    }
    else
    {
        if (!kIsNull(field->buffer) && field->capacity < bufferSize)
        {
            kAlloc_Free(obj->base.alloc, field->buffer);
            field->buffer = kNULL;
            field->capacity = 0;
        }

        if (kIsNull(field->buffer))
        {
            kCheck(kAlloc_Get(obj->base.alloc, bufferSize, &field->buffer));
            field->capacity = bufferSize;
        }
    }

    return kOK;
}

kFx(kStatus) kDataTreeItem_SetName(kDataTreeItem item, const kChar* str)
{
    kObj(kDataTreeItem, item);
    kDataTreeItemName* field = &obj->name;
    kSize bufferSize = kStrLength(str) + 1;

    if (bufferSize <= kCountOf(field->text))
    {
        if (!kIsNull(field->buffer))
        {
            kAlloc_Free(obj->base.alloc, field->buffer);
            field->buffer = kNULL;
            field->capacity = 0;
        }
        kCheck(kStrCopy(field->text, kCountOf(field->text), str));
    }
    else
    {
        kAlloc_Free(obj->base.alloc, field->buffer);
        field->buffer = kNULL;

        kCheck(kAlloc_Get(obj->base.alloc, bufferSize, &field->buffer));
        field->capacity = bufferSize;

        kCheck(kStrCopy(field->buffer, bufferSize, str));
    }

    return kOK;
}
