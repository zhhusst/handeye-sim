/** 
 * @file    kDataTree.cpp
 *
 * @internal
 * Copyright (C) 2008-2024 by LMI Technologies Inc.
 * Licensed under the MIT License.
 * Redistributed files must retain the above copyright notice.
 */
#include <kApi/Data/kDataTree.h>
#include <kApi/Data/kArrayList.h>
#include <kApi/Io/kFile.h>
#include <kApi/Io/kSerializer.h>
#include <kApi/Io/kDat6Serializer.h>
#include <kApi/Io/kDat5Serializer.h>

kBeginClassEx(k, kDataTree) 
    
    //serialization versions
    kAddPrivateVersionEx(kDataTree, "kdat5", "5.0.0.0", "128-0", WriteDat5V0, ReadDat5V0)
    kAddPrivateVersionEx(kDataTree, "kdat6", "5.7.1.0", "kDataTree-0", WriteDat6V0, ReadDat6V0)

    //special constructors
    kAddFrameworkConstructor(kDataTree, Construct)

    //virtual methods
    kAddVMethod(kDataTree, kObject, VRelease)
    kAddVMethod(kDataTree, kObject, VClone)
    kAddVMethod(kDataTree, kObject, VHasShared)
    kAddVMethod(kDataTree, kObject, VSize)
    kAddVMethod(kDataTree, kObject, VAllocTraits)

kEndClassEx() 

kFx(kStatus) kDataTree_Construct(kDataTree* tree, kObject allocator)
{
    kAlloc alloc = kAlloc_Fallback(allocator);
    kStatus status; 

    kCheck(kAlloc_GetObject(alloc, kTypeOf(kDataTree), tree)); 

    if (!kSuccess(status = kDataTree_Init(*tree, kTypeOf(kDataTree), alloc)))
    {
        kAlloc_FreeRef(alloc, tree); 
    }

    return status; 
}

kFx(kStatus) kDataTree_Init(kDataTree tree, kType classType, kAlloc allocator)
{
    kObjR(kDataTree, tree); 

    kCheck(kObject_Init(tree, classType, allocator)); 

    obj->root = kNULL; 

    return kOK; 
}

kFx(kStatus) kDataTree_VClone(kDataTree tree, kDataTree source, kAlloc valueAlloc, kObject context)
{
    kObj(kDataTree, tree); 
    
    return kObject_Clone(&obj->root, kDataTree_Root(source), kObject_Alloc(tree), valueAlloc, context);
}

kFx(kBool) kDataTree_VHasShared(kDataTree tree)
{
    kObj(kDataTree, tree); 

    return kObject_IsShared(tree) || (!kIsNull(obj->root) && kObject_HasShared(obj->root)); 
}

kFx(kSize) kDataTree_VSize(kDataTree tree)
{
    kObj(kDataTree, tree); 
    kSize size = sizeof(kDataTreeClass);

    if (!kIsNull(obj->root)) size += kObject_Size(obj->root);

    return size; 
}

kFx(kAllocTrait) kDataTree_VAllocTraits(kDataTree tree)
{
    kObj(kDataTree, tree); 
    kAllocTrait traits = kAlloc_Traits(kObject_Alloc(tree));

    if (!kIsNull(obj->root))
    {
        traits |= kObject_AllocTraits(obj->root);
    }

    return traits; 
}

kFx(kStatus) xkDataTree_WriteDat5V0(kDataTree tree, kSerializer serializer)
{
    kObj(kDataTree, tree); 
    return kSerializer_WriteObject(serializer, obj->root); 
}

kFx(kStatus) xkDataTree_ReadDat5V0(kDataTree tree, kSerializer serializer)
{
    kObj(kDataTree, tree); 
  
    kCheck(kSerializer_ReadObject(serializer, &obj->root, kObject_Alloc(tree)));

    return kOK; 
}

kFx(kStatus) xkDataTree_WriteDat6V0(kDataTree tree, kSerializer serializer)
{
    kObj(kDataTree, tree); 
    return kSerializer_WriteObject(serializer, obj->root);
}

kFx(kStatus) xkDataTree_ReadDat6V0(kDataTree tree, kSerializer serializer)
{
    kObj(kDataTree, tree); 
  
    kCheck(kSerializer_ReadObject(serializer, &obj->root, kObject_Alloc(tree)));

    return kOK; 
}

kFx(kStatus) kDataTree_VRelease(kDataTree tree)
{
    kObj(kDataTree, tree);

    kCheck(kDisposeRef(&obj->root));
    
    kCheck(kObject_VRelease(tree));
    
    return kOK;
}

kFx(kStatus) kDataTree_Clear(kDataTree tree)
{
    kObj(kDataTree, tree);
    return kDestroyRef(&obj->root); 
}

kFx(kStatus) kDataTree_Add(kDataTree tree, kDataTreeItem parent, const kChar* name, kDataTreeItem* item)
{
    return kDataTree_Insert(tree, parent, kNULL, name, item); 
}

kFx(kStatus) kDataTree_Insert(kDataTree tree, kDataTreeItem parent, kDataTreeItem before, const kChar* name, kDataTreeItem* item)
{
    kObj(kDataTree, tree);
    kDataTreeItem insertedItem = kNULL; 
    kStatus exception; 

    kTry
    {
        if (kIsNull(parent))
        {
            kTest(kDataTree_Clear(tree)); 
            kTest(kDataTreeItem_Construct(&insertedItem, name, obj->base.alloc));

            obj->root = insertedItem; 
        }
        else
        {
            kTest(kDataTreeItem_Construct(&insertedItem, name, obj->base.alloc));
            kTest(kDataTreeItem_Insert(parent, before, insertedItem)); 
        }

        if (!kIsNull(item))
        {
            *item = insertedItem; 
        }
    }
    kCatch(&exception) 
    {
        kObject_Destroy(insertedItem); 
        kEndCatch(exception); 
    }

    return kOK; 
}

kFx(kStatus) kDataTree_Move(kDataTree tree, kDataTreeItem source, kDataTreeItem destParent, kDataTreeItem destBefore)
{
    kObj(kDataTree, tree);
    kDataTreeItem ancestor = destParent; 

    kCheckArgs(!kIsNull(source) && !kIsNull(destParent)); 

    //verify that destination is not a child of source
    while (!kIsNull(ancestor))
    {
        if (ancestor == source)
        {
            return kERROR_PARAMETER; 
        }

        ancestor = kDataTreeItem_Parent(ancestor); 
    }
    
    kCheck(kDataTreeItem_Remove(source));
    kCheck(kDataTreeItem_Insert(destParent, destBefore, source)); 

    return kOK; 
}

kFx(kStatus) kDataTree_Delete(kDataTree tree, kDataTreeItem item)
{
    kObj(kDataTree, tree);

    if (!kIsNull(obj->root) && item == obj->root)
    {
        kCheck(kDataTree_Clear(tree)); 
    }
    else
    {
        kCheck(kDataTreeItem_Remove(item));
        kCheck(kObject_Destroy(item));          
    }

    return kOK; 
}

kFx(kStatus) kDataTree_SetItem32u(kDataTree tree, kDataTreeItem item, k32u value)
{
    return kDataTreeItem_SetValue(item, kTypeOf(k32u), &value); 
}

kFx(kStatus) kDataTree_SetItemSize(kDataTree tree, kDataTreeItem item, kSize value)
{
    k32u tempValue = (k32u) value;

    //important: kSize/k32u duality is required for compatibility
    return kDataTreeItem_SetValue(item, kTypeOf(k32u), &tempValue); 
}

kFx(kStatus) kDataTree_SetItem32s(kDataTree tree, kDataTreeItem item, k32s value)
{
    return kDataTreeItem_SetValue(item, kTypeOf(k32s), &value); 
}

kFx(kStatus) kDataTree_SetItemSSize(kDataTree tree, kDataTreeItem item, kSSize value)
{
    k32s tempValue = (k32s) value;

    //important: kSSize/k32s duality is required for compatibility
    return kDataTreeItem_SetValue(item, kTypeOf(k32s), &tempValue); 
}

kFx(kStatus) kDataTree_SetItemBool(kDataTree tree, kDataTreeItem item, kBool value)
{
    return kDataTreeItem_SetValue(item, kTypeOf(kBool), &value); 
}

kFx(kStatus) kDataTree_SetItem64u(kDataTree tree, kDataTreeItem item, k64u value)
{
    return kDataTreeItem_SetValue(item, kTypeOf(k64u), &value); 
}

kFx(kStatus) kDataTree_SetItem64s(kDataTree tree, kDataTreeItem item, k64s value)
{
    return kDataTreeItem_SetValue(item, kTypeOf(k64s), &value); 
}

kFx(kStatus) kDataTree_SetItem32f(kDataTree tree, kDataTreeItem item, k32f value)
{
    return kDataTreeItem_SetValue(item, kTypeOf(k32f), &value); 
}

kFx(kStatus) kDataTree_SetItem64f(kDataTree tree, kDataTreeItem item, k64f value)
{
    return kDataTreeItem_SetValue(item, kTypeOf(k64f), &value); 
}

kFx(kStatus) kDataTree_SetItemText(kDataTree tree, kDataTreeItem item, const kChar* value)
{
    return kDataTreeItem_SetText(item, value); 
}

kFx(kStatus) kDataTree_PutItemData(kDataTree tree, kDataTreeItem item, kObject value)
{
    return kDataTreeItem_SetData(item, value, kFALSE); 
}

kFx(kStatus) kDataTree_SetItemData(kDataTree tree, kDataTreeItem item, kObject value)
{
    return kDataTreeItem_SetData(item, value, kTRUE); 
}

kFx(kStatus) kDataTree_Item32u(kDataTree tree, kDataTreeItem item, k32u* value)
{
    return kDataTreeItem_Value(item, kTypeOf(k32u), value); 
}

kFx(kStatus) kDataTree_ItemSize(kDataTree tree, kDataTreeItem item, kSize* value)
{
    k32u tempValue;

    //important: kSize/k32u duality is required for compatibility
    kCheck(kDataTreeItem_Value(item, kTypeOf(k32u), &tempValue)); 

    *value = tempValue;
    return kOK;
}

kFx(kStatus) kDataTree_Item32s(kDataTree tree, kDataTreeItem item, k32s* value)
{
    return kDataTreeItem_Value(item, kTypeOf(k32s), value); 
}

kFx(kStatus) kDataTree_ItemSSize(kDataTree tree, kDataTreeItem item, kSSize* value)
{
    k32s tempValue;

    //important: kSSize/k32s duality is required for compatibility
    kCheck(kDataTreeItem_Value(item, kTypeOf(k32s), &tempValue)); 

    *value = tempValue;
    return kOK;
}

kFx(kStatus) kDataTree_ItemBool(kDataTree tree, kDataTreeItem item, kBool* value)
{
    return kDataTreeItem_Value(item, kTypeOf(kBool), value); 
}

kFx(kStatus) kDataTree_Item64u(kDataTree tree, kDataTreeItem item, k64u* value)
{
    return kDataTreeItem_Value(item, kTypeOf(k64u), value); 
}

kFx(kStatus) kDataTree_Item64s(kDataTree tree, kDataTreeItem item, k64s* value)
{
    return kDataTreeItem_Value(item, kTypeOf(k64s), value); 
}

kFx(kStatus) kDataTree_Item32f(kDataTree tree, kDataTreeItem item, k32f* value)
{
    return kDataTreeItem_Value(item, kTypeOf(k32f), value); 
}

kFx(kStatus) kDataTree_Item64f(kDataTree tree, kDataTreeItem item, k64f* value)
{
    return kDataTreeItem_Value(item, kTypeOf(k64f), value); 
}

kFx(kStatus) kDataTree_ItemText(kDataTree tree, kDataTreeItem item, kChar* value, k32u capacity)
{
    return kDataTreeItem_Text(item, value, capacity); 
}

kFx(kStatus) kDataTree_ItemData(kDataTree tree, kDataTreeItem item, kObject* value, kAlloc allocator)
{
    return kDataTreeItem_Data(item, value, allocator);
}
                
kFx(kStatus) kDataTree_SetChild32u(kDataTree tree, kDataTreeItem parent, const kChar* path, k32u value)
{
    return kDataTree_SetChildValue(tree, parent, path, kTypeOf(k32u), &value); 
}

kFx(kStatus) kDataTree_SetChildSize(kDataTree tree, kDataTreeItem parent, const kChar* path, kSize value)
{
    k32u tempValue = (k32u) value;

    //important: kSize/k32u duality is required for compatibility
    return kDataTree_SetChildValue(tree, parent, path, kTypeOf(k32u), &tempValue); 
}

kFx(kStatus) kDataTree_SetChild32s(kDataTree tree, kDataTreeItem parent, const kChar* path, k32s value)
{
    return kDataTree_SetChildValue(tree, parent, path, kTypeOf(k32s), &value); 
}

kFx(kStatus) kDataTree_SetChildSSize(kDataTree tree, kDataTreeItem parent, const kChar* path, kSSize value)
{
    k32s tempValue = (k32s) value;

    //important: kSSize/k32s duality is required for compatibility
    return kDataTree_SetChildValue(tree, parent, path, kTypeOf(k32s), &tempValue); 
}

kFx(kStatus) kDataTree_SetChildBool(kDataTree tree, kDataTreeItem parent, const kChar* path, kBool value)
{
    return kDataTree_SetChildValue(tree, parent, path, kTypeOf(kBool), &value); 
}
 
kFx(kStatus) kDataTree_SetChild64u(kDataTree tree, kDataTreeItem parent, const kChar* path, k64u value)
{
    return kDataTree_SetChildValue(tree, parent, path, kTypeOf(k64u), &value); 
}

kFx(kStatus) kDataTree_SetChild64s(kDataTree tree, kDataTreeItem parent, const kChar* path, k64s value)
{
    return kDataTree_SetChildValue(tree, parent, path, kTypeOf(k64s), &value); 
}

kFx(kStatus) kDataTree_SetChild32f(kDataTree tree, kDataTreeItem parent, const kChar* path, k32f value)
{
    return kDataTree_SetChildValue(tree, parent, path, kTypeOf(k32f), &value); 
}

kFx(kStatus) kDataTree_SetChild64f(kDataTree tree, kDataTreeItem parent, const kChar* path, k64f value)
{
    return kDataTree_SetChildValue(tree, parent, path, kTypeOf(k64f), &value); 
}

kFx(kStatus) kDataTree_SetChildValue(kDataTree tree, kDataTreeItem parent, const kChar* path, kType type, const void* value)
{
    kDataTreeItem item; 
    
    kCheck(kDataTree_EnsureExists(tree, parent, path, &item)); 
    kCheck(kDataTreeItem_SetValue(item, type, value)); 

    return kOK; 
}

kFx(kStatus) kDataTree_SetChildText(kDataTree tree, kDataTreeItem parent, const kChar* path, const kChar* value)
{
    kDataTreeItem item; 
    
    kCheck(kDataTree_EnsureExists(tree, parent, path, &item)); 
    kCheck(kDataTreeItem_SetText(item, value)); 

    return kOK; 
}

kFx(kStatus) kDataTree_PutChildData(kDataTree tree, kDataTreeItem parent, const kChar* path, kObject value)
{
    kDataTreeItem item; 
    
    kCheck(kDataTree_EnsureExists(tree, parent, path, &item)); 
    kCheck(kDataTreeItem_SetData(item, value, kFALSE)); 

    return kOK; 
}

kFx(kStatus) kDataTree_SetChildData(kDataTree tree, kDataTreeItem parent, const kChar* path, kObject value)
{
    kDataTreeItem item; 
    
    kCheck(kDataTree_EnsureExists(tree, parent, path, &item)); 
    kCheck(kDataTreeItem_SetData(item, value, kTRUE)); 

    return kOK; 
}

kFx(kStatus) kDataTree_Child32u(kDataTree tree, kDataTreeItem parent, const kChar* path, k32u* value)
{
    return kDataTree_ChildValue(tree, parent, path, kTypeOf(k32u), value); 
}

kFx(kStatus) kDataTree_ChildSize(kDataTree tree, kDataTreeItem parent, const kChar* path, kSize* value)
{
    k32u tempValue;

    //important: kSize/k32u duality is required for compatibility
    kCheck(kDataTree_ChildValue(tree, parent, path, kTypeOf(k32u), &tempValue)); 

    *value = tempValue;
    return kOK;
}

kFx(kStatus) kDataTree_Child32s(kDataTree tree, kDataTreeItem parent, const kChar* path, k32s* value)
{
    return kDataTree_ChildValue(tree, parent, path, kTypeOf(k32s), value); 
}

kFx(kStatus) kDataTree_ChildSSize(kDataTree tree, kDataTreeItem parent, const kChar* path, kSSize* value)
{
    k32s tempValue;

    //important: kSSize/k32s duality is required for compatibility
    kCheck(kDataTree_ChildValue(tree, parent, path, kTypeOf(k32s), &tempValue)); 

    *value = tempValue;
    return kOK;
}

kFx(kStatus) kDataTree_ChildBool(kDataTree tree, kDataTreeItem parent, const kChar* path, kBool* value)
{
    return kDataTree_ChildValue(tree, parent, path, kTypeOf(kBool), value); 
}

kFx(kStatus) kDataTree_Child64u(kDataTree tree, kDataTreeItem parent, const kChar* path, k64u* value)
{
    return kDataTree_ChildValue(tree, parent, path, kTypeOf(k64u), value); 
}
 
kFx(kStatus) kDataTree_Child64s(kDataTree tree, kDataTreeItem parent, const kChar* path, k64s* value)
{
    return kDataTree_ChildValue(tree, parent, path, kTypeOf(k64s), value); 
}

kFx(kStatus) kDataTree_Child32f(kDataTree tree, kDataTreeItem parent, const kChar* path, k32f* value)
{
    return kDataTree_ChildValue(tree, parent, path, kTypeOf(k32f), value); 
}

kFx(kStatus) kDataTree_Child64f(kDataTree tree, kDataTreeItem parent, const kChar* path, k64f* value)
{
    return kDataTree_ChildValue(tree, parent, path, kTypeOf(k64f), value); 
}

kFx(kStatus) kDataTree_ChildValue(kDataTree tree, kDataTreeItem parent, const kChar* path, kType type, void* value)
{
    kDataTreeItem item; 
    
    kCheck(kDataTree_ChildItem(tree, parent, path, &item)); 
    kCheck(kDataTreeItem_Value(item, type, value)); 

    return kOK; 
}

kFx(kStatus) kDataTree_ChildText(kDataTree tree, kDataTreeItem parent, const kChar* path, kChar* value, k32u capacity)
{
    kDataTreeItem item; 
    
    kCheck(kDataTree_ChildItem(tree, parent, path, &item)); 
    kCheck(kDataTreeItem_Text(item, value, capacity)); 

    return kOK; 
}

kFx(kStatus) kDataTree_ChildData(kDataTree tree, kDataTreeItem parent, const kChar* path, kObject* value, kAlloc allocator)
{
    kDataTreeItem item; 
    
    kCheck(kDataTree_ChildItem(tree, parent, path, &item)); 
    kCheck(kDataTreeItem_Data(item, value, allocator)); 

    return kOK; 
}

#ifdef K_MSVC
#   define strtok_safe_ strtok_s
#else
#   define strtok_safe_ strtok_r
#endif

kFx(kStatus) kDataTree_ChildItem(kDataTree tree, kDataTreeItem parent, const kChar* path, kDataTreeItem* item)
{
    kObj(kDataTree, tree); 
    kDataTreeItem result = parent; 
    const kChar* delimiters = "\\/"; 
    kChar* token = kNULL; 
    char* cookie = kNULL;
    kChar pathBuffer[1024];

    kCheck(kStrCopy(&pathBuffer[0], kCountOf(pathBuffer), path));

    token = strtok_safe_(&pathBuffer[0], delimiters, &cookie);

    //if parent argument is null, consume first token for root
    if (kIsNull(result))
    {
        kDataTreeItem root = kDataTree_Root(tree); 

        // If first token is root, then root must exist.
        if (kIsNull(root))
        {
            return kERROR_NOT_FOUND;
        }

        token = strtok_safe_(kNULL, delimiters, &cookie);
        result = root; 
    }

    while (!kIsNull(token))
    {
        kCheck(kDataTreeItem_Child(result, token, &result)); 
        token = strtok_safe_(kNULL, delimiters, &cookie);
    }

    *item = result; 

    return kOK; 
}

kFx(kStatus) kDataTree_EnsureExists(kDataTree tree, kDataTreeItem parent, const kChar* path, kDataTreeItem* item)
{
    kObj(kDataTree, tree); 
    kDataTreeItem node = parent; 
    kDataTreeItem child = parent; 
    const kChar* delimiters = "\\/"; 
    kChar* token = kNULL; 
    char* cookie = kNULL;
    kChar pathBuffer[1024];

    kCheck(kStrCopy(&pathBuffer[0], kCountOf(pathBuffer), path));

    token = strtok_safe_(&pathBuffer[0], delimiters, &cookie);

    while (!kIsNull(token))
    {
        if (!kSuccess(kDataTreeItem_Child(node, token, &child)))
        {
            kCheck(kDataTree_Add(tree, node, token, &child)); 
        }
        node = child; 
        token = strtok_safe_(kNULL, delimiters, &cookie);
    }

    *item = child; 

    return kOK; 
}

kFx(kDataTreeItem) kDataTree_Root(kDataTree tree)
{
    kObj(kDataTree, tree);
    return obj->root; 
}

kFx(kDataTreeItem) kDataTree_Parent(kDataTree tree, kDataTreeItem item)
{
    return kDataTreeItem_Parent(item);     
}

kFx(kDataTreeItem) kDataTree_FirstChild(kDataTree tree, kDataTreeItem item)
{
    return kDataTreeItem_FirstChild(item); 
}

kFx(kDataTreeItem) kDataTree_LastChild(kDataTree tree, kDataTreeItem item)
{
    return kDataTreeItem_LastChild(item); 
}

kFx(kDataTreeItem) kDataTree_NextSibling(kDataTree tree, kDataTreeItem item)
{
    return kDataTreeItem_NextSibling(item); 
}

kFx(kDataTreeItem) kDataTree_PreviousSibling(kDataTree tree, kDataTreeItem item)
{
    return kDataTreeItem_PreviousSibling(item); 
}

kFx(k32u) kDataTree_ChildCount(kDataTree tree, kDataTreeItem parent)
{
    kDataTreeItem childIt = kDataTree_FirstChild(tree, parent); 
    k32u count = 0; 

    while (!kIsNull(childIt))
    {
        count++; 
        childIt = kDataTree_NextSibling(tree, childIt); 
    }

    return count; 
}

kFx(kDataTreeItem) kDataTree_ChildAt(kDataTree tree, kDataTreeItem parent, k32u index)
{
    kDataTreeItem childIt = kDataTree_FirstChild(tree, parent); 
    k32u current = 0; 

    while (!kIsNull(childIt) && (current < index))
    {
        current++; 
        childIt = kDataTree_NextSibling(tree, childIt); 
    }

    return childIt; 
}

kFx(kStatus) kDataTree_SetItemName(kDataTree tree, kDataTreeItem item, const kChar* name)
{
    return kDataTreeItem_SetName(item, name); 
}

kFx(const kChar*) kDataTree_ItemName(kDataTree tree, kDataTreeItem item)
{
    return kDataTreeItem_Name(item); 
}

kFx(kObject) kDataTree_ItemContent(kDataTree tree, kDataTreeItem item)
{
    return kDataTreeItem_GetData(item); 
}

kFx(kStatus) kDataTree_SetItemString(kDataTree tree, kDataTreeItem item, const kChar* value)
{
    return kDataTree_SetItemText(tree, item, value); 
}

kFx(kStatus) kDataTree_ItemString(kDataTree tree, kDataTreeItem item, kChar* value, k32u capacity)
{
    return kDataTree_ItemText(tree, item, value, capacity); 
}

kFx(kStatus) kDataTree_SetChildString(kDataTree tree, kDataTreeItem parent, const kChar* path, const kChar* value)
{
    return kDataTree_SetChildText(tree, parent, path, value); 
}

kFx(kStatus) kDataTree_ChildString(kDataTree tree, kDataTreeItem parent, const kChar* path, kChar* value, k32u capacity)
{
    return kDataTree_ChildText(tree, parent, path, value, capacity); 
}

kFx(kObject) kDataTree_GetItemData(kDataTree tree, kDataTreeItem item)
{
    return kDataTree_ItemContent(tree, item); 
}
