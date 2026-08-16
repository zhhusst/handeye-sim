/** 
 * @file    kDataTreeItem.x.h
 *
 * @internal
 * Copyright (C) 2008-2024 by LMI Technologies Inc.  All rights reserved.
 * Licensed under the MIT License.
 * Redistributed files must retain the above copyright notice.
 */
#ifndef K_API_DATA_TREE_ITEM_X_H
#define K_API_DATA_TREE_ITEM_X_H

#include <kApi/Data/kDataTree.h>
#include <kApi/Data/kString.h>
#include <kApi/Io/kSerializer.h>

#define kDATA_TREE_ITEM_DEFAULT_TEXT_SIZE          (64)

typedef struct kDataTreeItemName 
{
    kChar text[kDATA_TREE_ITEM_DEFAULT_TEXT_SIZE];
    kChar* buffer;
    kSize capacity;
} kDataTreeItemName;

typedef struct kDataTreeItemClass 
{
    kObjectClass base; 
    kDataTreeItemName name;
    kObject value;
    kDataTreeItem parent; 
    kDataTreeItem firstChild; 
    kDataTreeItem previousSibling; 
    kDataTreeItem nextSibling; 
} kDataTreeItemClass;

kDeclareClassEx(k, kDataTreeItem, kObject)

kFx(kStatus) xkDataTreeItem_ConstructFramework(kDataTreeItem* item, kAlloc allocator); 

kFx(kStatus) kDataTreeItem_Construct(kDataTreeItem* item, const kChar* name, kObject allocator);
kFx(kStatus) kDataTreeItem_Init(kDataTreeItem item, kType classType, const kChar* name, kAlloc allocator);
kFx(kStatus) kDataTreeItem_VRelease(kDataTreeItem item);
kFx(kStatus) kDataTreeItem_VClone(kDataTreeItem item, kDataTreeItem source, kAlloc valueAlloc, kObject context);
kFx(kBool) kDataTreeItem_VHasShared(kDataTreeItem item); 
kFx(kSize) kDataTreeItem_VSize(kDataTreeItem item);
kFx(kAllocTrait) kDataTreeItem_VAllocTraits(kDataTreeItem item);

kFx(kStatus) xkDataTreeItem_WriteDat5V0(kDataTreeItem item, kSerializer serializer);
kFx(kStatus) xkDataTreeItem_ReadDat5V0(kDataTreeItem item, kSerializer serializer);
kFx(kStatus) xkDataTreeItem_WriteDat6V0(kDataTreeItem item, kSerializer serializer);
kFx(kStatus) xkDataTreeItem_ReadDat6V0(kDataTreeItem item, kSerializer serializer);

kFx(kStatus) kDataTreeItem_SetValue(kDataTreeItem item, kType type, const void* value); 
kFx(kStatus) kDataTreeItem_SetText(kDataTreeItem item, const kChar* value); 
kFx(kStatus) kDataTreeItem_SetData(kDataTreeItem item, kObject value, kBool clone); 

kFx(kStatus) kDataTreeItem_Value(kDataTreeItem item, kType type, void* value); 
kFx(kStatus) kDataTreeItem_Text(kDataTreeItem item, kChar* value, k32u capacity); 
kFx(kStatus) kDataTreeItem_Data(kDataTreeItem item, kObject* value, kAlloc allocator); 
kFx(kStatus) kDataTreeItem_Child(kDataTreeItem item, const kChar* childName, kDataTreeItem* child);

kFx(kStatus) kDataTreeItem_Insert(kDataTreeItem item, kDataTreeItem before, kDataTreeItem child);
kFx(kStatus) kDataTreeItem_Remove(kDataTreeItem item);

kFx(kDataTreeItem) kDataTreeItem_Parent(kDataTreeItem item); 
kFx(kDataTreeItem) kDataTreeItem_FirstChild(kDataTreeItem item); 
kFx(kDataTreeItem) kDataTreeItem_LastChild(kDataTreeItem item); 
kFx(kDataTreeItem) kDataTreeItem_PreviousSibling(kDataTreeItem item);
kFx(kDataTreeItem) kDataTreeItem_NextSibling(kDataTreeItem item); 

kFx(kObject) kDataTreeItem_GetData(kDataTreeItem item); 

kFx(kStatus) kDataTreeItem_Reserve(kDataTreeItem item, kSize size);
kFx(kStatus) kDataTreeItem_SetName(kDataTreeItem item, const kChar* str);

kInlineFx(kChar*) kDataTreeItem_Name(kDataTreeItem item)
{
    kObjR(kDataTreeItem, item); 

    return kIsNull(obj->name.buffer) ? obj->name.text : obj->name.buffer; 
}

#endif
