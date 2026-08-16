/** 
 * @file    kDataTree.x.h
 *
 * @internal
 * Copyright (C) 2008-2024 by LMI Technologies Inc.  All rights reserved.
 * Licensed under the MIT License.
 * Redistributed files must retain the above copyright notice.
 */
#ifndef K_API_DATA_TREE_X_H
#define K_API_DATA_TREE_X_H

#include <kApi/Io/kSerializer.h>
#include <kApi/Data/kDataTreeItem.x.h>

typedef struct kDataTreeClass 
{
    kObjectClass base; 
    kDataTreeItem root; 
} kDataTreeClass;

kDeclareClassEx(k, kDataTree, kObject)

kFx(kStatus) kDataTree_Init(kDataTree tree, kType classType, kAlloc allocator);
kFx(kStatus) kDataTree_VRelease(kDataTree tree);
kFx(kStatus) kDataTree_VClone(kDataTree tree, kDataTree source, kAlloc valueAlloc, kObject context);

kFx(kBool) kDataTree_VHasShared(kDataTree tree); 
kFx(kSize) kDataTree_VSize(kDataTree tree);
kFx(kAllocTrait) kDataTree_VAllocTraits(kDataTree tree);

kFx(kStatus) xkDataTree_WriteDat5V0(kDataTree tree, kSerializer serializer);
kFx(kStatus) xkDataTree_ReadDat5V0(kDataTree tree, kSerializer serializer);
kFx(kStatus) xkDataTree_WriteDat6V0(kDataTree tree, kSerializer serializer);
kFx(kStatus) xkDataTree_ReadDat6V0(kDataTree tree, kSerializer serializer);
kFx(kStatus) kDataTree_SetChildValue(kDataTree tree, kDataTreeItem parent, const kChar* path, kType type, const void* value); 
kFx(kStatus) kDataTree_EnsureExists(kDataTree tree, kDataTreeItem parent, const kChar* path, kDataTreeItem* item); 
kFx(kStatus) kDataTree_ChildValue(kDataTree tree, kDataTreeItem parent, const kChar* path, kType type, void* value); 

/* legacy */
kFx(kStatus) kDataTree_SetItemString(kDataTree tree, kDataTreeItem item, const kChar* value); 
kFx(kStatus) kDataTree_ItemString(kDataTree tree, kDataTreeItem item, kChar* value, k32u capacity); 
kFx(kStatus) kDataTree_SetChildString(kDataTree tree, kDataTreeItem parent, const kChar* path, const kChar* value); 
kFx(kStatus) kDataTree_ChildString(kDataTree tree, kDataTreeItem parent, const kChar* path, kChar* value, k32u capacity); 
kFx(kObject) kDataTree_GetItemData(kDataTree tree, kDataTreeItem item);
kFx(kStatus) kDataTree_SetItemData(kDataTree tree, kDataTreeItem item, kObject value);
kFx(kStatus) kDataTree_ItemData(kDataTree tree, kDataTreeItem item, kObject* value, kAlloc allocator);
kFx(kStatus) kDataTree_SetChildData(kDataTree tree, kDataTreeItem parent, const kChar* path, kObject value);
kFx(kStatus) kDataTree_ChildData(kDataTree tree, kDataTreeItem parent, const kChar* path, kObject* value, kAlloc allocator);

#endif
