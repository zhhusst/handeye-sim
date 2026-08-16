/** 
 * @file    kDataTree.h
 * @brief   Declares the kDataTree class. 
 *
 * @internal
 * Copyright (C) 2008-2024 by LMI Technologies Inc.  All rights reserved.
 * Licensed under the MIT License.
 * Redistributed files must retain the above copyright notice.
 */
#ifndef K_API_DATA_TREE_H
#define K_API_DATA_TREE_H

#include <kApi/kApiDef.h>

/**
 * @class       kDataTree
 * @extends     kObject
 * @ingroup     kApi-Data
 * @brief       Represents a tree of data objects.
 */
//typedef kObject kDataTree;            --forward-declared in kApiDef.x.h  

/**
 * @class       kDataTreeItem
 * @extends     kObject
 * @ingroup     kApi-Data
 * @brief       Represents an item within a tree of data objects.
 */
//typedef kObject kDataTreeItem;            --forward-declared in kApiDef.x.h  

/** 
 * Constructs a data tree object.
 *
 * The tree is initially empty; use kDataTree_Add with parent=kNULL to create a root node.
 *
 * @public              @memberof kDataTree
 * @param   tree        Destination for the constructed object handle.
 * @param   allocator   Memory allocator (or kNULL for default). 
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_Construct(kDataTree* tree, kAlloc allocator);

/** 
 * Inserts a new child item at the end of the specified parent's child list.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item, or kNULL to add a root node.   
 * @param   name        Name of the new item.  
 * @param   item        Destination for the item handle.  
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_Add(kDataTree tree, kDataTreeItem parent, const kChar* name, kDataTreeItem* item);

/** 
 * Inserts a new item before the specified sibling node.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item, or kNULL to insert a root node.   
 * @param   before      Sibling item, or kNULL to insert at end of specified parent's child list.   
 * @param   name        Name of the new item.  
 * @param   item        Destination for the item handle.  
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_Insert(kDataTree tree, kDataTreeItem parent, kDataTreeItem before, const kChar* name, kDataTreeItem* item);

/** 
 * Moves a node from one location in the tree to another location.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   source      Source node. 
 * @param   destParent  Destination parent node.   
 * @param   destBefore  Destination sibling item, or kNULL to insert at end of specified parent's child list.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_Move(kDataTree tree, kDataTreeItem source, kDataTreeItem destParent, kDataTreeItem destBefore);

/** 
 * Deletes an item from the tree.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_Delete(kDataTree tree, kDataTreeItem item);

/** 
 * Removes all elements from the tree. 
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_Clear(kDataTree tree);

/** 
 * Sets item content from a k32u value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @param   value       Numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_SetItem32u(kDataTree tree, kDataTreeItem item, k32u value);

/** 
 * Sets item content from a kSize value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @param   value       Numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_SetItemSize(kDataTree tree, kDataTreeItem item, kSize value);

/** 
 * Sets item content from a k32s value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @param   value       Numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_SetItem32s(kDataTree tree, kDataTreeItem item, k32s value);

/** 
 * Sets item content from a kSSize value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @param   value       Numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_SetItemSSize(kDataTree tree, kDataTreeItem item, kSSize value);

/** 
 * Sets item content from a kBool value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @param   value       Boolean value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_SetItemBool(kDataTree tree, kDataTreeItem item, kBool value);

/** 
 * Sets item content from a k64u value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @param   value       Numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_SetItem64u(kDataTree tree, kDataTreeItem item, k64u value);

/** 
 * Sets item content from a k64s value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @param   value       Numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_SetItem64s(kDataTree tree, kDataTreeItem item, k64s value);

/** 
 * Sets item content from a k32f value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @param   value       Numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_SetItem32f(kDataTree tree, kDataTreeItem item, k32f value);

/** 
 * Sets item content from a k64f value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @param   value       Numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_SetItem64f(kDataTree tree, kDataTreeItem item, k64f value);

/** 
 * Sets item content from a text value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @param   value       Numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_SetItemText(kDataTree tree, kDataTreeItem item, const kChar* value);

/** 
 * Sets item content to a particular data object.
 *
 * The data object passed as an argument to this function is copied by reference; ownership of the 
 * original object is transferred to the tree.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @param   value       Data object.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_PutItemData(kDataTree tree, kDataTreeItem item, kObject value);

/** 
 * Sets item content by cloning a data object.
 *
 * The data object passed as an argument to this function will be cloned; ownership of the 
 * original object is not transferred. 
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @param   value       Data object.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_SetItemData(kDataTree tree, kDataTreeItem item, kObject value);

/** 
 * Gets item content as a k32u value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @param   value       Receives content as a numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_Item32u(kDataTree tree, kDataTreeItem item, k32u* value);

/** 
 * Gets item content as a kSize value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @param   value       Receives content as a numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_ItemSize(kDataTree tree, kDataTreeItem item, kSize* value);

/** 
 * Gets item content as a k32s value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @param   value       Receives content as a numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_Item32s(kDataTree tree, kDataTreeItem item, k32s* value);

/** 
 * Gets item content as a kSSize value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @param   value       Receives content as a numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_ItemSSize(kDataTree tree, kDataTreeItem item, kSSize* value);

/** 
 * Gets item content as a kBool value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @param   value       Receives content as a boolean value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_ItemBool(kDataTree tree, kDataTreeItem item, kBool* value);

/** 
 * Gets item content as a k64u value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @param   value       Receives content as a numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_Item64u(kDataTree tree, kDataTreeItem item, k64u* value);

/** 
 * Gets item content as a k64s value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @param   value       Receives content as a numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_Item64s(kDataTree tree, kDataTreeItem item, k64s* value);

/** 
 * Gets item content as a k32f value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @param   value       Receives content as a numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_Item32f(kDataTree tree, kDataTreeItem item, k32f* value);

/** 
 * Gets item content as a k64f value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @param   value       Receives content as a numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_Item64f(kDataTree tree, kDataTreeItem item, k64f* value);

/** 
 * Gets item content as a text value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @param   value       Receives content as a text value.   
 * @param   capacity    Size, in characters, of the string buffer.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_ItemText(kDataTree tree, kDataTreeItem item, kChar* value, k32u capacity);

/** 
 * Gets item content as a data object.
 *
 * The data object returned by this function is a clone of the original; ownership of the 
 * cloned object is transferred to the caller.  Call kObject_Dispose to free the cloned object.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @param   value       Receives content as a data object.   
 * @param   allocator   Memory allocator (or kNULL for default).
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_ItemData(kDataTree tree, kDataTreeItem item, kObject* value, kAlloc allocator);

/** 
 * Sets child item content from a k32u value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item.   
 * @param   path        Path relative to the parent item.  
 * @param   value       Numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_SetChild32u(kDataTree tree, kDataTreeItem parent, const kChar* path, k32u value);

/** 
 * Sets child item content from a kSize value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item.   
 * @param   path        Path relative to the parent item.  
 * @param   value       Numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_SetChildSize(kDataTree tree, kDataTreeItem parent, const kChar* path, kSize value);

/** 
 * Sets child item content from a k32s value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item.   
 * @param   path        Path relative to the parent item.  
 * @param   value       Numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_SetChild32s(kDataTree tree, kDataTreeItem parent, const kChar* path, k32s value);

/** 
 * Sets child item content from a kSSize value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item.   
 * @param   path        Path relative to the parent item.  
 * @param   value       Numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_SetChildSSize(kDataTree tree, kDataTreeItem parent, const kChar* path, kSSize value);

/** 
 * Sets child item content from a kBool value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item.   
 * @param   path        Path relative to the parent item.  
 * @param   value       Boolean value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_SetChildBool(kDataTree tree, kDataTreeItem parent, const kChar* path, kBool value);

/** 
 * Sets child item content from a k64u value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item.   
 * @param   path        Path relative to the parent item.  
 * @param   value       Numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_SetChild64u(kDataTree tree, kDataTreeItem parent, const kChar* path, k64u value);

/** 
 * Sets child item content from a k64s value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item.   
 * @param   path        Path relative to the parent item.  
 * @param   value       Numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_SetChild64s(kDataTree tree, kDataTreeItem parent, const kChar* path, k64s value);

/** 
 * Sets child item content from a k32f value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item.   
 * @param   path        Path relative to the parent item.  
 * @param   value       Numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_SetChild32f(kDataTree tree, kDataTreeItem parent, const kChar* path, k32f value);

/** 
 * Sets child item content from a k64f value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item.   
 * @param   path        Path relative to the parent item.  
 * @param   value       Numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_SetChild64f(kDataTree tree, kDataTreeItem parent, const kChar* path, k64f value);

/** 
 * Sets child item content from a text value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item.   
 * @param   path        Path relative to the parent item.  
 * @param   value       Text value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_SetChildText(kDataTree tree, kDataTreeItem parent, const kChar* path, const kChar* value);

/** 
 * Sets child content to a particular data object.
 *
 * The data object passed as an argument to this function is copied by reference; ownership of the 
 * original object is transferred to the tree.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.
 * @param   parent      Parent item.
 * @param   path        Path relative to the parent item.
 * @param   value       Data object.
 * @return              Operation status.
 */
kFx(kStatus) kDataTree_PutChildData(kDataTree tree, kDataTreeItem parent, const kChar* path, kObject value);

/** 
 * Sets child content by cloning a data object.
 *
 * The data object passed as an argument to this function will be cloned; ownership of the 
 * original object is not transferred. 
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item.   
 * @param   path        Path relative to the parent item.  
 * @param   value       Data object.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_SetChildData(kDataTree tree, kDataTreeItem parent, const kChar* path, kObject value);

/** 
 * Gets child item content as a k32u value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item.   
 * @param   path        Path relative to the parent item.  
 * @param   value       Receives content as a numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_Child32u(kDataTree tree, kDataTreeItem parent, const kChar* path, k32u* value);

/** 
 * Gets child item content as a kSize value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item.   
 * @param   path        Path relative to the parent item.  
 * @param   value       Receives content as a numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_ChildSize(kDataTree tree, kDataTreeItem parent, const kChar* path, kSize* value);

/** 
 * Gets child item content as a k32s value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item.   
 * @param   path        Path relative to the parent item.  
 * @param   value       Receives content as a numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_Child32s(kDataTree tree, kDataTreeItem parent, const kChar* path, k32s* value);

/** 
 * Gets child item content as a kSSize value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item.   
 * @param   path        Path relative to the parent item.  
 * @param   value       Receives content as a numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_ChildSSize(kDataTree tree, kDataTreeItem parent, const kChar* path, kSSize* value);

/** 
 * Gets child item content as a kBool value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item.   
 * @param   path        Path relative to the parent item.  
 * @param   value       Receives content as a boolean value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_ChildBool(kDataTree tree, kDataTreeItem parent, const kChar* path, kBool* value);

/** 
 * Gets child item content as a k64u value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item.   
 * @param   path        Path relative to the parent item.  
 * @param   value       Receives content as a numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_Child64u(kDataTree tree, kDataTreeItem parent, const kChar* path, k64u* value);

/** 
 * Gets child item content as a k64s value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item.   
 * @param   path        Path relative to the parent item.  
 * @param   value       Receives content as a numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_Child64s(kDataTree tree, kDataTreeItem parent, const kChar* path, k64s* value);

/** 
 * Gets child item content as a k32f value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item.   
 * @param   path        Path relative to the parent item.  
 * @param   value       Receives content as a numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_Child32f(kDataTree tree, kDataTreeItem parent, const kChar* path, k32f* value);

/** 
 * Gets child item content as a k64f value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item.   
 * @param   path        Path relative to the parent item.  
 * @param   value       Receives content as a numeric value.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_Child64f(kDataTree tree, kDataTreeItem parent, const kChar* path, k64f* value);

/** 
 * Gets child item content as a text value.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item.   
 * @param   path        Path relative to the parent item.  
 * @param   value       Receives content as a text value.   
 * @param   capacity    Size, in characters, of the string buffer.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_ChildText(kDataTree tree, kDataTreeItem parent, const kChar* path, kChar* value, k32u capacity);

/** 
 * Gets child item content as a text value.
 *
 * The data object returned by this function is a clone of the original; ownership of the 
 * cloned object is transferred to the caller.  Call kObject_Dispose to free the cloned object.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item.   
 * @param   path        Path relative to the parent item.  
 * @param   value       Receives content as a data object.   
 * @param   allocator   Memory allocator (or kNULL for default). 
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_ChildData(kDataTree tree, kDataTreeItem parent, const kChar* path, kObject* value, kAlloc allocator);

/** 
 * Gets a child item by path. 
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item.   
 * @param   path        Path relative to the parent item.  
 * @param   item        Receives child item.   
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_ChildItem(kDataTree tree, kDataTreeItem parent, const kChar* path, kDataTreeItem* item);

/** 
 * Returns the root element of the data tree.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @return              Returns the root element of the data tree.
 */
kFx(kDataTreeItem) kDataTree_Root(kDataTree tree);

/** 
 * Returns the parent of the given tree item.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @return              Returns the parent of the given tree item.
 */
kFx(kDataTreeItem) kDataTree_Parent(kDataTree tree, kDataTreeItem item);

/** 
 * Returns the first child of the given tree item.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @return              Returns the first child of the given tree item.
 */
kFx(kDataTreeItem) kDataTree_FirstChild(kDataTree tree, kDataTreeItem item);

/** 
 * Returns the last child of the given tree item.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @return              Returns the last child of the given tree item.
 */
kFx(kDataTreeItem) kDataTree_LastChild(kDataTree tree, kDataTreeItem item);

/** 
 * Returns the next sibling of the given tree item.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @return              Returns the next sibling of the given tree item.
 */
kFx(kDataTreeItem) kDataTree_NextSibling(kDataTree tree, kDataTreeItem item);

/** 
 * Returns the previous sibling of the given tree item.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @return              Returns the previous sibling of the given tree item.
 */
kFx(kDataTreeItem) kDataTree_PreviousSibling(kDataTree tree, kDataTreeItem item);

/** 
 * Returns the number of child items for the given parent item.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item.   
 * @return              Returns the number of child items for the given parent item.
 */
kFx(k32u) kDataTree_ChildCount(kDataTree tree, kDataTreeItem parent);

/** 
 * Returns a child item at a specific index within the list of child items for the given parent item.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   parent      Parent item.   
 * @param   index       Child item index.  
 * @return              Returns a child item at a specific index within the list of child items for the given parent item.
 */
kFx(kDataTreeItem) kDataTree_ChildAt(kDataTree tree, kDataTreeItem parent, k32u index);

/** 
 * Sets the name of a tree item. 
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @param   name        Item name. 
 * @return              Operation status. 
 */
kFx(kStatus) kDataTree_SetItemName(kDataTree tree, kDataTreeItem item, const kChar* name);

/** 
 * Returns the name of a tree item.
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @return              Returns the name of a tree item.
 */
kFx(const kChar*) kDataTree_ItemName(kDataTree tree, kDataTreeItem item);

/** 
 * Returns the data object owned by a tree item. 
 *
 * The kDataTree_ItemData function creates a clone of the underlying data and returns 
 * the clone. The kDataTree_ItemContent function returns a reference to the underlying 
 * data object. Ownership of the underlying data object is not transferred by this function. 
 *
 * @public              @memberof kDataTree
 * @param   tree        Tree object.   
 * @param   item        Tree item.   
 * @return              Returns the data object owned by a tree item. 
 */
kFx(kObject) kDataTree_ItemContent(kDataTree tree, kDataTreeItem item);

#include <kApi/Data/kDataTree.x.h>

#endif
