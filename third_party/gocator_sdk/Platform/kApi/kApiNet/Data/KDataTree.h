// 
// KDataTree.h
//
// Copyright (C) 2014-2024 by LMI Technologies Inc.
// Licensed under the MIT License.
// Redistributed files must retain the above copyright notice.
// 
#ifndef K_API_NET_DATA_TREE_H
#define K_API_NET_DATA_TREE_H

#include <kApi/Data/kDataTree.h>
#include "kApiNet/KAlloc.h"
#include "kApiNet/Data/KString.h"
#include "kApiNet/Data/KDataTreeItem.h"

namespace Lmi3d
{
    namespace Zen
    {
        namespace Data
        {
            /// <summary>Represents a tree of data objects.</summary>
            public ref class KDataTree : public KObject
            {
                KDeclareAutoClass(KDataTree, kDataTree)

            public:
                /// <summary>Initializes a new instance of the KDataTree class with the specified Zen object handle.</summary>           
                /// <param name="handle">Zen object handle.</param>
                KDataTree(IntPtr handle)
                    : KObject(handle)
                {}

                /// <inheritdoc cref="KDataTree(IntPtr)" />
                ///
                /// <param name="refStyle">RefStyle for this object.</param>
                KDataTree(IntPtr handle, KRefStyle refStyle)
                    : KObject(handle, refStyle)
                {}

                /// <summary>Constructs a data tree object.</summary>
                /// 
                /// <remarks>The tree is initially empty; use kDataTree_Add with parent=kNULL to create a root 
                /// node.</remarks>
                KDataTree()
                {
                    kDataTree handle = kNULL;

                    KCheck(kDataTree_Construct(&handle, kNULL));

                    Handle = handle;
                }

                /// <inheritdoc cref="KDataTree()" />
                /// 
                /// <param name="allocator">Memory allocator (or null for default).</param>
                KDataTree(KAlloc^ allocator)
                {
                    kDataTree handle = kNULL;

                    KCheck(kDataTree_Construct(&handle, KToHandle(allocator)));

                    Handle = handle;
                }

                /// <inheritdoc cref="KDataTree(KAlloc^)"/>
                /// 
                /// <param name="refStyle">RefStyle for this object.</param>
                KDataTree(KAlloc^ allocator, KRefStyle refStyle)
                    : KObject(refStyle)
                {
                    kDataTree handle = kNULL;

                    KCheck(kDataTree_Construct(&handle, KToHandle(allocator)));

                    Handle = handle;
                }

                /// <summary>Inserts a new child item at the end of the specified parent's child list.</summary>
                /// 
                /// <param name="parent">Parent item, or kNULL to add a root node.</param>
                /// <param name="name">Name of the new item.</param>
                /// <returns>The new child item.</returns>
                KDataTreeItem^ Add(KDataTreeItem^ parent, String^ name)
                {
                    KString nameString(name);
                    kDataTreeItem item = kNULL;

                    KCheck(kDataTree_Add(Handle, KToHandle(parent), nameString.CharPtr, &item));

                    return KToObject<KDataTreeItem^>(item);
                }

                /// <summary>Inserts a new item before the specified sibling node.</summary>
                /// 
                /// <param name="parent">Parent item, or kNULL to insert a root node.</param>
                /// <param name="before">Sibling item, or kNULL to insert at end of specified parent's child list.</param>
                /// <param name="name">Name of the new item.</param>
                /// <returns>The new child item.</returns>
                KDataTreeItem^ Insert(KDataTreeItem^ parent, KDataTreeItem^ before, String^ name)
                {
                    KString nameString(name);
                    kDataTreeItem item = kNULL;

                    KCheck(kDataTree_Insert(Handle, KToHandle(parent), KToHandle(before), nameString.CharPtr, &item));

                    return KToObject<KDataTreeItem^>(item);
                }

                /// <summary>Moves a node from one location in the tree to another location.</summary>
                /// 
                /// <param name="source">Source node.</param>
                /// <param name="destParent">Destination parent node.</param>
                /// <param name="destBefore">Destination sibling item, or kNULL to insert at end of specified parent's child list.</param>
                void Move(KDataTreeItem^ source, KDataTreeItem^ destParent, KDataTreeItem^ destBefore)
                {
                    KCheck(kDataTree_Move(Handle, KToHandle(source), KToHandle(destParent), KToHandle(destBefore)));
                }

                /// <summary>Deletes an item from the tree.</summary>
                /// 
                /// <param name="item">Tree item.</param>
                void Delete(KDataTreeItem^ item)
                {
                    KCheck(kDataTree_Delete(Handle, KToHandle(item)));
                }

                /// <summary>Removes all elements from the tree.</summary>
                void Clear()
                {
                    KCheck(kDataTree_Clear(Handle));
                }

                /// <summary>Sets item content from a k32u value.</summary>
                /// 
                /// <param name="item">Tree item.</param>
                /// <param name="value">Numeric value.</param>
                void SetItem(KDataTreeItem^ item, k32u value)
                {
                    KCheck(kDataTree_SetItem32u(Handle, KToHandle(item), value));
                }

                ///// <summary>Sets item content from a kSize value.</summary>
                ///// 
                ///// <param name="item">Tree item.</param>
                ///// <param name="value">Numeric value.</param>
                //void SetItem(KDataTreeItem^ item, k64s value)
                //{
                //    //kFsFx(kStatus) kDataTree_SetItemSize(kDataTree tree, kDataTreeItem item, kSize value);
                //    KCheck(kDataTree_SetItemSize(Handle, KToHandle(item), K64sToSize(value)));
                //}

                /// <summary>Sets item content from a k32s value.</summary>
                /// 
                /// <param name="item">Tree item.</param>
                /// <param name="value">Numeric value.</param>
                void SetItem(KDataTreeItem^ item, k32s value)
                {
                    KCheck(kDataTree_SetItem32s(Handle, KToHandle(item), value));
                }

                /// <summary>Sets item content from a kSSize value.</summary>
                /// 
                /// <param name="item">Tree item.</param>
                /// <param name="value">Numeric value.</param>
                void SetItem(KDataTreeItem^ item, KSSize value)
                {
                    //kFsFx(kStatus) kDataTree_SetItemSSize(kDataTree tree, kDataTreeItem item, kSSize value);
                    KCheck(kDataTree_SetItemSSize(Handle, KToHandle(item), value));
                }

                /// <summary>Sets item content from a kBool value.</summary>
                /// 
                /// <param name="item">Tree item.</param>
                /// <param name="value">Boolean value.</param>
                void SetItem(KDataTreeItem^ item, bool value)
                {
                    //kFsFx(kStatus) kDataTree_SetItemBool(kDataTree tree, kDataTreeItem item, kBool value);
                    KCheck(kDataTree_SetItemBool(Handle, KToHandle(item), value));
                }

                /// <summary>Sets item content from a k64u value.</summary>
                /// 
                /// <param name="item">Tree item.</param>
                /// <param name="value">Numeric value.</param>
                void SetItem(KDataTreeItem^ item, k64u value)
                {
                    KCheck(kDataTree_SetItem64u(Handle, KToHandle(item), value));
                }

                /// <summary>Sets item content from a k64s value.</summary>
                /// 
                /// <param name="item">Tree item.</param>
                /// <param name="value">Numeric value.</param>
                void SetItem(KDataTreeItem^ item, k64s value)
                {
                    KCheck(kDataTree_SetItem64s(Handle, KToHandle(item), value));
                }

                /// <summary>Sets item content from a k32f value.</summary>
                /// 
                /// <param name="item">Tree item.</param>
                /// <param name="value">Numeric value.</param>
                void SetItem(KDataTreeItem^ item, k32f value)
                {
                    KCheck(kDataTree_SetItem32f(Handle, KToHandle(item), value));
                }

                /// <summary>Sets item content from a k64f value.</summary>
                /// 
                /// <param name="item">Tree item.</param>
                /// <param name="value">Numeric value.</param>
                void SetItem(KDataTreeItem^ item, k64f value)
                {
                    KCheck(kDataTree_SetItem64f(Handle, KToHandle(item), value));
                }

                /// <summary>Sets item content from a text value.</summary>
                /// 
                /// <param name="item">Tree item.</param>
                /// <param name="value">Numeric value.</param>
                void SetItemText(KDataTreeItem^ item, String^ value)
                {
                    KString valueString(value);

                    KCheck(kDataTree_SetItemText(Handle, KToHandle(item), valueString.CharPtr));
                }

                /// <summary>Sets item content to a particular data object.</summary>
                /// 
                /// <remarks>The data object passed as an argument to this function is copied by reference; ownership 
                /// of the original object is transferred to the tree.</remarks>
                /// 
                /// <param name="item">Tree item.</param>
                /// <param name="value">Data object.</param>
                void PutItemData(KDataTreeItem^ item, KObject^ value)
                {
                    KCheck(kDataTree_PutItemData(Handle, KToHandle(item), KToHandle(value)));
                }

                /// <summary>Sets item content by cloning a data object.</summary>
                /// 
                /// <remarks>The data object passed as an argument to this function will be cloned; ownership of 
                /// the original object is not transferred.</remarks>
                /// 
                /// <param name="item">Tree item.</param>
                /// <param name="value">Data object.</param>
                void SetItemData(KDataTreeItem^ item, KObject^ value)
                {
                    KCheck(kDataTree_SetItemData(Handle, KToHandle(item), KToHandle(value)));
                }

                /// <summary>Gets item content as a k32u value.</summary>
                /// 
                /// <param name="item">Tree item.</param>
                /// <returns>The content as a numeric value.</returns>
                k32u ItemUInt32(KDataTreeItem^ item)
                {
                    k32u value;

                    KCheck(kDataTree_Item32u(Handle, KToHandle(item), &value));

                    return value;
                }

                /// <summary>Gets item content as a k32s value.</summary>
                /// 
                /// <param name="item">Tree item.</param>
                /// <returns>The content as a numeric value.</returns>
                k32s ItemInt32(KDataTreeItem^ item)
                {
                    k32s value;

                    KCheck(kDataTree_Item32s(Handle, KToHandle(item), &value));

                    return value;
                }

                /// <summary>Gets item content as a kBool value.</summary>
                /// 
                /// <param name="item">Tree item.</param>
                /// <returns>The content as a boolean value.</returns>
                bool ItemBoolean(KDataTreeItem^ item)
                {
                    kBool value;

                    KCheck(kDataTree_ItemBool(Handle, KToHandle(item), &value));

                    return KToBool(value);
                }

                /// <summary>Gets item content as a k64u value.</summary>
                /// 
                /// <param name="item">Tree item.</param>
                /// <returns>The content as a numeric value.</returns>
                k64u ItemUInt64(KDataTreeItem^ item)
                {
                    k64u value;

                    KCheck(kDataTree_Item64u(Handle, KToHandle(item), &value));

                    return value;
                }

                /// <summary>Gets item content as a k64s value.</summary>
                /// 
                /// <param name="item">Tree item.</param>
                /// <returns>The content as a numeric value.</returns>
                k64s ItemInt64(KDataTreeItem^ item)
                {
                    k64s value;

                    KCheck(kDataTree_Item64s(Handle, KToHandle(item), &value));

                    return value;
                }

                /// <summary>Gets item content as a k32f value.</summary>
                /// 
                /// <param name="item">Tree item.</param>
                /// <returns>The content as a numeric value.</returns>
                k32f ItemFloat(KDataTreeItem^ item)
                {
                    k32f value;

                    KCheck(kDataTree_Item32f(Handle, KToHandle(item), &value));

                    return value;
                }

                /// <summary>Gets item content as a k64f value.</summary>
                /// 
                /// <param name="item">Tree item.</param>
                /// <returns>The content as a numeric value.</returns>
                k64f ItemDouble(KDataTreeItem^ item)
                {
                    k64f value;

                    KCheck(kDataTree_Item64f(Handle, KToHandle(item), &value));

                    return value;
                }

                /// <summary>Gets item content as a text value.</summary>
                /// 
                /// <param name="item">Tree item.</param>
                /// <returns>The content as a text value.</returns>
                String^ ItemString(KDataTreeItem^ item)
                {
                    kChar value[4096];

                    KCheck(kDataTree_ItemText(Handle, KToHandle(item), value, kCountOf(value)));

                    return KToString(value);
                }

                /// <summary>Gets item content as a data object.</summary>
                /// 
                /// <remarks>The data object returned by this function is a clone of the original; ownership of 
                /// the cloned object is transferred to the caller.  Call kObject_Dispose to free 
                /// the cloned object.</remarks>
                /// 
                /// <param name="item">Tree item.</param>
                /// <returns>The content as a data object.</returns>
                KObject^ ItemData(KDataTreeItem^ item)
                {
                    kObject value = kNULL;

                    KCheck(kDataTree_ItemData(Handle, KToHandle(item), &value, kNULL));

                    return KToObject<KObject^>(value);
                }

                /// <inheritdoc cref="ItemData(KDataTreeItem^)" />
                ///
                /// <param name="allocator">Memory allocator (or null for default).</param>
                KObject^ ItemData(KDataTreeItem^ item, KAlloc^ allocator)
                {
                    kObject value = kNULL;

                    KCheck(kDataTree_ItemData(Handle, KToHandle(item), &value, KToHandle(allocator)));

                    return KToObject<KObject^>(value);
                }

                ///
                /// <returns>The content as a data object.</returns>
                generic <typename T> where T : KObject
                T ItemData(KDataTreeItem^ item)
                {
                    kObject value;

                    KCheck(kDataTree_ItemData(Handle, KToHandle(item), &value, kNULL));

                    return KToObject<T>(value);
                }

                /// <summary>Sets child item content from a k32u value.</summary>
                /// 
                /// <param name="parent">Parent item.</param>
                /// <param name="path">Path relative to the parent item.</param>
                /// <param name="value">Numeric value.</param>
                void SetChild(KDataTreeItem^ parent, String^ path, k32u value)
                {
                    KString pathString(path);

                    KCheck(kDataTree_SetChild32u(Handle, KToHandle(parent), pathString.CharPtr, value));
                }

                /// <summary>Sets child item content from a k32s value.</summary>
                /// 
                /// <param name="parent">Parent item.</param>
                /// <param name="path">Path relative to the parent item.</param>
                /// <param name="value">Numeric value.</param>
                void SetChild(KDataTreeItem^ parent, String^ path, k32s value)
                {
                    KString pathString(path);

                    KCheck(kDataTree_SetChild32s(Handle, KToHandle(parent), pathString.CharPtr, value));
                }

                /// <summary>Sets child item content from a kSSize value.</summary>
                /// 
                /// <param name="parent">Parent item.</param>
                /// <param name="path">Path relative to the parent item.</param>
                /// <param name="value">Numeric value.</param>
                void SetChild(KDataTreeItem^ parent, String^ path, KSSize value)
                {
                    KString pathString(path);

                    KCheck(kDataTree_SetChildSSize(Handle, KToHandle(parent), pathString.CharPtr, value));
                }

                /// <summary>Sets child item content from a kBool value.</summary>
                /// 
                /// <param name="parent">Parent item.</param>
                /// <param name="path">Path relative to the parent item.</param>
                /// <param name="value">Boolean value.</param>
                void SetChild(KDataTreeItem^ parent, String^ path, bool value)
                {
                    KString pathString(path);

                    KCheck(kDataTree_SetChildBool(Handle, KToHandle(parent), pathString.CharPtr, value));
                }

                /// <summary>Sets child item content from a k64u value.</summary>
                /// 
                /// <param name="parent">Parent item.</param>
                /// <param name="path">Path relative to the parent item.</param>
                /// <param name="value">Numeric value.</param>
                void SetChild(KDataTreeItem^ parent, String^ path, k64u value)
                {
                    KString pathString(path);

                    KCheck(kDataTree_SetChild64u(Handle, KToHandle(parent), pathString.CharPtr, value));
                }

                /// <summary>Sets child item content from a k64s value.</summary>
                /// 
                /// <param name="parent">Parent item.</param>
                /// <param name="path">Path relative to the parent item.</param>
                /// <param name="value">Numeric value.</param>
                void SetChild(KDataTreeItem^ parent, String^ path, k64s value)
                {
                    KString pathString(path);

                    KCheck(kDataTree_SetChild64s(Handle, KToHandle(parent), pathString.CharPtr, value));
                }

                /// <summary>Sets child item content from a k32f value.</summary>
                /// 
                /// <param name="parent">Parent item.</param>
                /// <param name="path">Path relative to the parent item.</param>
                /// <param name="value">Numeric value.</param>
                void SetChild(KDataTreeItem^ parent, String^ path, k32f value)
                {
                    KString pathString(path);

                    KCheck(kDataTree_SetChild32f(Handle, KToHandle(parent), pathString.CharPtr, value));
                }

                /// <summary>Sets child item content from a k64f value.</summary>
                /// 
                /// <param name="parent">Parent item.</param>
                /// <param name="path">Path relative to the parent item.</param>
                /// <param name="value">Numeric value.</param>
                void SetChild(KDataTreeItem^ parent, String^ path, k64f value)
                {
                    KString pathString(path);

                    KCheck(kDataTree_SetChild64f(Handle, KToHandle(parent), pathString.CharPtr, value));
                }

                /// <summary>Sets child item content from a text value.</summary>
                /// 
                /// <param name="parent">Parent item.</param>
                /// <param name="path">Path relative to the parent item.</param>
                /// <param name="value">Text value.</param>
                void SetChild(KDataTreeItem^ parent, String^ path, String^ value)
                {
                    KString pathString(path);
                    KString valueString(value);

                    KCheck(kDataTree_SetChildText(Handle, KToHandle(parent), pathString.CharPtr, valueString.CharPtr));
                }

                /// <summary>Sets child content to a particular data object.</summary>
                /// 
                /// <remarks>The data object passed as an argument to this function is copied by reference; ownership 
                /// of the original object is transferred to the tree.</remarks>
                /// 
                /// <param name="parent">Parent item.</param>
                /// <param name="path">Path relative to the parent item.</param>
                /// <param name="value">Data object.</param>
                void PutChildData(KDataTreeItem^ parent, String^ path, KObject^ value)
                {
                    KString pathString(path);

                    KCheck(kDataTree_PutChildData(Handle, KToHandle(parent), pathString.CharPtr, KToHandle(value)));
                }

                /// <summary>Sets child content by cloning a data object.</summary>
                /// 
                /// <remarks>The data object passed as an argument to this function will be cloned; ownership of 
                /// the original object is not transferred.</remarks>
                /// 
                /// <param name="parent">Parent item.</param>
                /// <param name="path">Path relative to the parent item.</param>
                /// <param name="value">Data object.</param>
                void SetChild(KDataTreeItem^ parent, String^ path, KObject^ value)
                {
                    KString pathString(path);

                    KCheck(kDataTree_SetChildData(Handle, KToHandle(parent), pathString.CharPtr, KToHandle(value)));
                }

                /// <summary>Gets child item content as a k32u value.</summary>
                /// 
                /// <param name="parent">Parent item.</param>
                /// <param name="path">Path relative to the parent item.</param>
                /// <returns>The content as a numeric value.</returns>
                k32u GetUInt32(KDataTreeItem^ parent, String^ path)
                {
                    k32u value;
                    KString pathString(path);

                    KCheck(kDataTree_Child32u(Handle, KToHandle(parent), pathString.CharPtr, &value));

                    return value;
                }

                /// <summary>Gets child item content as a k32s value.</summary>
                /// 
                /// <param name="parent">Parent item.</param>
                /// <param name="path">Path relative to the parent item.</param>
                /// <returns>The content as a numeric value.</returns>
                k32s GetInt32(KDataTreeItem^ parent, String^ path)
                {
                    k32s value;
                    KString pathString(path);

                    KCheck(kDataTree_Child32s(Handle, KToHandle(parent), pathString.CharPtr, &value));

                    return value;
                }

                /// <summary>Gets child item content as a kBool value.</summary>
                /// 
                /// <param name="parent">Parent item.</param>
                /// <param name="path">Path relative to the parent item.</param>
                /// <returns>The content as a boolean value.</returns>
                bool GetBoolean(KDataTreeItem^ parent, String^ path)
                {
                    kBool value;
                    KString pathString(path);

                    KCheck(kDataTree_ChildBool(Handle, KToHandle(parent), pathString.CharPtr, &value));

                    return KToBool(value);
                }

                /// <summary>Gets child item content as a k64u value.</summary>
                /// 
                /// <param name="parent">Parent item.</param>
                /// <param name="path">Path relative to the parent item.</param>
                /// <returns>The content as a numeric value.</returns>
                k64u GetUInt64(KDataTreeItem^ parent, String^ path)
                {
                    k64u value;
                    KString pathString(path);

                    KCheck(kDataTree_Child64u(Handle, KToHandle(parent), pathString.CharPtr, &value));

                    return value;
                }

                /// <summary>Gets child item content as a k64s value.</summary>
                /// 
                /// <param name="parent">Parent item.</param>
                /// <param name="path">Path relative to the parent item.</param>
                /// <returns>The content as a numeric value.</returns>
                k64s GetInt64(KDataTreeItem^ parent, String^ path)
                {
                    k64s value;
                    KString pathString(path);

                    KCheck(kDataTree_Child64s(Handle, KToHandle(parent), pathString.CharPtr, &value));

                    return value;
                }

                /// <summary>Gets child item content as a k32f value.</summary>
                /// 
                /// <param name="parent">Parent item.</param>
                /// <param name="path">Path relative to the parent item.</param>
                /// <returns>The content as a numeric value.</returns>
                k32f GetFloat(KDataTreeItem^ parent, String^ path)
                {
                    k32f value;
                    KString pathString(path);

                    KCheck(kDataTree_Child32f(Handle, KToHandle(parent), pathString.CharPtr, &value));

                    return value;
                }

                /// <summary>Gets child item content as a k64f value.</summary>
                /// 
                /// <param name="parent">Parent item.</param>
                /// <param name="path">Path relative to the parent item.</param>
                /// <returns>The content as a numeric value.</returns>
                k64f GetDouble(KDataTreeItem^ parent, String^ path)
                {
                    k64f value;
                    KString pathString(path);

                    KCheck(kDataTree_Child64f(Handle, KToHandle(parent), pathString.CharPtr, &value));

                    return value;
                }

                /// <summary>Gets child item content as a text value.</summary>
                /// 
                /// <param name="parent">Parent item.</param>
                /// <param name="path">Path relative to the parent item.</param>
                /// <returns>The content as a text value.</returns>
                String^ GetString(KDataTreeItem^ parent, String^ path)
                {
                    kString valueString = kNULL;
                    KString pathString(path);

                    try 
                    {
                        KCheck(kDataTree_ChildData(Handle, KToHandle(parent), pathString.CharPtr, &valueString, kNULL));
                        return gcnew String(kString_Chars(valueString));
                    }
                    finally
                    {
                        kDisposeRef(&valueString);
                    }
                }

                /// <summary>Gets child item content as a data object.</summary>
                /// 
                /// <remarks>The data object returned by this function is a clone of the original; ownership of 
                /// the cloned object is transferred to the caller.  Call kObject_Dispose to free 
                /// the cloned object.</remarks>
                /// 
                /// <param name="parent">Parent item.</param>
                /// <param name="path">Path relative to the parent item.</param>
                /// <returns>The content as a data object.</returns>
                KObject^ GetData(KDataTreeItem^ parent, String^ path)
                {
                    kObject value;
                    KString pathString(path);

                    KCheck(kDataTree_ChildData(Handle, KToHandle(parent), pathString.CharPtr, &value, kNULL));

                    return KToObject<KObject^>(value);
                }

                ///
                /// <returns>The content as a data object.</returns>
                generic <typename T> where T : KObject
                T GetData(KDataTreeItem^ parent, String^ path) 
                {
                    kObject value;
                    KString pathString(path);

                    KCheck(kDataTree_ChildData(Handle, KToHandle(parent), pathString.CharPtr, &value, kNULL));

                    return KToObject<T>(value);
                }

                /// <inheritdoc cref="GetData(KDataTreeItem^, String^)" />
                ///
                /// <param name="allocator">Memory allocator (or null for default).</param>
                KObject^ GetData(KDataTreeItem^ parent, String^ path, KAlloc^ allocator)
                {
                    kObject value;
                    KString pathString(path);

                    KCheck(kDataTree_ChildData(Handle, KToHandle(parent), pathString.CharPtr, &value, KToHandle(allocator)));

                    return KToObject<KObject^>(value);
                }

                /// <summary>Gets a child item by path.</summary>
                /// 
                /// <param name="parent">Parent item.</param>
                /// <param name="path">Path relative to the parent item.</param>
                /// <returns>The child item.</returns>
                KDataTreeItem^ ChildItem(KDataTreeItem^ parent, String^ path)
                {
                    kDataTreeItem item = kNULL;
                    KString pathString(path);

                    KCheck(kDataTree_ChildItem(Handle, KToHandle(parent), pathString.CharPtr, &item));

                    return KToObject<KDataTreeItem^>(item);
                }

                /// <summary>Returns the root element of the data tree.</summary>
                /// 
                /// <returns>Returns the root element of the data tree.</returns>
                KDataTreeItem^ Root()
                {
                    //kFsFx(kDataTreeItem) kDataTree_Root(kDataTree tree);
                    return KToObject<KDataTreeItem^>(kDataTree_Root(Handle));
                }

                /// <summary>Returns the parent of the given tree item.</summary>
                /// 
                /// <param name="item">Tree item.</param>
                /// <returns>Returns the parent of the given tree item.</returns>
                KDataTreeItem^ parent(KDataTreeItem^ item)
                {
                    //kFsFx(kDataTreeItem) kDataTree_Parent(kDataTree tree, kDataTreeItem item);
                    return KToObject<KDataTreeItem^>(kDataTree_Parent(Handle, KToHandle(item)));
                }

                /// <summary>Returns the first child of the given tree item.</summary>
                /// 
                /// <param name="item">Tree item.</param>
                /// <returns>Returns the first child of the given tree item.</returns>
                KDataTreeItem^ FirstChild(KDataTreeItem^ item)
                {
                    //kFsFx(kDataTreeItem) kDataTree_FirstChild(kDataTree tree, kDataTreeItem item);
                    return KToObject<KDataTreeItem^>(kDataTree_FirstChild(Handle, KToHandle(item)));
                }

                /// <summary>Returns the last child of the given tree item.</summary>
                /// 
                /// <param name="item">Tree item.</param>
                /// <returns>Returns the last child of the given tree item.</returns>
                KDataTreeItem^ LastChild(KDataTreeItem^ item)
                {
                    return KToObject<KDataTreeItem^>(kDataTree_LastChild(Handle, KToHandle(item)));
                }

                /// <summary>Returns the next sibling of the given tree item.</summary>
                /// 
                /// <param name="item">Tree item.</param>
                /// <returns>Returns the next sibling of the given tree item.</returns>
                KDataTreeItem^ NextSibling(KDataTreeItem^ item)
                {
                    return KToObject<KDataTreeItem^>(kDataTree_NextSibling(Handle, KToHandle(item)));
                }

                /// <summary>Returns the previous sibling of the given tree item.</summary>
                /// 
                /// <param name="item">Tree item.</param>
                /// <returns>Returns the previous sibling of the given tree item.</returns>
                KDataTreeItem^ PreviousSibling(KDataTreeItem^ item)
                {
                    return KToObject<KDataTreeItem^>(kDataTree_PreviousSibling(Handle, KToHandle(item)));
                }

                /// <summary>Returns the number of child items for the given parent item.</summary>
                /// 
                /// <param name="parent">Parent item.</param>
                /// <returns>Returns the number of child items for the given parent item.</returns>
                k32u ChildCount(KDataTreeItem^ parent)
                {
                    return kDataTree_ChildCount(Handle, KToHandle(parent));
                }

                /// <summary>Returns a child item at a specific index within the list of child items for the given 
                /// parent item.</summary>
                /// 
                /// <param name="parent">Parent item.</param>
                /// <param name="index">Child item index.</param>
                /// <returns>Returns a child item at a specific index within the list of child items for the given parent item.</returns>
                KDataTreeItem^ ChildAt(KDataTreeItem^ parent, k32u index)
                {
                    kDataTreeItem item = kDataTree_ChildAt(Handle, KToHandle(parent), index);

                    return KToObject<KDataTreeItem^>(item);
                }

                /// <summary>Sets the name of a tree item.</summary>
                /// 
                /// <param name="item">Tree item.</param>
                /// <param name="name">Item name.</param>
                void SetItemName(KDataTreeItem^ item, String^ name)
                {
                    KString nameString(name);

                    KCheck(kDataTree_SetItemName(Handle, KToHandle(item), nameString.CharPtr));
                }

                /// <summary>Returns the name of a tree item.</summary>
                /// 
                /// <param name="item">Tree item.</param>
                /// <returns>Returns the name of a tree item.</returns>
                String^ ItemName(KDataTreeItem^ item)
                {
                    return KToString(kDataTree_ItemName(Handle, KToHandle(item)));
                }

                /// <summary>Returns the data object owned by a tree item.</summary>
                /// 
                /// <remarks>The kDataTree_ItemData function creates a clone of the underlying data and returns 
                /// the clone. The kDataTree_ItemContent function returns a reference to the underlying 
                /// data object. Ownership of the underlying data object is not transferred by this function.</remarks>
                /// 
                /// <param name="item">Tree item.</param>
                /// <returns>Returns the data object owned by a tree item.</returns>
                KObject^ ItemContent(KDataTreeItem^ item)
                {
                    kObject obj = kDataTree_ItemContent(Handle, KToHandle(item));

                    return KToObject<KObject^>(obj);
                }
            };
        }
    }
}

#endif