// 
// KDataTreeItem.h
//
// Copyright (C) 2014-2024 by LMI Technologies Inc.
// Licensed under the MIT License.
// Redistributed files must retain the above copyright notice.
// 
#ifndef K_API_NET_DATA_TREE_ITEM_H
#define K_API_NET_DATA_TREE_ITEM_H

#include <kApi/Data/kDataTreeItem.x.h>
#include "kApiNet/KAlloc.h"

namespace Lmi3d
{
    namespace Zen
    {
        namespace Data
        {
            /// <summary>Represents a tree of data objects.</summary>
            public ref class KDataTreeItem : public KObject
            {
                KDeclareNoneClass(KDataTreeItem, kDataTreeItem)

            public:
                /// <summary>Initializes a new instance of the KDataTreeItem class with the specified Zen object handle.</summary>           
                /// <param name="handle">Zen object handle.</param>
                KDataTreeItem(IntPtr handle)
                    : KObject(handle)
                {}

                property String^ Name
                {
                    String^ get()
                    {
                        return KToString(kDataTreeItem_Name(Handle));
                    }
                }

                property KObject^ Data
                {
                    KObject^ get()
                    {
                        kObject object = kDataTreeItem_GetData(Handle);

                        KAdjustRef(object, kTRUE, Nullable<KRefStyle>());
                        return KToObject<KObject^>(object);
                    }
                }
            };
        }
    }
}

#endif