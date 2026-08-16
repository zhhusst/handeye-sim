/** 
 * @file    kDynamicLib.h
 * @brief   Declares the kDynamicLib class. 
 *
 * @internal
 * Copyright (C) 2013-2024 by LMI Technologies Inc.
 * Licensed under the MIT License.
 * Redistributed files must retain the above copyright notice.
 */
#ifndef K_API_DYNAMIC_LIB_H
#define K_API_DYNAMIC_LIB_H

#include <kApi/kApiDef.h>

 /**
  * @class   kDynamicLib
  * @extends kValue
  * @ingroup kApi-Utils
  * @brief   Represents a kDynamicLib construction option.
  */
typedef k32s kDynamicLibOption;

/** @relates kDynamicLibOption @{ */
#define kDYNAMIC_LIB_OPTION_NONE                (0x0)       ///< No options.
#define kDYNAMIC_LIB_ALTERNATE_SEARCH_PATH      (0x1)       ///< Use alternate search path, supported by Windows, ignored elsewhere.
/** @} */

#include <kApi/Utils/kDynamicLib.x.h>

/**
 * @class   kDynamicLib
 * @extends kObject
 * @ingroup kApi-Utils
 * @brief   Represents a dynamically loaded library.
 * 
 * This class relies on OS support for dynamic loading, and is only available on platforms that 
 * provide underlying support.
 */
//typedef kObject kDynamicLib;             --forward-declared in kApiDef.x.h 

/** 
 * Constructs a kDynamicLib object.
 * 
 * @public              @memberof kDynamicLib
 * @param   library     Destination for the constructed object handle. 
 * @param   path        Path to the dynamic library. 
 * @param   allocator   Memory allocator (or kNULL for default). 
 * @return              Operation status. 
 */
kFx(kStatus) kDynamicLib_Construct(kDynamicLib* library, const kChar* path, kAlloc allocator);

/**
 * Constructs a kDynamicLib object with support for behavioral options.
 *
 * @public              @memberof kDynamicLib
 * @param   library     Destination for the constructed object handle.
 * @param   path        Path to the dynamic library.
 * @param   options     Bitset of library options. 
 * @param   allocator   Memory allocator (or kNULL for default).
 * @return              Operation status.
 */
kFx(kStatus) kDynamicLib_ConstructEx(kDynamicLib* library, const kChar* path, kDynamicLibOption options, kAlloc allocator);

/**
* Resolves a function pointer by name from the dynamic library.
*
* @public              @memberof kDynamicLib
* @param   library     Dynamic library object.
* @param   name        Function name.
* @param   function    Receives function pointer.
* @return              Operation status.
*/
kFx(kStatus) kDynamicLib_FindFunction(kDynamicLib library, const kChar* name, kFunction* function);

#endif
