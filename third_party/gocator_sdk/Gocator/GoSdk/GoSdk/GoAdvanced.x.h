/**
 * @file    GoAdvanced.x.h
 *
 * @internal
 * Copyright (C) 2016-2024 by LMI Technologies Inc.
 * Licensed under the MIT License.
 * Redistributed files must retain the above copyright notice.
 */
#ifndef GO_ADVANCED_X_H
#define GO_ADVANCED_X_H

#include <kApi/Data/kXml.h>

 // GOC-15004 Initial maximum is 1 but can be increased when future HDR modes require more parameters.
#define GO_MAX_NUM_HDR_PARAMETERS 1

typedef struct GoAdvancedClass
{
    kObjectClass base;

    kObject sensor;

    kXml xml;
    kXmlItem xmlItem;

    GoAdvancedType type;
    GoAdvancedType typeSystemValue;
    kBool typeUsed;

    GoElement32u spotThreshold;
    GoElement32u spotWidthMax;

    GoElement32u spotWidthMin;
    GoElement32u widthThreshold;
    GoElement32u spotSumMin;
    GoElement32u sobelEdgeWindow;

    GoSpotSelectionType spotSelectionType;
    GoSpotSelectionType systemSpotSelectionType;
    kBool spotSelectionTypeUsed;
    kArrayList spotSelectionTypeOptions; //of type GoSpotSelectionType

    k32u spotContinuitySortingMinimumSegmentSize;
    k32u spotContinuitySortingSearchWindowX;
    k32u spotContinuitySortingSearchWindowY;

    GoElement32u spotTranslucentSortingOpaqueWidth;
    GoElement32u spotTranslucentSortingTranslucentWidth;
    GoElement32u spotTranslucentSortingMinLength;
    GoElement32s spotTranslucentSortingThreadingMode;
    kArrayList spotTranslucentSortingThreadingModeOptions; //of type GoTranslucentThreadingMode

    GoElement64f cameraGainAnalog;
    GoElement64f cameraGainDigital;

    GoHdrMode hdrMode;
    kBool hdrModeUsed;
    kBool hdrModeUsedSet;
    kArrayList hdrModeOptions;
    k64f hdrParameters[GO_MAX_NUM_HDR_PARAMETERS];
    // Current count of number of valid entries in the hdrParameters array up to GO_MAX_NUM_HDR_PARAMETERS.
    kSize hdrParametersCount;
    kBool hdrParametersUsed;
    // Min/max limits of each parameter value.
    k64f hdrParametersValueMin;
    k64f hdrParametersValueMax;
    // Min/max of number of parameters allowed for the selected HDR mode.
    k32u hdrParametersCountMin;
    k32u hdrParametersCountMax;

    GoElement64f dynamicSensitivity;
    GoElement32u dynamicThreshold;
    GoElement32u gammaType;
    GoElementBool sensitivityCompensationEnabled;

    GoElement32u encoding;
    GoSurfacePhaseFilter phaseFilter;

    GoElement32u contrastThreshold;

    // Currently only used for G3 intensity compatibility between newer G3 hardware
    // revision compared to older G3 hardware revision.
    GoElementBool g3CompatibilityMode;

} GoAdvancedClass;

kDeclareClassEx(Go, GoAdvanced, kObject)

GoFx(kStatus) GoAdvanced_Construct(GoAdvanced* layout, kObject sensor, kAlloc allocator);

GoFx(kStatus) GoAdvanced_Init(GoAdvanced layout, kType type, kObject sensor, kAlloc alloc);
GoFx(kStatus) GoAdvanced_VRelease(GoAdvanced layout);

GoFx(kStatus) GoAdvanced_Read(GoAdvanced layout, kXml xml, kXmlItem item);
GoFx(kStatus) GoAdvanced_ReadSpotSettings(GoAdvanced advanced, kXml xml, kXmlItem item);
GoFx(kStatus) GoAdvanced_ReadCameraSettings(GoAdvanced advanced, kXml xml, kXmlItem item);
GoFx(kStatus) GoAdvanced_ReadDynamicExposureSettings(GoAdvanced advanced, kXml xml, kXmlItem item);
GoFx(kStatus) GoAdvanced_ReadIntensityCompatibility(GoAdvanced layout, kXml xml, kXmlItem item);
GoFx(kStatus) GoAdvanced_ReadTunableParams(GoAdvanced advanced, kChar* paramStr, GoElement32u* element, kXml xml, kXmlItem item);

GoFx(kStatus) GoAdvanced_Write(GoAdvanced layout, kXml xml, kXmlItem item);
GoFx(kStatus) GoAdvanced_WriteSpotSettings(GoAdvanced advanced, kXml xml, kXmlItem item);
GoFx(kStatus) GoAdvanced_WriteCameraSettings(GoAdvanced advanced, kXml xml, kXmlItem item);
GoFx(kStatus) GoAdvanced_WriteDynamicExposureSettings(GoAdvanced advanced, kXml xml, kXmlItem item);

#endif
