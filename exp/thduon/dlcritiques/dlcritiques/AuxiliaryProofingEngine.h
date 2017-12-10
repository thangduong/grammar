//--------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation.  All rights reserved.
//
//  File:       AuxiliaryProofingEngine.h
//
//  Contents:   Defines the API of proofing helper DLLs that augment
//              the grammar checker and other core proofing tools.
//
//--------------------------------------------------------------------------
#pragma once

#include <stdint.h>

struct AuxProofingCritique
{
    // The span within the input text that is being critiqued.
    // The length may be 0 in cases of pure insertions.
    uint32_t  iStartInputSpanToReplace;
    uint32_t cchInputSpanToReplace;

    // The single suggested replacement. May be the empty
    // string if pure deletion is being suggested.
    wchar_t  wzReplacementString[128];

    // The name of the critique type in a format that the host
    // proofing tool can understand. For the grammar checker,
    // this must be the internal critique name.
    wchar_t  wzCritiqueTypeName[128];
};


// This is the DLL export that analyzes an input to generate AuxProofingCritiques:
//
// uint32_t __stdcall GenerateCritiques(__in_z const wchar_t* wzInput,
//                                      __out_ecount(cCritiquesOut) AuxProofingCritique* rgCritiquesOut,
//                                      uint32_t cCritiquesOut) noexcept;
//
// Returns the number of critiques generated. Does not communicate internal errors -- returns 0,
// just as if no critiques were identified. If the buffer length is insufficient, then only the
// first cCritiquesOut critiques are returned and others are skipped.

using PFN_GenerateCritiques = uint32_t (__stdcall *)(__in_z const wchar_t* wzInput,
                                                     __out_ecount(cCritiquesOut) AuxProofingCritique* rgCritiquesOut,
                                                     uint32_t cCritiquesOut);


// This is the DLL export that ranks a provided list of alternative strings as potential replacements for
// the specified "blank" in the provided sentence.
//
// bool __stdcall GetFillInBlankProbs(__in_z const wchar_t* wzSentence,
//                                    int32_t iStartBlank,
//                                    uint32_t cchBlank,
//                                    __in_ecount(cChoices) const wchar_t** rgChoices,
//                                    uint32_t cChoices,
//                                    __out_ecount(cChoices + 1) float* rgProbsOut) noexcept;
//
// Returns a success code. Note that rgProbsOut[0] is the probability of the unchanged sentence. rgProbsOut[i]
// for i > 0 is the probability for choice i - 1.

using PFN_GetFillInBlankProbs = bool (__stdcall *)(__in_z const wchar_t* wzSentence,
                                                   uint32_t iStartBlank,
                                                   uint32_t cchBlank,
                                                   __in_ecount(cChoices) const wchar_t** rgChoices,
                                                   uint32_t cChoices,
                                                   __out_ecount(cChoices + 1) float* rgProbsOut);

