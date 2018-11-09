/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

folder: BC_Tensors


Tensor.h:
    
    Created type-aliases for Tensor_Base. These classes are the 'subclasses' of Tensor_Base.
    These are the 'user' classes.

Tesor_Base.h:

    The 'Base' class of the Tensors. 
    Tensor_Base inherits from Operations, Functions, Shaping, Utility and utilizes CRTP 
    to enable the superclasses to have access to all the methods of Tensor_Base.

    Functions, Operations, Shaping, Utility are wrapped in the 'module' namespace. 
    In practice, the entirety of each of these files could be placed within Tensor_Base.h

    They are seperated in their own files for purpose of clarity. 

Tensor_Common.h:

    defines:
    1) __BC_host_inline__ a forced inline macro for Host (not CUDA)
    2) template<int> class DISABLED; //used as an alternate parameter with std::conditional.
    DISABLED does not have a struct body, and should not. (It should never be used).
    3) template<class internal_t> auto make_tensor(internal_t) --a factory method of Tensor_Base

Tensor_Functions.h:

    Defines all methods that effect the internal_state of the Tensors.
    These methods are instant evaluated, and are not involved in lazy evaluation. 

Tensor_Operations.h:

    Includes all non-expression (non owning) classes in the Expression-Templates folder.
    Tensor_Operations creates the actual expression_templates and returns Tensor_Base objects. 
    All non utility methods in Tensor_Operations are lazy evaluated. 

Tensor_Shaping.h:

    Defines all access based methods, many of these methods are forwarded from the internal_type. 
    Internal_types may return internal values. 
    These methods have identical names except utilize an underscore first to represent 'internal' methosd. 

Tensor_Utility:

    Defines all IO based methods. These methosd are used to output to console and read/write from file .
