/***

Copyright (c) 2024, Snarkify, Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

#pragma once
namespace host {

namespace ff {

template<class Field, bool montgomery>
class impl;

template<class Field, bool montgomery>
class storage;

} // namespace ff

} // namespace host

namespace ec {

template<class Curve, bool montgomery>
class impl_xy;

template<class Curve, bool montgomery>
class storage_xy;

template<class Curve, bool montgomery>
class impl_xyzz;

template<class Curve, bool montgomery>
class storage_xyzz;

} // namespace ec

