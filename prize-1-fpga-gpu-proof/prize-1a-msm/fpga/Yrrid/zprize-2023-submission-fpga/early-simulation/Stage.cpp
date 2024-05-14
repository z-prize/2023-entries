/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

class Simulator;

class Stage {
  public:
  virtual bool run(Simulator* simulator, uint32_t cycle) = 0;
};

