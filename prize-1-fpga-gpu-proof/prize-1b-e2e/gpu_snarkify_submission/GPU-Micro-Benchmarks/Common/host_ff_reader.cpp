/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

#include <stdio.h>
#include <stdint.h>

namespace host {

namespace ff {

class reader {
  public:
  static const uint32_t MAX_LINE=4096;

  FILE*    file;
  uint8_t  buffer[MAX_LINE*2];
  bool     eof;
  uint32_t position;
  uint32_t available;

  reader(const char* path) {
    file=fopen(path, "r");
    if(file==NULL) {
      fprintf(stderr, "Failed to open '%s' for reading\n", path);
      eof=true;
      position=0;
      available=0;
    }
    else {
      eof=false;
      position=0;
      available=0;
      load();
    }
  }

  ~reader() {
    if(file!=NULL)
      fclose(file);
  }

  void load() {
    int32_t amount=4096;
    uint32_t start, maxRead;

    if(!eof && available<MAX_LINE) {
      start=position+available;
      start=(start>=MAX_LINE*2) ? start-MAX_LINE*2 : start;
      if(start<position)
        maxRead=position-start;
      else
        maxRead=MAX_LINE*2-start;
      maxRead=(maxRead>=4096) ? 4096 : maxRead;
      amount=fread(buffer+start, 1, maxRead, file);
      if(amount<=0)
        eof=true;
      available+=amount;
    }
  }

  char peek() {
    return (char)buffer[position];
  }

  char peek(uint32_t offset) {
    uint32_t where=position+offset;

    return buffer[where>=MAX_LINE*2 ? where-MAX_LINE*2 : where];
  }

  void advance() {
    available--;
    position=(position==MAX_LINE*2-1) ? 0 : position+1;
    if(!eof && available==0)
      load();
  }

  void advance(uint32_t amount) {
    available=available-amount;
    position=position+amount;
    position=(position<MAX_LINE*2) ? position : position-MAX_LINE*2;
  }

  char consume() {
    char next=(char)buffer[position];

    available--;
    position=(position==MAX_LINE*2-1) ? 0 : position+1;
    if(!eof && available==0)
      load();
    return next;
  }

  void skipDelimiters() {
    // always assume available>0
    while(available>0) {
      if(peek()<=' ')
        advance();
      else
        return;
    }
  }

  template<class Field, bool montgomery>
  bool readHex(storage<Field, montgomery>& s) {
    uint32_t start, length, offset;
    char     current;
    uint64_t add=0;

    for(uint32_t idx=0;idx<Field::packed_limbs;idx++)
      s.l[idx]=0L;

    skipDelimiters();
    if(eof && available==0)
      return false;
    load();
    if(peek(0)=='0' && peek(1)=='x')
      advance(2);
    offset=0;
    while(peek(offset)=='0')
      offset++;
    advance(offset);
    start=0;
    offset=0;
    while(offset-start<MAX_LINE-1) {
      current=peek(offset);
      if((current>='0' && current<='9') || (current>='a' && current<='f') || (current>='A' && current<='F'))
        offset++;
      else
        break;
    }
    length=offset-start;
    length=(length>=Field::packed_limbs*16) ? Field::packed_limbs*16 : length;
    for(uint32_t idx=0;idx<length;idx++) {
      current=peek(offset-idx-1);
      if(current>='0' && current<='9')
        add=current - '0';
      else if(current>='a' && current<='f')
        add=current - 'a' + 10;
      else if(current>='A' && current<='F')
        add=current - 'A' + 10;
      s.l[idx>>4] = s.l[idx>>4] | (add<<(idx & 0x0F)*4);
    }
    advance(offset-start);
    return true;
  }
};

}; // namespace ff

}; // namespace host
