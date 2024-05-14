/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

class HexReader {
   public:
   FILE*    dataFile;
   uint8_t  buffer[4096];
   char     copyBuffer[5*1024];
   uint32_t position;
   uint32_t last;
   bool     eof;

   HexReader(const char* path) {
     dataFile=fopen(path, "r");
     if(dataFile==NULL) {
       fprintf(stderr, "Unable to open '%s' for reading\n", path);
       exit(1);
     }
     position=0;
     last=0;
     eof=false;
   }

   ~HexReader() {
      if(dataFile!=NULL)
        fclose(dataFile);
   }

   void loadBuffer() {
     uint32_t amount;

     if(position>0 && position<last) {
       memcpy(buffer, buffer+position, last-position);
       last=last-position;
       position=0;
     }
     if(last==4096) {
       fprintf(stderr, "Line too long");
       exit(1);
     }
     amount=fread(buffer+last, 1, 4096-last, dataFile);
     if(amount==0)
       eof=true;
     last+=amount;
   }

   bool skipToHex() {
     while(true) {
       while(position<last) {
         if((buffer[position]>='0' && buffer[position]<='9') || (buffer[position]>='a' && buffer[position]<='f') ||
            (buffer[position]>='A' && buffer[position]<='F'))
           return true;
         position++;
       }
       if(eof)
         return false;
       loadBuffer();
     }
   }

   uint32_t hexCount(const char** hexBytes) {
     uint32_t from, to;

     *hexBytes=copyBuffer;
     while(true) {
       from=position;
       to=0;
       while(from<last) {
         if((buffer[from]>='0' && buffer[from]<='9') || (buffer[from]>='a' && buffer[from]<='f') || (buffer[from]>='A' && buffer[from]<='F'))
           copyBuffer[to++]=buffer[from++];
         else {
           copyBuffer[to++]=0;
           return from-position;
         }
       }
       if(eof) {
         copyBuffer[to++]=0;
         return from-position;
       }
       loadBuffer();
     }
   }

   void advance(uint32_t amount) {
     position+=amount;
   }
};


/*

int main() {
  ReadHex     reader("scalars.hex");
  const char* hexBytes;
  uint32_t    hexCount;

  while(true) {
    if(!reader.skipToHex())
      break;
    hexCount=reader.hexCount(&hexBytes);
    printf("%s\n", hexBytes);
    reader.advance(hexCount);
  }
}
*/