/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

namespace host {

template<class Field>
class ntt_context {
  public:
  impl<Field,true>* root_table;

  ntt_context() {
    root_table=NULL;
  }

  ntt_context(uint32_t max_size_in_bits) {
    root_table=NULL;
    initialize_root_table(max_size_in_bits);
  }

  ~ntt_context() {
    if(root_table!=NULL)
      free(root_table);
  }

  void initialize_root_table(uint32_t max_size_in_bits) {
    uint32_t         tables=(max_size_in_bits+9)/10;
    uint32_t         current_size=10, next=0;
    impl<Field,true> root, current;

    if(root_table!=NULL)
      free(root_table);

    root_table=(impl<Field,true>*)malloc(tables*1024*sizeof(current));
    for(uint32_t table=0;table<tables;table++) {
      current=impl<Field,true>::one();
      root=impl<Field,true>::generator(current_size);
      for(uint32_t idx=0;idx<1024;idx++) {
        root_table[next++]=current;
        current=current*root;
      }
      current_size+=10;
    }
  }

  impl<Field,true> root(const uint64_t idx, const uint32_t size_in_bits) {
    impl<Field,true> res;
    uint64_t         local_idx;

    if(size_in_bits<=10) {
      local_idx=(idx<<(10-size_in_bits)) & 0x3FFull;
      return root_table[local_idx];
    }
    else if(size_in_bits<=20) {
      local_idx=(idx<<(20-size_in_bits)) & 0xFFFFFull;
      res=root_table[local_idx>>10];
      return res*root_table[(local_idx & 0x3FF) + 1024];
    }
    else if(size_in_bits<=30) {
      local_idx=(idx<<(30-size_in_bits)) & 0x3FFFFFFFull;
      res=root_table[local_idx>>20];
      res=res*root_table[((local_idx>>10) & 0x3FF) + 1024];
      return res*root_table[(local_idx & 0x3FF) + 2048];
    }
    else if(size_in_bits<=40) {
      local_idx=(idx<<(40-size_in_bits)) & 0xFFFFFFFFFFull;
      res=root_table[local_idx>>30];
      res=res*root_table[((local_idx>>20) & 0x3FF) + 1024];
      res=res*root_table[((local_idx>>10) & 0x3FF) + 2048];
      return res*root_table[(local_idx & 0x3FF) + 3072];
    }
    return root_table[0];
  }

  uint32_t reverse_bits(const uint32_t in) {
    uint32_t out;

    out=(in>>16) | (in<<16);
    out=((out & 0x00FF00FF)<<8) | ((out & 0xFF00FF00)>>8);
    out=((out & 0x0F0F0F0F)<<4) | ((out & 0xF0F0F0F0)>>4);
    out=((out & 0x33333333)<<2) | ((out & 0xCCCCCCCC)>>2);
    out=((out & 0x55555555)<<1) | ((out & 0xAAAAAAAA)>>1);
    return out;
  }

  impl<Field,true> inverse_size(const uint32_t size_in_bits) {
    impl<Field,true>     inv;
    storage<Field,false> p_minus_one_div_size;

    // Note, (const_n-1) is evenly divisible by 2^size_in_bits, therefore:
    //
    //    const_n-1
    //    --------- * 2^s = -1
    //    2^s
    //
    // Thus,
    //
    //    const_n-1
    //    --------- = -1/2^s
    //    2^s
    //
    // Therefore,
    //
    //        const_n-1
    //  -1 *  --------- = inv(2^s)
    //        2^s

    mp_packed<Field>::shift_right_bits(p_minus_one_div_size.l, Field::const_storage_n.l, size_in_bits);
    inv=(impl<Field,true>)p_minus_one_div_size;
    return -inv;
  }

  void ntt_128(storage<Field,true>* out, const storage<Field,true>* in) {
    impl<Field,true> values[128], a, b;
    int32_t          current_size, table_factor, advance;
    int32_t          bit_reverse[8]={0, 4, 2, 6, 1, 5, 3, 7}, reversed;

    for(int i=0;i<128;i++)
      host::mp<Field>::unpack(values[i].l, in[i].l);

    current_size=128;
    table_factor=1;
    for(int pass=0;pass<7;pass++) {
      advance=current_size;
      current_size=current_size>>1;
      for(int base=0;base<128;base+=advance) {
        for(int i=0;i<current_size;i++) {
          a=values[base+i];
          b=values[base+current_size+i];
          values[base+i]=a+b;
          values[base+current_size+i]=(a-b)*root(i*table_factor & 0x7F, 7);
        }
      }
      table_factor=table_factor<<1;
    }

    for(int group=0;group<128;group+=16) {
      reversed=bit_reverse[group>>4];
      host::mp<Field>::pack(out[group + 0x00].l, values[0x00 + reversed].l);
      host::mp<Field>::pack(out[group + 0x01].l, values[0x40 + reversed].l);
      host::mp<Field>::pack(out[group + 0x02].l, values[0x20 + reversed].l);
      host::mp<Field>::pack(out[group + 0x03].l, values[0x60 + reversed].l);
      host::mp<Field>::pack(out[group + 0x04].l, values[0x10 + reversed].l);
      host::mp<Field>::pack(out[group + 0x05].l, values[0x50 + reversed].l);
      host::mp<Field>::pack(out[group + 0x06].l, values[0x30 + reversed].l);
      host::mp<Field>::pack(out[group + 0x07].l, values[0x70 + reversed].l);

      host::mp<Field>::pack(out[group + 0x08].l, values[0x08 + reversed].l);
      host::mp<Field>::pack(out[group + 0x09].l, values[0x48 + reversed].l);
      host::mp<Field>::pack(out[group + 0x0A].l, values[0x28 + reversed].l);
      host::mp<Field>::pack(out[group + 0x0B].l, values[0x68 + reversed].l);
      host::mp<Field>::pack(out[group + 0x0C].l, values[0x18 + reversed].l);
      host::mp<Field>::pack(out[group + 0x0D].l, values[0x58 + reversed].l);
      host::mp<Field>::pack(out[group + 0x0E].l, values[0x38 + reversed].l);
      host::mp<Field>::pack(out[group + 0x0F].l, values[0x78 + reversed].l);
    }
  }

  void ntt(storage<Field,true>* out, const storage<Field,true>* in, const uint32_t size_in_bits, const bool prints=true) {
    impl<Field,true>*   values;
    impl<Field,true>    a, b;
    storage<Field,true> temp;
    uint32_t            current_size, table_factor, advance, size=1<<size_in_bits, mask=size-1;

    values=(impl<Field,true>*)malloc(((size_t)size)*sizeof(a));

    for(uint32_t i=0;i<size;i++)
      host::mp<Field>::unpack(values[i].l, in[i].l);

    current_size=size;
    table_factor=1;
    for(uint32_t pass=0;pass<size_in_bits;pass++) {
      if(prints) {
        fprintf(stderr, "%d ", pass);
        fflush(stderr);
      }
      advance=current_size;
      current_size=current_size>>1;
      for(uint32_t base=0;base<size;base+=advance) {
        for(uint32_t i=0;i<current_size;i++) {
          a=values[base+i];
          b=values[base+current_size+i];
          values[base+i]=a+b;
          values[base+current_size+i]=(a-b)*root(i*table_factor & mask, size_in_bits);
        }
      }
      table_factor=table_factor<<1;
    }
    if(prints)
      fprintf(stderr, "\n");

    for(uint32_t i=0;i<size;i++) {
      host::mp<Field>::pack(temp.l, values[i].l);
      if(mp_packed<Field>::compare(temp.l, Field::const_storage_n.l)>=0)
        mp_packed<Field>::sub(temp.l, temp.l, Field::const_storage_n.l);
      out[reverse_bits(i)>>(32-size_in_bits)]=temp;
    }

    free(values);
  }

  void intt(storage<Field,true>* out, const storage<Field,true>* in, const uint32_t size_in_bits, const bool prints=true) {
    impl<Field,true>*   values;
    impl<Field,true>    a, b, inv;
    storage<Field,true> temp;
    uint32_t            current_size, table_factor, advance, size=1<<size_in_bits, mask=size-1;

    values=(impl<Field,true>*)malloc(((size_t)size)*sizeof(a));

    for(uint32_t i=0;i<size;i++)
      host::mp<Field>::unpack(values[i].l, in[i].l);

    current_size=size;
    table_factor=1;
    for(uint32_t pass=0;pass<size_in_bits;pass++) {
      if(prints) {
        fprintf(stderr, "%d ", pass);
        fflush(stderr);
      }
      advance=current_size;
      current_size=current_size>>1;
      for(uint32_t base=0;base<size;base+=advance) {
        for(uint32_t i=0;i<current_size;i++) {
          a=values[base+i];
          b=values[base+current_size+i];
          values[base+i]=a+b;
          values[base+current_size+i]=(a-b)*root(-i*table_factor & mask, size_in_bits);
        }
      }
      table_factor=table_factor<<1;
    }
    if(prints)
      fprintf(stderr, "\n");

    inv=inverse_size(size_in_bits);
    for(uint32_t i=0;i<size;i++) {
      values[i]=values[i] * inv;
      host::mp<Field>::pack(temp.l, values[i].l);
      if(mp_packed<Field>::compare(temp.l, Field::const_storage_n.l)>=0)
        mp_packed<Field>::sub(temp.l, temp.l, Field::const_storage_n.l);
      out[reverse_bits(i)>>(32-size_in_bits)]=temp;
    }

    free(values);
  }

  void intt_no_scale(storage<Field,true>* out, const storage<Field,true>* in, const uint32_t size_in_bits, const bool prints=true) {
    impl<Field,true>*   values;
    impl<Field,true>    a, b, inv;
    storage<Field,true> temp;
    uint32_t            current_size, table_factor, advance, size=1<<size_in_bits, mask=size-1;

    values=(impl<Field,true>*)malloc(((size_t)size)*sizeof(a));

    for(uint32_t i=0;i<size;i++)
      host::mp<Field>::unpack(values[i].l, in[i].l);

    current_size=size;
    table_factor=1;
    for(uint32_t pass=0;pass<size_in_bits;pass++) {
      if(prints) {
        fprintf(stderr, "%d ", pass);
        fflush(stderr);
      }
      advance=current_size;
      current_size=current_size>>1;
      for(uint32_t base=0;base<size;base+=advance) {
        for(uint32_t i=0;i<current_size;i++) {
          a=values[base+i];
          b=values[base+current_size+i];
          values[base+i]=a+b;
          values[base+current_size+i]=(a-b)*root(-i*table_factor & mask, size_in_bits);
        }
      }
      table_factor=table_factor<<1;
    }
    if(prints)
      fprintf(stderr, "\n");

    for(uint32_t i=0;i<size;i++) {
      host::mp<Field>::pack(temp.l, values[i].l);
      if(mp_packed<Field>::compare(temp.l, Field::const_storage_n.l)>=0)
        mp_packed<Field>::sub(temp.l, temp.l, Field::const_storage_n.l);
      out[reverse_bits(i)>>(32-size_in_bits)]=temp;
    }

    free(values);
  }

  void ntt(storage<Field,false>* out, const storage<Field,false>* in, const uint32_t size_in_bits, const bool prints=true) {
    ntt((storage<Field,true>*)out, (storage<Field,true>*)in, size_in_bits, prints);
  }

  void intt(storage<Field,false>* out, const storage<Field,false>* in, const uint32_t size_in_bits, const bool prints=true) {
    intt((storage<Field,true>*)out, (storage<Field,true>*)in, size_in_bits, prints);
  }

  void intt_no_scale(storage<Field,false>* out, const storage<Field,false>* in, const uint32_t size_in_bits, const bool prints=true) {
    intt_no_scale((storage<Field,true>*)out, (storage<Field,true>*)in, size_in_bits, prints);
  }

};

}; // namespace host
