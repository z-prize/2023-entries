#pragma once

#include "ff_storage.cuh"

namespace bls12_381{
  //__device__ __constant__ uint32_t inv_p = 0xfffcfffd;
  //__device__ __constant__ uint32_t inv_q = 0xffffffff;

  struct ff_config_base
  {
    // field structure size = 12 * 32 bit
    static constexpr unsigned limbs_count = 12;
    // modulus = 4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787

    static constexpr ff_storage<limbs_count> modulus = {0xFFFFAAAB, 0xB9FEFFFF, 0xB153FFFF, 0x1EABFFFE, 0xF6B0F624, 0x6730D2A0,
                                                        0xF38512BF, 0x64774B84, 0x434BACD7, 0x4B1BA7B6, 0x397FE69A, 0x1A0111EA};
    // modulus*2 = 8004819110443334786835579651471808313113765639878015770664116272248063300981675728885375258258031328075788545119574
    static constexpr ff_storage<limbs_count> modulus_2 = {0xFFFF5556, 0x73FDFFFF, 0x62A7FFFF, 0x3D57FFFD, 0xED61EC48, 0xCE61A541,
                                                          0xE70A257E, 0xC8EE9709, 0x869759AE, 0x96374F6C, 0x72FFCD34, 0x340223D4};
    // modulus*4 = 16009638220886669573671159302943616626227531279756031541328232544496126601963351457770750516516062656151577090239148
    static constexpr ff_storage<limbs_count> modulus_4 = {0xFFFEAAAC, 0xE7FBFFFF, 0xC54FFFFE, 0x7AAFFFFA, 0xDAC3D890, 0x9CC34A83,
                                                          0xCE144AFD, 0x91DD2E13, 0x0D2EB35D, 0x2C6E9ED9, 0xE5FF9A69, 0x680447A8};
    // modulus^2 =
    // 02A437A4B8C35FC74BD278EAA22F25E9E2DC90E50E7046B466E59E49349E8BD050A62CFD16DDCA6EF53149330978EF011D68619C86185C7B292E85A87091A04966BF91ED3E71B743162C338362113CFD7CED6B1D76382EAB26AA00001C718E39
    static constexpr ff_storage_wide<limbs_count> modulus_squared = {0x1C718E39, 0x26AA0000, 0x76382EAB, 0x7CED6B1D, 0x62113CFD, 0x162C3383,
                                                                     0x3E71B743, 0x66BF91ED, 0x7091A049, 0x292E85A8, 0x86185C7B, 0x1D68619C,
                                                                     0x0978EF01, 0xF5314933, 0x16DDCA6E, 0x50A62CFD, 0x349E8BD0, 0x66E59E49,
                                                                     0x0E7046B4, 0xE2DC90E5, 0xA22F25E9, 0x4BD278EA, 0xB8C35FC7, 0x02A437A4};

    // 2*modulus^2 =
    // 05486F497186BF8E97A4F1D5445E4BD3C5B921CA1CE08D68CDCB3C92693D17A0A14C59FA2DBB94DDEA62926612F1DE023AD0C3390C30B8F6525D0B50E1234092CD7F23DA7CE36E862C586706C42279FAF9DAD63AEC705D564D54000038E31C72
    static constexpr ff_storage_wide<limbs_count>
        modulus_squared_2 = {0x38E31C72, 0x4D540000, 0xEC705D56, 0xF9DAD63A, 0xC42279FA, 0x2C586706, 0x7CE36E86, 0xCD7F23DA,
                             0xE1234092, 0x525D0B50, 0x0C30B8F6, 0x3AD0C339, 0x12F1DE02, 0xEA629266, 0x2DBB94DD, 0xA14C59FA,
                             0x693D17A0, 0xCDCB3C92, 0x1CE08D68, 0xC5B921CA, 0x445E4BD3, 0x97A4F1D5, 0x7186BF8E, 0x05486F49};
    // 4*modulus^2
    // =0A90DE92E30D7F1D2F49E3AA88BC97A78B72439439C11AD19B967924D27A2F414298B3F45B7729BBD4C524CC25E3BC0475A18672186171ECA4BA16A1C24681259AFE47B4F9C6DD0C58B0CE0D8844F3F5F3B5AC75D8E0BAAC9AA8000071C638E4
    static constexpr ff_storage_wide<limbs_count> modulus_squared_4 = {
        0x71C638E4, 0x9AA80000, 0xD8E0BAAC, 0xF3B5AC75, 0x8844F3F5, 0x58B0CE0D, 0xF9C6DD0C, 0x9AFE47B4, 0xC2468125, 0xA4BA16A1, 0x186171EC, 0x75A18672,
        0x25E3BC04, 0xD4C524CC, 0x5B7729BB, 0x4298B3F4, 0xD27A2F41, 0x9B967924, 0x39C11AD1, 0x8B724394, 0x88BC97A7, 0x2F49E3AA, 0xE30D7F1D, 0x0A90DE92};

    // r2 = 0x11988FE592CAE3AA9A793E85B519952D67EB88A9939D83C08DE5476C4C95B6D50A76E6A609D104F1F4DF1F341C341746  2^768%N(Fq)
    static constexpr ff_storage<limbs_count> r2 = {0x1C341746, 0xF4DF1F34, 0x09D104F1, 0x0A76E6A6, 0x4C95B6D5, 0x8DE5476C,
                                                   0x939D83C0, 0x67EB88A9, 0xB519952D, 0x9A793E85, 0x92CAE3AA, 0x11988FE5};
    // inv
    static constexpr uint32_t inv = 0xfffcfffd;
    // 1 in montgomery form  2^384%N
    // 008D6661E2FDF49A4CF495BF803C84E87B4E97B76E7C63059F7DB3A98A7D3FF251409F837FFFFFB102CDFFFFFFFFFF68
    // 0x15f65ec3 fa80e493 5c071a97 a256ec6d 77ce5853 70525745 5f489857 53c758ba ebf4000b c40c0002 76090000 0002fffd
    static constexpr ff_storage<limbs_count> one = {0x0002fffd, 0x76090000, 0xc40c0002, 0xebf4000b, 0x53c758ba, 0x5f489857,
                                                    0x70525745, 0x77ce5853, 0xa256ec6d, 0x5c071a97, 0xfa80e493, 0x15f65ec3};

    static constexpr unsigned modulus_bits_count = 381;
    static constexpr unsigned B_VALUE = 4; // 对应的是椭圆曲线中的b值 ？？
  };

  // Can't make this a member of ff_config_p. nvcc does not allow __constant__ on members.
  // extern __device__ __constant__ uint32_t inv_p;

  struct ff_config_scalar
  {
    // field structure size = 8 * 32 bit
    static constexpr unsigned limbs_count = 8;
    // modulus = 52435875175126190479447740508185965837690552500527637822603658699938581184513
    static constexpr ff_storage<limbs_count> modulus = {0x00000001, 0xFFFFFFFF, 0xFFFE5BFE, 0x53BDA402, 0x09A1D805, 0x3339D808, 0x299D7D48, 0x73EDA753};
    // modulus*2 = 0xE7DB4EA6 533AFA90 6673B010 1343B00A A77B4805 FFFCB7FD FFFFFFFE 00000002
    static constexpr ff_storage<limbs_count> modulus_2 = {0x00000002, 0xFFFFFFFE, 0xFFFCB7FD, 0xA77B4805, 0x1343B00A, 0x6673B010, 0x533AFA90, 0xE7DB4EA6};

    // modulus*4 = 0x 1 CFB69D4C A675F520 CCE76020 26876015 4EF6900B FFF96FFB FFFFFFFC 00000004 , 切分出来是9个字段，与原有的moduls_4不匹配； 猜测此值不会被用到
    static constexpr ff_storage<limbs_count> modulus_4 = {0x00000004, 0xFFFFFFFC, 0xFFF96FFB, 0x4EF6900B, 0x26876015, 0xCCE76020, 0xA675F520, 0xCFB69D4C};

    // modulus^2 = 347F60F3F4BC2778520C13DBEF2CC20F0E9593F959934A1DC2611F6F4AA9C66148AA415E5433F7B8698AE0021CDBB005A77E9007FFFCB7FEFFFFFFFE00000001
    static constexpr ff_storage_wide<limbs_count> modulus_squared = {0x00000001, 0xFFFFFFFE, 0xFFFCB7FE, 0xA77E9007, 0x1CDBB005, 0x698AE002,
                                                                     0x5433F7B8, 0x48AA415E, 0x4AA9C661, 0xC2611F6F, 0x59934A1D, 0x0E9593F9,
                                                                     0xEF2CC20F, 0x520C13DB, 0xF4BC2778, 0x347F60F3};
    // 2*modulus^2 = 68FEC1E7E9784EF0A41827B7DE59841E1D2B27F2B326943B84C23EDE95538CC2915482BCA867EF70D315C00439B7600B4EFD200FFFF96FFDFFFFFFFC00000002
    static constexpr ff_storage_wide<limbs_count> modulus_squared_2 = {0x00000002, 0xFFFFFFFC, 0xFFF96FFD, 0x4EFD200F, 0x39B7600B, 0xD315C004,
                                                                       0xA867EF70, 0x915482BC, 0x95538CC2, 0x84C23EDE, 0xB326943B, 0x1D2B27F2,
                                                                       0xDE59841E, 0xA41827B7, 0xE9784EF0, 0x68FEC1E7};

    // 4*modulus^2 = D1FD83CFD2F09DE148304F6FBCB3083C3A564FE5664D287709847DBD2AA7198522A9057950CFDEE1A62B8008736EC0169DFA401FFFF2DFFBFFFFFFF800000004
    static constexpr ff_storage_wide<limbs_count> modulus_squared_4 = {0x00000004, 0xFFFFFFF8, 0xFFF2DFFB, 0x9DFA401F, 0x736EC016, 0xA62B8008,
                                                                       0x50CFDEE1, 0x22A90579, 0x2AA71985, 0x09847DBD, 0x664D2877, 0x3A564FE5,
                                                                       0xBCB3083C, 0x48304F6F, 0xD2F09DE1, 0xD1FD83CF};

    // r2 = 0748D9D9 9F59FF11 05D31496 7254398F 2B6CEDCB 87925C23 C999E990 F3F29C6D  2^512%N(Fr)  存储r2需要64*4 bits，64*4的2倍就是r2的值
    static constexpr ff_storage<limbs_count> r2 = {0xF3F29C6D, 0xC999E990, 0x87925C23, 0x2B6CEDCB, 0x7254398F, 0x05D31496, 0x9F59FF11, 0x0748D9D9};

    // inv
    static constexpr uint32_t inv = 0xffffffff; // todo ??????

    // 1 in montgomery form  ==> 2^256 % Fr; 1824B159ACC5056F998C4FEFECBC4FF55884B7FA0003480200000001FFFFFFFE
    static constexpr ff_storage<limbs_count> one = {0xFFFFFFFE, 0x00000001, 0x00034802, 0x5884B7FA, 0xECBC4FF5, 0x998C4FEF, 0xACC5056F, 0x1824B159};        
    static constexpr ff_storage<limbs_count> zero = {0, 0, 0, 0, 0, 0, 0, 0};
    static constexpr ff_storage<limbs_count> m_one = { 3, 4294967293, 4294644732, 4214811656, 484804623, 2578286616, 2094561240, 1539896825 };
    static constexpr ff_storage<limbs_count> k1 = { 4294967281, 14, 1612815, 400778195, 1870944176, 4288436103, 2412095684, 890450464};
    static constexpr ff_storage<limbs_count> k2 = { 4294967268, 27, 3010588, 3611430828, 4065091434, 1705795358, 1925598234, 1375843047};
    static constexpr ff_storage<limbs_count> k3 = { 4294967259, 36, 3978277, 3851897745, 2610677562, 2560870102, 4231849104, 1051119866};
    
    static constexpr unsigned modulus_bits_count = 255;
    // log2 of order of omega
    static constexpr unsigned omega_log_order = 32; // todo 确定
    // k = (modulus - 1) / (2^omega_log_order)
    //   = (modulus - 1) / (2^28)
    //   = 31458071430878231019708607117762962473094088342614709689971929251840
    //  = 31458071430878228999849010692650307322271690337212454074694020104192
    // omega generator is 22
    static constexpr unsigned omega_generator = 7; // todo 查询曲线参数
    // omega = generator^k % modulus
    //       = 22^31458071430878231019708607117762962473094088342614709689971929251840 % modulus
    //       = 121723987773120995826174558753991820398372762714909599931916967971437184138

    // omega in montgomery form  ==> omega * 2^256 %N(Fr)

    // omega 381  0xb9b58d8c5f0e466a5b1b4c801819d7ec0af53ae352a31e645bf3adda19e9b27b
    // omega in montgomery form  =>  (omega << 256) %Fr = 72EADF779178BACF5AFBE64B9D50A6D5D51D61A402D4BD47E6237A3B6D0DA3D8
    //                                                   0x16a2a19edfe81f20d09b681922c813b4b63683508c2280b93829971f439f0d2b
    static constexpr ff_storage<limbs_count> omega = {0x6D0DA3D8, 0xE6237A3B, 0x02D4BD47, 0xD51D61A4, 0x9D50A6D5, 0x5AFBE64B, 0x9178BACF, 0x72EADF77};

    // inverse of 2 in montgomery form  , 先求2逆， 假设等于X， X*2^256%N(Fr)
    //  inverse of 2  = 26217937587563095239723870254092982918845276250263818911301829349969290592257
    // inverse of 2 in montgomery form = ( 先求2逆， 假设等于X， X*2^256%N(Fr)) = 0xc1258acd66282b7ccc627f7f65e27faac425bfd0001a40100000000ffffffff
    static constexpr ff_storage<limbs_count> two_inv = {0xFFFFFFFF, 0x00000000, 0x0001A401, 0xAC425BFD, 0xF65E27FA, 0xCCC627F7, 0xD66282B7, 0x0C1258AC};
  };

  // Can't make this a member of ff_config_q. nvcc does not allow __constant__ on members.
  // extern __device__ __constant__ uint32_t inv_q;

} // namespace bls12_381

#define ff_config_p bls12_381::ff_config_base
#define ff_config_q bls12_381::ff_config_scalar
#define inv_p 0xfffcfffd
#define inv_q 0xffffffff
