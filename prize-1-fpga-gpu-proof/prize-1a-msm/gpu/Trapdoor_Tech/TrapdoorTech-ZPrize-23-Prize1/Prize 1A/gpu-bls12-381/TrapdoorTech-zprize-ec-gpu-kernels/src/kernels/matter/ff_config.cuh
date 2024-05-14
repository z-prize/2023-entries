namespace bls12_curve
{

#if defined(BLS12_381)
  __device__ __constant__ uint32_t inv_p = 0xfffcfffd;
  __device__ __constant__ uint32_t inv_q = 0xffffffff;

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
    static constexpr unsigned B_VALUE = 4;
  };

  // Can't make this a member of ff_config_p. nvcc does not allow __constant__ on members.

  struct ff_config_scalar
  {
    // field structure size = 8 * 32 bit
    static constexpr unsigned limbs_count = 8;
    // modulus = 52435875175126190479447740508185965837690552500527637822603658699938581184513
    static constexpr ff_storage<limbs_count> modulus = {0x00000001, 0xFFFFFFFF, 0xFFFE5BFE, 0x53BDA402, 0x09A1D805, 0x3339D808, 0x299D7D48, 0x73EDA753};
    // modulus*2 = 0xE7DB4EA6 533AFA90 6673B010 1343B00A A77B4805 FFFCB7FD FFFFFFFE 00000002
    static constexpr ff_storage<limbs_count> modulus_2 = {0x00000002, 0xFFFFFFFE, 0xFFFCB7FD, 0xA77B4805, 0x1343B00A, 0x6673B010, 0x533AFA90, 0xE7DB4EA6};

    // modulus*4 = 0x 1 CFB69D4C A675F520 CCE76020 26876015 4EF6900B FFF96FFB FFFFFFFC 00000004 
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

    // r2 = 0748D9D9 9F59FF11 05D31496 7254398F 2B6CEDCB 87925C23 C999E990 F3F29C6D  2^512%N(Fr)
    static constexpr ff_storage<limbs_count> r2 = {0xF3F29C6D, 0xC999E990, 0x87925C23, 0x2B6CEDCB, 0x7254398F, 0x05D31496, 0x9F59FF11, 0x0748D9D9};

    // inv
    static constexpr uint32_t inv = 0xffffffff;

    // 1 in montgomery form  ==> 2^256 % Fr; 1824B159ACC5056F998C4FEFECBC4FF55884B7FA0003480200000001FFFFFFFE
    static constexpr ff_storage<limbs_count> one = {0xFFFFFFFE, 0x00000001, 0x00034802, 0x5884B7FA, 0xECBC4FF5, 0x998C4FEF, 0xACC5056F, 0x1824B159};
    static constexpr unsigned modulus_bits_count = 255;
    // log2 of order of omega
    static constexpr unsigned omega_log_order = 32;
    // k = (modulus - 1) / (2^omega_log_order)
    //   = (modulus - 1) / (2^28)
    //   = 31458071430878231019708607117762962473094088342614709689971929251840
    //  = 31458071430878228999849010692650307322271690337212454074694020104192
    // omega generator is 22
    static constexpr unsigned omega_generator = 7;
    // omega = generator^k % modulus
    //       = 22^31458071430878231019708607117762962473094088342614709689971929251840 % modulus
    //       = 121723987773120995826174558753991820398372762714909599931916967971437184138

    // omega in montgomery form  ==> omega * 2^256 %N(Fr)

    // omega 381  0xb9b58d8c5f0e466a5b1b4c801819d7ec0af53ae352a31e645bf3adda19e9b27b
    // omega in montgomery form  =>  (omega << 256) %Fr = 72EADF779178BACF5AFBE64B9D50A6D5D51D61A402D4BD47E6237A3B6D0DA3D8
    //                                                   0x16a2a19edfe81f20d09b681922c813b4b63683508c2280b93829971f439f0d2b
    static constexpr ff_storage<limbs_count> omega = {0x6D0DA3D8, 0xE6237A3B, 0x02D4BD47, 0xD51D61A4, 0x9D50A6D5, 0x5AFBE64B, 0x9178BACF, 0x72EADF77};

    // inverse of 2 in montgomery form  
    //  inverse of 2  = 26217937587563095239723870254092982918845276250263818911301829349969290592257
    // inverse of 2 in montgomery form = 0xc1258acd66282b7ccc627f7f65e27faac425bfd0001a40100000000ffffffff
    static constexpr ff_storage<limbs_count> two_inv = {0xFFFFFFFF, 0x00000000, 0x0001A401, 0xAC425BFD, 0xF65E27FA, 0xCCC627F7, 0xD66282B7, 0x0C1258AC};
  };

// Can't make this a member of ff_config_q. nvcc does not allow __constant__ on members.
#elif defined(BLS12_377)
  __device__ __constant__ uint32_t inv_p = 0xffffffff;
  __device__ __constant__ uint32_t inv_q = 0xffffffff;
  // bls12_377
  struct ff_config_base
  {
    // field structure size = 12 * 32 bit
    static constexpr unsigned limbs_count = 12;
    // modulus = 258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177
    static constexpr ff_storage<limbs_count> modulus = {0x00000001, 0x8508c000, 0x30000000, 0x170b5d44, 0xba094800, 0x1ef3622f,
                                                        0x00f5138f, 0x1a22d9f3, 0x6ca1493b, 0xc63b05c0, 0x17c510ea, 0x01ae3a46};
    // modulus*2 = 517328852025938188021305467389787067072787025509829321079768525333440936696681645549937776279146720248880642916354
    static constexpr ff_storage<limbs_count> modulus_2 = {0x00000002, 0x0a118000, 0x60000001, 0x2e16ba88, 0x74129000, 0x3de6c45f,
                                                          0x01ea271e, 0x3445b3e6, 0xd9429276, 0x8c760b80, 0x2f8a21d5, 0x035c748c};
    // modulus*4 = 1034657704051876376042610934779574134145574051019658642159537050666881873393363291099875552558293440497761285832708
    static constexpr ff_storage<limbs_count> modulus_4 = {0x00000004, 0x14230000, 0xc0000002, 0x5c2d7510, 0xe8252000, 0x7bcd88be,
                                                          0x03d44e3c, 0x688b67cc, 0xb28524ec, 0x18ec1701, 0x5f1443ab, 0x06b8e918};
    // modulus^2
    static constexpr ff_storage_wide<limbs_count> modulus_squared = {
        0x00000001, 0x0a118000, 0xf0000001, 0x7338d254, 0x2e1bd800, 0x4ada268f, 0x35f1c09a, 0x6bcbfbd2, 0x58638c9d, 0x318324b9, 0x8bb70ae0, 0x460aaaaa,
        0x502a4d6c, 0xc014e712, 0xb90660cd, 0x09d018af, 0x3dda4d5c, 0x1f5e7141, 0xa4aee93f, 0x4bb8b87d, 0xb361263c, 0x2256913b, 0xd0bbaffb, 0x0002d307};
    // 2*modulus^2
    static constexpr ff_storage_wide<limbs_count> modulus_squared_2 = {
        0x00000002, 0x14230000, 0xe0000002, 0xe671a4a9, 0x5c37b000, 0x95b44d1e, 0x6be38134, 0xd797f7a4, 0xb0c7193a, 0x63064972, 0x176e15c0, 0x8c155555,
        0xa0549ad8, 0x8029ce24, 0x720cc19b, 0x13a0315f, 0x7bb49ab8, 0x3ebce282, 0x495dd27e, 0x977170fb, 0x66c24c78, 0x44ad2277, 0xa1775ff6, 0x0005a60f};
    // 4*modulus^2
    static constexpr ff_storage_wide<limbs_count> modulus_squared_4 = {
        0x00000004, 0x28460000, 0xc0000004, 0xcce34953, 0xb86f6001, 0x2b689a3c, 0xd7c70269, 0xaf2fef48, 0x618e3275, 0xc60c92e5, 0x2edc2b80, 0x182aaaaa,
        0x40a935b1, 0x00539c49, 0xe4198337, 0x274062be, 0xf7693570, 0x7d79c504, 0x92bba4fc, 0x2ee2e1f6, 0xcd8498f1, 0x895a44ee, 0x42eebfec, 0x000b4c1f};
    // r2 = 66127428376872697816332570116866232405230528984664918319606315420233909940404532140033099444330447428417853902114
    static constexpr ff_storage<limbs_count> r2 = {0x9400cd22, 0xb786686c, 0xb00431b1, 0x0329fcaa, 0x62d6b46d, 0x22a5f111,
                                                   0x827dc3ac, 0xbfdf7d03, 0x41790bf9, 0x837e92f0, 0x1e914b88, 0x006dfccb};
    // inv
    static constexpr uint32_t inv = 0xffffffff;
    // 1 in montgomery form
    static constexpr ff_storage<limbs_count> one = {0xffffff68, 0x02cdffff, 0x7fffffb1, 0x51409f83, 0x8a7d3ff2, 0x9f7db3a9,
                                                    0x6e7c6305, 0x7b4e97b7, 0x803c84e8, 0x4cf495bf, 0xe2fdf49a, 0x008d6661};
    static constexpr unsigned modulus_bits_count = 377;
    static constexpr unsigned B_VALUE = 1;
  };

  // Can't make this a member of ff_config_p. nvcc does not allow __constant__ on members.

  struct ff_config_scalar
  {
    // field structure size = 8 * 32 bit
    static constexpr unsigned limbs_count = 8;
    // modulus = 8444461749428370424248824938781546531375899335154063827935233455917409239041
    static constexpr ff_storage<limbs_count> modulus = {0x00000001, 0x0a118000, 0xd0000001, 0x59aa76fe, 0x5c37b001, 0x60b44d1e, 0x9a2ca556, 0x12ab655e};
    // modulus*2 = 16888923498856740848497649877563093062751798670308127655870466911834818478082
    static constexpr ff_storage<limbs_count> modulus_2 = {0x00000002, 0x14230000, 0xa0000002, 0xb354edfd, 0xb86f6002, 0xc1689a3c, 0x34594aac, 0x2556cabd};
    // modulus*4 = 33777846997713481696995299755126186125503597340616255311740933823669636956164
    static constexpr ff_storage<limbs_count> modulus_4 = {0x00000004, 0x28460000, 0x40000004, 0x66a9dbfb, 0x70dec005, 0x82d13479, 0x68b29559, 0x4aad957a};
    // modulus^2
    static constexpr ff_storage_wide<limbs_count> modulus_squared = {0x00000001, 0x14230000, 0xe0000002, 0xc7dd4d2f, 0x8585d003, 0x08ee1bd4,
                                                                     0xe57fc56e, 0x7e7557e3, 0x483a709d, 0x1fdebb41, 0x5678f4e6, 0x8ea77334,
                                                                     0xc19c3ec5, 0xd717de29, 0xe2340781, 0x015c8d01};
    // 2*modulus^2
    static constexpr ff_storage_wide<limbs_count> modulus_squared_2 = {0x00000002, 0x28460000, 0xc0000004, 0x8fba9a5f, 0x0b0ba007, 0x11dc37a9,
                                                                       0xcaff8adc, 0xfceaafc7, 0x9074e13a, 0x3fbd7682, 0xacf1e9cc, 0x1d4ee668,
                                                                       0x83387d8b, 0xae2fbc53, 0xc4680f03, 0x02b91a03};
    // 4*modulus^2
    static constexpr ff_storage_wide<limbs_count> modulus_squared_4 = {0x00000004, 0x508c0000, 0x80000008, 0x1f7534bf, 0x1617400f, 0x23b86f52,
                                                                       0x95ff15b8, 0xf9d55f8f, 0x20e9c275, 0x7f7aed05, 0x59e3d398, 0x3a9dccd1,
                                                                       0x0670fb16, 0x5c5f78a7, 0x88d01e07, 0x05723407};
    // r2 = 508595941311779472113692600146818027278633330499214071737745792929336755579
    static constexpr ff_storage<limbs_count> r2 = {0xb861857b, 0x25d577ba, 0x8860591f, 0xcc2c27b5, 0xe5dc8593, 0xa7cc008f, 0xeff1c939, 0x011fdae7};
    // inv
    static constexpr uint32_t inv = 0xffffffff;
    // 1 in montgomery form
    static constexpr ff_storage<limbs_count> one = {0xfffffff3, 0x7d1c7fff, 0x6ffffff2, 0x7257f50f, 0x512c0fee, 0x16d81575, 0x2bbb9a9d, 0x0d4bda32};
    static constexpr unsigned modulus_bits_count = 253;
    // log2 of order of omega
    static constexpr unsigned omega_log_order = 28;
    // k = (modulus - 1) / (2^omega_log_order)
    //   = (modulus - 1) / (2^28)
    //   = 31458071430878231019708607117762962473094088342614709689971929251840
    // omega generator is 22
    static constexpr unsigned omega_generator = 22;
    // omega = generator^k % modulus
    //       = 22^31458071430878231019708607117762962473094088342614709689971929251840 % modulus
    //       = 121723987773120995826174558753991820398372762714909599931916967971437184138
    // omega in montgomery form
    static constexpr ff_storage<limbs_count> omega = {0x06a79893, 0x18edfc4b, 0xd9942118, 0xffe5168f, 0xb4ab2c5c, 0x76b85d81, 0x47c2ecc0, 0x0a439040};
    // inverse of 2 in montgomery form
    static constexpr ff_storage<limbs_count> two_inv = {0xfffffffa, 0xc396ffff, 0x1ffffff9, 0xe6013607, 0xd6b1dff7, 0xbbc63149, 0x62f41ff9, 0x0ffb9fc8};
  };

  // Can't make this a member of ff_config_q. nvcc does not allow __constant__ on members.

#endif

} // namespace bls12_381

#define ff_config_p bls12_curve::ff_config_base
#define ff_config_q bls12_curve::ff_config_scalar
#define inv_p bls12_curve::inv_p
#define inv_q bls12_curve::inv_q
