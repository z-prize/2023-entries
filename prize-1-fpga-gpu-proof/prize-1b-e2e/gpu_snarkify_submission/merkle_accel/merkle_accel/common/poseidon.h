/***

Copyright (c) 2024, Ingonyama, Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#pragma once
#include <cstdint>

class PoseidonMDS {
  public:
  static constexpr uint64_t POSEIDON_MDS[9][4] = {
    {0xAAAAAAAA00000001, 0xE27E6D5755543D54, 0xCCD13AB0066BE558, 0x4D491A377113A8DA,},
    {0x3FFFFFFF40000001, 0xFECE3B023FFEC4FF, 0x266B620607396203, 0x56F23D7E5F361DF6,},
    {0xCCCCCCCC33333334, 0x323E959B66656A65, 0x51EF819E6C2DE803, 0x458E97984C2B4B2B,},
    {0x3FFFFFFF40000001, 0xFECE3B023FFEC4FF, 0x266B620607396203, 0x56F23D7E5F361DF6,},
    {0xCCCCCCCC33333334, 0x323E959B66656A65, 0x51EF819E6C2DE803, 0x458E97984C2B4B2B,},
    {0xD555555480000001, 0x1B1E08AD2AA94CA9, 0x8005895C0806DEAF, 0x609B60C54D589311,},
    {0xCCCCCCCC33333334, 0x323E959B66656A65, 0x51EF819E6C2DE803, 0x458E97984C2B4B2B,},
    {0xD555555480000001, 0x1B1E08AD2AA94CA9, 0x8005895C0806DEAF, 0x609B60C54D589311,},
    {0xDB6DB6DB24924925, 0xAA362EDC49241A48, 0x57C7624B7077624A, 0x211F5460E7519182,}
  };
};

class PoseidonRC {
  public:
  static constexpr uint64_t POSEIDON_RC[189][4] = {
    {0xA7320A009D6ED3D8, 0x501EAC92A2758B36, 0x23BD51861DBB4A24, 0x669F064BFA3AE17A,},
    {0x6E72AB842B9D3F15, 0xEEE67C9645AC2734, 0xE4537FF2C84FA66C, 0x0A61A8DEFBACCA36,},
    {0x1E234771128165D5, 0x17292B4E1AAA4930, 0xF91B1E8A45DF275B, 0x21E9CEFA24B89D09,},
    {0xC161D2B03886EE3D, 0x5E4EBC5102B7E51D, 0xB56E1026A391BE2D, 0x4E2377D2F5CB43B2,},
    {0x0A8119099CB02EA3, 0x9F01A3B06FA57C24, 0x49762A88D8D5D087, 0x02255D26879B6D0D,},
    {0x16F4DD3D288D8C5B, 0xF2E8BAEEC680DDFC, 0x2845F98EB5DB8645, 0x27AA46E0263DDC66,},
    {0x90CB8624FCEF0186, 0x6F4875C83D73DFF8, 0x02BE301FD89B3034, 0x62E1FA5861A7807D,},
    {0x6A772C143CFB0CA0, 0x01907FB5E36E40F9, 0x1C14BC471FDD3BC9, 0x19F00D6D8A121F27,},
    {0x7E65D565609EAFE6, 0xD3BE0CF50259F3EF, 0xF23E96DF7F8F3C4D, 0x3499F695552F2C56,},
    {0x8F7C7A2B0157E0F5, 0x6F45F12F89B00716, 0x3941406ECDB90707, 0x236D5CF282247AAF,},
    {0x82D1D844649B4FD1, 0x1993992832094A97, 0x0ED41B01F9228F81, 0x445DAF9D58581E5B,},
    {0xCF29A71C0A8426BF, 0x678BD7E144DD9214, 0x224D031A69EF9B23, 0x21A58E9955E3645B,},
    {0xBD522C81D998A723, 0x3CD7BB8F18A0817C, 0x20CB0C4D6EDA2D5A, 0x48D3F5C7CCC92284,},
    {0x278D9C83899AB5CB, 0x0E714B39873CF991, 0x7B414EAD7CA52AF6, 0x5135F0A3315E5CC2,},
    {0xC77D3A2F1613A743, 0xD175FF96A11F8BD9, 0x6AE0B7E714CB38BC, 0x1D5284A657F69B5C,},
    {0x8F9C9BBC28024722, 0x44690AC5F14FE403, 0x21D1685651777023, 0x49C2165A0680F7A1,},
    {0x62E8546AAF307170, 0x3DDD5335F4FC71C2, 0xF1A79060BAD37F89, 0x5E8EABC6A3ADAE0E,},
    {0x036125902138F079, 0x9317A90CD6793532, 0x0D487222B6F385F6, 0x3264D05B6C40C40C,},
    {0x8DCE723994810339, 0xAD02B541140A2800, 0x0331890376F382FB, 0x727F2A5B23ECC141,},
    {0xD70587F1E7AEE758, 0xE8CB7B9A6EB7C667, 0x238BCAF2D615CA05, 0x1F00C3B0F6EBF427,},
    {0x8E540A80D926D400, 0x46E4119098B17503, 0xF18232D1F5AD341F, 0x3EF895995E82A276,},
    {0x60C5C9D8762968CC, 0x4E00F0729E0F9CF3, 0xED53B49DFC211988, 0x5606F5AAA845B400,},
    {0x3A93A4DC666F5808, 0xAF8127F93BA64590, 0xDE9CFEE7219A9448, 0x106150F58DCCB090,},
    {0x74249E62D15CF96F, 0xA3BE1F4CD4439855, 0x034590FE35014BCA, 0x3117BBB789D8A521,},
    {0xFB8C48F46662A7B6, 0x4F76C1D8AD154C7B, 0x7FBA4557EAD8A640, 0x5404CA4C386646A1,},
    {0x619D4E628F977920, 0x9E1E3EBD68A7C72B, 0x3E1957E0AEE03024, 0x561682D9CCC4D703,},
    {0x721B3A2E044EC332, 0x2E9435E4B74FFE14, 0xD42473DFB8DC2A9B, 0x609EBF7A728ACDF6,},
    {0x6846F58C08E45D2E, 0xAB384102429C4C94, 0xCC01B4D06B0915C3, 0x206C2444ABBAF221,},
    {0x81206F3F416EDF91, 0x54561DD1498A5006, 0x78FEFC1DC0B0D94D, 0x41E35E0864E292C3,},
    {0x7C681E1743AF3517, 0x2EC41A7370438AC3, 0xE05FDEBC7E85F5E4, 0x65963F69684A9459,},
    {0x06F24E236980012F, 0xBF3F7ADE7172C92B, 0x04B6D28122541C92, 0x384C77B79A3F1CC0,},
    {0x9A6C29B6DA32C1A3, 0x35191258C6FC7C41, 0x6DB2C72C52F05D79, 0x1D64115835442574,},
    {0x6605E726E94BADDD, 0xE5B284A0EFB6C166, 0x32D5B018FB1AAE24, 0x4F804C762F12827C,},
    {0xEA7C313D50266E6F, 0xE38A1ECCABBCFA18, 0x2C508BFFB262BEA2, 0x02C302E7C7980011,},
    {0xA035E35114759AB4, 0x2F125299604678EC, 0xCBD7741FAA335B57, 0x2118BE89BCF376E2,},
    {0x9F59447024B9F2CF, 0xC31023B4D8B7C363, 0xB9D65F5E9815FE36, 0x33652630502C8A44,},
    {0xEFCC1F686F2DCB90, 0x73796E0936B2CBF1, 0xB371052083FB04A2, 0x4B15C0ED31137BE5,},
    {0x2799BF8985852053, 0x9986FAD1B8EC46D4, 0x0A14DAA6FCE4A658, 0x3F86A1801F46D7F3,},
    {0x3D2CE7DB33D2A857, 0xF720160B08B9EB34, 0xFBDE85C5E5FB483E, 0x05A46F165678E990,},
    {0x1E0AA0A2177246C4, 0x11D23FA38B1079CE, 0x98D96BB31BC7C0D0, 0x3B8F9DB5BF15E2DC,},
    {0x50315E6F520F17BC, 0x6B24D9D232DFC84B, 0x7889BD6497D7F93C, 0x161E9F7D4EF018F4,},
    {0x8B365DA04EB4152B, 0xC914D49EDE4B2716, 0xF21A27D1C865DC75, 0x264A98F297E5B58B,},
    {0xA0D5DBEF34C7A11E, 0x643C7A233C75EB2A, 0x986BED3C9A386D77, 0x38802665B39DF51C,},
    {0x4E9282AC50284C0C, 0x49C6CBEBC2E860F5, 0xB5E2A9BE09CC870B, 0x67FCCA14EF5D1F6F,},
    {0x0A63A7C908629BE8, 0x75FD28C689BA05D9, 0x7AD993A80B2973E3, 0x2CA0307CCF7F7B3F,},
    {0xC9B647FF9C4035EA, 0x1422706ABA7C8612, 0x85D03D9B56361C72, 0x1F3072135CCF376C,},
    {0xB32B1A1C8F3A516A, 0xCEC86CAA1C1B266B, 0x3020D2BEC31D919A, 0x4F6851C88570A777,},
    {0xFDB4D3E313D14AC6, 0x15D63F99383B45CE, 0x4CF655F50389E658, 0x54B838CF161D883E,},
    {0x73424EE64276F9DB, 0xC9EB9A59F79703D4, 0x6EF747BC68544191, 0x3CE7859D585DAADA,},
    {0x43B14EA4F49B4030, 0xF675D4645F690777, 0xB65492F03154F11A, 0x57C14154A0017A88,},
    {0x078C72E4EACFECB0, 0x46FFD16AE1C18DCE, 0xF6D62BE4BFAF7B3C, 0x5F68E7F4C3235215,},
    {0x848212B2DB53C67B, 0x77256399101A131B, 0xAA78D1F8B241188A, 0x2D993835B6108C60,},
    {0xBD2AD14948E8F255, 0x56648C8E4CC192B2, 0x5079C1A03038AD2E, 0x5E4A9799BBB8F9B3,},
    {0x2A8B436FA8420A33, 0xE4345898EACEE8BB, 0x3C5A062EEA003EA9, 0x62F6AEAD0E19B6E3,},
    {0x079A37C72CD24CDB, 0xD0438A961B2EEB7E, 0x3C5849B3989781FD, 0x1FDADD9C8DA40632,},
    {0x393A6C147CF4ACAB, 0x631B76A0DE563282, 0xB44B8B636523ABAD, 0x359359B7346F649B,},
    {0x11AEB774D42AC2B6, 0x6814E9CF4AC6124B, 0x371913C37BDC1C00, 0x15675DF989854F06,},
    {0xF4E83AFCF50F4EE0, 0x8E511A30E570D5A9, 0x4CA7E8F75AB7F775, 0x3670C3A64B4BA269,},
    {0x25C78BFBEDB87171, 0x7B254FA0B52E3181, 0xF3EA660B391D16DA, 0x66B4703BF0415B2A,},
    {0xA33727CBE3C4C1F3, 0x3A8FFF249BE3466E, 0xF253D0B44363D5B5, 0x72F66342D2B390FD,},
    {0x11A5AC9E39DB1395, 0x35CC9A0F3B3E3A8C, 0xEF963771323962A4, 0x37CF10985D6BA77E,},
    {0x1E31446A160192B7, 0x33D48FC90E3F1E33, 0x9D0FBA99491C34BF, 0x27E4538F90EC0B00,},
    {0xAB11F05968D9EB6F, 0x28F10F229EB5B027, 0x7281CD1AE0F2E828, 0x1B711906C22AC993,},
    {0x4C52F982F38008E5, 0xD5AB4A8E15AB2291, 0xDDEC5898A33119B5, 0x54A3606A3E8C5AB2,},
    {0x195ED82B9B2F055E, 0x3FDD3DE93F4A1595, 0xBDE9C7CA41EBFA2F, 0x0E0AEB0F0E759A2B,},
    {0x6F4F11466086EFDC, 0xF167D6D1D054109A, 0x62F720BDA04E4E56, 0x29839536E73AE403,},
    {0x6854D29D5ED64B2F, 0xC4C96BA1168E4EC4, 0xB46EF89751C6B7B6, 0x54D8E3EC7800FFB1,},
    {0x3B604F3A5AA931D0, 0xE38B708774C505C9, 0xADD5C7BD132146CC, 0x0DD33054F93FA501,},
    {0xCD37A4F4B7FCF66E, 0xFED3780B9A4EDA0C, 0x8831ABC2F58A332E, 0x58ED1F752578C70C,},
    {0xD85CCC676E321EA1, 0xE599B9C364C98607, 0xF7E4362B9823267D, 0x59C0BD12AC0F1E8D,},
    {0x82A6A2FBD555A72F, 0xBDE5B65AF599E483, 0xB20642577C774662, 0x0734E8779B78F832,},
    {0xE45D8353C4F27D6D, 0x1F5CA5C88A670CDC, 0xD57278A61087F017, 0x4296C2CD2D63615A,},
    {0x9B829EEAA0F7A28D, 0xF5B8AB6A706C5B8B, 0x4FD71C2B5BD64640, 0x70CA4F659AC36545,},
    {0xC41854F5CD16A79D, 0x5750B5EA5F076CA7, 0xACC8BCA7B14DD4D1, 0x2B3ED2067E88294C,},
    {0xBA3E679B2E928FBA, 0x1EF353729BE365A5, 0x1DF557808F34D2CB, 0x1F6005F1C1E35F72,},
    {0x2550F30BAB0E7487, 0xD1682C5A32872218, 0xB3CE0D7D0B87B726, 0x5FD220812F0B9285,},
    {0xC2CE7D8871F179A0, 0x99D3A583DC70D473, 0x888AD3B41C7CB129, 0x29FCC8A49F9ACE62,},
    {0x109C6B543A8A344A, 0xF8E71F47CA60CC69, 0x2E31B1A4E134FD4D, 0x62A4E582D88CEAC8,},
    {0x50528998CF81E018, 0xA61462A70C1E1757, 0x959787E2181DCC89, 0x31C4D94CE026BAD4,},
    {0x718BA2C11362F807, 0xB79A074B8A98B32F, 0xB511580861CC406B, 0x3F26B82B525E9E40,},
    {0x243A3BF3B1177F2C, 0x765FF0FD99B11ED4, 0x36F7F9CC8179904C, 0x65013234A39C87F8,},
    {0x92CC6F22772459F8, 0x6C1E221B6E852122, 0x4157AB4063E0FCC6, 0x0C22DB7080672B16,},
    {0x1ACE65E23661AC39, 0x7C1E269A14A8FB09, 0xBBB9A3256E3535D6, 0x1757F9FAC9967EED,},
    {0xB14D6E3BF1CACA54, 0x86999662EC4C7620, 0xBE40825C6EEB0F7C, 0x504CEF8899122412,},
    {0x26B8C98DD993F8ED, 0x4E144FCB881EF394, 0xD2F30A79FB29B92E, 0x376DE57ED93DE38C,},
    {0x175BBF19C7C4C1D4, 0x533B982A05D2FA88, 0x6243D96EA8C49E84, 0x0DBA6D221E369078,},
    {0xF2BD7B9F7D6E1BDE, 0xA1556AB8D94994B3, 0x90923711F781D49D, 0x444130BFB285396E,},
    {0x92BDC36FC935200D, 0xA5907D668CE5266F, 0xB5E54D09BC30397D, 0x423B9D3D069FBE61,},
    {0x468F9851606155E9, 0xA9752C74F2931F3B, 0xA96924CFAD5663BE, 0x1DA3918F5CD423CB,},
    {0xB4CFF11862CFAEB4, 0xD86F0DD0715432DC, 0xED2E74387F326B92, 0x53022CE831F40D5C,},
    {0x7125626EC14ECDC1, 0x46525455DBD68715, 0xC1568823DDDF0607, 0x69E2FE14A6875A92,},
    {0x286526CFD9CB57BD, 0xEE0E4D88B3C54135, 0x8C71DFE13FBFAE2A, 0x2AC84677966E174C,},
    {0x483D4EAFCAA9C226, 0xE72A364858597EB3, 0x34C8135147704FD7, 0x06FBB5301E4CFFA5,},
    {0xE006A20319240F7C, 0x250984A3F71277B9, 0xF54893808DBE78C4, 0x5CE6E13A1AD45CD2,},
    {0xF3617E4E91046232, 0x9598D2529A481FBD, 0x31C8D6AB9862C501, 0x582FD75A0CF50BB0,},
    {0xCEC3750C3FFB9365, 0x74E44C34DA109C5A, 0x8E7E4A22010DBA78, 0x04F4473B814BEF66,},
    {0x7C765B067A1CF3CD, 0x8B643EADF8BF659F, 0x8F3BD861CB5B0EE3, 0x207C2CBA5430A95A,},
    {0x5454CE254E63D716, 0xBE911E532783C3FF, 0xFB0CC9B1CA9FCCFA, 0x6D0E096809B5E23F,},
    {0xBEBDD8B9CA1AAC95, 0x53C2028C1D1A4A71, 0x2E9D4D05B3371494, 0x05969D41CC3BBAAA,},
    {0xC3EEE2BFAD77AA9C, 0xF60337EE3BF60B37, 0x6A925C8C3ED6D343, 0x19548FF7A77670D8,},
    {0xF763DCB37B25CA02, 0x02D60A48149081A3, 0xDC44E77CA7B3C89F, 0x5796BC6126B98754,},
    {0x3AF65A51FCF3DC0F, 0x457A950AAFC7330F, 0x62D2E74E63852475, 0x4AA80AF316C7A2B6,},
    {0x727C1342EEA142FF, 0x8BE9253C3981E851, 0x1E3B72596A048996, 0x19CC9DF4E33ED774,},
    {0xCD841E09BEAA7E97, 0x2921F6DFBB7D0B51, 0x98FCB59EE514B2DD, 0x1CEE1165C26E9A56,},
    {0x5E7B7036A6219619, 0xA7E0A03CFE4D4258, 0xB9C1AAC0F1A014D7, 0x28AD294F58C5C3E4,},
    {0x12A4F952EC79D318, 0x602E31685EF3FD73, 0xA7D1F0BE28C3C389, 0x63591B7786B99B3E,},
    {0x9CB16EB18B76414A, 0x7E1B563D3B8000A3, 0x48299F14738EA081, 0x3592AB44A12EA166,},
    {0xE960235B33D34D13, 0x4E95A792E79BAB0A, 0xF90824119B64D79F, 0x375513F5F15360C7,},
    {0xC53494281D4A52F6, 0x1201785B2E32A321, 0xC9D29C71A556F541, 0x6AAFF1F098EAE8D9,},
    {0xB48C98691379F7AE, 0x1DE166EFDE36E51B, 0xB43FED4A90B702F6, 0x4A60DB5B4FDA6BD3,},
    {0xB43CBDA39895EF7A, 0x367FB501470A428C, 0xE327E1C2AA7F9764, 0x16A315ECE8D14BF6,},
    {0x58C7921DC3D1C02F, 0x0AD227E88CF0C920, 0x3604E949D226C07F, 0x4732994724DA4762,},
    {0x7B3CE4C99C9BAD4B, 0xF456A4235F900F03, 0x523ADB5CDD45376A, 0x66CFB45F6647A095,},
    {0x3D884E00A5DE1ABC, 0x11219BC1B936B000, 0x62D16496529992AF, 0x5534431C0D6F15F8,},
    {0xDEB9126C7E897EE8, 0xD1F313DD66643EEB, 0x76091FFBB6F3A1B9, 0x2AF548FBE4C0E021,},
    {0xE153FF959B58FA15, 0x35BB2C6312A4D74F, 0x5931CA15C5E11F7A, 0x46C38ECE952E8661,},
    {0xDB312592A8D7C86A, 0x1AB406D91046F32F, 0x6D5029BAE57752E7, 0x0D58DBDE54644527,},
    {0x1B8432EBC923E6AB, 0xF7568A2D7619DAE9, 0x428BB21A79D56951, 0x4A61E2F96F08BCFD,},
    {0xEFE68237B208AC59, 0x762A62F0F5A78E94, 0x70159FAAA5A21BE4, 0x59C58D944CB5FA9B,},
    {0x553B03AD0ED33FCC, 0x28AD37A1069F8FCE, 0x5D54BEC400E5F251, 0x1DE64BAB1CAE8DC3,},
    {0xDD771ABDB276ED2F, 0x97A3A12440DC35EB, 0xD5F3A90120495A4C, 0x532D70F037959493,},
    {0x5926B68D82EF116A, 0x68E180184449717A, 0x2E9679FFD214CE9D, 0x2FA34FCD7AC076E0,},
    {0x929569552CF616B6, 0x9644A930EEC30E9E, 0x43940441CE2A8773, 0x739AFA5F847A858B,},
    {0x12E241CE51351030, 0xF062AF17D068B275, 0xF34724EA1476973F, 0x0F2EC24A54A31240,},
    {0xF8385CAB0A1B0E5D, 0xFF171EDE3B3D01D4, 0x58E4D806B6F0B7D6, 0x5A927A7FF5FF45E0,},
    {0x9320B6C5D9D0DA0D, 0xF9D915A75570BF5F, 0x31B019A0DD8CB14B, 0x29D8D55CEC33000E,},
    {0x02434CA27A071EED, 0xAAF60AB638036498, 0xE8B35C912A6B82F4, 0x37A368626670937D,},
    {0x0290BB19350F572C, 0xAEB79CCD643E9E1C, 0x72088B70537DB5CE, 0x32F8EF20DF6C4865,},
    {0x5850A4D0C7477E0C, 0x2E6AAD615EB472FD, 0xAA065F7BDD558148, 0x009DF831FA9CEDAC,},
    {0xCD75ACFB523D917F, 0x25E745FEEA3BB289, 0xCA347A6D8653D426, 0x557F47B72F848625,},
    {0xAA379722820B7466, 0x5BA4C8FD4CD0247F, 0xDC88C156D33BD39B, 0x6B0844F5EB2FC982,},
    {0x17AC724296F8DC2B, 0x7B21D12E0610AC13, 0x9F38D4980B01355D, 0x227DEE5B1C7F8C40,},
    {0x4EF532BB4B4D5F7B, 0xC38272D01437F999, 0x87153DD07B1FB944, 0x46074918624B162C,},
    {0x8B8BE56B19D9DB33, 0xF848CE6E831D8455, 0x72B68288CDA1CB70, 0x2527405B6D64E041,},
    {0x3B6433B3D177D9B6, 0x89E4CC960F495774, 0xAFC6CC5C571FE59E, 0x43A67538A944E568,},
    {0x2173B78D6D3547B8, 0xAD61B520D1537726, 0x448747E6340E908B, 0x3DFA278C4F8D8744,},
    {0x225B7D11CA1174AC, 0x986C435B8C5C094E, 0x57E7595D0778318D, 0x28470D01333EEB18,},
    {0x7260CB3F7F8DB9C0, 0x9B611DD3B9919667, 0xC393BA38B3B15E68, 0x15FD96CD7931143E,},
    {0x3AB814EC29364DAF, 0x98DEE91162C63CCA, 0x4158D15D9A8F2FB2, 0x44068FD9A1C6015B,},
    {0xA5BFDBEE41171729, 0x3F9B84C22E511C0A, 0x672C388D3DBF6A35, 0x4C59F63D303F727C,},
    {0xDDDE44DF43FD7706, 0xA86E9F036A004BE5, 0x3E53E1EF86DEF6B0, 0x3101540E1D7F0A92,},
    {0x7BCFE39D8B8F236E, 0x442ED279FFFDFEE2, 0xAF0F408FD1B18CD9, 0x22E2471398685D19,},
    {0x42D54C51B323F6D9, 0xCCB97AC0EA73D68B, 0xF274AF47EA80443A, 0x5C586D10AE52A9E1,},
    {0xA914C79ED039AD85, 0x7C458E224E82BE2A, 0x14BDDCE789FCA3B6, 0x35B59F926A693AF1,},
    {0x204577A73448BA0C, 0x0861D7BEF7C63C25, 0x54D60365FE40B612, 0x0B2D458EE73F4E60,},
    {0x99368269DE3B94B5, 0x075DA766FC20154F, 0x8E317874AB8670A7, 0x02F95CF5081A78C4,},
    {0x4BF5E49115618534, 0x8184F51A6319DAB2, 0xB61AEDFA14DE3D4D, 0x71C801487FCBBC03,},
    {0xE841C1F933CAF041, 0x23B3DDBC1399DC8C, 0xE05ECA41A102DC88, 0x307B41C27156AC0B,},
    {0x0ECD17DD31D253A5, 0xC6C638AE8481C320, 0x57AC98C9314454DF, 0x02FC9786AB1B4484,},
    {0x81FFA016AF40AC74, 0xE213D4A5FC9755C9, 0x724F86B5B0D55904, 0x4409AB4F6C0F069D,},
    {0x5A1DB30DEAAD785D, 0x80ABE3F48F875148, 0x20EBFEFFF311BFE2, 0x39176C107168EB66,},
    {0x5F066214CCCB06A6, 0xC5F02234708B45EA, 0x1169E30799579971, 0x204F0AC20ECE2CC1,},
    {0x7CB65C5CC8C4EB9B, 0x713717267D5192F7, 0x707A40C835FA6A27, 0x739FC006D3B54D5A,},
    {0x63962E6F87D09ACF, 0xDCD6268085760C85, 0x3961190A8236D7E7, 0x15A6DF2116720700,},
    {0x7B88931A9757B5D8, 0x2466B79CBACAAD34, 0xABA417215372CBB9, 0x40376B10DA48099F,},
    {0xE0204E472BF111EB, 0x9D7E5BCD4660D06A, 0x2B9D47032EACD650, 0x5D32AA7E6FA4E2E5,},
    {0x04EA09F94AB0817C, 0x81E0050668E9665F, 0xE14CB877CD268D97, 0x2C0E9F70303690B7,},
    {0x7BA53AEB3CDFFD79, 0x787E0B25D99596E3, 0xFC0530A3064FF80B, 0x231DDA54EE054A0A,},
    {0x6F58D466F9D99AF4, 0xBEE858B2434639F6, 0x6DC8FF296A065973, 0x274B78559E51E542,},
    {0xFD0E837C669AB727, 0xE3C066DAC732B888, 0x21C627895438811A, 0x5992BB8BA0CA2568,},
    {0xC592DE58D6A35F56, 0xDE9E4C4226799A5D, 0x5DEF858CAF2FC0B2, 0x597A123502768A98,},
    {0xC1B7228C8E335E54, 0xA5B9387ED0A3B7A8, 0x735178AD5F7BF87C, 0x2A0C027D819603F0,},
    {0x28C75762E728ECC4, 0xFFC2C54B92903D42, 0xCE033CA67F075830, 0x62BD0E7766514106,},
    {0xCCBCF2A2E647D1DC, 0xA34C01CF6A3B1AB9, 0x1F1D9D57F69C56FC, 0x00FB6BA60EFE3256,},
    {0xCAEF03014D4B2BD0, 0xF923280145073DB0, 0x4FFCD66850045257, 0x450C9B21E12734BB,},
    {0xB20F8A1CC3E0CD73, 0xB76F0001CDBEFECD, 0xF694C6D6ED8797CC, 0x5A496FFF496AEBE0,},
    {0xA0FA47D2D91D4933, 0x8D027F108E760055, 0x08DB33A151EB2E3E, 0x601352AA918622A2,},
    {0xCC810B45C6289EC4, 0x51D9E30528C92087, 0xE509CFC2BB12EB22, 0x59EDA8EC0F8CF0CF,},
    {0xD8D56E8DE98A95C4, 0x9660BC94C20B02C5, 0x1E8DF72386202423, 0x47F1310D0E239C2C,},
    {0xBD68D5A917CE5BC5, 0x9A5F410227E6EC8C, 0x96D1B6436B68E047, 0x6F53B661D4B62205,},
    {0x69BA10EADF521F06, 0x55061D6A07D1A8C9, 0xB46A8C1E78A6F4FD, 0x6A0728EFD48227AA,},
    {0xF47FBC7CE454FC0D, 0x2628E2897B39F3C8, 0xD5436B55CE0C1FE8, 0x2DE99C53670CDB1E,},
    {0x7DF972615989D00E, 0x9BCCBC5BAF65A986, 0xE422036BC48DBF99, 0x35A1BAD35F8A78A9,},
    {0xBCF9EBC8E7C723F6, 0xB6D6DCBC1420A0C6, 0x1BCF24D7ADD25FEE, 0x529D9E8D93678BFE,},
    {0xB40532B86AA85C9A, 0x7A54E6F9A278A05D, 0x216B0A5819603758, 0x69CF3756CE67CD72,},
    {0x002ABD6203FF2686, 0x6216613FB89B262E, 0x71FD4572FD69B6D5, 0x4D850BAB7657C1AC,},
    {0x4513030AA7F18968, 0x45E37A625E349161, 0x6E24D1117B25343E, 0x2784E7EF9AC462B5,},
    {0xD3BDB0393EC5A7E9, 0x3CE8D1F5ADBADF22, 0x2A9DA9721A114021, 0x11BCAB21E844DA9A,},
    {0xBC7D64462213F45C, 0xE04C456CC781E0F8, 0xDD1DC466421A8891, 0x1AA8AFC39349A7C2,},
    {0xC172AC9894C225D8, 0x4F4E71D012107432, 0x032EE88DA1CC50CC, 0x1CDBA5A6FF825AFF,},
    {0x466A3733A62C6787, 0x33ABE87A4B1DC8B2, 0x28093E2F98736DE4, 0x15090A6658F804A4,},
    {0x667CA55DB404F474, 0xACCBBC2BC0C94629, 0xCE8C00EAE2043A7F, 0x3A1864E0CA051A15,},
    {0xB83DF6A7C4A649C3, 0x220E2777F5A1FD5B, 0xA9047F1DB8A13338, 0x2E2B4CBA4FDD3698,},
    {0x3965DA155ED9D012, 0x67E700372A8943AE, 0x2159173B507F73DB, 0x3D316650B675ECB7,},
    {0xE4B2EF5865202D92, 0x8A4611741C12A4C3, 0x545406C32EB2CC45, 0x08D5B91751A690A7,},
    {0xD414E393E5BAECC1, 0xE1D2071489AD43E6, 0xCBD798B894F344AC, 0x1662A1E3946393E6,},
    {0x573FBCD6CB536BCC, 0xCC6A96C1D3DFF54D, 0x50A4ACBBE5D904DC, 0x6B98576FE63BA1E8,},
    {0xE2AEADE6F7A5A190, 0x6EB6917BE79AD083, 0x25E262AE021BECDF, 0x54A91D320B6372D4,},
    {0x0BE0ABE065395E5C, 0xED12E38EC24D5BB8, 0x1A917A05466E5884, 0x60DFBFA5D5DD0635,}
  };
};