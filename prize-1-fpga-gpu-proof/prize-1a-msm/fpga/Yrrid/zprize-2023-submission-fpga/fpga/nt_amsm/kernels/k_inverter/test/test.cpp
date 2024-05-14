/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#include <iostream>
#include <string>
#include <gmp.h>
#include "k_inverter.h"

std::string mpz_to_string(const mpz_t &number) {
    int base = 10;
    char *buffer = new char[mpz_sizeinbase(number, base) + 2];
    mpz_get_str(buffer, base, number);
    std::string result(buffer);
    delete[] buffer;
    return result;
}

void from_mont(mpz_t result, const mpz_t x, const mpz_t R_inv, const mpz_t P) {
    mpz_mul(result, x, R_inv);
    mpz_mod(result, result, P);
}

void to_mont(mpz_t result, const mpz_t x, const mpz_t R, const mpz_t P) {
    mpz_mul(result, x, R);
    mpz_mod(result, result, P);
}

int main() {
    hls::stream<field_t> a_in, b_in, c_in;
    hls::stream<field_t> a_out, b_out, c_out;

    k_bls12_381_inverter(a_in, b_in, c_in, a_out, b_out, c_out);

    //    std::cout << "-- [k_bls12_381_inverter basic test] -------------------------------------------" << std::endl;
    //    for (int i = 0; i < 42; i++) {
    //        a_in << base_field_t<bls12_381>::MONT_ONE;
    //        b_in << base_field_t<bls12_381>::MONT_ONE;
    //        c_in << base_field_t<bls12_381>::MONT_ONE;
    //    }
    //
    //    for (int i = 0; i < 42; i++) {
    //        assert(a_out.read() % base_field_t<bls12_381>::P == base_field_t<bls12_381>::MONT_ONE);
    //        assert(b_out.read() % base_field_t<bls12_381>::P == base_field_t<bls12_381>::MONT_ONE);
    //        assert(c_out.read() % base_field_t<bls12_381>::P == base_field_t<bls12_381>::MONT_ONE);
    //    }
    //
    //    for (int i = 0; i < 42; i++) {
    //        a_in << 1;
    //        b_in << 1;
    //        c_in << 1;
    //    }
    //
    //    for (int i = 0; i < 42; i++) {
    //        assert(a_out.read() % base_field_t<bls12_381>::P ==
    //               field_t("0x11988fe592cae3aa9a793e85b519952d67eb88a9939d83c08de5476c4c95b6d50a76e6a609d104f1f4df1f341c341746",
    //                       16));
    //        assert(b_out.read() % base_field_t<bls12_381>::P ==
    //               field_t("0x11988fe592cae3aa9a793e85b519952d67eb88a9939d83c08de5476c4c95b6d50a76e6a609d104f1f4df1f341c341746",
    //                       16));
    //        assert(c_out.read() % base_field_t<bls12_381>::P ==
    //               field_t("0x11988fe592cae3aa9a793e85b519952d67eb88a9939d83c08de5476c4c95b6d50a76e6a609d104f1f4df1f341c341746",
    //                       16));
    //    }
    //
    //    for (int i = 0; i < 42; i++) {
    //        a_in << 2;
    //        b_in << 2;
    //        c_in << 2;
    //    }
    //
    //    for (int i = 0; i < 42; i++) {
    //        assert(a_out.read() % base_field_t<bls12_381>::P ==
    //               field_t("0x8cc47f2c96571d54d3c9f42da8cca96b3f5c454c9cec1e046f2a3b6264adb6a853b735304e88278fa6f8f9a0e1a0ba3",
    //                       16));
    //        assert(b_out.read() % base_field_t<bls12_381>::P ==
    //               field_t("0x8cc47f2c96571d54d3c9f42da8cca96b3f5c454c9cec1e046f2a3b6264adb6a853b735304e88278fa6f8f9a0e1a0ba3",
    //                       16));
    //        assert(c_out.read() % base_field_t<bls12_381>::P ==
    //               field_t("0x8cc47f2c96571d54d3c9f42da8cca96b3f5c454c9cec1e046f2a3b6264adb6a853b735304e88278fa6f8f9a0e1a0ba3",
    //                       16));
    //    }
    //
    //    for (int i = 0; i < 42; i++) {
    //        a_in << field_t(
    //                "0x22B685ED3CE8D30A5F74073753931CF5932CED301ECACFBA16B429BBED32A7C8E01C2715C7171229E95B50E8E7CEC76",
    //                16);
    //        b_in << field_t(
    //                "0x22B685ED3CE8D30A5F74073753931CF5932CED301ECACFBA16B429BBED32A7C8E01C2715C7171229E95B50E8E7CEC76",
    //                16);
    //        c_in << field_t(
    //                "0x22B685ED3CE8D30A5F74073753931CF5932CED301ECACFBA16B429BBED32A7C8E01C2715C7171229E95B50E8E7CEC76",
    //                16);
    //    }
    //
    //    for (int i = 0; i < 42; i++) {
    //        assert(a_out.read() % base_field_t<bls12_381>::P ==
    //               field_t("0x183e48182fb623ae9606b992a131883f92c2ee6ca7165d186c7d3059d8cf668a15ada39e5417bdce7115b513817d9442",
    //                       16));
    //        assert(b_out.read() % base_field_t<bls12_381>::P ==
    //               field_t("0x183e48182fb623ae9606b992a131883f92c2ee6ca7165d186c7d3059d8cf668a15ada39e5417bdce7115b513817d9442",
    //                       16));
    //        assert(c_out.read() % base_field_t<bls12_381>::P ==
    //               field_t("0x183e48182fb623ae9606b992a131883f92c2ee6ca7165d186c7d3059d8cf668a15ada39e5417bdce7115b513817d9442",
    //                       16));
    //    }
    //
    //    std::cout << "PASSED" << std::endl;
    //    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    //
    //    std::cout << "-- [k_bls12_381_inverter advanced test] ----------------------------------------" << std::endl;
    //    field_t test_0[42] = {
    //            field_t("0x1", 16),
    //            field_t("0x2", 16),
    //            field_t("0x3", 16),
    //            field_t("0x4", 16),
    //            field_t("0x5", 16),
    //            field_t("0x6", 16),
    //            field_t("0x7", 16),
    //            field_t("0x8", 16),
    //            field_t("0x9", 16),
    //            field_t("0xa", 16),
    //            field_t("0xb", 16),
    //            field_t("0xc", 16),
    //            field_t("0xd", 16),
    //            field_t("0xe", 16),
    //            field_t("0xf", 16),
    //            field_t("0x10", 16),
    //            field_t("0x11", 16),
    //            field_t("0x12", 16),
    //            field_t("0x13", 16),
    //            field_t("0x14", 16),
    //            field_t("0x15", 16),
    //            field_t("0x16", 16),
    //            field_t("0x17", 16),
    //            field_t("0x18", 16),
    //            field_t("0x19", 16),
    //            field_t("0x1a", 16),
    //            field_t("0x1b", 16),
    //            field_t("0x1c", 16),
    //            field_t("0x1d", 16),
    //            field_t("0x1e", 16),
    //            field_t("0x1f", 16),
    //            field_t("0x20", 16),
    //            field_t("0x21", 16),
    //            field_t("0x22", 16),
    //            field_t("0x23", 16),
    //            field_t("0x24", 16),
    //            field_t("0x25", 16),
    //            field_t("0x26", 16),
    //            field_t("0x27", 16),
    //            field_t("0x28", 16),
    //            field_t("0x29", 16),
    //            field_t("0x2a", 16),
    //    };
    //
    //    field_t golden_0[42] = {
    //            field_t("0x11988fe592cae3aa9a793e85b519952d67eb88a9939d83c08de5476c4c95b6d50a76e6a609d104f1f4df1f341c341746",
    //                    16),
    //            field_t("0x8cc47f2c96571d54d3c9f42da8cca96b3f5c454c9cec1e046f2a3b6264adb6a853b735304e88278fa6f8f9a0e1a0ba3",
    //                    16),
    //            field_t("0xe888b454418ee16f731a213fd771601997646ba2d0b877ffc5cb359c1178efdb860f78c3e61ac508f9f5fbc096695fb",
    //                    16),
    //            field_t("0x1166acee8172ac37cc2c237c8eec3bb70c3687ecdea9ea4fd711bb2b8e7de8c751f3b9a8db1e413c5a3747cd070cdb27",
    //                    16),
    //            field_t("0x384e994508efa5552183fb4576b843c47fbe821ea52b3f34f94417c0f5124910217c7bacec36763975fd30a6c0a6b0e",
    //                    16),
    //            field_t("0x1444ce97becc6a58a126a4e52061616c7ef6c91f90484d1fb1c6c2fd5be44290eb867bc577dad62824cf2fde04b32053",
    //                    16),
    //            field_t("0x63a84d48ae61ce5455e6a089133096e665742e2134e157ffe70df2677533d482a72b33c1abc254718fb290771be4047",
    //                    16),
    //            field_t("0x15b3df6c5d7949690ba3e599691bf4473856e9b8e9177e879f2146e642976f75b84fdcd3c639209e0a1b23e6838642e9",
    //                    16),
    //            field_t("0x162e3a5de7b2e9192f22fb2ad6af7a9020cc49ec06b1e454ee3f72de8f7dd3c1fc92fd2de0588ec55689ca940321f91b",
    //                    16),
    //            field_t("0x1c274ca28477d2aa90c1fda2bb5c21e23fdf410f52959f9a7ca20be07a89248810be3dd6761b3b1cbafe98536053587",
    //                    16),
    //            field_t("0x1998171f61271c9b0f3bfdd9c199932ac5b3af824b13a85de5aa966efb084fc18395ac946b5e92d44ce77334861d392",
    //                    16),
    //            field_t("0x1722f040fc2628797621264db1d68721f1b70a5241e6afef8c7bcacf294a9c5a85193de214976b13ef6717ef0259657f",
    //                    16),
    //            field_t("0x135b3edb331962643fe16539b2ac832274ee8d9078de65e217472128ff7256020258d6a967d414126c37db04022b292d",
    //                    16),
    //            field_t("0x101dcb5f623301bfc83d08df6a3f5b22e56747338369941fb2d0d8e3b70219b6248f599d660812a3697d1483b8def579",
    //                    16),
    //            field_t("0x128259cd9684ed2df81a850af4ab9f4e5af8d50e9b1ef32609fca23f543bb0486a7a97e810792276591f4658ceadeacc",
    //                    16),
    //            field_t("0x17da78ab4b7c9801ab5fc6a7d633d08f4e671a9eee4e48a383290cc39ca432cceb7dee693bc6904ee20d11f341c2f6ca",
    //                    16),
    //            field_t("0x198273adb739048c31a8b09549fd6f36ce14b8871b0e0a560f1d51f837faa71f77d60d9010e2e22c17cff2c6d47b3896",
    //                    16),
    //            field_t("0x1817a624109967d9bd1f51708cfd93b3c2a1cab87d1b7b8a2ab822bfc31764f30d9f7e9648d647628844654a0190d1e3",
    //                    16),
    //            field_t("0xbe00f1ddc915f9ac970576140d05e19b681fe91bf255fab913b9256f2a33b5dca18c8c4d1eb0042839f88610ef520aa",
    //                    16),
    //            field_t("0xde1c35a30e3b1e27a13e3c83780b77ac43a9fcaf457365c877d79af7f2cc4364fdbf1ee0c5ad9d8c2d774c29b027019",
    //                    16),
    //            field_t("0x213819c2e4cb44c6c74ce02db110324ccc7c0f6066f5c7fff7af50cd271146d637b911408e961c25da90dad25ea156d",
    //                    16),
    //            field_t("0xccc0b8fb0938e4d879dfeece0ccc99562d9d7c12589d42ef2d54b377d8427e0c1cad64a35af496a2673b99a430e9c9",
    //                    16),
    //            field_t("0xf768ad126e1cf059557af6cd4ddf8dfe44289ab900f47fa1f1a6f06a5065b6b975b41ad6fc942ff3c1f6587d4b429d3",
    //                    16),
    //            field_t("0x189201159ad30789e09e6701fa9119fcab172aeb9ab5e15779d64eb80ffdc93f51e29ef062f5b589d4b30bf7812c8815",
    //                    16),
    //            field_t("0xb42eb7434fcbaaaa04d98a77e24da5a7ff2e6d2edd5730a9840d18cfdd0750339e5b255c8d7b13eb132a3548cee236",
    //                    16),
    //            field_t("0x16ae2862b64ca47f457e8677fafc17fcecb2ec8ab631bc50bf3bf9e4fb11a61310826b540c940a09131b6d82011569ec",
    //                    16),
    //            field_t("0x18baca10c8e63cc4971e18dd1f179c14f893a051f9e9589be98b080ad44a9558bdf8ff0e6baada4198d7ee3156606f7b",
    //                    16),
    //            field_t("0x150f6ea4cdd9742d09ac584ad6c583fd24ef495c3b77536f8d00d5c256d987ed219dacce0bae095191be0a41dc6f5012",
    //                    16),
    //            field_t("0x1455854fac44b6ca619d930a7716fb061201849fac2c972de93e996c308b68c1a4dedbcfc4a808ff4eec73d5a8b21858",
    //                    16),
    //            field_t("0x9412ce6cb427696fc0d42857a55cfa72d7c6a874d8f799304fe511faa1dd824353d4bf4083c913b2c8fa32c6756f566",
    //                    16),
    //            field_t("0x23ecbfdbe1f5007331e78311ab3207ab740d7bb7791cba734866abb4c31475386720770a0670028d1d595a6d79e8fe4",
    //                    16),
    //            field_t("0xbed3c55a5be4c00d5afe353eb19e847a7338d4f77272451c1948661ce52196675bef7349de34827710688f9a0e17b65",
    //                    16),
    //            field_t("0x11de8c6ccdb0bfaa17b90518b63afba07c6df0ab593e7556e43ec4e2f45b7b16c73073978dc9f8643d997d11182062f8",
    //                    16),
    //            field_t("0xcc139d6db9c824618d4584aa4feb79b670a5c438d87052b078ea8fc1bfd538fbbeb06c8087171160be7f9636a3d9c4b",
    //                    16),
    //            field_t("0xba5bb54ff9462052beabeb104c213d2a30df8626544d8995c2a1a7b47576eb67b288a3eb2ada10e1c31d5017d25eab9",
    //                    16),
    //            field_t("0x190c5c07250ca73a041d7c936824a045938c8b1eb8504724c8f47ab05ce42d8b9625bf4a7d1523b12121b2a500c83e47",
    //                    16),
    //            field_t("0x13739b572ded31e3e7f4dcf0666f43e3983ad304bc9cb74bb0004f86398c196c394274eec6db29a57996990f3f081563",
    //                    16),
    //            field_t("0x5f0078eee48afcd64b82bb0a0682f0cdb40ff48df92afd5c89dc92b79519daee50c646268f5802141cfc430877a9055",
    //                    16),
    //            field_t("0xf1ec59724331854d8ff044ffca80ffdf321f30724212835d4d2a698a76119620b019ce2b30d5c060cbcf3ac00b8f148",
    //                    16),
    //            field_t("0x13f16aa23531cc3e6297c5bf3d6632291458f5a7f3ee248df75726283aeedd2d3743f8f65ed76cec3e6b3a614d810d62",
    //                    16),
    //            field_t("0x254f8552d4042222eba0dcbea1f2304614ce9a3b7c2a6ed3cf66ecffaf161657f60a7f7775c6a4452055e6b6ad589ef",
    //                    16),
    //            field_t("0xe0a49c333e64d735bc83adc8f2e57fe189f863d7cfa379fb355e3d6e4910548c113c8895d1eb0e10bd406d692f4e00c",
    //                    16),
    //    };
    //
    //    for (const auto &x: test_0) {
    //        a_in << x;
    //        b_in << x;
    //        c_in << x;
    //    }
    //
    //    for (const auto &x: golden_0) {
    //        assert(a_out.read() % base_field_t<bls12_381>::P == x);
    //        assert(b_out.read() % base_field_t<bls12_381>::P == x);
    //        assert(c_out.read() % base_field_t<bls12_381>::P == x);
    //    }
    //
    //    field_t test_a[42] = {
    //            field_t("0xf1ca20c7311d8a3c2ce6f447ed4d57b1e2feb89414c343c1027c4d1c386bbc4cd613e30d8f16adf91b7584a2265b1f6",
    //                    16),
    //            field_t("0x137021ce6ec9d28663ca828dd5f4b3b2e4b06ce60741c7a87ce42c8218072e8c35bf992dc9e9c616612e7696a6cecc1c",
    //                    16),
    //            field_t("0x34571e3f1fd42a29755d4c13a902931cd447e35b8b6d8fe442e3d437204e52db2221a58008a05a6c4647159c324c986",
    //                    16),
    //            field_t("0x15f7acff619699cfe1988ad9f06c144a025b413f8a9a021ea648a7dd06839eb905b6e6e307d4bedc51431193e6c3f33a",
    //                    16),
    //            field_t("0x11b106917eed8d14f06d3fef701966a0c381e88f38c0c8fd8712b8bc076f3787b9d179e06c0fd4f5f8130c4237730ee0",
    //                    16),
    //            field_t("0xd51589705805975ed2f89d94a2f20aaf3c64af775a89294c2cd789a380208a9ad45f23d3b1a11df587fd2803bab6c3a",
    //                    16),
    //            field_t("0x97c07b6dc2574bdb94067edfe175330a11d459a2f978d8719999e3fa46d6753ec148cb48e73ca47ea90a8f0d66b829f",
    //                    16),
    //            field_t("0x103f383e6c0f3459f79b17aeefba91fc803468b6b610a9f7f9270f4eb8b333a8e5446dd4552b82f6be3edc0a1ef2a4f1",
    //                    16),
    //            field_t("0x154594356a107b75677f6cbdcc22af58be6521cc3e2434e37af027bc08d6af57da71144896c8da1964b2d2bc815a47c6",
    //                    16),
    //            field_t("0x153d8100705fca161622bd795fec898fbcfbb050acab1a6bc69d4bd8b3fa7aa7e1fab9d78c7e134f5dfbd3d12c4a3699",
    //                    16),
    //            field_t("0xf04abad07923986bb968a437d5c8dfc5eda92d864ac5db9d707107e855c384429e821a4c74803e31ba1621582283d16",
    //                    16),
    //            field_t("0x5651fde2b9c014ea5ac06d864c2f2e39403560d97dae38d9d643c25fbb230bbd92a4aa2b410d93c4efbc8d60b21fbad",
    //                    16),
    //            field_t("0xcf14b543b6fe5078c5fe8f8dc3bf364eb8ac8ce8a245e6b33138131c541013d0326324dfb695ffb3a1890c78092b4d5",
    //                    16),
    //            field_t("0x137c79d98c497c68a8c24d4244ef7febe8e5b4617589a82b5a702cfa93ea5c4ed8f33418f3d4e7115804f92283868a2a",
    //                    16),
    //            field_t("0x10666643bd91a1b7f03edca7e2dcaa37f463b337d20b5d59db610487c89da11b62397bc701762741bab9f87ff5059286",
    //                    16),
    //            field_t("0xbabeaeddeb8fc4c7b297d0b0e5e18baf320cd576d14475b349aae908fb5262cc703806984c8199921167d8fcf23cae9",
    //                    16),
    //            field_t("0xcfb75589890086a17b9af5b569643d037cdff7c240d4969d495dd81355c53f0e642f43328ad088ded3c9691eb79fb",
    //                    16),
    //            field_t("0x1454f5c33ac7652ccdf8440407295e4299901c0475491bc354c56c9a9cc9af4ec9546b439f9d01298a449ebe89d9bf03",
    //                    16),
    //            field_t("0x1821f54803ba33db73f7ba8e0445d656de3a5db5154ed51212093d26ac512b01f18dd1eed77c96c0084f3dd6415af342",
    //                    16),
    //            field_t("0x55beab311cbc2884a5012dc582c18c92f429ce59ff3078fcc1b0c3e1c07724e44c5b4763fe31d0347fc816ac16e2285",
    //                    16),
    //            field_t("0x167be8947467537a4b63e0efb62ac1fea5f09e6345ddb87da81aa40a2b0b8c12f3b37f32870266c44155d7ef28dd37ec",
    //                    16),
    //            field_t("0x844fbc43023580ccbd3f5e06bc1538557e54acc62f5680c4fdf8e1a060cea631d3b993f79490eab7f1a355e526eb524",
    //                    16),
    //            field_t("0x1033af94b46108cc721754ef2904acecf5bb9188b80599e9090b20bb257e845465b675cd0492c4f539b21c95055455e9",
    //                    16),
    //            field_t("0xe6d528f843fdda7b1eedaffcc3d5506a17a4340f9c08feffa1b1bf13879399bd50e00978b7199cd6d39eb43ad9ceddf",
    //                    16),
    //            field_t("0xda43e99a185cc8ea8ea37f7523d2a54cdaaac43936aa40cacc66a576518093d07dbf924a6048457861e02ec39235bc1",
    //                    16),
    //            field_t("0x2723f37dbc799b0121b28004e6f5a940c250a03e023033d364e433ff7c882f4202cc8284c717095bcc99ae80f0c8a8a",
    //                    16),
    //            field_t("0x11f16570022bc32021615022409a8a78909ff4976a8a43ef28804790be6c6fe94c41d9c0f07534feeacc110e4f73fd95",
    //                    16),
    //            field_t("0x32b373d58d07674334de73d60c290d00994940e82458cc89f7a7dafb43adc4fc7af3626f9495568deb0e066de26e656",
    //                    16),
    //            field_t("0xc7b603faa7c314bf01dbf291abb8ba37e0ab2ed31b1c27e976699cc6ed5d1bfe585552fac954ab592c9357d34accd79",
    //                    16),
    //            field_t("0x505dc1704a1bde44806aa81e65150b566fec086df2296509cb471a55349da4804673b757ff2e341810d2e304bcb6b23",
    //                    16),
    //            field_t("0x887755836891eeb6de2b33b56cef8ec2298bdb1c85f0d46903715c8fcaf4a5acfa6cf3e53e6d093db87872d336b1a46",
    //                    16),
    //            field_t("0x11193b51afe673f6d6730839e1e48557ea190b2a58068a9d8c31406deea3d685611575c2d67393d618ae013eaca9167a",
    //                    16),
    //            field_t("0x55466012b711343220d672b15ad9a9d0a57af35b9b8163510b8fe223c11654988534206fc4a447ec49872c67c081bb8",
    //                    16),
    //            field_t("0xad7df475e3c536c415ac400d75470808181e84d99a74924550d40ddc2557035449c4ca23685156b89c80c4de9367eda",
    //                    16),
    //            field_t("0xfa430dae323ce54b7115c02f44d7e40c78fec459a9e994cf1a9a658de0f39a73c35612e4a8d15d81d296588571ceeef",
    //                    16),
    //            field_t("0x273ee2260c73494ed192da3c82ad58996605d959d7cd4f61d5c482557450e6520012170d418f7af25b7501ac9c1fff0",
    //                    16),
    //            field_t("0x1118f01e907f96694ba955f3e40961505d698c8b44480030f3c668b114ed204990e32e82394553538cdece75921ebce7",
    //                    16),
    //            field_t("0x13a33dc8032b73284bb57b5cd3e89d320bb662a8c979cb061b943cfc46f57327e592067375305db71d43d1ffecd1345f",
    //                    16),
    //            field_t("0x1922ea543d589cab301ba9880a3efb80ca357568e2934bf1d37c99611d775b7c69dd649317788b9503b96d91aba018eb",
    //                    16),
    //            field_t("0x119b50144b452123d17f6494e8c2d2198afd2973f8633958ce75f4ba60d6c766f6f62c28e927db486f62e63a1a5356b6",
    //                    16),
    //            field_t("0x192df4eb02b087f806faadb10a248cff51423286a6ecc31f35263b4519a2105c50806f017a1d556cb62c228e40df7c9b",
    //                    16),
    //            field_t("0x20ddc55101e75eb6607b61550332cb8642a357c732902f451fbfcc798b8da9fb9fad67e4ba927c3ecf45ccbfb8a99a3",
    //                    16),
    //    };
    //
    //    field_t golden_a[42] = {
    //            field_t("0x83ba3c6346c33fea69f7c6b905cf1b20f0f3bdb4f109cee4cd57c155ef8cc6f02e5be80313073d2cdacce7253b1269e",
    //                    16),
    //            field_t("0xd64a4c5074383e10cedff11de3e7f5633d8fb792d48ef8d3ca8e64046e6b209adc13cc2a46117f10b3e22a5aaeb341e",
    //                    16),
    //            field_t("0xbbaaf862d6b23d24aaf05f6909106be657da3ef3bbd0e98e0f16a9faffeee7445c33e0b03590167722a9ec30c152482",
    //                    16),
    //            field_t("0xdf071482604d362ee88959f9dca20c7a1590080ae12b4651901540352f600bf396c9f7ec968adbb31d6d614a125e1de",
    //                    16),
    //            field_t("0x128ab94fc94efbd2cfb961e31cbfc65d52bcb1979ea06a57a219d19835f24cba08496cacc597b0375fe27c735a65e43f",
    //                    16),
    //            field_t("0x2bd4d9217bbed19da1495775ce5a85b3919efca39792037849c4d110f250b9e04c7772cb4e8eb5e8f847a5a1cd2fbda",
    //                    16),
    //            field_t("0x10af04a4045b87e1231143776fda7dc6d5be309e9f84a4248c22943fcb668b02995fdb1f234ed61e7acf84fd8ff617d0",
    //                    16),
    //            field_t("0x16f8ae7e71cc17a974972facb5bac32b5d7b8c5caa0ef1ced57c1e82a4cf1bc0de4379f84734660d591cae44d4dfaa88",
    //                    16),
    //            field_t("0x7efce41ae4c7701ce9153e8a544d6b98e5b329146af4a44c3b4ef9a8a41efc0a8fae8bb82c3270e0f27c9adf2458077",
    //                    16),
    //            field_t("0x18219aadc32f27fb7c710146fd680636881c1ab9fb2104e1a3f0a5f9bc3dea041dfc0cc76ac746068580a4bf6c30f1bc",
    //                    16),
    //            field_t("0xbf41b9b9fbd7dc0f00370f673200e28f6ad7569317638acdf5b140313d4f47eca88f7bfdb28fb9db13963aabee96d75",
    //                    16),
    //            field_t("0xee8f5e7d6ed58180addb2baccc4e195c62731cfcf80ebc1dd2f60e68f537df4b84297bddb16c0bfba4c2555427fb8ea",
    //                    16),
    //            field_t("0x352bb8e07e2aaf14ab3955cdfd3eabe4addbe3ec8b9f919ddb9cd21268f40a3900df5ac49d165dd94d01bb47a180377",
    //                    16),
    //            field_t("0xdbf08bda005188636e898ddafac1a202ca4d328566de4e800c0c74d9ab5542c40a06e2de6c7d53c0400f723f1d3ce91",
    //                    16),
    //            field_t("0x181b270685b8cab3dfda3ffd55255ce72fc692c368acf33371e1a820ea91f84e74b880e6bce03877ac13bc5596659ee4",
    //                    16),
    //            field_t("0x16aa720e489fbe9050ad47b37730d1f13e32fa8b29a7383c76a9ee862d3b623f8f3ef208b0cf350d9ba9c8cc7d42ccc4",
    //                    16),
    //            field_t("0x1448c514ce2635bb09b1f769f4492f9bd1bb79bcde250043a2af5a975b90a9be6393b2f5f7f6a00f4a3ba10a3043110a",
    //                    16),
    //            field_t("0xa1bf4fb8c6fe63063f7c6b6be75442d85241a2f28396c0eb3904d211cdcaa7655ee2a1cf3ea45a582a8161fb0c2fd3",
    //                    16),
    //            field_t("0x90b81c4e1713cdf724c8049c6d37b5a78600670f3a903f7e63a53a980c2771c4f56cebc4027e7a86f44371c26d2f597",
    //                    16),
    //            field_t("0x19b8bb05fe9f8d4a5aa3384cff750b05c28e81bda80e8dd01938afafcae2882fb460f11f183f4145da07f356d73bb19e",
    //                    16),
    //            field_t("0x12f161c6e4a85102fd2188dbdd17218008cfc89001b602b951b42198799839f933ab35bc02b6574fb8f9765a7988dc8f",
    //                    16),
    //            field_t("0x14f3012531484d4439b77395bc4a36e6ac6932b8d88c11ca6054a1fd8146d9c6fc122e2d7cfa65bf91eb8469745395a7",
    //                    16),
    //            field_t("0x49fec30f32b3717293fcda507b975497e87ea3a9bf8a866e91b107836d23306a238a27f3a7f55eb0937ebc989492330",
    //                    16),
    //            field_t("0x38c704d4fe02129fe5e30df74aabea428e7717ab898d31ae007b98fcc89ca05530047f6b03bd72be62588a8ce684705",
    //                    16),
    //            field_t("0x89216667476a32c6f0ab93bbdfb2dc4d260e286c24f99a131be27b376cf5c45bf973b358301948e8a88c158c842705b",
    //                    16),
    //            field_t("0x2ebd26c8b931b53b487d3e5be4c2c526a181eee1c9361c20033d76840a5be18d254c67d26432b756a5f64a04530f983",
    //                    16),
    //            field_t("0x23ec8ba2d73cd6d13e5361413dc6943bb6641ddc6bf3b79397054b93bf9bd734ef1c518c0488e98fa386d6f0ab1b513",
    //                    16),
    //            field_t("0x170d7a6f5b579ad9b1700e980edeaf9f9fcbd54105facc9be23a510b08f6193336637c5caaf53560d3760871cb3f9165",
    //                    16),
    //            field_t("0x38c7607758e257d8735cfa231c131c5a566552c2c47dc89ca57d1d264b3093976938d31ea47e3dcdea1bc2137c9e0d7",
    //                    16),
    //            field_t("0x42e819d4f4779861e535d5d8dac52fb54bdcd006d982040a122932ba6b09fc07d4e4d02ecd1a489d8019b0631055acb",
    //                    16),
    //            field_t("0x19cc05df16f197658898f1a353690c24b76a7ceb2c9be1200f4942233fea37b00f3c48369e0499353298b4eafaac7c6",
    //                    16),
    //            field_t("0x13b67424955ab81fc276034257c63c547c4fb591866f0cb9972955e37e6b38629f7e2a819cbff17d39d591f64033ca91",
    //                    16),
    //            field_t("0xa867f823fa88cb9f47c5089d338abb76aa9601c8a96787dddabb8fe446be0e1c96b10e122a73371480e59b252910098",
    //                    16),
    //            field_t("0x29c2d6b5b60abe0162d4fe800e5b668786eec4f9e35b5a8bb86334fea71a29e3db2c689d0344264fc82dafe95ff994e",
    //                    16),
    //            field_t("0xb416dc48b50e2b3ecb573fe793a1cdef57c6277a16f62265ac9211ca21b303bc80444294f90b23db65f1838cb314d68",
    //                    16),
    //            field_t("0x8dbeb80c4376570becae0073e7ac742f338d751e010addd56bd9007cb786b72b412c52b3f1f2577fae03b0f010bb4b9",
    //                    16),
    //            field_t("0x6512936c03a7dfbb1320d2228baa14f045f9794f6cbdba8052396bd301b0da40df2de089fd1c26afada32e08eed1b50",
    //                    16),
    //            field_t("0x1df6c0af47a4341fce3848b3008d00c7e90a450e8f14324679b326e6181dc626877ad7f091f41f8ddc339d09a9ffe13",
    //                    16),
    //            field_t("0x1ca25aa8900497b25c7711cfe795c57c5e9f25cf0c724b6e7c24f0d080bb55a2c6321d04ed6d46939e62ea399d34292",
    //                    16),
    //            field_t("0x5f7799489ca3b99f391ffe6de1e1973af1494e8fb0c905733a128f4da8becc7d1624410b0ade1852fbfbb522d24ebcc",
    //                    16),
    //            field_t("0xac39e77fcecba7924109ed67c47585a24307b15b79baa5f1d4533040d831d6f299461d101ad846e61db347af8f6ff2b",
    //                    16),
    //            field_t("0x7a908113cd992ba129c386fcc80f6933a0fb96c0bba5c5dd3acd33cae3fa771090a22f98dd59acc9c068033eeafeef0",
    //                    16),
    //    };
    //
    //    field_t test_b[42] = {
    //            field_t("0x19e3045fbc6887782b491044d5e341245c6e433715ba2bdd177219d30e7a269fd95bafc8f2a4d27bdcf4bb99f4bea974",
    //                    16),
    //            field_t("0x5a7b0a9061b90303b08c6e33c7295782d6c797f8f7d9b782a1be9cd8697bbd0e2520e33e44c50556c71c4a66148a870",
    //                    16),
    //            field_t("0x12fdd56cca2ce6bc5d3fd983c34c769fe89204e2e8168561867e5e15bc01bfce6a27e0dfcbf8754472154e76e4c11ab3",
    //                    16),
    //            field_t("0x17a287f5b714210c665d7435c1066932f4767f26294365b2721dea3bf63f23d0dbe53fcafb2147df5ca495fa5a91c89c",
    //                    16),
    //            field_t("0x197a9a7cd4dec9ef83f0be4e80371eb97f81375eecc1cb6347733e847d718d733ff98ff387c56473a7a83ee0761ebfd3",
    //                    16),
    //            field_t("0x11d7b7fceb9ac688b9d39cca91551e8259cc60b17604e4b4e73695c3e652c71a74667bffe202849da9643a295a9ac6df",
    //                    16),
    //            field_t("0xfa51f2635339774bb1e386c4fd5079e681b8f5896838b769da59b74a6c3181c81e220df848b1df78feb994a81167347",
    //                    16),
    //            field_t("0x44fdd6b9d7d01f5769da05d205bbfcc8c69069134bccd3e1cf4f589f8e4ce0af29d115ef24bd625dd961e6830b54fb",
    //                    16),
    //            field_t("0x12ef358345e9dd320c855fdfa7251af0930cdbd30f0ad2a81b2d19a2beaa14a7ff3fe32a30ffc4eed0a7bd04e85bfcde",
    //                    16),
    //            field_t("0xb86e8e85cc36c270e8a35b10828d569c268a20eb78ac332e5e138e26c4454b90f756132e16dce72f18e859835e1f292",
    //                    16),
    //            field_t("0xfdc9fb105d83e85e951862f0981aebc1b00d92838e766ef9b6bf2d037fe2e20b6a8464174e75a5f834da70569c019",
    //                    16),
    //            field_t("0x11a5ea4f72daf0a54ef25c0707e338687d1f71575653a45c49390aa51cf5192bbf67da14be11d56ba0b4a2969d8055aa",
    //                    16),
    //            field_t("0x4e91ba3b4917fc09f20dbb0dcc93f0e66dfe717c17313394391b6e2e6eacb0f0bb7be72bd6d25009aeb7fa0c4169b15",
    //                    16),
    //            field_t("0x193aa63572a47403063238da1a1fe3f9d6a179fa50f96cd4aff9261aa92c0e6f17ec940639bc2ccdf572df00790813e4",
    //                    16),
    //            field_t("0x769b7e5a294523d74115c86188b10442bb3b36f29421c4021b7379f0897246a40c270b00e893302aba9e7b823fc5ad3",
    //                    16),
    //            field_t("0x25a80a071d7b14eb6c004cc3b8367dc3f2bb31efe9934ad0809eae3ef232a32b5459d83fbc46f1aea990e94821d4607",
    //                    16),
    //            field_t("0x15e7ebaf41b1256d5c1dc12fb5a1ae519fb8883accda6559caa538a09fc9370d3a6b86a7975b54a31497024640332b07",
    //                    16),
    //            field_t("0x1062f9741c75f67e290535d868a24b7f627f285509167d4126af8090013c3273c02c6b9586b4625b475b51096c4ad653",
    //                    16),
    //            field_t("0xc8369f37a47ce41aeffafc3b45402ac02659fe2e87d4150511baeb198ababb1a16daff3da95cd2167b75dfb948f82b",
    //                    16),
    //            field_t("0x1856384de8147dc9af479f2936631b3e6147db98a44a4d468918b6824f4a353e7430051376e31f5aab63ad02854efa61",
    //                    16),
    //            field_t("0x9f9a935dbe3b837ad45a77cd7acfcba9cd831118026938ebad83042e64c3e094d2c3a6866aa110edcb1f9a6b031f3e",
    //                    16),
    //            field_t("0x9dd62875f4a3ff5eeb4de7afbc80976b0bdb4fe4a21f7035dd1d1839c4a67c31e5bfa6bebe42b82f5ee773384eaed20",
    //                    16),
    //            field_t("0x13a2418296b7ff15e3b904bb354fab10769d70687c61838ca325c0ae6921f4be0f5b8e2c73907cfccfe330cd04065c82",
    //                    16),
    //            field_t("0xfb1ba8ec146a389381b37241398aa12b93b8e54ef6de2014e4a4f6a5f768331062e2b1048cbc656015d313712e3db4d",
    //                    16),
    //            field_t("0x6c8c8f2f5acd6d1641f6abda418067b55a0b0a6d99e3ea39dd5a943149c59af1f7a35fc1f2c53494110c7681f29a9d6",
    //                    16),
    //            field_t("0x94dd8e97f799eafb4af8aa6b9387e620b0ddfd1c6f75c81786a7648a8beb0039e412c9d0650b1531afb9094b151ad76",
    //                    16),
    //            field_t("0x1706870c7a415782dda4a33d86bbd79d7bf4a20644d96a455ffa441acc790cf4243725d175004afff5a8513f5b8aaa47",
    //                    16),
    //            field_t("0x50139913b4bee51650936624bf8b43aae2321e6d60476177dfbad9cee9b14f66b415a7ecde9e144ba588a82f4670328",
    //                    16),
    //            field_t("0x174a251395fd948c1595661deb2812efb2ff0a7aaddc220eb2140e476d7ab8b88c6e800b4268636f98b7df4f7d214e98",
    //                    16),
    //            field_t("0x92f02268624857a2c2af60d70583376545484cfae5c812fe2999fc1abb51d18b559e8ca3b50aaf263fdf8f24bdfb990",
    //                    16),
    //            field_t("0x7f48a0584396eee542f18a9189d94396c784059c17a9f18f807214ef32f2f10fc9779298a77fac227f0ce161cb0a313",
    //                    16),
    //            field_t("0x19e952b30785c4f2b807c3ef70f162c07776fa45250f5218ba9acb5192ccfd66c3aba2f4c8669cb35bca24abfec109fc",
    //                    16),
    //            field_t("0x81e7c9367a87d17462198bf7b8263990db6eaa3829a999364923e422e304ca0bc9bdc7fe1becaea621cc2b4985cadfc",
    //                    16),
    //            field_t("0x16d305ba54aa8ea28c2e111af64426d65c2ff4ed78f5b4d4a5dea412b4949aea69825e3fefc9998bbb3308f3b5e0b621",
    //                    16),
    //            field_t("0x60313e19f0a2d2e88692706399986ccb9cd52cbda217886d1d68ccdc2f8cc9414d78322a8927986e976c587bee07a22",
    //                    16),
    //            field_t("0x10c11cc176ecabad501c709102f898ebea3b776da284462fe185133afa2c02f261ef2a6aaabeab54d0f866ac6718e996",
    //                    16),
    //            field_t("0x6edca73f1777e346725e00704631dd7182c8eb9d0e9f5042d680ac5a66bf90e775fef72e21ac47be81e5b84b629e04e",
    //                    16),
    //            field_t("0x11d8de9bd0cc8d5263d6004419b7932af8fb9528e27a5ed8372af4d0e7c3401562c2d9009b3fe92291bb840cbabe20f4",
    //                    16),
    //            field_t("0x1394e592cddf5b307d63c5693105c746948979669613006debe007a9be86e2aa463fb7d63314ecfbcd3520dbc45a2919",
    //                    16),
    //            field_t("0x16ce8d16779301792c6ee00e90eb93d08370e22040f1e6977b6809796f34b4d4adbf03f59cc13f0e022c06e7234ba74a",
    //                    16),
    //            field_t("0x15692b56d6522e9e887d94b87c44b0afe7da77d100c70d2659a92c2a12a4aef8c299cf2cf77ef20df8ee4777347ab734",
    //                    16),
    //            field_t("0x1bb04d22286db61b83551abdd30aefcd55c21c9c99481d4a008a77daccc5a7d415adfd1678cb80bff6b2fa5c8ccdd23",
    //                    16),
    //    };
    //
    //    field_t golden_b[42] = {
    //            field_t("0x8654b52d0707000817c169806f1b85af7d28543cbff20352f67a96e96458146ad68ece82432ebb10cd58e48bf8c4c2d",
    //                    16),
    //            field_t("0x94ca5866b075048e0d064c262f431961884cd1c04eff9a7560b2cbacc6501aad8aed89af8e8616f1c84ac58673364a1",
    //                    16),
    //            field_t("0xe55a6a8fff2693dd12fd2359b4a4d817cc0a28c95f5efef8b5814fb83a4090d84ed11df536f84ab373863504377da6d",
    //                    16),
    //            field_t("0x133f7e7840d1939dac2be20f882214b1021b5f8a52bd41533fbc59609d8071a018b20498dc30a599aa65c2e783ef53c5",
    //                    16),
    //            field_t("0x160e0f8becd3b7a59b9f9130a6b1b9c21f96b34335cea28cbbb14cfe362d8b7297afde99b8be00e923feb148ff4f7370",
    //                    16),
    //            field_t("0x10faf7e2ae70e2a23268298adacc90f508b9062d2bb82b5fe4a585135d0024167cdccbf4409f83e81f08e7ed2c865cd0",
    //                    16),
    //            field_t("0x16718a17ed4b00071c469575e5cf6c102458005aa745a28257239f485d49d66c12c8d3367762d201e3fb893aa181bd2c",
    //                    16),
    //            field_t("0x68a2441d6f44e6bcb752fee35b3cfc5216484bd62e95ee469ce6d32bc43c5cb56647216434524b75d672c216a5815df",
    //                    16),
    //            field_t("0x7630121e906aac667d6616a830c2da4f5b1b94575c5313e3ed4d85b64b13bb1cd47b6ecd244f0c8ab509774395ed6f0",
    //                    16),
    //            field_t("0x390fcbe6661899f777907b25dc76f3a0d3ffa0f8d7a5dcc379509bca6794e64365cc0c37e9888a0ec243f6ebd3b888c",
    //                    16),
    //            field_t("0x144ab79f34c5ef22b6df5c0cd35696b2d2c1a01b606d8287f8276507b17cc38968e72723e0ba6be24fa2b86f3cf3b5d4",
    //                    16),
    //            field_t("0xafc37d3f6403df04e60854d7bef3f3a2a7fe1573e47a76afa3bae5e1fef7d985153817afca6d76e1dd4bb3d601610d9",
    //                    16),
    //            field_t("0x41fb99316d9de8a97fb5871f81f6ad6828ef2d666545feb1bf7bca8464b114520a2cadadc22152105624c445a74ab71",
    //                    16),
    //            field_t("0x12a8d78ccccfae3cf118d28093abdea7c91b73dd03949d03ab20b0aa0a7428d3f68d74d98ff8d6397d3118590d216761",
    //                    16),
    //            field_t("0x92686dcef27a1a8a026933b1375c309d4f57abb89ae309837250bfbe61bf6f6b9571dc80cbf31cc04544e5e1df7d07b",
    //                    16),
    //            field_t("0x1888e64129095d7a761942076ea5b3e734a208d8ca89d74fab135b1c816712ab2b553ade566d8ae9dab633263f6a28cd",
    //                    16),
    //            field_t("0x15f1755e993624fe52b33d4ab22be7c69146d17d8afdca6b49b10e574d2832e65986bf7ce6d3c10808e5d0406e6f9183",
    //                    16),
    //            field_t("0x1037606640b77709e8736edf566c9ff60bdb5b426207cef04630fd5cd8451fd54d2f65f3c4472928614097d59a1cc542",
    //                    16),
    //            field_t("0xdf23a1e35e2888fed81e5c9b978aa02e85ccc7476f64744ccc1d2306f75f5f7c44e67a258b9ce215576113f679bd573",
    //                    16),
    //            field_t("0x18e21ded4b702831343277f374d23b0355c388eff54f7f85bc41d2258efdd69c1c9051f7b32c6b444bb075c414054dd9",
    //                    16),
    //            field_t("0x46cc638743285f38a4af68dfb3dac39ff8e6e526a7c45a023fc1e5a6f2d8b06c2c4131c08c5d71e92b02fa9277ea44f",
    //                    16),
    //            field_t("0x1873fc8bd43033682c56a8357e9278175726f93aa9b3d5330b83cafcee48120abeeae1521d3c5c673e113d0369c60c49",
    //                    16),
    //            field_t("0x1ec0dbf42b8e942f7fd560b0f47b8e1b36de5ae34cccd0316727933b533450fd2651a85de74b7da2a42ffe44151e4e8",
    //                    16),
    //            field_t("0x218f595216b6b349187b930393419aea74ea1d0f3644b0975f373bca1e107edb5e8614b117cab8387910f7cc6925848",
    //                    16),
    //            field_t("0x4cc574df58470f3e6fdfeb15473ff07f67df0e52c7be3e53b6374a8a0bea0a90e26dad748c8380aaaad15a3786b6993",
    //                    16),
    //            field_t("0x199419a6a191583ca29d6e713bf00fab93ccd0788cc73fbe0bd641b9ee10b409b95420567402aac7edc23f72558dd697",
    //                    16),
    //            field_t("0x14a9e696f1f136a69b821edcde4b0e0ef6115dd70067687de3f93594fa7b2b98c4581d39a641217e3f7022171693c22f",
    //                    16),
    //            field_t("0x17bb22fce2d160f6baaf834a73b7906faa7fcda682d00293fd84f5b6967973409c249039796fe90601d9e9ba7cf969d1",
    //                    16),
    //            field_t("0x5e0ae528180bbb1236a984149e0bc7b60327e1e4fa2fc2d587fe5ce6843c9df7f036c67cd00662fb15f2c3ac7e65944",
    //                    16),
    //            field_t("0x7e47ccd8bb723a87d5e8baa7b7acee31f9169d31c733b9bcb54d637623898b0d0758feee87634c8c068f02219a72d91",
    //                    16),
    //            field_t("0x2ef2172e43236dd504783e9c1c2fd1ea9dde1b098d5b9531d453fadb07df5be739263b4cadf436d41a8bff83eeb98d0",
    //                    16),
    //            field_t("0x17f68c627d70897c9b448cada3667df2006174b850f1fb6d6c0c6b1f04ccce82712e01d7d468d15150ba32ec84715675",
    //                    16),
    //            field_t("0x1af78ce61024ab6d61000a3dca2b8c5db4b8c0e228d56ba49fd67d1e3c2b7fcd385c13c3b48b7d4233ef2100a55ab40",
    //                    16),
    //            field_t("0x157d3813c20f59c74045d2ccff4e00fb2a70f6a2911a6894a688a4cda99f2ae6fafb8ff4746a4c252f65253b64ed37f2",
    //                    16),
    //            field_t("0x11445efca953711431aa8df9ecf4d53be1231572fa2a151bb25d804c7d965972d986cd0f6dee22bc6493121a923dee46",
    //                    16),
    //            field_t("0x11d8477eec7a958d98588addb401a154c0020c71a7dc529211eb3e574d62d4f32982fe70377795c6e5a008d8bf5d7d04",
    //                    16),
    //            field_t("0x89cc4988338053328d5eeac8eab1bcb8adb6ac6048534d664e8a31c91faa5a22ece9965fdd49ec4ef466ee3d01c2a58",
    //                    16),
    //            field_t("0x170b4849e2043e120c406a3a2ce815c807fb5ca8e1822da14872270ff1222a1a34eaf1796ecb2859947312c745ce6ff8",
    //                    16),
    //            field_t("0xb9a0dd2aa97e3e30334ca428e9c429acee56bfd2c74f18960a4428cd110eec23c844412f0b680b73c44081582459c3a",
    //                    16),
    //            field_t("0x1111f23772f0f716dec6078274fa0c8dadd45989f585a0f05414f6bbea2b1e2384f7071992d55f6db5a44dcbfcf6301b",
    //                    16),
    //            field_t("0x7e3454c7325b280d613c5ae1980165862aa54018bd21e29d80f77e24acb78827019abb827055a10c3b2f8a4206030e9",
    //                    16),
    //            field_t("0x15d7faefcaec3452aabb64739e4d9f9fef8d9dce4571999f9fd87845fc68f31dd1e4baa278052b69327e3dd8958c6eca",
    //                    16),
    //    };
    //
    //    field_t test_c[42] = {
    //            field_t("0x1361124710c67fd994b2b8fda02f34a6795b929e9a9a80fdea7b5bf55eb561a4216363698b529b4a97b750923ceb3ffe",
    //                    16),
    //            field_t("0x114fa87678633074b7970386fee29476311624273bfd1d338d0038ec42650644781f9c58d6645fa9e8a8529f035efa26",
    //                    16),
    //            field_t("0x15e871a97524d6af51e8722c21b609228ce6f2410645d51c6f8da3eabe19f5803e0a813bdc2ae9963d2e49085ef3431",
    //                    16),
    //            field_t("0xc674a1de0f9e038eb8f624fb804d820984181177906159644f9794cdd933160d2d5844307f062cec7b317d94d1fe0a0",
    //                    16),
    //            field_t("0x44af313ef829c88f6ced90a71d2af7293b05a04cd085b71ba6676b3651c52536d4b9adbebcd1f5ec9c18070b6d1308a",
    //                    16),
    //            field_t("0x18ed57e86fa84dcaac0ae4e2f729b4c8420b0ebe378c74dc7eb0adf422cedafb092fdddf18f2c41c5d92b243e0fd67de",
    //                    16),
    //            field_t("0xd0adc8b95c76ab488bafad959d5450592f3277b62c82185d55ec1a581daad106bd0638b4d100d8fdaf0105ba06c05a2",
    //                    16),
    //            field_t("0x1362df01fdb17f5447997b6bdb3d115007564931edcf6109ea6d5547ae96619356363b4be779c4703b7dae0495918695",
    //                    16),
    //            field_t("0x35496c891b1078e926baeafe79a27e68ab12c32f6f22f41538e504edc52bdcab2d87d5e29c0e596b2109307abd8952d",
    //                    16),
    //            field_t("0xf6c8e3c103ef3c21fdaf62548f2f8ed445fad2a92d3043afcf249f3d4e441c3a20ab57c360c4979a7cf94d7b6bcb650",
    //                    16),
    //            field_t("0xa4ddee26988f4fe5a8181b691406be110d7c25ccf3d0b35815a3d516a91f397bc73a83fd63ed5ba385ac4bda9bf98d",
    //                    16),
    //            field_t("0xc16fa050b813439c2fa7b1f9d5200ef9ae085bf0b500a3f1e715c0bdf6da8e16a4a5ed7c4cf8b966d59298c4b3c74f8",
    //                    16),
    //            field_t("0x1270467ff9e48403c67523f81633acf47715c45fb0af1e3ec007b1be18302948d04999d54b9693c961cadbcb7ebb70d",
    //                    16),
    //            field_t("0x954e387686e80a9f8af8c793287d050f2ead0a808085f68891ba6ad998a0e311badb4f513b45a3901da01354f468978",
    //                    16),
    //            field_t("0x46d2abcf56ab44e5c35d7ed5057326c56fe09f7de26c45bfad9d3a90add12e3b09258ce27fca832436c6d2a9c4792db",
    //                    16),
    //            field_t("0x15ca835a987c88bbdde8bcb9a4d5e41562dd8a70852380c4deb135fa75dd67de6072c48f60b6cbb1dc98da8ae58b7c6b",
    //                    16),
    //            field_t("0x16e65ed2b866517ea260db3c6e6291d24573f54181cc8265cfbf40b8f0cc8de3f90ee1f29ec096091a4236678f2bbba4",
    //                    16),
    //            field_t("0x193a8b3802eee0ab56c2adc08c65f0674d90f55185689935421b8cb9fa50ecd76ffc71e44d14075defba436b3cd5b002",
    //                    16),
    //            field_t("0x1445e6ae0f616fb4221de112a1d6956c96d604649da4ef01606363ab05222fb2509bbd4d947899a4fcc9e97f6a4b398a",
    //                    16),
    //            field_t("0xfaa01efbcefd0a747679714b4fab1019bde81635a427c37ead6b3cbade562bc5a58b185775c303c551b7f9da0996d53",
    //                    16),
    //            field_t("0x98e6db674d0df35a0c2995f40498cb35e819615f69b31ce0570ceeead0faadaf47076520f81f60c96e1689405adc012",
    //                    16),
    //            field_t("0x873cff5987aa6bdd805f5d25e80dfffc2134f15500b2f292f6c48f65d2c29382d6b76db51ed2f1599f8eee797b95810",
    //                    16),
    //            field_t("0x434995bbc344f4baf091db491bae46af8abffd606e44edfd0247e4cc5b3b5d31ad8df8e608d9499c98c9e514ce74655",
    //                    16),
    //            field_t("0x14c905fc6f6b8421ad9593b42ff9134d53e9cfd23d1b208544f5f725cdc656fba75a68a138f83d748000b3d94f5d410d",
    //                    16),
    //            field_t("0x19e99847703cff0b39763c0bd562ce04acc80ab55570e103f2fb6eee526c5cc599c90e881a124c1518d675a4b2b47ae8",
    //                    16),
    //            field_t("0x8a8c114737b6ed79182c3c8e288b16437d02410a675a109bdf84ab55632a44614777e962b56363cf5efd434db045aaf",
    //                    16),
    //            field_t("0x12650cfad6a66353d6118814ce88f3e750ad12d330d884adf52407cd8795ad0f08ae412f1ef491a6c9794969399b6cae",
    //                    16),
    //            field_t("0x12dd06bc5864742b9e8c8b63ce66e9ee15e58ecba4560002d3f44c52cea663ee57116d4c4751d092dd1d40962eff8330",
    //                    16),
    //            field_t("0x94af2566aba54efa25994fc58aaac8176f7f138456bb11bd997c6f7cb3a88f684b5b4de4abcc4e46bd881fd21334eb1",
    //                    16),
    //            field_t("0x108c01038daf051b79e4444ed6897d8fc5ab8f2f33dc30a8f1233c76f31b6928298956cfca65f8e9f66ad57e1464135",
    //                    16),
    //            field_t("0x11fe7550b8a276b3e99c6c8cf68bc281eb8143249799084f8911b0496b3952ddba4a636116ce129dc8d4dd13a3b3bc5",
    //                    16),
    //            field_t("0x19454d693b337fb6e0d0eb1e651171de230ffbce5856cfa32ced3f5ec81bf908325f276b196b0c7cd8e5f01e752f00e",
    //                    16),
    //            field_t("0x112f12fb86640cb0051490eaa9b38f203d3221cc4cc576f280d0dfba2bfc7ffd1eeda989becbde017b25f34a035d7018",
    //                    16),
    //            field_t("0x114fb6cfdc960f12f8d45cb940a230e6201a95cc5762e3571d140ed89cb6c63de9bfec51f06516210da1920569eb8cb5",
    //                    16),
    //            field_t("0x3d09293d11a8404e33c37f188ddf9181f49e090328475a738868e9b5a124b1d0fb5d240c846756acfc1d5507a299d75",
    //                    16),
    //            field_t("0xf999c6801ebd454ebb679b4d2d0d09720e469aace595c72e3bf018debf8e3d946150f34caab02c83d4d071b2bda7713",
    //                    16),
    //            field_t("0x10a1273b86faea979e3b164d44c20f283f8de0e1457a46a7c1a9425a0cc8557466789723dcd06050922631c6a0ec66f4",
    //                    16),
    //            field_t("0x17a58aa207a1cdec6767d960e0992e3db65d2a400768817d1cc755ac6c88cfe52b7bdbe790ff9b20d0c8ea76c48ae1a",
    //                    16),
    //            field_t("0xfae02bf808aefcf83f18d61160c7c39b674c4f4dabd2a4c08736a21f985732a7b99a1261183c1860cc1e0331fe78155",
    //                    16),
    //            field_t("0x87a4f745c5fa7d24ddab100962c470663bf2ffea59c217962c3995a59ee1cce125fdb0f50884d442833e1d550de9399",
    //                    16),
    //            field_t("0x197309b461574803b9191d5cb74e950400e4a64e8e36f2c720ab0e211fae68cf6dbf42c0542aaf09fcef0f2a30eabfee",
    //                    16),
    //            field_t("0x145fa73b615906a78a943011c859e78da6782c0b9abc3e5b75f828935f8eec2c0aff87582db5db0591157d5f1474683b",
    //                    16),
    //    };
    //
    //    field_t golden_c[42] = {
    //            field_t("0x12878f9b74162a999aa6c5f884e9200a570ac2c911b6aecf8bdae8c3ccaab1ea8c4871e339d8a37f6db077fea7a034f0",
    //                    16),
    //            field_t("0x154cb5591a89ea72ee8ae5eeb5bd2f51de1501a0ec26d485a762f709e0ae2ddc2799e23f213e1d360ccadd9437626f15",
    //                    16),
    //            field_t("0xe3283ad26de18a86603de14b0a134e0027e15c1ed094234f41808ab2a07b2b649b253d9059ba673f099d7e56f7745cb",
    //                    16),
    //            field_t("0x526338dd0a9aa5612f63ac9df14582e7421a736b9aa92a2ed99834f1dcedbf60677b7a35da0665c2ca4f54ee547ea43",
    //                    16),
    //            field_t("0x1262851cf65aaa2aa41f20bdb13138592c7742168c7301a8670ea14670a6e3fcf472b120ef2a6d479f8cd05f961f89a0",
    //                    16),
    //            field_t("0xae0e07e21a3e2da4de49e918409d29097fce8e49b1678c90216b632669d61636f24a3fea3db9ffd78610f5c8f5212b0",
    //                    16),
    //            field_t("0x59c34ef0c2d3cea58fa3da28ed660c62377367de39d7622eb76cced581639892d91233f2c8943add9b12b2cce64cce9",
    //                    16),
    //            field_t("0x16998d3741910eca0501602bd6196d16377e61b058fdd6552318fe049dfedba00560e3f5847ea79a4f564aa669e8c900",
    //                    16),
    //            field_t("0x14b22f6136039bfba2ae9137b4636ec08d95346f524e3ce4d6302d5ab5ef95f2a2afd905845ee374e33930c41a546891",
    //                    16),
    //            field_t("0xa0c82ceff0a822b79ae7aab145b14d2ab8e7a5bdcf57105264c4bcfa6b6033a512fad26f1050d38e5c0c562dcf28393",
    //                    16),
    //            field_t("0x122c3d543e9047783a2b545a68b9a9419f0e880e80ef45f9279cb2c08ffe9f45ba9e3a539d5ee0b9c68687d8e1f6880",
    //                    16),
    //            field_t("0xc2b9d152683cfad96c6f94030a1a87d76a4e633c3f7c501b1fc4a4b15ead44e6355587caeb748a6199dfa47a62dcf05",
    //                    16),
    //            field_t("0xaa75facde204fc0382a32b7434fd3e8805e63ed3191219a26a1548eea293e0a3ab5013b679f56e6a5f0fcaacf61ba5f",
    //                    16),
    //            field_t("0x135e68d34e35b7c36b7549d457ddab3de54f05af6f1a9ba74e9321523fcbed85485494334c528a9d02697ba4a99d7d99",
    //                    16),
    //            field_t("0x14eb9f73f623f81614d53fc6b8dc9feeb79653c56d279579554fb2b6eb363ef90625202095a45e2c8ca1a7fcc205fde7",
    //                    16),
    //            field_t("0xa6d1198633a773822089f4cfc422cec51fb06933d84d623ba3fdae0337bb7777e90c860b835659e9b8708cf4a39a482",
    //                    16),
    //            field_t("0x4f571cafddf4cfde47e04bc29e7bda1c73c5c4620293b6c159ce433225082a4f2724d3e00cd9e62b4984b014a3d9204",
    //                    16),
    //            field_t("0x149c8e570ad39d0fbe273a131c66a2ff0ae21a361e1564a5840d877abf5f2feee812d17a056292f083d661a95d6de00",
    //                    16),
    //            field_t("0xf9613527da09b794aa1172f675250ed6d2904dcefa4df26dedca46b265970c727242c93222162fc55089ec9402bb6a1",
    //                    16),
    //            field_t("0x4bf8caf8a0dd66cdcb0856f91a8672171e06ee51771aa302de372efe9c327a34886c26ad85c8cc90c15232bc0d1f6ea",
    //                    16),
    //            field_t("0x7a2f3ee4ced896c93b86b3c8cf7dbb1e04b5da8af7c191298fc5d0a85b64cae841b14d924bbecf48cea7b64510c3c7c",
    //                    16),
    //            field_t("0x14ddb74e384e6b63b99b9549651e6d88daf470d63f3cf0cc9bbdda95d575c0f6621371dd0e208362973f2c9080bcc849",
    //                    16),
    //            field_t("0x15dd875a558b3d4e0b033e68b6e24c3b3d732e25fecb6b26b75f2e5695cf75865000aaa0603f634fdd4be50ed73aa3c3",
    //                    16),
    //            field_t("0x65328b8720569311e29a58d30c27a3df0d9bfe6bb856a1c7ee72c8194bbf77bd984cf1cc0386863e562c31ace8673b2",
    //                    16),
    //            field_t("0xc79d0a7ca2a83423173727ece4e81ee508fac5ff9d80bf2c7f3d17b137b65d5e543b1da24371ecefd36c0b2ef195f84",
    //                    16),
    //            field_t("0x60bb0389d6deca20d20588bc7e5463455c580abd0d0c4ae75e7a32b718b7b0ae4b0403ce3c84dbe78f57b18c4e64d98",
    //                    16),
    //            field_t("0xaf5d616e114e0301f56689fc041790db8c4ef8b9aa661794521851188504a77f1d49c7f0bb691f97f93bb5aaa860430",
    //                    16),
    //            field_t("0xab7b60e57f09f6eb138053ea712363f0771430e10da57d867461714cf89f51a9a7f9074928c1acc221e43ad749ac22c",
    //                    16),
    //            field_t("0x2030898ed5454f6f5098682b2a31ca47532390f0d81e14160496a3ed198d84ff33e117719cc12a2546eb60c1f252841",
    //                    16),
    //            field_t("0x1914e8265563d1aea1dc686112ea98fb99a0184c3cd6687d3d7b29d9e0863e6fa85f047a688d1d3407d0df5ec68f7b7d",
    //                    16),
    //            field_t("0x15f69a7986b65da028b42d7e52dc6a277ed43fac9e8fcea5f1af388154fe93908918952143df6caba8e83ce94e761d24",
    //                    16),
    //            field_t("0x6f25f59bf20c38b5f6bf45e4fb78b90691fae4f9e994dabe5a5629a27732abcf4ec2173bc0618f26edf9727bb49bc3c",
    //                    16),
    //            field_t("0x187b8c6c5b7f296d58cff5a153456f8a6e012fa48dfc1d7276c44617a2c37d1c79a60f0b47a4188e12c04ea363237351",
    //                    16),
    //            field_t("0x1642ab3e21d52c10b89c2a6b9cd22efd82620eb371eddc238ef83c9516c060eef6fc51ed304bee81025f470bac71175a",
    //                    16),
    //            field_t("0xca9c06f42061af50b87a7a3f199aa0c483e6af1795d87dd8213e5f1a5733159fc47fb26b61fc04d885507f931e409d1",
    //                    16),
    //            field_t("0x113dc6b906a829281950d90d3bcdc13ea3a9727fea59ef4082c22a0c9a02c454f328bdb5f23a32b111b4ff74ec9ef74d",
    //                    16),
    //            field_t("0x14fcb41fc717adf314774bd72c84590ce3293656474b89a8840f622fc93b563699e5b177c3e2cd9ca1ba9bbbd65080bc",
    //                    16),
    //            field_t("0x10f82472944de34cde7d3ccb873b62b48ce18938285cb45ac0674989e28c786dbc741f6b6305da4d321f496d237fe385",
    //                    16),
    //            field_t("0xc39322b49db94a366d9081cea1df40beaa41d8075312331dc0cc8312ed6074b674490c878805849dfffb45e691f985a",
    //                    16),
    //            field_t("0x13c313f10c5e270b2ef57cac89cf35b65d26b289e208d8219d6a3123cc9d0c031cddf3dd2e5837dd2574ac8584492db9",
    //                    16),
    //            field_t("0x153bcb971f4c449b6c6923a00b842c8dd469cbd14cf2d4ba7a9b65d4b0de7aaf4f859a71974c559e27b5e50c7e357015",
    //                    16),
    //            field_t("0x73b337e1e4d4caeccbe818e33e00be032d84d9f7bf32a7a74fdfd74c8545873e469468a1a726d1db60bd1903b3379f2",
    //                    16),
    //    };
    //
    //    for (int i = 0; i < 42; i++) {
    //        a_in << test_a[i];
    //        b_in << test_b[i];
    //        c_in << test_c[i];
    //    }
    //
    //    for (int i = 0; i < 42; i++) {
    //        assert(a_out.read() % base_field_t<bls12_381>::P == golden_a[i]);
    //        assert(b_out.read() % base_field_t<bls12_381>::P == golden_b[i]);
    //        assert(c_out.read() % base_field_t<bls12_381>::P == golden_c[i]);
    //    }
    //
    //    std::cout << "PASSED" << std::endl;
    //    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    std::cout << "-- [k_bls12_381_inverter random test] ------------------------------------------" << std::endl;
    mpz_t P, R, R_inv;
    mpz_init_set_str(P,
                     "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab",
                     16);
    mpz_init_set_str(R,
                     "15f65ec3fa80e4935c071a97a256ec6d77ce5853705257455f48985753c758baebf4000bc40c0002760900000002fffd",
                     16);
    mpz_init_set_str(R_inv,
                     "14fec701e8fb0ce9ed5e64273c4f538b1797ab1458a88de9343ea97914956dc87fe11274d898fafbf4d38259380b4820",
                     16);

    gmp_randstate_t state;
    gmp_randinit_default(state);

    constexpr int samples = 42 * 200;
    static field_t test_v[3][samples];
    static field_t golden_v[3][samples];

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < samples; j++) {
            mpz_t sample, inv;
            mpz_init(sample);
            mpz_init(inv);
            mpz_urandomm(sample, state, P);
            test_v[i][j] = field_t(mpz_to_string(sample).c_str());
            from_mont(sample, sample, R_inv, P);
            mpz_invert(inv, sample, P);
            to_mont(inv, inv, R, P);
            golden_v[i][j] = field_t(mpz_to_string(inv).c_str());
            mpz_clear(sample);
            mpz_clear(inv);
        }
    }

    for (int i = 0; i < samples; i++) {
        a_in << test_v[0][i];
        b_in << test_v[1][i];
        c_in << test_v[2][i];
    }

    for (int i = 0; i < samples; i++) {
        assert(a_out.read() % base_field_t<bls12_381>::P == golden_v[0][i]);
        assert(b_out.read() % base_field_t<bls12_381>::P == golden_v[1][i]);
        assert(c_out.read() % base_field_t<bls12_381>::P == golden_v[2][i]);
    }

    mpz_clear(P);
    gmp_randclear(state);
    std::cout << "PASSED" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    return 0;
}
