`include "basic.vh"
`include "env.vh"
//for zprize msm

//when 16=15, CORE_NUM=4, so SBITMULTISIZE = 64 for DDR alignment
///define SBITMULTISIZE       (16*4)

//so means: find the Smallest bit of One value
//sz means: find the Smallest bit of Zero value
//OK array contains bits, 
//when so_or_sz == 1, value is One can be choosen and clear to 0
//when so_or_sz == 0, value is Zero can be choosen and set to 1
//when set, set the corresponding bit to 1
//when clr, clr the corresponding bit to 0
//OK array select so, use two pipeline for timing
//pipeline A: OK array, do so for every sub-array bank 
//pipeline B: do so for array of bank
//pipeline B1: output the index of so

module zprize_so_pipe import zprize_param::*;#(
    parameter so_or_sz    = 1,      //1 for so, 0 for sz
    parameter reset_value = 0
)
(
    input   [(1<<9)-1:0]    set_clr_onehot,             //when so_or_sz == 1, set, when so_or_sz == 0, clear
    `MAX_FANOUT32 output logic            valid_B1,
    input                   ready_B1,
    output logic [9-1:0] index_B1,           //output the index of choosen onehot bit
    input                   clk,
    input                   rstN
);

    
////////////////////////////////////////////////////////////////////////////////////////////////////////
//Declaration
////////////////////////////////////////////////////////////////////////////////////////////////////////
    logic                   clean_bit;                      //when OK bit is choosen, set to clean_bit
    logic   [(1<<9)-1:0]    OK_A;
    logic   [(1<<9)-1:0]    OK_so_onehot_A;
    logic   [(1<<9)-1:0]    OK_sz_onehot_A;
    logic   [(1<<9)-1:0]    OK_onehot_A;
    logic                   ready_0_A;
    logic                   ready_1_A;
    logic                   ready_2_A;
    logic                   ready_3_A;
    logic                   ready_4_A;
    logic                   ready_5_A;
    logic                   ready_6_A;
    logic                   ready_7_A;
    logic                   ready_8_A;
    logic                   ready_9_A;
    logic                   ready_10_A;
    logic                   ready_11_A;
    logic                   ready_12_A;
    logic                   ready_13_A;
    logic                   ready_14_A;
    logic                   ready_15_A;
    logic                   ready_16_A;
    logic                   ready_17_A;
    logic                   ready_18_A;
    logic                   ready_19_A;
    logic                   ready_20_A;
    logic                   ready_21_A;
    logic                   ready_22_A;
    logic                   ready_23_A;
    logic                   ready_24_A;
    logic                   ready_25_A;
    logic                   ready_26_A;
    logic                   ready_27_A;
    logic                   ready_28_A;
    logic                   ready_29_A;
    logic                   ready_30_A;
    logic                   ready_31_A;
    logic                   ready_32_A;
    logic                   ready_33_A;
    logic                   ready_34_A;
    logic                   ready_35_A;
    logic                   ready_36_A;
    logic                   ready_37_A;
    logic                   ready_38_A;
    logic                   ready_39_A;
    logic                   ready_40_A;
    logic                   ready_41_A;
    logic                   ready_42_A;
    logic                   ready_43_A;
    logic                   ready_44_A;
    logic                   ready_45_A;
    logic                   ready_46_A;
    logic                   ready_47_A;
    logic                   ready_48_A;
    logic                   ready_49_A;
    logic                   ready_50_A;
    logic                   ready_51_A;
    logic                   ready_52_A;
    logic                   ready_53_A;
    logic                   ready_54_A;
    logic                   ready_55_A;
    logic                   ready_56_A;
    logic                   ready_57_A;
    logic                   ready_58_A;
    logic                   ready_59_A;
    logic                   ready_60_A;
    logic                   ready_61_A;
    logic                   ready_62_A;
    logic                   ready_63_A;
    logic                   ready_64_A;
    logic                   ready_65_A;
    logic                   ready_66_A;
    logic                   ready_67_A;
    logic                   ready_68_A;
    logic                   ready_69_A;
    logic                   ready_70_A;
    logic                   ready_71_A;
    logic                   ready_72_A;
    logic                   ready_73_A;
    logic                   ready_74_A;
    logic                   ready_75_A;
    logic                   ready_76_A;
    logic                   ready_77_A;
    logic                   ready_78_A;
    logic                   ready_79_A;
    logic                   ready_80_A;
    logic                   ready_81_A;
    logic                   ready_82_A;
    logic                   ready_83_A;
    logic                   ready_84_A;
    logic                   ready_85_A;
    logic                   ready_86_A;
    logic                   ready_87_A;
    logic                   ready_88_A;
    logic                   ready_89_A;
    logic                   ready_90_A;
    logic                   ready_91_A;
    logic                   ready_92_A;
    logic                   ready_93_A;
    logic                   ready_94_A;
    logic                   ready_95_A;
    logic                   ready_96_A;
    logic                   ready_97_A;
    logic                   ready_98_A;
    logic                   ready_99_A;
    logic                   ready_100_A;
    logic                   ready_101_A;
    logic                   ready_102_A;
    logic                   ready_103_A;
    logic                   ready_104_A;
    logic                   ready_105_A;
    logic                   ready_106_A;
    logic                   ready_107_A;
    logic                   ready_108_A;
    logic                   ready_109_A;
    logic                   ready_110_A;
    logic                   ready_111_A;
    logic                   ready_112_A;
    logic                   ready_113_A;
    logic                   ready_114_A;
    logic                   ready_115_A;
    logic                   ready_116_A;
    logic                   ready_117_A;
    logic                   ready_118_A;
    logic                   ready_119_A;
    logic                   ready_120_A;
    logic                   ready_121_A;
    logic                   ready_122_A;
    logic                   ready_123_A;
    logic                   ready_124_A;
    logic                   ready_125_A;
    logic                   ready_126_A;
    logic                   ready_127_A;
    logic                   ready_128_A;
    logic                   ready_129_A;
    logic                   ready_130_A;
    logic                   ready_131_A;
    logic                   ready_132_A;
    logic                   ready_133_A;
    logic                   ready_134_A;
    logic                   ready_135_A;
    logic                   ready_136_A;
    logic                   ready_137_A;
    logic                   ready_138_A;
    logic                   ready_139_A;
    logic                   ready_140_A;
    logic                   ready_141_A;
    logic                   ready_142_A;
    logic                   ready_143_A;
    logic                   ready_144_A;
    logic                   ready_145_A;
    logic                   ready_146_A;
    logic                   ready_147_A;
    logic                   ready_148_A;
    logic                   ready_149_A;
    logic                   ready_150_A;
    logic                   ready_151_A;
    logic                   ready_152_A;
    logic                   ready_153_A;
    logic                   ready_154_A;
    logic                   ready_155_A;
    logic                   ready_156_A;
    logic                   ready_157_A;
    logic                   ready_158_A;
    logic                   ready_159_A;
    logic                   ready_160_A;
    logic                   ready_161_A;
    logic                   ready_162_A;
    logic                   ready_163_A;
    logic                   ready_164_A;
    logic                   ready_165_A;
    logic                   ready_166_A;
    logic                   ready_167_A;
    logic                   ready_168_A;
    logic                   ready_169_A;
    logic                   ready_170_A;
    logic                   ready_171_A;
    logic                   ready_172_A;
    logic                   ready_173_A;
    logic                   ready_174_A;
    logic                   ready_175_A;
    logic                   ready_176_A;
    logic                   ready_177_A;
    logic                   ready_178_A;
    logic                   ready_179_A;
    logic                   ready_180_A;
    logic                   ready_181_A;
    logic                   ready_182_A;
    logic                   ready_183_A;
    logic                   ready_184_A;
    logic                   ready_185_A;
    logic                   ready_186_A;
    logic                   ready_187_A;
    logic                   ready_188_A;
    logic                   ready_189_A;
    logic                   ready_190_A;
    logic                   ready_191_A;
    logic                   ready_192_A;
    logic                   ready_193_A;
    logic                   ready_194_A;
    logic                   ready_195_A;
    logic                   ready_196_A;
    logic                   ready_197_A;
    logic                   ready_198_A;
    logic                   ready_199_A;
    logic                   ready_200_A;
    logic                   ready_201_A;
    logic                   ready_202_A;
    logic                   ready_203_A;
    logic                   ready_204_A;
    logic                   ready_205_A;
    logic                   ready_206_A;
    logic                   ready_207_A;
    logic                   ready_208_A;
    logic                   ready_209_A;
    logic                   ready_210_A;
    logic                   ready_211_A;
    logic                   ready_212_A;
    logic                   ready_213_A;
    logic                   ready_214_A;
    logic                   ready_215_A;
    logic                   ready_216_A;
    logic                   ready_217_A;
    logic                   ready_218_A;
    logic                   ready_219_A;
    logic                   ready_220_A;
    logic                   ready_221_A;
    logic                   ready_222_A;
    logic                   ready_223_A;
    logic                   ready_224_A;
    logic                   ready_225_A;
    logic                   ready_226_A;
    logic                   ready_227_A;
    logic                   ready_228_A;
    logic                   ready_229_A;
    logic                   ready_230_A;
    logic                   ready_231_A;
    logic                   ready_232_A;
    logic                   ready_233_A;
    logic                   ready_234_A;
    logic                   ready_235_A;
    logic                   ready_236_A;
    logic                   ready_237_A;
    logic                   ready_238_A;
    logic                   ready_239_A;
    logic                   ready_240_A;
    logic                   ready_241_A;
    logic                   ready_242_A;
    logic                   ready_243_A;
    logic                   ready_244_A;
    logic                   ready_245_A;
    logic                   ready_246_A;
    logic                   ready_247_A;
    logic                   ready_248_A;
    logic                   ready_249_A;
    logic                   ready_250_A;
    logic                   ready_251_A;
    logic                   ready_252_A;
    logic                   ready_253_A;
    logic                   ready_254_A;
    logic                   ready_255_A;
    logic                   ready_256_A;
    logic                   ready_257_A;
    logic                   ready_258_A;
    logic                   ready_259_A;
    logic                   ready_260_A;
    logic                   ready_261_A;
    logic                   ready_262_A;
    logic                   ready_263_A;
    logic                   ready_264_A;
    logic                   ready_265_A;
    logic                   ready_266_A;
    logic                   ready_267_A;
    logic                   ready_268_A;
    logic                   ready_269_A;
    logic                   ready_270_A;
    logic                   ready_271_A;
    logic                   ready_272_A;
    logic                   ready_273_A;
    logic                   ready_274_A;
    logic                   ready_275_A;
    logic                   ready_276_A;
    logic                   ready_277_A;
    logic                   ready_278_A;
    logic                   ready_279_A;
    logic                   ready_280_A;
    logic                   ready_281_A;
    logic                   ready_282_A;
    logic                   ready_283_A;
    logic                   ready_284_A;
    logic                   ready_285_A;
    logic                   ready_286_A;
    logic                   ready_287_A;
    logic                   ready_288_A;
    logic                   ready_289_A;
    logic                   ready_290_A;
    logic                   ready_291_A;
    logic                   ready_292_A;
    logic                   ready_293_A;
    logic                   ready_294_A;
    logic                   ready_295_A;
    logic                   ready_296_A;
    logic                   ready_297_A;
    logic                   ready_298_A;
    logic                   ready_299_A;
    logic                   ready_300_A;
    logic                   ready_301_A;
    logic                   ready_302_A;
    logic                   ready_303_A;
    logic                   ready_304_A;
    logic                   ready_305_A;
    logic                   ready_306_A;
    logic                   ready_307_A;
    logic                   ready_308_A;
    logic                   ready_309_A;
    logic                   ready_310_A;
    logic                   ready_311_A;
    logic                   ready_312_A;
    logic                   ready_313_A;
    logic                   ready_314_A;
    logic                   ready_315_A;
    logic                   ready_316_A;
    logic                   ready_317_A;
    logic                   ready_318_A;
    logic                   ready_319_A;
    logic                   ready_320_A;
    logic                   ready_321_A;
    logic                   ready_322_A;
    logic                   ready_323_A;
    logic                   ready_324_A;
    logic                   ready_325_A;
    logic                   ready_326_A;
    logic                   ready_327_A;
    logic                   ready_328_A;
    logic                   ready_329_A;
    logic                   ready_330_A;
    logic                   ready_331_A;
    logic                   ready_332_A;
    logic                   ready_333_A;
    logic                   ready_334_A;
    logic                   ready_335_A;
    logic                   ready_336_A;
    logic                   ready_337_A;
    logic                   ready_338_A;
    logic                   ready_339_A;
    logic                   ready_340_A;
    logic                   ready_341_A;
    logic                   ready_342_A;
    logic                   ready_343_A;
    logic                   ready_344_A;
    logic                   ready_345_A;
    logic                   ready_346_A;
    logic                   ready_347_A;
    logic                   ready_348_A;
    logic                   ready_349_A;
    logic                   ready_350_A;
    logic                   ready_351_A;
    logic                   ready_352_A;
    logic                   ready_353_A;
    logic                   ready_354_A;
    logic                   ready_355_A;
    logic                   ready_356_A;
    logic                   ready_357_A;
    logic                   ready_358_A;
    logic                   ready_359_A;
    logic                   ready_360_A;
    logic                   ready_361_A;
    logic                   ready_362_A;
    logic                   ready_363_A;
    logic                   ready_364_A;
    logic                   ready_365_A;
    logic                   ready_366_A;
    logic                   ready_367_A;
    logic                   ready_368_A;
    logic                   ready_369_A;
    logic                   ready_370_A;
    logic                   ready_371_A;
    logic                   ready_372_A;
    logic                   ready_373_A;
    logic                   ready_374_A;
    logic                   ready_375_A;
    logic                   ready_376_A;
    logic                   ready_377_A;
    logic                   ready_378_A;
    logic                   ready_379_A;
    logic                   ready_380_A;
    logic                   ready_381_A;
    logic                   ready_382_A;
    logic                   ready_383_A;
    logic                   ready_384_A;
    logic                   ready_385_A;
    logic                   ready_386_A;
    logic                   ready_387_A;
    logic                   ready_388_A;
    logic                   ready_389_A;
    logic                   ready_390_A;
    logic                   ready_391_A;
    logic                   ready_392_A;
    logic                   ready_393_A;
    logic                   ready_394_A;
    logic                   ready_395_A;
    logic                   ready_396_A;
    logic                   ready_397_A;
    logic                   ready_398_A;
    logic                   ready_399_A;
    logic                   ready_400_A;
    logic                   ready_401_A;
    logic                   ready_402_A;
    logic                   ready_403_A;
    logic                   ready_404_A;
    logic                   ready_405_A;
    logic                   ready_406_A;
    logic                   ready_407_A;
    logic                   ready_408_A;
    logic                   ready_409_A;
    logic                   ready_410_A;
    logic                   ready_411_A;
    logic                   ready_412_A;
    logic                   ready_413_A;
    logic                   ready_414_A;
    logic                   ready_415_A;
    logic                   ready_416_A;
    logic                   ready_417_A;
    logic                   ready_418_A;
    logic                   ready_419_A;
    logic                   ready_420_A;
    logic                   ready_421_A;
    logic                   ready_422_A;
    logic                   ready_423_A;
    logic                   ready_424_A;
    logic                   ready_425_A;
    logic                   ready_426_A;
    logic                   ready_427_A;
    logic                   ready_428_A;
    logic                   ready_429_A;
    logic                   ready_430_A;
    logic                   ready_431_A;
    logic                   ready_432_A;
    logic                   ready_433_A;
    logic                   ready_434_A;
    logic                   ready_435_A;
    logic                   ready_436_A;
    logic                   ready_437_A;
    logic                   ready_438_A;
    logic                   ready_439_A;
    logic                   ready_440_A;
    logic                   ready_441_A;
    logic                   ready_442_A;
    logic                   ready_443_A;
    logic                   ready_444_A;
    logic                   ready_445_A;
    logic                   ready_446_A;
    logic                   ready_447_A;
    logic                   ready_448_A;
    logic                   ready_449_A;
    logic                   ready_450_A;
    logic                   ready_451_A;
    logic                   ready_452_A;
    logic                   ready_453_A;
    logic                   ready_454_A;
    logic                   ready_455_A;
    logic                   ready_456_A;
    logic                   ready_457_A;
    logic                   ready_458_A;
    logic                   ready_459_A;
    logic                   ready_460_A;
    logic                   ready_461_A;
    logic                   ready_462_A;
    logic                   ready_463_A;
    logic                   ready_464_A;
    logic                   ready_465_A;
    logic                   ready_466_A;
    logic                   ready_467_A;
    logic                   ready_468_A;
    logic                   ready_469_A;
    logic                   ready_470_A;
    logic                   ready_471_A;
    logic                   ready_472_A;
    logic                   ready_473_A;
    logic                   ready_474_A;
    logic                   ready_475_A;
    logic                   ready_476_A;
    logic                   ready_477_A;
    logic                   ready_478_A;
    logic                   ready_479_A;
    logic                   ready_480_A;
    logic                   ready_481_A;
    logic                   ready_482_A;
    logic                   ready_483_A;
    logic                   ready_484_A;
    logic                   ready_485_A;
    logic                   ready_486_A;
    logic                   ready_487_A;
    logic                   ready_488_A;
    logic                   ready_489_A;
    logic                   ready_490_A;
    logic                   ready_491_A;
    logic                   ready_492_A;
    logic                   ready_493_A;
    logic                   ready_494_A;
    logic                   ready_495_A;
    logic                   ready_496_A;
    logic                   ready_497_A;
    logic                   ready_498_A;
    logic                   ready_499_A;
    logic                   ready_500_A;
    logic                   ready_501_A;
    logic                   ready_502_A;
    logic                   ready_503_A;
    logic                   ready_504_A;
    logic                   ready_505_A;
    logic                   ready_506_A;
    logic                   ready_507_A;
    logic                   ready_508_A;
    logic                   ready_509_A;
    logic                   ready_510_A;
    logic                   ready_511_A;
    logic                   valid_bank0_A;
    logic                   ready_bank0_A;
    logic                   valid_bank1_A;
    logic                   ready_bank1_A;
    logic                   valid_bank2_A;
    logic                   ready_bank2_A;
    logic                   valid_bank3_A;
    logic                   ready_bank3_A;
    logic                   valid_bank4_A;
    logic                   ready_bank4_A;
    logic                   valid_bank5_A;
    logic                   ready_bank5_A;
    logic                   valid_bank6_A;
    logic                   ready_bank6_A;
    logic                   valid_bank7_A;
    logic                   ready_bank7_A;
    logic                   valid_bank8_A;
    logic                   ready_bank8_A;
    logic                   valid_bank9_A;
    logic                   ready_bank9_A;
    logic                   valid_bank10_A;
    logic                   ready_bank10_A;
    logic                   valid_bank11_A;
    logic                   ready_bank11_A;
    logic                   valid_bank12_A;
    logic                   ready_bank12_A;
    logic                   valid_bank13_A;
    logic                   ready_bank13_A;
    logic                   valid_bank14_A;
    logic                   ready_bank14_A;
    logic                   valid_bank15_A;
    logic                   ready_bank15_A;
    logic                   valid_bank16_A;
    logic                   ready_bank16_A;
    logic                   valid_bank17_A;
    logic                   ready_bank17_A;
    logic                   valid_bank18_A;
    logic                   ready_bank18_A;
    logic                   valid_bank19_A;
    logic                   ready_bank19_A;
    logic                   valid_bank20_A;
    logic                   ready_bank20_A;
    logic                   valid_bank21_A;
    logic                   ready_bank21_A;
    logic                   valid_bank22_A;
    logic                   ready_bank22_A;
    logic                   valid_bank23_A;
    logic                   ready_bank23_A;
    logic                   valid_bank24_A;
    logic                   ready_bank24_A;
    logic                   valid_bank25_A;
    logic                   ready_bank25_A;
    logic                   valid_bank26_A;
    logic                   ready_bank26_A;
    logic                   valid_bank27_A;
    logic                   ready_bank27_A;
    logic                   valid_bank28_A;
    logic                   ready_bank28_A;
    logic                   valid_bank29_A;
    logic                   ready_bank29_A;
    logic                   valid_bank30_A;
    logic                   ready_bank30_A;
    logic                   valid_bank31_A;
    logic                   ready_bank31_A;

    logic                   valid_B;
    logic                   ready_B;

    logic   [(1<<9)-1:0]    OK_onehot_B;
    logic                   valid_bank0_B;
    logic                   ready_bank0_B;
    logic                   valid_bank1_B;
    logic                   ready_bank1_B;
    logic                   valid_bank2_B;
    logic                   ready_bank2_B;
    logic                   valid_bank3_B;
    logic                   ready_bank3_B;
    logic                   valid_bank4_B;
    logic                   ready_bank4_B;
    logic                   valid_bank5_B;
    logic                   ready_bank5_B;
    logic                   valid_bank6_B;
    logic                   ready_bank6_B;
    logic                   valid_bank7_B;
    logic                   ready_bank7_B;
    logic                   valid_bank8_B;
    logic                   ready_bank8_B;
    logic                   valid_bank9_B;
    logic                   ready_bank9_B;
    logic                   valid_bank10_B;
    logic                   ready_bank10_B;
    logic                   valid_bank11_B;
    logic                   ready_bank11_B;
    logic                   valid_bank12_B;
    logic                   ready_bank12_B;
    logic                   valid_bank13_B;
    logic                   ready_bank13_B;
    logic                   valid_bank14_B;
    logic                   ready_bank14_B;
    logic                   valid_bank15_B;
    logic                   ready_bank15_B;
    logic                   valid_bank16_B;
    logic                   ready_bank16_B;
    logic                   valid_bank17_B;
    logic                   ready_bank17_B;
    logic                   valid_bank18_B;
    logic                   ready_bank18_B;
    logic                   valid_bank19_B;
    logic                   ready_bank19_B;
    logic                   valid_bank20_B;
    logic                   ready_bank20_B;
    logic                   valid_bank21_B;
    logic                   ready_bank21_B;
    logic                   valid_bank22_B;
    logic                   ready_bank22_B;
    logic                   valid_bank23_B;
    logic                   ready_bank23_B;
    logic                   valid_bank24_B;
    logic                   ready_bank24_B;
    logic                   valid_bank25_B;
    logic                   ready_bank25_B;
    logic                   valid_bank26_B;
    logic                   ready_bank26_B;
    logic                   valid_bank27_B;
    logic                   ready_bank27_B;
    logic                   valid_bank28_B;
    logic                   ready_bank28_B;
    logic                   valid_bank29_B;
    logic                   ready_bank29_B;
    logic                   valid_bank30_B;
    logic                   ready_bank30_B;
    logic                   valid_bank31_B;
    logic                   ready_bank31_B;

    logic   [(1<<(9-4))-1:0]   valid_bank_d_B;
    logic   [(1<<(9-4))-1:0]   valid_bank_onehot_B;

    logic   [4-1:0] index_lowbit_B;
    
    logic   [4-1:0] index_lowbit_B1;
    logic   [(9-4)-1:0] index_highbit_B1;
    logic   [(1<<(9-4))-1:0]   valid_bank_onehot_B1;

////////////////////////////////////////////////////////////////////////////////////////////////////////
//Implementation
////////////////////////////////////////////////////////////////////////////////////////////////////////
    assign clean_bit = !so_or_sz;

////////////////////////////////////////////////////////////////////////////////////////////////////////
//OK array
////////////////////////////////////////////////////////////////////////////////////////////////////////
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[0] <= `CQ reset_value;                        
        else if(ready_0_A)
            OK_A[0] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[0] <= `CQ so_or_sz ? OK_A[0] || set_clr_onehot[0] : OK_A[0] && !set_clr_onehot[0];       //update

assign ready_0_A = OK_onehot_A[0] && (valid_bank0_B ? ready_bank0_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[0] |-> !OK_A[0])
`assert_clk(!so_or_sz && set_clr_onehot[0] |->  OK_A[0])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[1] <= `CQ reset_value;                        
        else if(ready_1_A)
            OK_A[1] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[1] <= `CQ so_or_sz ? OK_A[1] || set_clr_onehot[1] : OK_A[1] && !set_clr_onehot[1];       //update

assign ready_1_A = OK_onehot_A[1] && (valid_bank0_B ? ready_bank0_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[1] |-> !OK_A[1])
`assert_clk(!so_or_sz && set_clr_onehot[1] |->  OK_A[1])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[2] <= `CQ reset_value;                        
        else if(ready_2_A)
            OK_A[2] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[2] <= `CQ so_or_sz ? OK_A[2] || set_clr_onehot[2] : OK_A[2] && !set_clr_onehot[2];       //update

assign ready_2_A = OK_onehot_A[2] && (valid_bank0_B ? ready_bank0_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[2] |-> !OK_A[2])
`assert_clk(!so_or_sz && set_clr_onehot[2] |->  OK_A[2])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[3] <= `CQ reset_value;                        
        else if(ready_3_A)
            OK_A[3] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[3] <= `CQ so_or_sz ? OK_A[3] || set_clr_onehot[3] : OK_A[3] && !set_clr_onehot[3];       //update

assign ready_3_A = OK_onehot_A[3] && (valid_bank0_B ? ready_bank0_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[3] |-> !OK_A[3])
`assert_clk(!so_or_sz && set_clr_onehot[3] |->  OK_A[3])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[4] <= `CQ reset_value;                        
        else if(ready_4_A)
            OK_A[4] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[4] <= `CQ so_or_sz ? OK_A[4] || set_clr_onehot[4] : OK_A[4] && !set_clr_onehot[4];       //update

assign ready_4_A = OK_onehot_A[4] && (valid_bank0_B ? ready_bank0_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[4] |-> !OK_A[4])
`assert_clk(!so_or_sz && set_clr_onehot[4] |->  OK_A[4])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[5] <= `CQ reset_value;                        
        else if(ready_5_A)
            OK_A[5] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[5] <= `CQ so_or_sz ? OK_A[5] || set_clr_onehot[5] : OK_A[5] && !set_clr_onehot[5];       //update

assign ready_5_A = OK_onehot_A[5] && (valid_bank0_B ? ready_bank0_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[5] |-> !OK_A[5])
`assert_clk(!so_or_sz && set_clr_onehot[5] |->  OK_A[5])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[6] <= `CQ reset_value;                        
        else if(ready_6_A)
            OK_A[6] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[6] <= `CQ so_or_sz ? OK_A[6] || set_clr_onehot[6] : OK_A[6] && !set_clr_onehot[6];       //update

assign ready_6_A = OK_onehot_A[6] && (valid_bank0_B ? ready_bank0_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[6] |-> !OK_A[6])
`assert_clk(!so_or_sz && set_clr_onehot[6] |->  OK_A[6])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[7] <= `CQ reset_value;                        
        else if(ready_7_A)
            OK_A[7] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[7] <= `CQ so_or_sz ? OK_A[7] || set_clr_onehot[7] : OK_A[7] && !set_clr_onehot[7];       //update

assign ready_7_A = OK_onehot_A[7] && (valid_bank0_B ? ready_bank0_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[7] |-> !OK_A[7])
`assert_clk(!so_or_sz && set_clr_onehot[7] |->  OK_A[7])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[8] <= `CQ reset_value;                        
        else if(ready_8_A)
            OK_A[8] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[8] <= `CQ so_or_sz ? OK_A[8] || set_clr_onehot[8] : OK_A[8] && !set_clr_onehot[8];       //update

assign ready_8_A = OK_onehot_A[8] && (valid_bank0_B ? ready_bank0_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[8] |-> !OK_A[8])
`assert_clk(!so_or_sz && set_clr_onehot[8] |->  OK_A[8])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[9] <= `CQ reset_value;                        
        else if(ready_9_A)
            OK_A[9] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[9] <= `CQ so_or_sz ? OK_A[9] || set_clr_onehot[9] : OK_A[9] && !set_clr_onehot[9];       //update

assign ready_9_A = OK_onehot_A[9] && (valid_bank0_B ? ready_bank0_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[9] |-> !OK_A[9])
`assert_clk(!so_or_sz && set_clr_onehot[9] |->  OK_A[9])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[10] <= `CQ reset_value;                        
        else if(ready_10_A)
            OK_A[10] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[10] <= `CQ so_or_sz ? OK_A[10] || set_clr_onehot[10] : OK_A[10] && !set_clr_onehot[10];       //update

assign ready_10_A = OK_onehot_A[10] && (valid_bank0_B ? ready_bank0_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[10] |-> !OK_A[10])
`assert_clk(!so_or_sz && set_clr_onehot[10] |->  OK_A[10])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[11] <= `CQ reset_value;                        
        else if(ready_11_A)
            OK_A[11] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[11] <= `CQ so_or_sz ? OK_A[11] || set_clr_onehot[11] : OK_A[11] && !set_clr_onehot[11];       //update

assign ready_11_A = OK_onehot_A[11] && (valid_bank0_B ? ready_bank0_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[11] |-> !OK_A[11])
`assert_clk(!so_or_sz && set_clr_onehot[11] |->  OK_A[11])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[12] <= `CQ reset_value;                        
        else if(ready_12_A)
            OK_A[12] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[12] <= `CQ so_or_sz ? OK_A[12] || set_clr_onehot[12] : OK_A[12] && !set_clr_onehot[12];       //update

assign ready_12_A = OK_onehot_A[12] && (valid_bank0_B ? ready_bank0_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[12] |-> !OK_A[12])
`assert_clk(!so_or_sz && set_clr_onehot[12] |->  OK_A[12])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[13] <= `CQ reset_value;                        
        else if(ready_13_A)
            OK_A[13] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[13] <= `CQ so_or_sz ? OK_A[13] || set_clr_onehot[13] : OK_A[13] && !set_clr_onehot[13];       //update

assign ready_13_A = OK_onehot_A[13] && (valid_bank0_B ? ready_bank0_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[13] |-> !OK_A[13])
`assert_clk(!so_or_sz && set_clr_onehot[13] |->  OK_A[13])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[14] <= `CQ reset_value;                        
        else if(ready_14_A)
            OK_A[14] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[14] <= `CQ so_or_sz ? OK_A[14] || set_clr_onehot[14] : OK_A[14] && !set_clr_onehot[14];       //update

assign ready_14_A = OK_onehot_A[14] && (valid_bank0_B ? ready_bank0_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[14] |-> !OK_A[14])
`assert_clk(!so_or_sz && set_clr_onehot[14] |->  OK_A[14])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[15] <= `CQ reset_value;                        
        else if(ready_15_A)
            OK_A[15] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[15] <= `CQ so_or_sz ? OK_A[15] || set_clr_onehot[15] : OK_A[15] && !set_clr_onehot[15];       //update

assign ready_15_A = OK_onehot_A[15] && (valid_bank0_B ? ready_bank0_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[15] |-> !OK_A[15])
`assert_clk(!so_or_sz && set_clr_onehot[15] |->  OK_A[15])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[16] <= `CQ reset_value;                        
        else if(ready_16_A)
            OK_A[16] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[16] <= `CQ so_or_sz ? OK_A[16] || set_clr_onehot[16] : OK_A[16] && !set_clr_onehot[16];       //update

assign ready_16_A = OK_onehot_A[16] && (valid_bank1_B ? ready_bank1_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[16] |-> !OK_A[16])
`assert_clk(!so_or_sz && set_clr_onehot[16] |->  OK_A[16])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[17] <= `CQ reset_value;                        
        else if(ready_17_A)
            OK_A[17] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[17] <= `CQ so_or_sz ? OK_A[17] || set_clr_onehot[17] : OK_A[17] && !set_clr_onehot[17];       //update

assign ready_17_A = OK_onehot_A[17] && (valid_bank1_B ? ready_bank1_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[17] |-> !OK_A[17])
`assert_clk(!so_or_sz && set_clr_onehot[17] |->  OK_A[17])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[18] <= `CQ reset_value;                        
        else if(ready_18_A)
            OK_A[18] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[18] <= `CQ so_or_sz ? OK_A[18] || set_clr_onehot[18] : OK_A[18] && !set_clr_onehot[18];       //update

assign ready_18_A = OK_onehot_A[18] && (valid_bank1_B ? ready_bank1_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[18] |-> !OK_A[18])
`assert_clk(!so_or_sz && set_clr_onehot[18] |->  OK_A[18])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[19] <= `CQ reset_value;                        
        else if(ready_19_A)
            OK_A[19] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[19] <= `CQ so_or_sz ? OK_A[19] || set_clr_onehot[19] : OK_A[19] && !set_clr_onehot[19];       //update

assign ready_19_A = OK_onehot_A[19] && (valid_bank1_B ? ready_bank1_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[19] |-> !OK_A[19])
`assert_clk(!so_or_sz && set_clr_onehot[19] |->  OK_A[19])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[20] <= `CQ reset_value;                        
        else if(ready_20_A)
            OK_A[20] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[20] <= `CQ so_or_sz ? OK_A[20] || set_clr_onehot[20] : OK_A[20] && !set_clr_onehot[20];       //update

assign ready_20_A = OK_onehot_A[20] && (valid_bank1_B ? ready_bank1_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[20] |-> !OK_A[20])
`assert_clk(!so_or_sz && set_clr_onehot[20] |->  OK_A[20])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[21] <= `CQ reset_value;                        
        else if(ready_21_A)
            OK_A[21] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[21] <= `CQ so_or_sz ? OK_A[21] || set_clr_onehot[21] : OK_A[21] && !set_clr_onehot[21];       //update

assign ready_21_A = OK_onehot_A[21] && (valid_bank1_B ? ready_bank1_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[21] |-> !OK_A[21])
`assert_clk(!so_or_sz && set_clr_onehot[21] |->  OK_A[21])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[22] <= `CQ reset_value;                        
        else if(ready_22_A)
            OK_A[22] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[22] <= `CQ so_or_sz ? OK_A[22] || set_clr_onehot[22] : OK_A[22] && !set_clr_onehot[22];       //update

assign ready_22_A = OK_onehot_A[22] && (valid_bank1_B ? ready_bank1_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[22] |-> !OK_A[22])
`assert_clk(!so_or_sz && set_clr_onehot[22] |->  OK_A[22])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[23] <= `CQ reset_value;                        
        else if(ready_23_A)
            OK_A[23] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[23] <= `CQ so_or_sz ? OK_A[23] || set_clr_onehot[23] : OK_A[23] && !set_clr_onehot[23];       //update

assign ready_23_A = OK_onehot_A[23] && (valid_bank1_B ? ready_bank1_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[23] |-> !OK_A[23])
`assert_clk(!so_or_sz && set_clr_onehot[23] |->  OK_A[23])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[24] <= `CQ reset_value;                        
        else if(ready_24_A)
            OK_A[24] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[24] <= `CQ so_or_sz ? OK_A[24] || set_clr_onehot[24] : OK_A[24] && !set_clr_onehot[24];       //update

assign ready_24_A = OK_onehot_A[24] && (valid_bank1_B ? ready_bank1_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[24] |-> !OK_A[24])
`assert_clk(!so_or_sz && set_clr_onehot[24] |->  OK_A[24])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[25] <= `CQ reset_value;                        
        else if(ready_25_A)
            OK_A[25] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[25] <= `CQ so_or_sz ? OK_A[25] || set_clr_onehot[25] : OK_A[25] && !set_clr_onehot[25];       //update

assign ready_25_A = OK_onehot_A[25] && (valid_bank1_B ? ready_bank1_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[25] |-> !OK_A[25])
`assert_clk(!so_or_sz && set_clr_onehot[25] |->  OK_A[25])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[26] <= `CQ reset_value;                        
        else if(ready_26_A)
            OK_A[26] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[26] <= `CQ so_or_sz ? OK_A[26] || set_clr_onehot[26] : OK_A[26] && !set_clr_onehot[26];       //update

assign ready_26_A = OK_onehot_A[26] && (valid_bank1_B ? ready_bank1_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[26] |-> !OK_A[26])
`assert_clk(!so_or_sz && set_clr_onehot[26] |->  OK_A[26])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[27] <= `CQ reset_value;                        
        else if(ready_27_A)
            OK_A[27] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[27] <= `CQ so_or_sz ? OK_A[27] || set_clr_onehot[27] : OK_A[27] && !set_clr_onehot[27];       //update

assign ready_27_A = OK_onehot_A[27] && (valid_bank1_B ? ready_bank1_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[27] |-> !OK_A[27])
`assert_clk(!so_or_sz && set_clr_onehot[27] |->  OK_A[27])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[28] <= `CQ reset_value;                        
        else if(ready_28_A)
            OK_A[28] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[28] <= `CQ so_or_sz ? OK_A[28] || set_clr_onehot[28] : OK_A[28] && !set_clr_onehot[28];       //update

assign ready_28_A = OK_onehot_A[28] && (valid_bank1_B ? ready_bank1_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[28] |-> !OK_A[28])
`assert_clk(!so_or_sz && set_clr_onehot[28] |->  OK_A[28])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[29] <= `CQ reset_value;                        
        else if(ready_29_A)
            OK_A[29] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[29] <= `CQ so_or_sz ? OK_A[29] || set_clr_onehot[29] : OK_A[29] && !set_clr_onehot[29];       //update

assign ready_29_A = OK_onehot_A[29] && (valid_bank1_B ? ready_bank1_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[29] |-> !OK_A[29])
`assert_clk(!so_or_sz && set_clr_onehot[29] |->  OK_A[29])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[30] <= `CQ reset_value;                        
        else if(ready_30_A)
            OK_A[30] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[30] <= `CQ so_or_sz ? OK_A[30] || set_clr_onehot[30] : OK_A[30] && !set_clr_onehot[30];       //update

assign ready_30_A = OK_onehot_A[30] && (valid_bank1_B ? ready_bank1_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[30] |-> !OK_A[30])
`assert_clk(!so_or_sz && set_clr_onehot[30] |->  OK_A[30])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[31] <= `CQ reset_value;                        
        else if(ready_31_A)
            OK_A[31] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[31] <= `CQ so_or_sz ? OK_A[31] || set_clr_onehot[31] : OK_A[31] && !set_clr_onehot[31];       //update

assign ready_31_A = OK_onehot_A[31] && (valid_bank1_B ? ready_bank1_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[31] |-> !OK_A[31])
`assert_clk(!so_or_sz && set_clr_onehot[31] |->  OK_A[31])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[32] <= `CQ reset_value;                        
        else if(ready_32_A)
            OK_A[32] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[32] <= `CQ so_or_sz ? OK_A[32] || set_clr_onehot[32] : OK_A[32] && !set_clr_onehot[32];       //update

assign ready_32_A = OK_onehot_A[32] && (valid_bank2_B ? ready_bank2_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[32] |-> !OK_A[32])
`assert_clk(!so_or_sz && set_clr_onehot[32] |->  OK_A[32])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[33] <= `CQ reset_value;                        
        else if(ready_33_A)
            OK_A[33] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[33] <= `CQ so_or_sz ? OK_A[33] || set_clr_onehot[33] : OK_A[33] && !set_clr_onehot[33];       //update

assign ready_33_A = OK_onehot_A[33] && (valid_bank2_B ? ready_bank2_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[33] |-> !OK_A[33])
`assert_clk(!so_or_sz && set_clr_onehot[33] |->  OK_A[33])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[34] <= `CQ reset_value;                        
        else if(ready_34_A)
            OK_A[34] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[34] <= `CQ so_or_sz ? OK_A[34] || set_clr_onehot[34] : OK_A[34] && !set_clr_onehot[34];       //update

assign ready_34_A = OK_onehot_A[34] && (valid_bank2_B ? ready_bank2_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[34] |-> !OK_A[34])
`assert_clk(!so_or_sz && set_clr_onehot[34] |->  OK_A[34])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[35] <= `CQ reset_value;                        
        else if(ready_35_A)
            OK_A[35] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[35] <= `CQ so_or_sz ? OK_A[35] || set_clr_onehot[35] : OK_A[35] && !set_clr_onehot[35];       //update

assign ready_35_A = OK_onehot_A[35] && (valid_bank2_B ? ready_bank2_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[35] |-> !OK_A[35])
`assert_clk(!so_or_sz && set_clr_onehot[35] |->  OK_A[35])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[36] <= `CQ reset_value;                        
        else if(ready_36_A)
            OK_A[36] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[36] <= `CQ so_or_sz ? OK_A[36] || set_clr_onehot[36] : OK_A[36] && !set_clr_onehot[36];       //update

assign ready_36_A = OK_onehot_A[36] && (valid_bank2_B ? ready_bank2_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[36] |-> !OK_A[36])
`assert_clk(!so_or_sz && set_clr_onehot[36] |->  OK_A[36])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[37] <= `CQ reset_value;                        
        else if(ready_37_A)
            OK_A[37] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[37] <= `CQ so_or_sz ? OK_A[37] || set_clr_onehot[37] : OK_A[37] && !set_clr_onehot[37];       //update

assign ready_37_A = OK_onehot_A[37] && (valid_bank2_B ? ready_bank2_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[37] |-> !OK_A[37])
`assert_clk(!so_or_sz && set_clr_onehot[37] |->  OK_A[37])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[38] <= `CQ reset_value;                        
        else if(ready_38_A)
            OK_A[38] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[38] <= `CQ so_or_sz ? OK_A[38] || set_clr_onehot[38] : OK_A[38] && !set_clr_onehot[38];       //update

assign ready_38_A = OK_onehot_A[38] && (valid_bank2_B ? ready_bank2_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[38] |-> !OK_A[38])
`assert_clk(!so_or_sz && set_clr_onehot[38] |->  OK_A[38])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[39] <= `CQ reset_value;                        
        else if(ready_39_A)
            OK_A[39] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[39] <= `CQ so_or_sz ? OK_A[39] || set_clr_onehot[39] : OK_A[39] && !set_clr_onehot[39];       //update

assign ready_39_A = OK_onehot_A[39] && (valid_bank2_B ? ready_bank2_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[39] |-> !OK_A[39])
`assert_clk(!so_or_sz && set_clr_onehot[39] |->  OK_A[39])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[40] <= `CQ reset_value;                        
        else if(ready_40_A)
            OK_A[40] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[40] <= `CQ so_or_sz ? OK_A[40] || set_clr_onehot[40] : OK_A[40] && !set_clr_onehot[40];       //update

assign ready_40_A = OK_onehot_A[40] && (valid_bank2_B ? ready_bank2_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[40] |-> !OK_A[40])
`assert_clk(!so_or_sz && set_clr_onehot[40] |->  OK_A[40])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[41] <= `CQ reset_value;                        
        else if(ready_41_A)
            OK_A[41] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[41] <= `CQ so_or_sz ? OK_A[41] || set_clr_onehot[41] : OK_A[41] && !set_clr_onehot[41];       //update

assign ready_41_A = OK_onehot_A[41] && (valid_bank2_B ? ready_bank2_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[41] |-> !OK_A[41])
`assert_clk(!so_or_sz && set_clr_onehot[41] |->  OK_A[41])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[42] <= `CQ reset_value;                        
        else if(ready_42_A)
            OK_A[42] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[42] <= `CQ so_or_sz ? OK_A[42] || set_clr_onehot[42] : OK_A[42] && !set_clr_onehot[42];       //update

assign ready_42_A = OK_onehot_A[42] && (valid_bank2_B ? ready_bank2_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[42] |-> !OK_A[42])
`assert_clk(!so_or_sz && set_clr_onehot[42] |->  OK_A[42])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[43] <= `CQ reset_value;                        
        else if(ready_43_A)
            OK_A[43] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[43] <= `CQ so_or_sz ? OK_A[43] || set_clr_onehot[43] : OK_A[43] && !set_clr_onehot[43];       //update

assign ready_43_A = OK_onehot_A[43] && (valid_bank2_B ? ready_bank2_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[43] |-> !OK_A[43])
`assert_clk(!so_or_sz && set_clr_onehot[43] |->  OK_A[43])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[44] <= `CQ reset_value;                        
        else if(ready_44_A)
            OK_A[44] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[44] <= `CQ so_or_sz ? OK_A[44] || set_clr_onehot[44] : OK_A[44] && !set_clr_onehot[44];       //update

assign ready_44_A = OK_onehot_A[44] && (valid_bank2_B ? ready_bank2_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[44] |-> !OK_A[44])
`assert_clk(!so_or_sz && set_clr_onehot[44] |->  OK_A[44])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[45] <= `CQ reset_value;                        
        else if(ready_45_A)
            OK_A[45] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[45] <= `CQ so_or_sz ? OK_A[45] || set_clr_onehot[45] : OK_A[45] && !set_clr_onehot[45];       //update

assign ready_45_A = OK_onehot_A[45] && (valid_bank2_B ? ready_bank2_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[45] |-> !OK_A[45])
`assert_clk(!so_or_sz && set_clr_onehot[45] |->  OK_A[45])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[46] <= `CQ reset_value;                        
        else if(ready_46_A)
            OK_A[46] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[46] <= `CQ so_or_sz ? OK_A[46] || set_clr_onehot[46] : OK_A[46] && !set_clr_onehot[46];       //update

assign ready_46_A = OK_onehot_A[46] && (valid_bank2_B ? ready_bank2_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[46] |-> !OK_A[46])
`assert_clk(!so_or_sz && set_clr_onehot[46] |->  OK_A[46])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[47] <= `CQ reset_value;                        
        else if(ready_47_A)
            OK_A[47] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[47] <= `CQ so_or_sz ? OK_A[47] || set_clr_onehot[47] : OK_A[47] && !set_clr_onehot[47];       //update

assign ready_47_A = OK_onehot_A[47] && (valid_bank2_B ? ready_bank2_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[47] |-> !OK_A[47])
`assert_clk(!so_or_sz && set_clr_onehot[47] |->  OK_A[47])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[48] <= `CQ reset_value;                        
        else if(ready_48_A)
            OK_A[48] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[48] <= `CQ so_or_sz ? OK_A[48] || set_clr_onehot[48] : OK_A[48] && !set_clr_onehot[48];       //update

assign ready_48_A = OK_onehot_A[48] && (valid_bank3_B ? ready_bank3_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[48] |-> !OK_A[48])
`assert_clk(!so_or_sz && set_clr_onehot[48] |->  OK_A[48])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[49] <= `CQ reset_value;                        
        else if(ready_49_A)
            OK_A[49] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[49] <= `CQ so_or_sz ? OK_A[49] || set_clr_onehot[49] : OK_A[49] && !set_clr_onehot[49];       //update

assign ready_49_A = OK_onehot_A[49] && (valid_bank3_B ? ready_bank3_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[49] |-> !OK_A[49])
`assert_clk(!so_or_sz && set_clr_onehot[49] |->  OK_A[49])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[50] <= `CQ reset_value;                        
        else if(ready_50_A)
            OK_A[50] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[50] <= `CQ so_or_sz ? OK_A[50] || set_clr_onehot[50] : OK_A[50] && !set_clr_onehot[50];       //update

assign ready_50_A = OK_onehot_A[50] && (valid_bank3_B ? ready_bank3_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[50] |-> !OK_A[50])
`assert_clk(!so_or_sz && set_clr_onehot[50] |->  OK_A[50])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[51] <= `CQ reset_value;                        
        else if(ready_51_A)
            OK_A[51] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[51] <= `CQ so_or_sz ? OK_A[51] || set_clr_onehot[51] : OK_A[51] && !set_clr_onehot[51];       //update

assign ready_51_A = OK_onehot_A[51] && (valid_bank3_B ? ready_bank3_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[51] |-> !OK_A[51])
`assert_clk(!so_or_sz && set_clr_onehot[51] |->  OK_A[51])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[52] <= `CQ reset_value;                        
        else if(ready_52_A)
            OK_A[52] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[52] <= `CQ so_or_sz ? OK_A[52] || set_clr_onehot[52] : OK_A[52] && !set_clr_onehot[52];       //update

assign ready_52_A = OK_onehot_A[52] && (valid_bank3_B ? ready_bank3_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[52] |-> !OK_A[52])
`assert_clk(!so_or_sz && set_clr_onehot[52] |->  OK_A[52])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[53] <= `CQ reset_value;                        
        else if(ready_53_A)
            OK_A[53] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[53] <= `CQ so_or_sz ? OK_A[53] || set_clr_onehot[53] : OK_A[53] && !set_clr_onehot[53];       //update

assign ready_53_A = OK_onehot_A[53] && (valid_bank3_B ? ready_bank3_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[53] |-> !OK_A[53])
`assert_clk(!so_or_sz && set_clr_onehot[53] |->  OK_A[53])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[54] <= `CQ reset_value;                        
        else if(ready_54_A)
            OK_A[54] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[54] <= `CQ so_or_sz ? OK_A[54] || set_clr_onehot[54] : OK_A[54] && !set_clr_onehot[54];       //update

assign ready_54_A = OK_onehot_A[54] && (valid_bank3_B ? ready_bank3_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[54] |-> !OK_A[54])
`assert_clk(!so_or_sz && set_clr_onehot[54] |->  OK_A[54])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[55] <= `CQ reset_value;                        
        else if(ready_55_A)
            OK_A[55] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[55] <= `CQ so_or_sz ? OK_A[55] || set_clr_onehot[55] : OK_A[55] && !set_clr_onehot[55];       //update

assign ready_55_A = OK_onehot_A[55] && (valid_bank3_B ? ready_bank3_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[55] |-> !OK_A[55])
`assert_clk(!so_or_sz && set_clr_onehot[55] |->  OK_A[55])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[56] <= `CQ reset_value;                        
        else if(ready_56_A)
            OK_A[56] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[56] <= `CQ so_or_sz ? OK_A[56] || set_clr_onehot[56] : OK_A[56] && !set_clr_onehot[56];       //update

assign ready_56_A = OK_onehot_A[56] && (valid_bank3_B ? ready_bank3_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[56] |-> !OK_A[56])
`assert_clk(!so_or_sz && set_clr_onehot[56] |->  OK_A[56])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[57] <= `CQ reset_value;                        
        else if(ready_57_A)
            OK_A[57] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[57] <= `CQ so_or_sz ? OK_A[57] || set_clr_onehot[57] : OK_A[57] && !set_clr_onehot[57];       //update

assign ready_57_A = OK_onehot_A[57] && (valid_bank3_B ? ready_bank3_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[57] |-> !OK_A[57])
`assert_clk(!so_or_sz && set_clr_onehot[57] |->  OK_A[57])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[58] <= `CQ reset_value;                        
        else if(ready_58_A)
            OK_A[58] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[58] <= `CQ so_or_sz ? OK_A[58] || set_clr_onehot[58] : OK_A[58] && !set_clr_onehot[58];       //update

assign ready_58_A = OK_onehot_A[58] && (valid_bank3_B ? ready_bank3_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[58] |-> !OK_A[58])
`assert_clk(!so_or_sz && set_clr_onehot[58] |->  OK_A[58])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[59] <= `CQ reset_value;                        
        else if(ready_59_A)
            OK_A[59] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[59] <= `CQ so_or_sz ? OK_A[59] || set_clr_onehot[59] : OK_A[59] && !set_clr_onehot[59];       //update

assign ready_59_A = OK_onehot_A[59] && (valid_bank3_B ? ready_bank3_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[59] |-> !OK_A[59])
`assert_clk(!so_or_sz && set_clr_onehot[59] |->  OK_A[59])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[60] <= `CQ reset_value;                        
        else if(ready_60_A)
            OK_A[60] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[60] <= `CQ so_or_sz ? OK_A[60] || set_clr_onehot[60] : OK_A[60] && !set_clr_onehot[60];       //update

assign ready_60_A = OK_onehot_A[60] && (valid_bank3_B ? ready_bank3_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[60] |-> !OK_A[60])
`assert_clk(!so_or_sz && set_clr_onehot[60] |->  OK_A[60])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[61] <= `CQ reset_value;                        
        else if(ready_61_A)
            OK_A[61] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[61] <= `CQ so_or_sz ? OK_A[61] || set_clr_onehot[61] : OK_A[61] && !set_clr_onehot[61];       //update

assign ready_61_A = OK_onehot_A[61] && (valid_bank3_B ? ready_bank3_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[61] |-> !OK_A[61])
`assert_clk(!so_or_sz && set_clr_onehot[61] |->  OK_A[61])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[62] <= `CQ reset_value;                        
        else if(ready_62_A)
            OK_A[62] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[62] <= `CQ so_or_sz ? OK_A[62] || set_clr_onehot[62] : OK_A[62] && !set_clr_onehot[62];       //update

assign ready_62_A = OK_onehot_A[62] && (valid_bank3_B ? ready_bank3_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[62] |-> !OK_A[62])
`assert_clk(!so_or_sz && set_clr_onehot[62] |->  OK_A[62])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[63] <= `CQ reset_value;                        
        else if(ready_63_A)
            OK_A[63] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[63] <= `CQ so_or_sz ? OK_A[63] || set_clr_onehot[63] : OK_A[63] && !set_clr_onehot[63];       //update

assign ready_63_A = OK_onehot_A[63] && (valid_bank3_B ? ready_bank3_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[63] |-> !OK_A[63])
`assert_clk(!so_or_sz && set_clr_onehot[63] |->  OK_A[63])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[64] <= `CQ reset_value;                        
        else if(ready_64_A)
            OK_A[64] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[64] <= `CQ so_or_sz ? OK_A[64] || set_clr_onehot[64] : OK_A[64] && !set_clr_onehot[64];       //update

assign ready_64_A = OK_onehot_A[64] && (valid_bank4_B ? ready_bank4_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[64] |-> !OK_A[64])
`assert_clk(!so_or_sz && set_clr_onehot[64] |->  OK_A[64])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[65] <= `CQ reset_value;                        
        else if(ready_65_A)
            OK_A[65] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[65] <= `CQ so_or_sz ? OK_A[65] || set_clr_onehot[65] : OK_A[65] && !set_clr_onehot[65];       //update

assign ready_65_A = OK_onehot_A[65] && (valid_bank4_B ? ready_bank4_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[65] |-> !OK_A[65])
`assert_clk(!so_or_sz && set_clr_onehot[65] |->  OK_A[65])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[66] <= `CQ reset_value;                        
        else if(ready_66_A)
            OK_A[66] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[66] <= `CQ so_or_sz ? OK_A[66] || set_clr_onehot[66] : OK_A[66] && !set_clr_onehot[66];       //update

assign ready_66_A = OK_onehot_A[66] && (valid_bank4_B ? ready_bank4_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[66] |-> !OK_A[66])
`assert_clk(!so_or_sz && set_clr_onehot[66] |->  OK_A[66])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[67] <= `CQ reset_value;                        
        else if(ready_67_A)
            OK_A[67] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[67] <= `CQ so_or_sz ? OK_A[67] || set_clr_onehot[67] : OK_A[67] && !set_clr_onehot[67];       //update

assign ready_67_A = OK_onehot_A[67] && (valid_bank4_B ? ready_bank4_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[67] |-> !OK_A[67])
`assert_clk(!so_or_sz && set_clr_onehot[67] |->  OK_A[67])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[68] <= `CQ reset_value;                        
        else if(ready_68_A)
            OK_A[68] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[68] <= `CQ so_or_sz ? OK_A[68] || set_clr_onehot[68] : OK_A[68] && !set_clr_onehot[68];       //update

assign ready_68_A = OK_onehot_A[68] && (valid_bank4_B ? ready_bank4_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[68] |-> !OK_A[68])
`assert_clk(!so_or_sz && set_clr_onehot[68] |->  OK_A[68])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[69] <= `CQ reset_value;                        
        else if(ready_69_A)
            OK_A[69] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[69] <= `CQ so_or_sz ? OK_A[69] || set_clr_onehot[69] : OK_A[69] && !set_clr_onehot[69];       //update

assign ready_69_A = OK_onehot_A[69] && (valid_bank4_B ? ready_bank4_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[69] |-> !OK_A[69])
`assert_clk(!so_or_sz && set_clr_onehot[69] |->  OK_A[69])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[70] <= `CQ reset_value;                        
        else if(ready_70_A)
            OK_A[70] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[70] <= `CQ so_or_sz ? OK_A[70] || set_clr_onehot[70] : OK_A[70] && !set_clr_onehot[70];       //update

assign ready_70_A = OK_onehot_A[70] && (valid_bank4_B ? ready_bank4_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[70] |-> !OK_A[70])
`assert_clk(!so_or_sz && set_clr_onehot[70] |->  OK_A[70])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[71] <= `CQ reset_value;                        
        else if(ready_71_A)
            OK_A[71] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[71] <= `CQ so_or_sz ? OK_A[71] || set_clr_onehot[71] : OK_A[71] && !set_clr_onehot[71];       //update

assign ready_71_A = OK_onehot_A[71] && (valid_bank4_B ? ready_bank4_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[71] |-> !OK_A[71])
`assert_clk(!so_or_sz && set_clr_onehot[71] |->  OK_A[71])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[72] <= `CQ reset_value;                        
        else if(ready_72_A)
            OK_A[72] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[72] <= `CQ so_or_sz ? OK_A[72] || set_clr_onehot[72] : OK_A[72] && !set_clr_onehot[72];       //update

assign ready_72_A = OK_onehot_A[72] && (valid_bank4_B ? ready_bank4_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[72] |-> !OK_A[72])
`assert_clk(!so_or_sz && set_clr_onehot[72] |->  OK_A[72])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[73] <= `CQ reset_value;                        
        else if(ready_73_A)
            OK_A[73] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[73] <= `CQ so_or_sz ? OK_A[73] || set_clr_onehot[73] : OK_A[73] && !set_clr_onehot[73];       //update

assign ready_73_A = OK_onehot_A[73] && (valid_bank4_B ? ready_bank4_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[73] |-> !OK_A[73])
`assert_clk(!so_or_sz && set_clr_onehot[73] |->  OK_A[73])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[74] <= `CQ reset_value;                        
        else if(ready_74_A)
            OK_A[74] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[74] <= `CQ so_or_sz ? OK_A[74] || set_clr_onehot[74] : OK_A[74] && !set_clr_onehot[74];       //update

assign ready_74_A = OK_onehot_A[74] && (valid_bank4_B ? ready_bank4_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[74] |-> !OK_A[74])
`assert_clk(!so_or_sz && set_clr_onehot[74] |->  OK_A[74])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[75] <= `CQ reset_value;                        
        else if(ready_75_A)
            OK_A[75] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[75] <= `CQ so_or_sz ? OK_A[75] || set_clr_onehot[75] : OK_A[75] && !set_clr_onehot[75];       //update

assign ready_75_A = OK_onehot_A[75] && (valid_bank4_B ? ready_bank4_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[75] |-> !OK_A[75])
`assert_clk(!so_or_sz && set_clr_onehot[75] |->  OK_A[75])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[76] <= `CQ reset_value;                        
        else if(ready_76_A)
            OK_A[76] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[76] <= `CQ so_or_sz ? OK_A[76] || set_clr_onehot[76] : OK_A[76] && !set_clr_onehot[76];       //update

assign ready_76_A = OK_onehot_A[76] && (valid_bank4_B ? ready_bank4_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[76] |-> !OK_A[76])
`assert_clk(!so_or_sz && set_clr_onehot[76] |->  OK_A[76])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[77] <= `CQ reset_value;                        
        else if(ready_77_A)
            OK_A[77] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[77] <= `CQ so_or_sz ? OK_A[77] || set_clr_onehot[77] : OK_A[77] && !set_clr_onehot[77];       //update

assign ready_77_A = OK_onehot_A[77] && (valid_bank4_B ? ready_bank4_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[77] |-> !OK_A[77])
`assert_clk(!so_or_sz && set_clr_onehot[77] |->  OK_A[77])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[78] <= `CQ reset_value;                        
        else if(ready_78_A)
            OK_A[78] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[78] <= `CQ so_or_sz ? OK_A[78] || set_clr_onehot[78] : OK_A[78] && !set_clr_onehot[78];       //update

assign ready_78_A = OK_onehot_A[78] && (valid_bank4_B ? ready_bank4_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[78] |-> !OK_A[78])
`assert_clk(!so_or_sz && set_clr_onehot[78] |->  OK_A[78])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[79] <= `CQ reset_value;                        
        else if(ready_79_A)
            OK_A[79] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[79] <= `CQ so_or_sz ? OK_A[79] || set_clr_onehot[79] : OK_A[79] && !set_clr_onehot[79];       //update

assign ready_79_A = OK_onehot_A[79] && (valid_bank4_B ? ready_bank4_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[79] |-> !OK_A[79])
`assert_clk(!so_or_sz && set_clr_onehot[79] |->  OK_A[79])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[80] <= `CQ reset_value;                        
        else if(ready_80_A)
            OK_A[80] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[80] <= `CQ so_or_sz ? OK_A[80] || set_clr_onehot[80] : OK_A[80] && !set_clr_onehot[80];       //update

assign ready_80_A = OK_onehot_A[80] && (valid_bank5_B ? ready_bank5_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[80] |-> !OK_A[80])
`assert_clk(!so_or_sz && set_clr_onehot[80] |->  OK_A[80])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[81] <= `CQ reset_value;                        
        else if(ready_81_A)
            OK_A[81] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[81] <= `CQ so_or_sz ? OK_A[81] || set_clr_onehot[81] : OK_A[81] && !set_clr_onehot[81];       //update

assign ready_81_A = OK_onehot_A[81] && (valid_bank5_B ? ready_bank5_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[81] |-> !OK_A[81])
`assert_clk(!so_or_sz && set_clr_onehot[81] |->  OK_A[81])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[82] <= `CQ reset_value;                        
        else if(ready_82_A)
            OK_A[82] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[82] <= `CQ so_or_sz ? OK_A[82] || set_clr_onehot[82] : OK_A[82] && !set_clr_onehot[82];       //update

assign ready_82_A = OK_onehot_A[82] && (valid_bank5_B ? ready_bank5_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[82] |-> !OK_A[82])
`assert_clk(!so_or_sz && set_clr_onehot[82] |->  OK_A[82])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[83] <= `CQ reset_value;                        
        else if(ready_83_A)
            OK_A[83] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[83] <= `CQ so_or_sz ? OK_A[83] || set_clr_onehot[83] : OK_A[83] && !set_clr_onehot[83];       //update

assign ready_83_A = OK_onehot_A[83] && (valid_bank5_B ? ready_bank5_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[83] |-> !OK_A[83])
`assert_clk(!so_or_sz && set_clr_onehot[83] |->  OK_A[83])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[84] <= `CQ reset_value;                        
        else if(ready_84_A)
            OK_A[84] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[84] <= `CQ so_or_sz ? OK_A[84] || set_clr_onehot[84] : OK_A[84] && !set_clr_onehot[84];       //update

assign ready_84_A = OK_onehot_A[84] && (valid_bank5_B ? ready_bank5_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[84] |-> !OK_A[84])
`assert_clk(!so_or_sz && set_clr_onehot[84] |->  OK_A[84])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[85] <= `CQ reset_value;                        
        else if(ready_85_A)
            OK_A[85] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[85] <= `CQ so_or_sz ? OK_A[85] || set_clr_onehot[85] : OK_A[85] && !set_clr_onehot[85];       //update

assign ready_85_A = OK_onehot_A[85] && (valid_bank5_B ? ready_bank5_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[85] |-> !OK_A[85])
`assert_clk(!so_or_sz && set_clr_onehot[85] |->  OK_A[85])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[86] <= `CQ reset_value;                        
        else if(ready_86_A)
            OK_A[86] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[86] <= `CQ so_or_sz ? OK_A[86] || set_clr_onehot[86] : OK_A[86] && !set_clr_onehot[86];       //update

assign ready_86_A = OK_onehot_A[86] && (valid_bank5_B ? ready_bank5_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[86] |-> !OK_A[86])
`assert_clk(!so_or_sz && set_clr_onehot[86] |->  OK_A[86])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[87] <= `CQ reset_value;                        
        else if(ready_87_A)
            OK_A[87] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[87] <= `CQ so_or_sz ? OK_A[87] || set_clr_onehot[87] : OK_A[87] && !set_clr_onehot[87];       //update

assign ready_87_A = OK_onehot_A[87] && (valid_bank5_B ? ready_bank5_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[87] |-> !OK_A[87])
`assert_clk(!so_or_sz && set_clr_onehot[87] |->  OK_A[87])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[88] <= `CQ reset_value;                        
        else if(ready_88_A)
            OK_A[88] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[88] <= `CQ so_or_sz ? OK_A[88] || set_clr_onehot[88] : OK_A[88] && !set_clr_onehot[88];       //update

assign ready_88_A = OK_onehot_A[88] && (valid_bank5_B ? ready_bank5_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[88] |-> !OK_A[88])
`assert_clk(!so_or_sz && set_clr_onehot[88] |->  OK_A[88])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[89] <= `CQ reset_value;                        
        else if(ready_89_A)
            OK_A[89] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[89] <= `CQ so_or_sz ? OK_A[89] || set_clr_onehot[89] : OK_A[89] && !set_clr_onehot[89];       //update

assign ready_89_A = OK_onehot_A[89] && (valid_bank5_B ? ready_bank5_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[89] |-> !OK_A[89])
`assert_clk(!so_or_sz && set_clr_onehot[89] |->  OK_A[89])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[90] <= `CQ reset_value;                        
        else if(ready_90_A)
            OK_A[90] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[90] <= `CQ so_or_sz ? OK_A[90] || set_clr_onehot[90] : OK_A[90] && !set_clr_onehot[90];       //update

assign ready_90_A = OK_onehot_A[90] && (valid_bank5_B ? ready_bank5_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[90] |-> !OK_A[90])
`assert_clk(!so_or_sz && set_clr_onehot[90] |->  OK_A[90])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[91] <= `CQ reset_value;                        
        else if(ready_91_A)
            OK_A[91] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[91] <= `CQ so_or_sz ? OK_A[91] || set_clr_onehot[91] : OK_A[91] && !set_clr_onehot[91];       //update

assign ready_91_A = OK_onehot_A[91] && (valid_bank5_B ? ready_bank5_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[91] |-> !OK_A[91])
`assert_clk(!so_or_sz && set_clr_onehot[91] |->  OK_A[91])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[92] <= `CQ reset_value;                        
        else if(ready_92_A)
            OK_A[92] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[92] <= `CQ so_or_sz ? OK_A[92] || set_clr_onehot[92] : OK_A[92] && !set_clr_onehot[92];       //update

assign ready_92_A = OK_onehot_A[92] && (valid_bank5_B ? ready_bank5_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[92] |-> !OK_A[92])
`assert_clk(!so_or_sz && set_clr_onehot[92] |->  OK_A[92])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[93] <= `CQ reset_value;                        
        else if(ready_93_A)
            OK_A[93] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[93] <= `CQ so_or_sz ? OK_A[93] || set_clr_onehot[93] : OK_A[93] && !set_clr_onehot[93];       //update

assign ready_93_A = OK_onehot_A[93] && (valid_bank5_B ? ready_bank5_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[93] |-> !OK_A[93])
`assert_clk(!so_or_sz && set_clr_onehot[93] |->  OK_A[93])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[94] <= `CQ reset_value;                        
        else if(ready_94_A)
            OK_A[94] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[94] <= `CQ so_or_sz ? OK_A[94] || set_clr_onehot[94] : OK_A[94] && !set_clr_onehot[94];       //update

assign ready_94_A = OK_onehot_A[94] && (valid_bank5_B ? ready_bank5_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[94] |-> !OK_A[94])
`assert_clk(!so_or_sz && set_clr_onehot[94] |->  OK_A[94])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[95] <= `CQ reset_value;                        
        else if(ready_95_A)
            OK_A[95] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[95] <= `CQ so_or_sz ? OK_A[95] || set_clr_onehot[95] : OK_A[95] && !set_clr_onehot[95];       //update

assign ready_95_A = OK_onehot_A[95] && (valid_bank5_B ? ready_bank5_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[95] |-> !OK_A[95])
`assert_clk(!so_or_sz && set_clr_onehot[95] |->  OK_A[95])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[96] <= `CQ reset_value;                        
        else if(ready_96_A)
            OK_A[96] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[96] <= `CQ so_or_sz ? OK_A[96] || set_clr_onehot[96] : OK_A[96] && !set_clr_onehot[96];       //update

assign ready_96_A = OK_onehot_A[96] && (valid_bank6_B ? ready_bank6_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[96] |-> !OK_A[96])
`assert_clk(!so_or_sz && set_clr_onehot[96] |->  OK_A[96])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[97] <= `CQ reset_value;                        
        else if(ready_97_A)
            OK_A[97] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[97] <= `CQ so_or_sz ? OK_A[97] || set_clr_onehot[97] : OK_A[97] && !set_clr_onehot[97];       //update

assign ready_97_A = OK_onehot_A[97] && (valid_bank6_B ? ready_bank6_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[97] |-> !OK_A[97])
`assert_clk(!so_or_sz && set_clr_onehot[97] |->  OK_A[97])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[98] <= `CQ reset_value;                        
        else if(ready_98_A)
            OK_A[98] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[98] <= `CQ so_or_sz ? OK_A[98] || set_clr_onehot[98] : OK_A[98] && !set_clr_onehot[98];       //update

assign ready_98_A = OK_onehot_A[98] && (valid_bank6_B ? ready_bank6_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[98] |-> !OK_A[98])
`assert_clk(!so_or_sz && set_clr_onehot[98] |->  OK_A[98])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[99] <= `CQ reset_value;                        
        else if(ready_99_A)
            OK_A[99] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[99] <= `CQ so_or_sz ? OK_A[99] || set_clr_onehot[99] : OK_A[99] && !set_clr_onehot[99];       //update

assign ready_99_A = OK_onehot_A[99] && (valid_bank6_B ? ready_bank6_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[99] |-> !OK_A[99])
`assert_clk(!so_or_sz && set_clr_onehot[99] |->  OK_A[99])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[100] <= `CQ reset_value;                        
        else if(ready_100_A)
            OK_A[100] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[100] <= `CQ so_or_sz ? OK_A[100] || set_clr_onehot[100] : OK_A[100] && !set_clr_onehot[100];       //update

assign ready_100_A = OK_onehot_A[100] && (valid_bank6_B ? ready_bank6_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[100] |-> !OK_A[100])
`assert_clk(!so_or_sz && set_clr_onehot[100] |->  OK_A[100])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[101] <= `CQ reset_value;                        
        else if(ready_101_A)
            OK_A[101] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[101] <= `CQ so_or_sz ? OK_A[101] || set_clr_onehot[101] : OK_A[101] && !set_clr_onehot[101];       //update

assign ready_101_A = OK_onehot_A[101] && (valid_bank6_B ? ready_bank6_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[101] |-> !OK_A[101])
`assert_clk(!so_or_sz && set_clr_onehot[101] |->  OK_A[101])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[102] <= `CQ reset_value;                        
        else if(ready_102_A)
            OK_A[102] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[102] <= `CQ so_or_sz ? OK_A[102] || set_clr_onehot[102] : OK_A[102] && !set_clr_onehot[102];       //update

assign ready_102_A = OK_onehot_A[102] && (valid_bank6_B ? ready_bank6_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[102] |-> !OK_A[102])
`assert_clk(!so_or_sz && set_clr_onehot[102] |->  OK_A[102])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[103] <= `CQ reset_value;                        
        else if(ready_103_A)
            OK_A[103] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[103] <= `CQ so_or_sz ? OK_A[103] || set_clr_onehot[103] : OK_A[103] && !set_clr_onehot[103];       //update

assign ready_103_A = OK_onehot_A[103] && (valid_bank6_B ? ready_bank6_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[103] |-> !OK_A[103])
`assert_clk(!so_or_sz && set_clr_onehot[103] |->  OK_A[103])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[104] <= `CQ reset_value;                        
        else if(ready_104_A)
            OK_A[104] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[104] <= `CQ so_or_sz ? OK_A[104] || set_clr_onehot[104] : OK_A[104] && !set_clr_onehot[104];       //update

assign ready_104_A = OK_onehot_A[104] && (valid_bank6_B ? ready_bank6_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[104] |-> !OK_A[104])
`assert_clk(!so_or_sz && set_clr_onehot[104] |->  OK_A[104])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[105] <= `CQ reset_value;                        
        else if(ready_105_A)
            OK_A[105] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[105] <= `CQ so_or_sz ? OK_A[105] || set_clr_onehot[105] : OK_A[105] && !set_clr_onehot[105];       //update

assign ready_105_A = OK_onehot_A[105] && (valid_bank6_B ? ready_bank6_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[105] |-> !OK_A[105])
`assert_clk(!so_or_sz && set_clr_onehot[105] |->  OK_A[105])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[106] <= `CQ reset_value;                        
        else if(ready_106_A)
            OK_A[106] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[106] <= `CQ so_or_sz ? OK_A[106] || set_clr_onehot[106] : OK_A[106] && !set_clr_onehot[106];       //update

assign ready_106_A = OK_onehot_A[106] && (valid_bank6_B ? ready_bank6_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[106] |-> !OK_A[106])
`assert_clk(!so_or_sz && set_clr_onehot[106] |->  OK_A[106])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[107] <= `CQ reset_value;                        
        else if(ready_107_A)
            OK_A[107] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[107] <= `CQ so_or_sz ? OK_A[107] || set_clr_onehot[107] : OK_A[107] && !set_clr_onehot[107];       //update

assign ready_107_A = OK_onehot_A[107] && (valid_bank6_B ? ready_bank6_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[107] |-> !OK_A[107])
`assert_clk(!so_or_sz && set_clr_onehot[107] |->  OK_A[107])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[108] <= `CQ reset_value;                        
        else if(ready_108_A)
            OK_A[108] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[108] <= `CQ so_or_sz ? OK_A[108] || set_clr_onehot[108] : OK_A[108] && !set_clr_onehot[108];       //update

assign ready_108_A = OK_onehot_A[108] && (valid_bank6_B ? ready_bank6_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[108] |-> !OK_A[108])
`assert_clk(!so_or_sz && set_clr_onehot[108] |->  OK_A[108])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[109] <= `CQ reset_value;                        
        else if(ready_109_A)
            OK_A[109] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[109] <= `CQ so_or_sz ? OK_A[109] || set_clr_onehot[109] : OK_A[109] && !set_clr_onehot[109];       //update

assign ready_109_A = OK_onehot_A[109] && (valid_bank6_B ? ready_bank6_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[109] |-> !OK_A[109])
`assert_clk(!so_or_sz && set_clr_onehot[109] |->  OK_A[109])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[110] <= `CQ reset_value;                        
        else if(ready_110_A)
            OK_A[110] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[110] <= `CQ so_or_sz ? OK_A[110] || set_clr_onehot[110] : OK_A[110] && !set_clr_onehot[110];       //update

assign ready_110_A = OK_onehot_A[110] && (valid_bank6_B ? ready_bank6_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[110] |-> !OK_A[110])
`assert_clk(!so_or_sz && set_clr_onehot[110] |->  OK_A[110])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[111] <= `CQ reset_value;                        
        else if(ready_111_A)
            OK_A[111] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[111] <= `CQ so_or_sz ? OK_A[111] || set_clr_onehot[111] : OK_A[111] && !set_clr_onehot[111];       //update

assign ready_111_A = OK_onehot_A[111] && (valid_bank6_B ? ready_bank6_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[111] |-> !OK_A[111])
`assert_clk(!so_or_sz && set_clr_onehot[111] |->  OK_A[111])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[112] <= `CQ reset_value;                        
        else if(ready_112_A)
            OK_A[112] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[112] <= `CQ so_or_sz ? OK_A[112] || set_clr_onehot[112] : OK_A[112] && !set_clr_onehot[112];       //update

assign ready_112_A = OK_onehot_A[112] && (valid_bank7_B ? ready_bank7_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[112] |-> !OK_A[112])
`assert_clk(!so_or_sz && set_clr_onehot[112] |->  OK_A[112])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[113] <= `CQ reset_value;                        
        else if(ready_113_A)
            OK_A[113] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[113] <= `CQ so_or_sz ? OK_A[113] || set_clr_onehot[113] : OK_A[113] && !set_clr_onehot[113];       //update

assign ready_113_A = OK_onehot_A[113] && (valid_bank7_B ? ready_bank7_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[113] |-> !OK_A[113])
`assert_clk(!so_or_sz && set_clr_onehot[113] |->  OK_A[113])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[114] <= `CQ reset_value;                        
        else if(ready_114_A)
            OK_A[114] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[114] <= `CQ so_or_sz ? OK_A[114] || set_clr_onehot[114] : OK_A[114] && !set_clr_onehot[114];       //update

assign ready_114_A = OK_onehot_A[114] && (valid_bank7_B ? ready_bank7_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[114] |-> !OK_A[114])
`assert_clk(!so_or_sz && set_clr_onehot[114] |->  OK_A[114])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[115] <= `CQ reset_value;                        
        else if(ready_115_A)
            OK_A[115] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[115] <= `CQ so_or_sz ? OK_A[115] || set_clr_onehot[115] : OK_A[115] && !set_clr_onehot[115];       //update

assign ready_115_A = OK_onehot_A[115] && (valid_bank7_B ? ready_bank7_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[115] |-> !OK_A[115])
`assert_clk(!so_or_sz && set_clr_onehot[115] |->  OK_A[115])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[116] <= `CQ reset_value;                        
        else if(ready_116_A)
            OK_A[116] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[116] <= `CQ so_or_sz ? OK_A[116] || set_clr_onehot[116] : OK_A[116] && !set_clr_onehot[116];       //update

assign ready_116_A = OK_onehot_A[116] && (valid_bank7_B ? ready_bank7_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[116] |-> !OK_A[116])
`assert_clk(!so_or_sz && set_clr_onehot[116] |->  OK_A[116])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[117] <= `CQ reset_value;                        
        else if(ready_117_A)
            OK_A[117] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[117] <= `CQ so_or_sz ? OK_A[117] || set_clr_onehot[117] : OK_A[117] && !set_clr_onehot[117];       //update

assign ready_117_A = OK_onehot_A[117] && (valid_bank7_B ? ready_bank7_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[117] |-> !OK_A[117])
`assert_clk(!so_or_sz && set_clr_onehot[117] |->  OK_A[117])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[118] <= `CQ reset_value;                        
        else if(ready_118_A)
            OK_A[118] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[118] <= `CQ so_or_sz ? OK_A[118] || set_clr_onehot[118] : OK_A[118] && !set_clr_onehot[118];       //update

assign ready_118_A = OK_onehot_A[118] && (valid_bank7_B ? ready_bank7_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[118] |-> !OK_A[118])
`assert_clk(!so_or_sz && set_clr_onehot[118] |->  OK_A[118])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[119] <= `CQ reset_value;                        
        else if(ready_119_A)
            OK_A[119] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[119] <= `CQ so_or_sz ? OK_A[119] || set_clr_onehot[119] : OK_A[119] && !set_clr_onehot[119];       //update

assign ready_119_A = OK_onehot_A[119] && (valid_bank7_B ? ready_bank7_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[119] |-> !OK_A[119])
`assert_clk(!so_or_sz && set_clr_onehot[119] |->  OK_A[119])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[120] <= `CQ reset_value;                        
        else if(ready_120_A)
            OK_A[120] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[120] <= `CQ so_or_sz ? OK_A[120] || set_clr_onehot[120] : OK_A[120] && !set_clr_onehot[120];       //update

assign ready_120_A = OK_onehot_A[120] && (valid_bank7_B ? ready_bank7_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[120] |-> !OK_A[120])
`assert_clk(!so_or_sz && set_clr_onehot[120] |->  OK_A[120])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[121] <= `CQ reset_value;                        
        else if(ready_121_A)
            OK_A[121] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[121] <= `CQ so_or_sz ? OK_A[121] || set_clr_onehot[121] : OK_A[121] && !set_clr_onehot[121];       //update

assign ready_121_A = OK_onehot_A[121] && (valid_bank7_B ? ready_bank7_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[121] |-> !OK_A[121])
`assert_clk(!so_or_sz && set_clr_onehot[121] |->  OK_A[121])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[122] <= `CQ reset_value;                        
        else if(ready_122_A)
            OK_A[122] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[122] <= `CQ so_or_sz ? OK_A[122] || set_clr_onehot[122] : OK_A[122] && !set_clr_onehot[122];       //update

assign ready_122_A = OK_onehot_A[122] && (valid_bank7_B ? ready_bank7_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[122] |-> !OK_A[122])
`assert_clk(!so_or_sz && set_clr_onehot[122] |->  OK_A[122])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[123] <= `CQ reset_value;                        
        else if(ready_123_A)
            OK_A[123] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[123] <= `CQ so_or_sz ? OK_A[123] || set_clr_onehot[123] : OK_A[123] && !set_clr_onehot[123];       //update

assign ready_123_A = OK_onehot_A[123] && (valid_bank7_B ? ready_bank7_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[123] |-> !OK_A[123])
`assert_clk(!so_or_sz && set_clr_onehot[123] |->  OK_A[123])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[124] <= `CQ reset_value;                        
        else if(ready_124_A)
            OK_A[124] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[124] <= `CQ so_or_sz ? OK_A[124] || set_clr_onehot[124] : OK_A[124] && !set_clr_onehot[124];       //update

assign ready_124_A = OK_onehot_A[124] && (valid_bank7_B ? ready_bank7_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[124] |-> !OK_A[124])
`assert_clk(!so_or_sz && set_clr_onehot[124] |->  OK_A[124])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[125] <= `CQ reset_value;                        
        else if(ready_125_A)
            OK_A[125] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[125] <= `CQ so_or_sz ? OK_A[125] || set_clr_onehot[125] : OK_A[125] && !set_clr_onehot[125];       //update

assign ready_125_A = OK_onehot_A[125] && (valid_bank7_B ? ready_bank7_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[125] |-> !OK_A[125])
`assert_clk(!so_or_sz && set_clr_onehot[125] |->  OK_A[125])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[126] <= `CQ reset_value;                        
        else if(ready_126_A)
            OK_A[126] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[126] <= `CQ so_or_sz ? OK_A[126] || set_clr_onehot[126] : OK_A[126] && !set_clr_onehot[126];       //update

assign ready_126_A = OK_onehot_A[126] && (valid_bank7_B ? ready_bank7_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[126] |-> !OK_A[126])
`assert_clk(!so_or_sz && set_clr_onehot[126] |->  OK_A[126])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[127] <= `CQ reset_value;                        
        else if(ready_127_A)
            OK_A[127] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[127] <= `CQ so_or_sz ? OK_A[127] || set_clr_onehot[127] : OK_A[127] && !set_clr_onehot[127];       //update

assign ready_127_A = OK_onehot_A[127] && (valid_bank7_B ? ready_bank7_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[127] |-> !OK_A[127])
`assert_clk(!so_or_sz && set_clr_onehot[127] |->  OK_A[127])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[128] <= `CQ reset_value;                        
        else if(ready_128_A)
            OK_A[128] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[128] <= `CQ so_or_sz ? OK_A[128] || set_clr_onehot[128] : OK_A[128] && !set_clr_onehot[128];       //update

assign ready_128_A = OK_onehot_A[128] && (valid_bank8_B ? ready_bank8_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[128] |-> !OK_A[128])
`assert_clk(!so_or_sz && set_clr_onehot[128] |->  OK_A[128])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[129] <= `CQ reset_value;                        
        else if(ready_129_A)
            OK_A[129] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[129] <= `CQ so_or_sz ? OK_A[129] || set_clr_onehot[129] : OK_A[129] && !set_clr_onehot[129];       //update

assign ready_129_A = OK_onehot_A[129] && (valid_bank8_B ? ready_bank8_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[129] |-> !OK_A[129])
`assert_clk(!so_or_sz && set_clr_onehot[129] |->  OK_A[129])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[130] <= `CQ reset_value;                        
        else if(ready_130_A)
            OK_A[130] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[130] <= `CQ so_or_sz ? OK_A[130] || set_clr_onehot[130] : OK_A[130] && !set_clr_onehot[130];       //update

assign ready_130_A = OK_onehot_A[130] && (valid_bank8_B ? ready_bank8_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[130] |-> !OK_A[130])
`assert_clk(!so_or_sz && set_clr_onehot[130] |->  OK_A[130])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[131] <= `CQ reset_value;                        
        else if(ready_131_A)
            OK_A[131] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[131] <= `CQ so_or_sz ? OK_A[131] || set_clr_onehot[131] : OK_A[131] && !set_clr_onehot[131];       //update

assign ready_131_A = OK_onehot_A[131] && (valid_bank8_B ? ready_bank8_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[131] |-> !OK_A[131])
`assert_clk(!so_or_sz && set_clr_onehot[131] |->  OK_A[131])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[132] <= `CQ reset_value;                        
        else if(ready_132_A)
            OK_A[132] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[132] <= `CQ so_or_sz ? OK_A[132] || set_clr_onehot[132] : OK_A[132] && !set_clr_onehot[132];       //update

assign ready_132_A = OK_onehot_A[132] && (valid_bank8_B ? ready_bank8_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[132] |-> !OK_A[132])
`assert_clk(!so_or_sz && set_clr_onehot[132] |->  OK_A[132])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[133] <= `CQ reset_value;                        
        else if(ready_133_A)
            OK_A[133] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[133] <= `CQ so_or_sz ? OK_A[133] || set_clr_onehot[133] : OK_A[133] && !set_clr_onehot[133];       //update

assign ready_133_A = OK_onehot_A[133] && (valid_bank8_B ? ready_bank8_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[133] |-> !OK_A[133])
`assert_clk(!so_or_sz && set_clr_onehot[133] |->  OK_A[133])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[134] <= `CQ reset_value;                        
        else if(ready_134_A)
            OK_A[134] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[134] <= `CQ so_or_sz ? OK_A[134] || set_clr_onehot[134] : OK_A[134] && !set_clr_onehot[134];       //update

assign ready_134_A = OK_onehot_A[134] && (valid_bank8_B ? ready_bank8_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[134] |-> !OK_A[134])
`assert_clk(!so_or_sz && set_clr_onehot[134] |->  OK_A[134])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[135] <= `CQ reset_value;                        
        else if(ready_135_A)
            OK_A[135] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[135] <= `CQ so_or_sz ? OK_A[135] || set_clr_onehot[135] : OK_A[135] && !set_clr_onehot[135];       //update

assign ready_135_A = OK_onehot_A[135] && (valid_bank8_B ? ready_bank8_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[135] |-> !OK_A[135])
`assert_clk(!so_or_sz && set_clr_onehot[135] |->  OK_A[135])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[136] <= `CQ reset_value;                        
        else if(ready_136_A)
            OK_A[136] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[136] <= `CQ so_or_sz ? OK_A[136] || set_clr_onehot[136] : OK_A[136] && !set_clr_onehot[136];       //update

assign ready_136_A = OK_onehot_A[136] && (valid_bank8_B ? ready_bank8_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[136] |-> !OK_A[136])
`assert_clk(!so_or_sz && set_clr_onehot[136] |->  OK_A[136])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[137] <= `CQ reset_value;                        
        else if(ready_137_A)
            OK_A[137] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[137] <= `CQ so_or_sz ? OK_A[137] || set_clr_onehot[137] : OK_A[137] && !set_clr_onehot[137];       //update

assign ready_137_A = OK_onehot_A[137] && (valid_bank8_B ? ready_bank8_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[137] |-> !OK_A[137])
`assert_clk(!so_or_sz && set_clr_onehot[137] |->  OK_A[137])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[138] <= `CQ reset_value;                        
        else if(ready_138_A)
            OK_A[138] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[138] <= `CQ so_or_sz ? OK_A[138] || set_clr_onehot[138] : OK_A[138] && !set_clr_onehot[138];       //update

assign ready_138_A = OK_onehot_A[138] && (valid_bank8_B ? ready_bank8_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[138] |-> !OK_A[138])
`assert_clk(!so_or_sz && set_clr_onehot[138] |->  OK_A[138])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[139] <= `CQ reset_value;                        
        else if(ready_139_A)
            OK_A[139] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[139] <= `CQ so_or_sz ? OK_A[139] || set_clr_onehot[139] : OK_A[139] && !set_clr_onehot[139];       //update

assign ready_139_A = OK_onehot_A[139] && (valid_bank8_B ? ready_bank8_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[139] |-> !OK_A[139])
`assert_clk(!so_or_sz && set_clr_onehot[139] |->  OK_A[139])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[140] <= `CQ reset_value;                        
        else if(ready_140_A)
            OK_A[140] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[140] <= `CQ so_or_sz ? OK_A[140] || set_clr_onehot[140] : OK_A[140] && !set_clr_onehot[140];       //update

assign ready_140_A = OK_onehot_A[140] && (valid_bank8_B ? ready_bank8_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[140] |-> !OK_A[140])
`assert_clk(!so_or_sz && set_clr_onehot[140] |->  OK_A[140])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[141] <= `CQ reset_value;                        
        else if(ready_141_A)
            OK_A[141] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[141] <= `CQ so_or_sz ? OK_A[141] || set_clr_onehot[141] : OK_A[141] && !set_clr_onehot[141];       //update

assign ready_141_A = OK_onehot_A[141] && (valid_bank8_B ? ready_bank8_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[141] |-> !OK_A[141])
`assert_clk(!so_or_sz && set_clr_onehot[141] |->  OK_A[141])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[142] <= `CQ reset_value;                        
        else if(ready_142_A)
            OK_A[142] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[142] <= `CQ so_or_sz ? OK_A[142] || set_clr_onehot[142] : OK_A[142] && !set_clr_onehot[142];       //update

assign ready_142_A = OK_onehot_A[142] && (valid_bank8_B ? ready_bank8_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[142] |-> !OK_A[142])
`assert_clk(!so_or_sz && set_clr_onehot[142] |->  OK_A[142])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[143] <= `CQ reset_value;                        
        else if(ready_143_A)
            OK_A[143] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[143] <= `CQ so_or_sz ? OK_A[143] || set_clr_onehot[143] : OK_A[143] && !set_clr_onehot[143];       //update

assign ready_143_A = OK_onehot_A[143] && (valid_bank8_B ? ready_bank8_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[143] |-> !OK_A[143])
`assert_clk(!so_or_sz && set_clr_onehot[143] |->  OK_A[143])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[144] <= `CQ reset_value;                        
        else if(ready_144_A)
            OK_A[144] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[144] <= `CQ so_or_sz ? OK_A[144] || set_clr_onehot[144] : OK_A[144] && !set_clr_onehot[144];       //update

assign ready_144_A = OK_onehot_A[144] && (valid_bank9_B ? ready_bank9_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[144] |-> !OK_A[144])
`assert_clk(!so_or_sz && set_clr_onehot[144] |->  OK_A[144])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[145] <= `CQ reset_value;                        
        else if(ready_145_A)
            OK_A[145] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[145] <= `CQ so_or_sz ? OK_A[145] || set_clr_onehot[145] : OK_A[145] && !set_clr_onehot[145];       //update

assign ready_145_A = OK_onehot_A[145] && (valid_bank9_B ? ready_bank9_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[145] |-> !OK_A[145])
`assert_clk(!so_or_sz && set_clr_onehot[145] |->  OK_A[145])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[146] <= `CQ reset_value;                        
        else if(ready_146_A)
            OK_A[146] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[146] <= `CQ so_or_sz ? OK_A[146] || set_clr_onehot[146] : OK_A[146] && !set_clr_onehot[146];       //update

assign ready_146_A = OK_onehot_A[146] && (valid_bank9_B ? ready_bank9_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[146] |-> !OK_A[146])
`assert_clk(!so_or_sz && set_clr_onehot[146] |->  OK_A[146])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[147] <= `CQ reset_value;                        
        else if(ready_147_A)
            OK_A[147] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[147] <= `CQ so_or_sz ? OK_A[147] || set_clr_onehot[147] : OK_A[147] && !set_clr_onehot[147];       //update

assign ready_147_A = OK_onehot_A[147] && (valid_bank9_B ? ready_bank9_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[147] |-> !OK_A[147])
`assert_clk(!so_or_sz && set_clr_onehot[147] |->  OK_A[147])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[148] <= `CQ reset_value;                        
        else if(ready_148_A)
            OK_A[148] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[148] <= `CQ so_or_sz ? OK_A[148] || set_clr_onehot[148] : OK_A[148] && !set_clr_onehot[148];       //update

assign ready_148_A = OK_onehot_A[148] && (valid_bank9_B ? ready_bank9_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[148] |-> !OK_A[148])
`assert_clk(!so_or_sz && set_clr_onehot[148] |->  OK_A[148])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[149] <= `CQ reset_value;                        
        else if(ready_149_A)
            OK_A[149] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[149] <= `CQ so_or_sz ? OK_A[149] || set_clr_onehot[149] : OK_A[149] && !set_clr_onehot[149];       //update

assign ready_149_A = OK_onehot_A[149] && (valid_bank9_B ? ready_bank9_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[149] |-> !OK_A[149])
`assert_clk(!so_or_sz && set_clr_onehot[149] |->  OK_A[149])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[150] <= `CQ reset_value;                        
        else if(ready_150_A)
            OK_A[150] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[150] <= `CQ so_or_sz ? OK_A[150] || set_clr_onehot[150] : OK_A[150] && !set_clr_onehot[150];       //update

assign ready_150_A = OK_onehot_A[150] && (valid_bank9_B ? ready_bank9_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[150] |-> !OK_A[150])
`assert_clk(!so_or_sz && set_clr_onehot[150] |->  OK_A[150])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[151] <= `CQ reset_value;                        
        else if(ready_151_A)
            OK_A[151] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[151] <= `CQ so_or_sz ? OK_A[151] || set_clr_onehot[151] : OK_A[151] && !set_clr_onehot[151];       //update

assign ready_151_A = OK_onehot_A[151] && (valid_bank9_B ? ready_bank9_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[151] |-> !OK_A[151])
`assert_clk(!so_or_sz && set_clr_onehot[151] |->  OK_A[151])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[152] <= `CQ reset_value;                        
        else if(ready_152_A)
            OK_A[152] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[152] <= `CQ so_or_sz ? OK_A[152] || set_clr_onehot[152] : OK_A[152] && !set_clr_onehot[152];       //update

assign ready_152_A = OK_onehot_A[152] && (valid_bank9_B ? ready_bank9_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[152] |-> !OK_A[152])
`assert_clk(!so_or_sz && set_clr_onehot[152] |->  OK_A[152])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[153] <= `CQ reset_value;                        
        else if(ready_153_A)
            OK_A[153] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[153] <= `CQ so_or_sz ? OK_A[153] || set_clr_onehot[153] : OK_A[153] && !set_clr_onehot[153];       //update

assign ready_153_A = OK_onehot_A[153] && (valid_bank9_B ? ready_bank9_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[153] |-> !OK_A[153])
`assert_clk(!so_or_sz && set_clr_onehot[153] |->  OK_A[153])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[154] <= `CQ reset_value;                        
        else if(ready_154_A)
            OK_A[154] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[154] <= `CQ so_or_sz ? OK_A[154] || set_clr_onehot[154] : OK_A[154] && !set_clr_onehot[154];       //update

assign ready_154_A = OK_onehot_A[154] && (valid_bank9_B ? ready_bank9_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[154] |-> !OK_A[154])
`assert_clk(!so_or_sz && set_clr_onehot[154] |->  OK_A[154])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[155] <= `CQ reset_value;                        
        else if(ready_155_A)
            OK_A[155] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[155] <= `CQ so_or_sz ? OK_A[155] || set_clr_onehot[155] : OK_A[155] && !set_clr_onehot[155];       //update

assign ready_155_A = OK_onehot_A[155] && (valid_bank9_B ? ready_bank9_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[155] |-> !OK_A[155])
`assert_clk(!so_or_sz && set_clr_onehot[155] |->  OK_A[155])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[156] <= `CQ reset_value;                        
        else if(ready_156_A)
            OK_A[156] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[156] <= `CQ so_or_sz ? OK_A[156] || set_clr_onehot[156] : OK_A[156] && !set_clr_onehot[156];       //update

assign ready_156_A = OK_onehot_A[156] && (valid_bank9_B ? ready_bank9_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[156] |-> !OK_A[156])
`assert_clk(!so_or_sz && set_clr_onehot[156] |->  OK_A[156])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[157] <= `CQ reset_value;                        
        else if(ready_157_A)
            OK_A[157] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[157] <= `CQ so_or_sz ? OK_A[157] || set_clr_onehot[157] : OK_A[157] && !set_clr_onehot[157];       //update

assign ready_157_A = OK_onehot_A[157] && (valid_bank9_B ? ready_bank9_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[157] |-> !OK_A[157])
`assert_clk(!so_or_sz && set_clr_onehot[157] |->  OK_A[157])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[158] <= `CQ reset_value;                        
        else if(ready_158_A)
            OK_A[158] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[158] <= `CQ so_or_sz ? OK_A[158] || set_clr_onehot[158] : OK_A[158] && !set_clr_onehot[158];       //update

assign ready_158_A = OK_onehot_A[158] && (valid_bank9_B ? ready_bank9_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[158] |-> !OK_A[158])
`assert_clk(!so_or_sz && set_clr_onehot[158] |->  OK_A[158])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[159] <= `CQ reset_value;                        
        else if(ready_159_A)
            OK_A[159] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[159] <= `CQ so_or_sz ? OK_A[159] || set_clr_onehot[159] : OK_A[159] && !set_clr_onehot[159];       //update

assign ready_159_A = OK_onehot_A[159] && (valid_bank9_B ? ready_bank9_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[159] |-> !OK_A[159])
`assert_clk(!so_or_sz && set_clr_onehot[159] |->  OK_A[159])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[160] <= `CQ reset_value;                        
        else if(ready_160_A)
            OK_A[160] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[160] <= `CQ so_or_sz ? OK_A[160] || set_clr_onehot[160] : OK_A[160] && !set_clr_onehot[160];       //update

assign ready_160_A = OK_onehot_A[160] && (valid_bank10_B ? ready_bank10_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[160] |-> !OK_A[160])
`assert_clk(!so_or_sz && set_clr_onehot[160] |->  OK_A[160])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[161] <= `CQ reset_value;                        
        else if(ready_161_A)
            OK_A[161] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[161] <= `CQ so_or_sz ? OK_A[161] || set_clr_onehot[161] : OK_A[161] && !set_clr_onehot[161];       //update

assign ready_161_A = OK_onehot_A[161] && (valid_bank10_B ? ready_bank10_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[161] |-> !OK_A[161])
`assert_clk(!so_or_sz && set_clr_onehot[161] |->  OK_A[161])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[162] <= `CQ reset_value;                        
        else if(ready_162_A)
            OK_A[162] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[162] <= `CQ so_or_sz ? OK_A[162] || set_clr_onehot[162] : OK_A[162] && !set_clr_onehot[162];       //update

assign ready_162_A = OK_onehot_A[162] && (valid_bank10_B ? ready_bank10_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[162] |-> !OK_A[162])
`assert_clk(!so_or_sz && set_clr_onehot[162] |->  OK_A[162])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[163] <= `CQ reset_value;                        
        else if(ready_163_A)
            OK_A[163] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[163] <= `CQ so_or_sz ? OK_A[163] || set_clr_onehot[163] : OK_A[163] && !set_clr_onehot[163];       //update

assign ready_163_A = OK_onehot_A[163] && (valid_bank10_B ? ready_bank10_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[163] |-> !OK_A[163])
`assert_clk(!so_or_sz && set_clr_onehot[163] |->  OK_A[163])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[164] <= `CQ reset_value;                        
        else if(ready_164_A)
            OK_A[164] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[164] <= `CQ so_or_sz ? OK_A[164] || set_clr_onehot[164] : OK_A[164] && !set_clr_onehot[164];       //update

assign ready_164_A = OK_onehot_A[164] && (valid_bank10_B ? ready_bank10_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[164] |-> !OK_A[164])
`assert_clk(!so_or_sz && set_clr_onehot[164] |->  OK_A[164])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[165] <= `CQ reset_value;                        
        else if(ready_165_A)
            OK_A[165] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[165] <= `CQ so_or_sz ? OK_A[165] || set_clr_onehot[165] : OK_A[165] && !set_clr_onehot[165];       //update

assign ready_165_A = OK_onehot_A[165] && (valid_bank10_B ? ready_bank10_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[165] |-> !OK_A[165])
`assert_clk(!so_or_sz && set_clr_onehot[165] |->  OK_A[165])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[166] <= `CQ reset_value;                        
        else if(ready_166_A)
            OK_A[166] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[166] <= `CQ so_or_sz ? OK_A[166] || set_clr_onehot[166] : OK_A[166] && !set_clr_onehot[166];       //update

assign ready_166_A = OK_onehot_A[166] && (valid_bank10_B ? ready_bank10_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[166] |-> !OK_A[166])
`assert_clk(!so_or_sz && set_clr_onehot[166] |->  OK_A[166])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[167] <= `CQ reset_value;                        
        else if(ready_167_A)
            OK_A[167] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[167] <= `CQ so_or_sz ? OK_A[167] || set_clr_onehot[167] : OK_A[167] && !set_clr_onehot[167];       //update

assign ready_167_A = OK_onehot_A[167] && (valid_bank10_B ? ready_bank10_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[167] |-> !OK_A[167])
`assert_clk(!so_or_sz && set_clr_onehot[167] |->  OK_A[167])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[168] <= `CQ reset_value;                        
        else if(ready_168_A)
            OK_A[168] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[168] <= `CQ so_or_sz ? OK_A[168] || set_clr_onehot[168] : OK_A[168] && !set_clr_onehot[168];       //update

assign ready_168_A = OK_onehot_A[168] && (valid_bank10_B ? ready_bank10_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[168] |-> !OK_A[168])
`assert_clk(!so_or_sz && set_clr_onehot[168] |->  OK_A[168])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[169] <= `CQ reset_value;                        
        else if(ready_169_A)
            OK_A[169] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[169] <= `CQ so_or_sz ? OK_A[169] || set_clr_onehot[169] : OK_A[169] && !set_clr_onehot[169];       //update

assign ready_169_A = OK_onehot_A[169] && (valid_bank10_B ? ready_bank10_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[169] |-> !OK_A[169])
`assert_clk(!so_or_sz && set_clr_onehot[169] |->  OK_A[169])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[170] <= `CQ reset_value;                        
        else if(ready_170_A)
            OK_A[170] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[170] <= `CQ so_or_sz ? OK_A[170] || set_clr_onehot[170] : OK_A[170] && !set_clr_onehot[170];       //update

assign ready_170_A = OK_onehot_A[170] && (valid_bank10_B ? ready_bank10_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[170] |-> !OK_A[170])
`assert_clk(!so_or_sz && set_clr_onehot[170] |->  OK_A[170])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[171] <= `CQ reset_value;                        
        else if(ready_171_A)
            OK_A[171] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[171] <= `CQ so_or_sz ? OK_A[171] || set_clr_onehot[171] : OK_A[171] && !set_clr_onehot[171];       //update

assign ready_171_A = OK_onehot_A[171] && (valid_bank10_B ? ready_bank10_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[171] |-> !OK_A[171])
`assert_clk(!so_or_sz && set_clr_onehot[171] |->  OK_A[171])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[172] <= `CQ reset_value;                        
        else if(ready_172_A)
            OK_A[172] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[172] <= `CQ so_or_sz ? OK_A[172] || set_clr_onehot[172] : OK_A[172] && !set_clr_onehot[172];       //update

assign ready_172_A = OK_onehot_A[172] && (valid_bank10_B ? ready_bank10_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[172] |-> !OK_A[172])
`assert_clk(!so_or_sz && set_clr_onehot[172] |->  OK_A[172])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[173] <= `CQ reset_value;                        
        else if(ready_173_A)
            OK_A[173] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[173] <= `CQ so_or_sz ? OK_A[173] || set_clr_onehot[173] : OK_A[173] && !set_clr_onehot[173];       //update

assign ready_173_A = OK_onehot_A[173] && (valid_bank10_B ? ready_bank10_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[173] |-> !OK_A[173])
`assert_clk(!so_or_sz && set_clr_onehot[173] |->  OK_A[173])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[174] <= `CQ reset_value;                        
        else if(ready_174_A)
            OK_A[174] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[174] <= `CQ so_or_sz ? OK_A[174] || set_clr_onehot[174] : OK_A[174] && !set_clr_onehot[174];       //update

assign ready_174_A = OK_onehot_A[174] && (valid_bank10_B ? ready_bank10_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[174] |-> !OK_A[174])
`assert_clk(!so_or_sz && set_clr_onehot[174] |->  OK_A[174])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[175] <= `CQ reset_value;                        
        else if(ready_175_A)
            OK_A[175] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[175] <= `CQ so_or_sz ? OK_A[175] || set_clr_onehot[175] : OK_A[175] && !set_clr_onehot[175];       //update

assign ready_175_A = OK_onehot_A[175] && (valid_bank10_B ? ready_bank10_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[175] |-> !OK_A[175])
`assert_clk(!so_or_sz && set_clr_onehot[175] |->  OK_A[175])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[176] <= `CQ reset_value;                        
        else if(ready_176_A)
            OK_A[176] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[176] <= `CQ so_or_sz ? OK_A[176] || set_clr_onehot[176] : OK_A[176] && !set_clr_onehot[176];       //update

assign ready_176_A = OK_onehot_A[176] && (valid_bank11_B ? ready_bank11_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[176] |-> !OK_A[176])
`assert_clk(!so_or_sz && set_clr_onehot[176] |->  OK_A[176])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[177] <= `CQ reset_value;                        
        else if(ready_177_A)
            OK_A[177] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[177] <= `CQ so_or_sz ? OK_A[177] || set_clr_onehot[177] : OK_A[177] && !set_clr_onehot[177];       //update

assign ready_177_A = OK_onehot_A[177] && (valid_bank11_B ? ready_bank11_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[177] |-> !OK_A[177])
`assert_clk(!so_or_sz && set_clr_onehot[177] |->  OK_A[177])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[178] <= `CQ reset_value;                        
        else if(ready_178_A)
            OK_A[178] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[178] <= `CQ so_or_sz ? OK_A[178] || set_clr_onehot[178] : OK_A[178] && !set_clr_onehot[178];       //update

assign ready_178_A = OK_onehot_A[178] && (valid_bank11_B ? ready_bank11_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[178] |-> !OK_A[178])
`assert_clk(!so_or_sz && set_clr_onehot[178] |->  OK_A[178])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[179] <= `CQ reset_value;                        
        else if(ready_179_A)
            OK_A[179] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[179] <= `CQ so_or_sz ? OK_A[179] || set_clr_onehot[179] : OK_A[179] && !set_clr_onehot[179];       //update

assign ready_179_A = OK_onehot_A[179] && (valid_bank11_B ? ready_bank11_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[179] |-> !OK_A[179])
`assert_clk(!so_or_sz && set_clr_onehot[179] |->  OK_A[179])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[180] <= `CQ reset_value;                        
        else if(ready_180_A)
            OK_A[180] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[180] <= `CQ so_or_sz ? OK_A[180] || set_clr_onehot[180] : OK_A[180] && !set_clr_onehot[180];       //update

assign ready_180_A = OK_onehot_A[180] && (valid_bank11_B ? ready_bank11_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[180] |-> !OK_A[180])
`assert_clk(!so_or_sz && set_clr_onehot[180] |->  OK_A[180])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[181] <= `CQ reset_value;                        
        else if(ready_181_A)
            OK_A[181] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[181] <= `CQ so_or_sz ? OK_A[181] || set_clr_onehot[181] : OK_A[181] && !set_clr_onehot[181];       //update

assign ready_181_A = OK_onehot_A[181] && (valid_bank11_B ? ready_bank11_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[181] |-> !OK_A[181])
`assert_clk(!so_or_sz && set_clr_onehot[181] |->  OK_A[181])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[182] <= `CQ reset_value;                        
        else if(ready_182_A)
            OK_A[182] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[182] <= `CQ so_or_sz ? OK_A[182] || set_clr_onehot[182] : OK_A[182] && !set_clr_onehot[182];       //update

assign ready_182_A = OK_onehot_A[182] && (valid_bank11_B ? ready_bank11_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[182] |-> !OK_A[182])
`assert_clk(!so_or_sz && set_clr_onehot[182] |->  OK_A[182])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[183] <= `CQ reset_value;                        
        else if(ready_183_A)
            OK_A[183] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[183] <= `CQ so_or_sz ? OK_A[183] || set_clr_onehot[183] : OK_A[183] && !set_clr_onehot[183];       //update

assign ready_183_A = OK_onehot_A[183] && (valid_bank11_B ? ready_bank11_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[183] |-> !OK_A[183])
`assert_clk(!so_or_sz && set_clr_onehot[183] |->  OK_A[183])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[184] <= `CQ reset_value;                        
        else if(ready_184_A)
            OK_A[184] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[184] <= `CQ so_or_sz ? OK_A[184] || set_clr_onehot[184] : OK_A[184] && !set_clr_onehot[184];       //update

assign ready_184_A = OK_onehot_A[184] && (valid_bank11_B ? ready_bank11_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[184] |-> !OK_A[184])
`assert_clk(!so_or_sz && set_clr_onehot[184] |->  OK_A[184])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[185] <= `CQ reset_value;                        
        else if(ready_185_A)
            OK_A[185] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[185] <= `CQ so_or_sz ? OK_A[185] || set_clr_onehot[185] : OK_A[185] && !set_clr_onehot[185];       //update

assign ready_185_A = OK_onehot_A[185] && (valid_bank11_B ? ready_bank11_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[185] |-> !OK_A[185])
`assert_clk(!so_or_sz && set_clr_onehot[185] |->  OK_A[185])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[186] <= `CQ reset_value;                        
        else if(ready_186_A)
            OK_A[186] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[186] <= `CQ so_or_sz ? OK_A[186] || set_clr_onehot[186] : OK_A[186] && !set_clr_onehot[186];       //update

assign ready_186_A = OK_onehot_A[186] && (valid_bank11_B ? ready_bank11_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[186] |-> !OK_A[186])
`assert_clk(!so_or_sz && set_clr_onehot[186] |->  OK_A[186])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[187] <= `CQ reset_value;                        
        else if(ready_187_A)
            OK_A[187] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[187] <= `CQ so_or_sz ? OK_A[187] || set_clr_onehot[187] : OK_A[187] && !set_clr_onehot[187];       //update

assign ready_187_A = OK_onehot_A[187] && (valid_bank11_B ? ready_bank11_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[187] |-> !OK_A[187])
`assert_clk(!so_or_sz && set_clr_onehot[187] |->  OK_A[187])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[188] <= `CQ reset_value;                        
        else if(ready_188_A)
            OK_A[188] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[188] <= `CQ so_or_sz ? OK_A[188] || set_clr_onehot[188] : OK_A[188] && !set_clr_onehot[188];       //update

assign ready_188_A = OK_onehot_A[188] && (valid_bank11_B ? ready_bank11_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[188] |-> !OK_A[188])
`assert_clk(!so_or_sz && set_clr_onehot[188] |->  OK_A[188])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[189] <= `CQ reset_value;                        
        else if(ready_189_A)
            OK_A[189] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[189] <= `CQ so_or_sz ? OK_A[189] || set_clr_onehot[189] : OK_A[189] && !set_clr_onehot[189];       //update

assign ready_189_A = OK_onehot_A[189] && (valid_bank11_B ? ready_bank11_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[189] |-> !OK_A[189])
`assert_clk(!so_or_sz && set_clr_onehot[189] |->  OK_A[189])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[190] <= `CQ reset_value;                        
        else if(ready_190_A)
            OK_A[190] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[190] <= `CQ so_or_sz ? OK_A[190] || set_clr_onehot[190] : OK_A[190] && !set_clr_onehot[190];       //update

assign ready_190_A = OK_onehot_A[190] && (valid_bank11_B ? ready_bank11_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[190] |-> !OK_A[190])
`assert_clk(!so_or_sz && set_clr_onehot[190] |->  OK_A[190])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[191] <= `CQ reset_value;                        
        else if(ready_191_A)
            OK_A[191] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[191] <= `CQ so_or_sz ? OK_A[191] || set_clr_onehot[191] : OK_A[191] && !set_clr_onehot[191];       //update

assign ready_191_A = OK_onehot_A[191] && (valid_bank11_B ? ready_bank11_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[191] |-> !OK_A[191])
`assert_clk(!so_or_sz && set_clr_onehot[191] |->  OK_A[191])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[192] <= `CQ reset_value;                        
        else if(ready_192_A)
            OK_A[192] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[192] <= `CQ so_or_sz ? OK_A[192] || set_clr_onehot[192] : OK_A[192] && !set_clr_onehot[192];       //update

assign ready_192_A = OK_onehot_A[192] && (valid_bank12_B ? ready_bank12_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[192] |-> !OK_A[192])
`assert_clk(!so_or_sz && set_clr_onehot[192] |->  OK_A[192])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[193] <= `CQ reset_value;                        
        else if(ready_193_A)
            OK_A[193] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[193] <= `CQ so_or_sz ? OK_A[193] || set_clr_onehot[193] : OK_A[193] && !set_clr_onehot[193];       //update

assign ready_193_A = OK_onehot_A[193] && (valid_bank12_B ? ready_bank12_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[193] |-> !OK_A[193])
`assert_clk(!so_or_sz && set_clr_onehot[193] |->  OK_A[193])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[194] <= `CQ reset_value;                        
        else if(ready_194_A)
            OK_A[194] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[194] <= `CQ so_or_sz ? OK_A[194] || set_clr_onehot[194] : OK_A[194] && !set_clr_onehot[194];       //update

assign ready_194_A = OK_onehot_A[194] && (valid_bank12_B ? ready_bank12_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[194] |-> !OK_A[194])
`assert_clk(!so_or_sz && set_clr_onehot[194] |->  OK_A[194])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[195] <= `CQ reset_value;                        
        else if(ready_195_A)
            OK_A[195] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[195] <= `CQ so_or_sz ? OK_A[195] || set_clr_onehot[195] : OK_A[195] && !set_clr_onehot[195];       //update

assign ready_195_A = OK_onehot_A[195] && (valid_bank12_B ? ready_bank12_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[195] |-> !OK_A[195])
`assert_clk(!so_or_sz && set_clr_onehot[195] |->  OK_A[195])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[196] <= `CQ reset_value;                        
        else if(ready_196_A)
            OK_A[196] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[196] <= `CQ so_or_sz ? OK_A[196] || set_clr_onehot[196] : OK_A[196] && !set_clr_onehot[196];       //update

assign ready_196_A = OK_onehot_A[196] && (valid_bank12_B ? ready_bank12_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[196] |-> !OK_A[196])
`assert_clk(!so_or_sz && set_clr_onehot[196] |->  OK_A[196])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[197] <= `CQ reset_value;                        
        else if(ready_197_A)
            OK_A[197] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[197] <= `CQ so_or_sz ? OK_A[197] || set_clr_onehot[197] : OK_A[197] && !set_clr_onehot[197];       //update

assign ready_197_A = OK_onehot_A[197] && (valid_bank12_B ? ready_bank12_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[197] |-> !OK_A[197])
`assert_clk(!so_or_sz && set_clr_onehot[197] |->  OK_A[197])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[198] <= `CQ reset_value;                        
        else if(ready_198_A)
            OK_A[198] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[198] <= `CQ so_or_sz ? OK_A[198] || set_clr_onehot[198] : OK_A[198] && !set_clr_onehot[198];       //update

assign ready_198_A = OK_onehot_A[198] && (valid_bank12_B ? ready_bank12_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[198] |-> !OK_A[198])
`assert_clk(!so_or_sz && set_clr_onehot[198] |->  OK_A[198])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[199] <= `CQ reset_value;                        
        else if(ready_199_A)
            OK_A[199] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[199] <= `CQ so_or_sz ? OK_A[199] || set_clr_onehot[199] : OK_A[199] && !set_clr_onehot[199];       //update

assign ready_199_A = OK_onehot_A[199] && (valid_bank12_B ? ready_bank12_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[199] |-> !OK_A[199])
`assert_clk(!so_or_sz && set_clr_onehot[199] |->  OK_A[199])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[200] <= `CQ reset_value;                        
        else if(ready_200_A)
            OK_A[200] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[200] <= `CQ so_or_sz ? OK_A[200] || set_clr_onehot[200] : OK_A[200] && !set_clr_onehot[200];       //update

assign ready_200_A = OK_onehot_A[200] && (valid_bank12_B ? ready_bank12_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[200] |-> !OK_A[200])
`assert_clk(!so_or_sz && set_clr_onehot[200] |->  OK_A[200])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[201] <= `CQ reset_value;                        
        else if(ready_201_A)
            OK_A[201] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[201] <= `CQ so_or_sz ? OK_A[201] || set_clr_onehot[201] : OK_A[201] && !set_clr_onehot[201];       //update

assign ready_201_A = OK_onehot_A[201] && (valid_bank12_B ? ready_bank12_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[201] |-> !OK_A[201])
`assert_clk(!so_or_sz && set_clr_onehot[201] |->  OK_A[201])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[202] <= `CQ reset_value;                        
        else if(ready_202_A)
            OK_A[202] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[202] <= `CQ so_or_sz ? OK_A[202] || set_clr_onehot[202] : OK_A[202] && !set_clr_onehot[202];       //update

assign ready_202_A = OK_onehot_A[202] && (valid_bank12_B ? ready_bank12_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[202] |-> !OK_A[202])
`assert_clk(!so_or_sz && set_clr_onehot[202] |->  OK_A[202])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[203] <= `CQ reset_value;                        
        else if(ready_203_A)
            OK_A[203] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[203] <= `CQ so_or_sz ? OK_A[203] || set_clr_onehot[203] : OK_A[203] && !set_clr_onehot[203];       //update

assign ready_203_A = OK_onehot_A[203] && (valid_bank12_B ? ready_bank12_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[203] |-> !OK_A[203])
`assert_clk(!so_or_sz && set_clr_onehot[203] |->  OK_A[203])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[204] <= `CQ reset_value;                        
        else if(ready_204_A)
            OK_A[204] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[204] <= `CQ so_or_sz ? OK_A[204] || set_clr_onehot[204] : OK_A[204] && !set_clr_onehot[204];       //update

assign ready_204_A = OK_onehot_A[204] && (valid_bank12_B ? ready_bank12_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[204] |-> !OK_A[204])
`assert_clk(!so_or_sz && set_clr_onehot[204] |->  OK_A[204])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[205] <= `CQ reset_value;                        
        else if(ready_205_A)
            OK_A[205] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[205] <= `CQ so_or_sz ? OK_A[205] || set_clr_onehot[205] : OK_A[205] && !set_clr_onehot[205];       //update

assign ready_205_A = OK_onehot_A[205] && (valid_bank12_B ? ready_bank12_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[205] |-> !OK_A[205])
`assert_clk(!so_or_sz && set_clr_onehot[205] |->  OK_A[205])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[206] <= `CQ reset_value;                        
        else if(ready_206_A)
            OK_A[206] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[206] <= `CQ so_or_sz ? OK_A[206] || set_clr_onehot[206] : OK_A[206] && !set_clr_onehot[206];       //update

assign ready_206_A = OK_onehot_A[206] && (valid_bank12_B ? ready_bank12_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[206] |-> !OK_A[206])
`assert_clk(!so_or_sz && set_clr_onehot[206] |->  OK_A[206])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[207] <= `CQ reset_value;                        
        else if(ready_207_A)
            OK_A[207] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[207] <= `CQ so_or_sz ? OK_A[207] || set_clr_onehot[207] : OK_A[207] && !set_clr_onehot[207];       //update

assign ready_207_A = OK_onehot_A[207] && (valid_bank12_B ? ready_bank12_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[207] |-> !OK_A[207])
`assert_clk(!so_or_sz && set_clr_onehot[207] |->  OK_A[207])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[208] <= `CQ reset_value;                        
        else if(ready_208_A)
            OK_A[208] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[208] <= `CQ so_or_sz ? OK_A[208] || set_clr_onehot[208] : OK_A[208] && !set_clr_onehot[208];       //update

assign ready_208_A = OK_onehot_A[208] && (valid_bank13_B ? ready_bank13_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[208] |-> !OK_A[208])
`assert_clk(!so_or_sz && set_clr_onehot[208] |->  OK_A[208])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[209] <= `CQ reset_value;                        
        else if(ready_209_A)
            OK_A[209] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[209] <= `CQ so_or_sz ? OK_A[209] || set_clr_onehot[209] : OK_A[209] && !set_clr_onehot[209];       //update

assign ready_209_A = OK_onehot_A[209] && (valid_bank13_B ? ready_bank13_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[209] |-> !OK_A[209])
`assert_clk(!so_or_sz && set_clr_onehot[209] |->  OK_A[209])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[210] <= `CQ reset_value;                        
        else if(ready_210_A)
            OK_A[210] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[210] <= `CQ so_or_sz ? OK_A[210] || set_clr_onehot[210] : OK_A[210] && !set_clr_onehot[210];       //update

assign ready_210_A = OK_onehot_A[210] && (valid_bank13_B ? ready_bank13_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[210] |-> !OK_A[210])
`assert_clk(!so_or_sz && set_clr_onehot[210] |->  OK_A[210])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[211] <= `CQ reset_value;                        
        else if(ready_211_A)
            OK_A[211] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[211] <= `CQ so_or_sz ? OK_A[211] || set_clr_onehot[211] : OK_A[211] && !set_clr_onehot[211];       //update

assign ready_211_A = OK_onehot_A[211] && (valid_bank13_B ? ready_bank13_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[211] |-> !OK_A[211])
`assert_clk(!so_or_sz && set_clr_onehot[211] |->  OK_A[211])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[212] <= `CQ reset_value;                        
        else if(ready_212_A)
            OK_A[212] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[212] <= `CQ so_or_sz ? OK_A[212] || set_clr_onehot[212] : OK_A[212] && !set_clr_onehot[212];       //update

assign ready_212_A = OK_onehot_A[212] && (valid_bank13_B ? ready_bank13_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[212] |-> !OK_A[212])
`assert_clk(!so_or_sz && set_clr_onehot[212] |->  OK_A[212])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[213] <= `CQ reset_value;                        
        else if(ready_213_A)
            OK_A[213] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[213] <= `CQ so_or_sz ? OK_A[213] || set_clr_onehot[213] : OK_A[213] && !set_clr_onehot[213];       //update

assign ready_213_A = OK_onehot_A[213] && (valid_bank13_B ? ready_bank13_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[213] |-> !OK_A[213])
`assert_clk(!so_or_sz && set_clr_onehot[213] |->  OK_A[213])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[214] <= `CQ reset_value;                        
        else if(ready_214_A)
            OK_A[214] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[214] <= `CQ so_or_sz ? OK_A[214] || set_clr_onehot[214] : OK_A[214] && !set_clr_onehot[214];       //update

assign ready_214_A = OK_onehot_A[214] && (valid_bank13_B ? ready_bank13_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[214] |-> !OK_A[214])
`assert_clk(!so_or_sz && set_clr_onehot[214] |->  OK_A[214])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[215] <= `CQ reset_value;                        
        else if(ready_215_A)
            OK_A[215] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[215] <= `CQ so_or_sz ? OK_A[215] || set_clr_onehot[215] : OK_A[215] && !set_clr_onehot[215];       //update

assign ready_215_A = OK_onehot_A[215] && (valid_bank13_B ? ready_bank13_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[215] |-> !OK_A[215])
`assert_clk(!so_or_sz && set_clr_onehot[215] |->  OK_A[215])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[216] <= `CQ reset_value;                        
        else if(ready_216_A)
            OK_A[216] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[216] <= `CQ so_or_sz ? OK_A[216] || set_clr_onehot[216] : OK_A[216] && !set_clr_onehot[216];       //update

assign ready_216_A = OK_onehot_A[216] && (valid_bank13_B ? ready_bank13_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[216] |-> !OK_A[216])
`assert_clk(!so_or_sz && set_clr_onehot[216] |->  OK_A[216])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[217] <= `CQ reset_value;                        
        else if(ready_217_A)
            OK_A[217] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[217] <= `CQ so_or_sz ? OK_A[217] || set_clr_onehot[217] : OK_A[217] && !set_clr_onehot[217];       //update

assign ready_217_A = OK_onehot_A[217] && (valid_bank13_B ? ready_bank13_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[217] |-> !OK_A[217])
`assert_clk(!so_or_sz && set_clr_onehot[217] |->  OK_A[217])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[218] <= `CQ reset_value;                        
        else if(ready_218_A)
            OK_A[218] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[218] <= `CQ so_or_sz ? OK_A[218] || set_clr_onehot[218] : OK_A[218] && !set_clr_onehot[218];       //update

assign ready_218_A = OK_onehot_A[218] && (valid_bank13_B ? ready_bank13_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[218] |-> !OK_A[218])
`assert_clk(!so_or_sz && set_clr_onehot[218] |->  OK_A[218])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[219] <= `CQ reset_value;                        
        else if(ready_219_A)
            OK_A[219] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[219] <= `CQ so_or_sz ? OK_A[219] || set_clr_onehot[219] : OK_A[219] && !set_clr_onehot[219];       //update

assign ready_219_A = OK_onehot_A[219] && (valid_bank13_B ? ready_bank13_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[219] |-> !OK_A[219])
`assert_clk(!so_or_sz && set_clr_onehot[219] |->  OK_A[219])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[220] <= `CQ reset_value;                        
        else if(ready_220_A)
            OK_A[220] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[220] <= `CQ so_or_sz ? OK_A[220] || set_clr_onehot[220] : OK_A[220] && !set_clr_onehot[220];       //update

assign ready_220_A = OK_onehot_A[220] && (valid_bank13_B ? ready_bank13_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[220] |-> !OK_A[220])
`assert_clk(!so_or_sz && set_clr_onehot[220] |->  OK_A[220])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[221] <= `CQ reset_value;                        
        else if(ready_221_A)
            OK_A[221] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[221] <= `CQ so_or_sz ? OK_A[221] || set_clr_onehot[221] : OK_A[221] && !set_clr_onehot[221];       //update

assign ready_221_A = OK_onehot_A[221] && (valid_bank13_B ? ready_bank13_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[221] |-> !OK_A[221])
`assert_clk(!so_or_sz && set_clr_onehot[221] |->  OK_A[221])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[222] <= `CQ reset_value;                        
        else if(ready_222_A)
            OK_A[222] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[222] <= `CQ so_or_sz ? OK_A[222] || set_clr_onehot[222] : OK_A[222] && !set_clr_onehot[222];       //update

assign ready_222_A = OK_onehot_A[222] && (valid_bank13_B ? ready_bank13_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[222] |-> !OK_A[222])
`assert_clk(!so_or_sz && set_clr_onehot[222] |->  OK_A[222])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[223] <= `CQ reset_value;                        
        else if(ready_223_A)
            OK_A[223] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[223] <= `CQ so_or_sz ? OK_A[223] || set_clr_onehot[223] : OK_A[223] && !set_clr_onehot[223];       //update

assign ready_223_A = OK_onehot_A[223] && (valid_bank13_B ? ready_bank13_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[223] |-> !OK_A[223])
`assert_clk(!so_or_sz && set_clr_onehot[223] |->  OK_A[223])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[224] <= `CQ reset_value;                        
        else if(ready_224_A)
            OK_A[224] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[224] <= `CQ so_or_sz ? OK_A[224] || set_clr_onehot[224] : OK_A[224] && !set_clr_onehot[224];       //update

assign ready_224_A = OK_onehot_A[224] && (valid_bank14_B ? ready_bank14_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[224] |-> !OK_A[224])
`assert_clk(!so_or_sz && set_clr_onehot[224] |->  OK_A[224])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[225] <= `CQ reset_value;                        
        else if(ready_225_A)
            OK_A[225] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[225] <= `CQ so_or_sz ? OK_A[225] || set_clr_onehot[225] : OK_A[225] && !set_clr_onehot[225];       //update

assign ready_225_A = OK_onehot_A[225] && (valid_bank14_B ? ready_bank14_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[225] |-> !OK_A[225])
`assert_clk(!so_or_sz && set_clr_onehot[225] |->  OK_A[225])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[226] <= `CQ reset_value;                        
        else if(ready_226_A)
            OK_A[226] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[226] <= `CQ so_or_sz ? OK_A[226] || set_clr_onehot[226] : OK_A[226] && !set_clr_onehot[226];       //update

assign ready_226_A = OK_onehot_A[226] && (valid_bank14_B ? ready_bank14_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[226] |-> !OK_A[226])
`assert_clk(!so_or_sz && set_clr_onehot[226] |->  OK_A[226])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[227] <= `CQ reset_value;                        
        else if(ready_227_A)
            OK_A[227] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[227] <= `CQ so_or_sz ? OK_A[227] || set_clr_onehot[227] : OK_A[227] && !set_clr_onehot[227];       //update

assign ready_227_A = OK_onehot_A[227] && (valid_bank14_B ? ready_bank14_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[227] |-> !OK_A[227])
`assert_clk(!so_or_sz && set_clr_onehot[227] |->  OK_A[227])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[228] <= `CQ reset_value;                        
        else if(ready_228_A)
            OK_A[228] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[228] <= `CQ so_or_sz ? OK_A[228] || set_clr_onehot[228] : OK_A[228] && !set_clr_onehot[228];       //update

assign ready_228_A = OK_onehot_A[228] && (valid_bank14_B ? ready_bank14_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[228] |-> !OK_A[228])
`assert_clk(!so_or_sz && set_clr_onehot[228] |->  OK_A[228])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[229] <= `CQ reset_value;                        
        else if(ready_229_A)
            OK_A[229] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[229] <= `CQ so_or_sz ? OK_A[229] || set_clr_onehot[229] : OK_A[229] && !set_clr_onehot[229];       //update

assign ready_229_A = OK_onehot_A[229] && (valid_bank14_B ? ready_bank14_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[229] |-> !OK_A[229])
`assert_clk(!so_or_sz && set_clr_onehot[229] |->  OK_A[229])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[230] <= `CQ reset_value;                        
        else if(ready_230_A)
            OK_A[230] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[230] <= `CQ so_or_sz ? OK_A[230] || set_clr_onehot[230] : OK_A[230] && !set_clr_onehot[230];       //update

assign ready_230_A = OK_onehot_A[230] && (valid_bank14_B ? ready_bank14_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[230] |-> !OK_A[230])
`assert_clk(!so_or_sz && set_clr_onehot[230] |->  OK_A[230])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[231] <= `CQ reset_value;                        
        else if(ready_231_A)
            OK_A[231] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[231] <= `CQ so_or_sz ? OK_A[231] || set_clr_onehot[231] : OK_A[231] && !set_clr_onehot[231];       //update

assign ready_231_A = OK_onehot_A[231] && (valid_bank14_B ? ready_bank14_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[231] |-> !OK_A[231])
`assert_clk(!so_or_sz && set_clr_onehot[231] |->  OK_A[231])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[232] <= `CQ reset_value;                        
        else if(ready_232_A)
            OK_A[232] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[232] <= `CQ so_or_sz ? OK_A[232] || set_clr_onehot[232] : OK_A[232] && !set_clr_onehot[232];       //update

assign ready_232_A = OK_onehot_A[232] && (valid_bank14_B ? ready_bank14_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[232] |-> !OK_A[232])
`assert_clk(!so_or_sz && set_clr_onehot[232] |->  OK_A[232])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[233] <= `CQ reset_value;                        
        else if(ready_233_A)
            OK_A[233] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[233] <= `CQ so_or_sz ? OK_A[233] || set_clr_onehot[233] : OK_A[233] && !set_clr_onehot[233];       //update

assign ready_233_A = OK_onehot_A[233] && (valid_bank14_B ? ready_bank14_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[233] |-> !OK_A[233])
`assert_clk(!so_or_sz && set_clr_onehot[233] |->  OK_A[233])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[234] <= `CQ reset_value;                        
        else if(ready_234_A)
            OK_A[234] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[234] <= `CQ so_or_sz ? OK_A[234] || set_clr_onehot[234] : OK_A[234] && !set_clr_onehot[234];       //update

assign ready_234_A = OK_onehot_A[234] && (valid_bank14_B ? ready_bank14_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[234] |-> !OK_A[234])
`assert_clk(!so_or_sz && set_clr_onehot[234] |->  OK_A[234])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[235] <= `CQ reset_value;                        
        else if(ready_235_A)
            OK_A[235] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[235] <= `CQ so_or_sz ? OK_A[235] || set_clr_onehot[235] : OK_A[235] && !set_clr_onehot[235];       //update

assign ready_235_A = OK_onehot_A[235] && (valid_bank14_B ? ready_bank14_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[235] |-> !OK_A[235])
`assert_clk(!so_or_sz && set_clr_onehot[235] |->  OK_A[235])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[236] <= `CQ reset_value;                        
        else if(ready_236_A)
            OK_A[236] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[236] <= `CQ so_or_sz ? OK_A[236] || set_clr_onehot[236] : OK_A[236] && !set_clr_onehot[236];       //update

assign ready_236_A = OK_onehot_A[236] && (valid_bank14_B ? ready_bank14_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[236] |-> !OK_A[236])
`assert_clk(!so_or_sz && set_clr_onehot[236] |->  OK_A[236])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[237] <= `CQ reset_value;                        
        else if(ready_237_A)
            OK_A[237] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[237] <= `CQ so_or_sz ? OK_A[237] || set_clr_onehot[237] : OK_A[237] && !set_clr_onehot[237];       //update

assign ready_237_A = OK_onehot_A[237] && (valid_bank14_B ? ready_bank14_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[237] |-> !OK_A[237])
`assert_clk(!so_or_sz && set_clr_onehot[237] |->  OK_A[237])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[238] <= `CQ reset_value;                        
        else if(ready_238_A)
            OK_A[238] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[238] <= `CQ so_or_sz ? OK_A[238] || set_clr_onehot[238] : OK_A[238] && !set_clr_onehot[238];       //update

assign ready_238_A = OK_onehot_A[238] && (valid_bank14_B ? ready_bank14_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[238] |-> !OK_A[238])
`assert_clk(!so_or_sz && set_clr_onehot[238] |->  OK_A[238])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[239] <= `CQ reset_value;                        
        else if(ready_239_A)
            OK_A[239] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[239] <= `CQ so_or_sz ? OK_A[239] || set_clr_onehot[239] : OK_A[239] && !set_clr_onehot[239];       //update

assign ready_239_A = OK_onehot_A[239] && (valid_bank14_B ? ready_bank14_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[239] |-> !OK_A[239])
`assert_clk(!so_or_sz && set_clr_onehot[239] |->  OK_A[239])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[240] <= `CQ reset_value;                        
        else if(ready_240_A)
            OK_A[240] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[240] <= `CQ so_or_sz ? OK_A[240] || set_clr_onehot[240] : OK_A[240] && !set_clr_onehot[240];       //update

assign ready_240_A = OK_onehot_A[240] && (valid_bank15_B ? ready_bank15_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[240] |-> !OK_A[240])
`assert_clk(!so_or_sz && set_clr_onehot[240] |->  OK_A[240])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[241] <= `CQ reset_value;                        
        else if(ready_241_A)
            OK_A[241] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[241] <= `CQ so_or_sz ? OK_A[241] || set_clr_onehot[241] : OK_A[241] && !set_clr_onehot[241];       //update

assign ready_241_A = OK_onehot_A[241] && (valid_bank15_B ? ready_bank15_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[241] |-> !OK_A[241])
`assert_clk(!so_or_sz && set_clr_onehot[241] |->  OK_A[241])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[242] <= `CQ reset_value;                        
        else if(ready_242_A)
            OK_A[242] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[242] <= `CQ so_or_sz ? OK_A[242] || set_clr_onehot[242] : OK_A[242] && !set_clr_onehot[242];       //update

assign ready_242_A = OK_onehot_A[242] && (valid_bank15_B ? ready_bank15_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[242] |-> !OK_A[242])
`assert_clk(!so_or_sz && set_clr_onehot[242] |->  OK_A[242])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[243] <= `CQ reset_value;                        
        else if(ready_243_A)
            OK_A[243] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[243] <= `CQ so_or_sz ? OK_A[243] || set_clr_onehot[243] : OK_A[243] && !set_clr_onehot[243];       //update

assign ready_243_A = OK_onehot_A[243] && (valid_bank15_B ? ready_bank15_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[243] |-> !OK_A[243])
`assert_clk(!so_or_sz && set_clr_onehot[243] |->  OK_A[243])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[244] <= `CQ reset_value;                        
        else if(ready_244_A)
            OK_A[244] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[244] <= `CQ so_or_sz ? OK_A[244] || set_clr_onehot[244] : OK_A[244] && !set_clr_onehot[244];       //update

assign ready_244_A = OK_onehot_A[244] && (valid_bank15_B ? ready_bank15_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[244] |-> !OK_A[244])
`assert_clk(!so_or_sz && set_clr_onehot[244] |->  OK_A[244])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[245] <= `CQ reset_value;                        
        else if(ready_245_A)
            OK_A[245] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[245] <= `CQ so_or_sz ? OK_A[245] || set_clr_onehot[245] : OK_A[245] && !set_clr_onehot[245];       //update

assign ready_245_A = OK_onehot_A[245] && (valid_bank15_B ? ready_bank15_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[245] |-> !OK_A[245])
`assert_clk(!so_or_sz && set_clr_onehot[245] |->  OK_A[245])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[246] <= `CQ reset_value;                        
        else if(ready_246_A)
            OK_A[246] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[246] <= `CQ so_or_sz ? OK_A[246] || set_clr_onehot[246] : OK_A[246] && !set_clr_onehot[246];       //update

assign ready_246_A = OK_onehot_A[246] && (valid_bank15_B ? ready_bank15_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[246] |-> !OK_A[246])
`assert_clk(!so_or_sz && set_clr_onehot[246] |->  OK_A[246])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[247] <= `CQ reset_value;                        
        else if(ready_247_A)
            OK_A[247] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[247] <= `CQ so_or_sz ? OK_A[247] || set_clr_onehot[247] : OK_A[247] && !set_clr_onehot[247];       //update

assign ready_247_A = OK_onehot_A[247] && (valid_bank15_B ? ready_bank15_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[247] |-> !OK_A[247])
`assert_clk(!so_or_sz && set_clr_onehot[247] |->  OK_A[247])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[248] <= `CQ reset_value;                        
        else if(ready_248_A)
            OK_A[248] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[248] <= `CQ so_or_sz ? OK_A[248] || set_clr_onehot[248] : OK_A[248] && !set_clr_onehot[248];       //update

assign ready_248_A = OK_onehot_A[248] && (valid_bank15_B ? ready_bank15_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[248] |-> !OK_A[248])
`assert_clk(!so_or_sz && set_clr_onehot[248] |->  OK_A[248])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[249] <= `CQ reset_value;                        
        else if(ready_249_A)
            OK_A[249] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[249] <= `CQ so_or_sz ? OK_A[249] || set_clr_onehot[249] : OK_A[249] && !set_clr_onehot[249];       //update

assign ready_249_A = OK_onehot_A[249] && (valid_bank15_B ? ready_bank15_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[249] |-> !OK_A[249])
`assert_clk(!so_or_sz && set_clr_onehot[249] |->  OK_A[249])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[250] <= `CQ reset_value;                        
        else if(ready_250_A)
            OK_A[250] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[250] <= `CQ so_or_sz ? OK_A[250] || set_clr_onehot[250] : OK_A[250] && !set_clr_onehot[250];       //update

assign ready_250_A = OK_onehot_A[250] && (valid_bank15_B ? ready_bank15_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[250] |-> !OK_A[250])
`assert_clk(!so_or_sz && set_clr_onehot[250] |->  OK_A[250])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[251] <= `CQ reset_value;                        
        else if(ready_251_A)
            OK_A[251] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[251] <= `CQ so_or_sz ? OK_A[251] || set_clr_onehot[251] : OK_A[251] && !set_clr_onehot[251];       //update

assign ready_251_A = OK_onehot_A[251] && (valid_bank15_B ? ready_bank15_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[251] |-> !OK_A[251])
`assert_clk(!so_or_sz && set_clr_onehot[251] |->  OK_A[251])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[252] <= `CQ reset_value;                        
        else if(ready_252_A)
            OK_A[252] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[252] <= `CQ so_or_sz ? OK_A[252] || set_clr_onehot[252] : OK_A[252] && !set_clr_onehot[252];       //update

assign ready_252_A = OK_onehot_A[252] && (valid_bank15_B ? ready_bank15_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[252] |-> !OK_A[252])
`assert_clk(!so_or_sz && set_clr_onehot[252] |->  OK_A[252])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[253] <= `CQ reset_value;                        
        else if(ready_253_A)
            OK_A[253] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[253] <= `CQ so_or_sz ? OK_A[253] || set_clr_onehot[253] : OK_A[253] && !set_clr_onehot[253];       //update

assign ready_253_A = OK_onehot_A[253] && (valid_bank15_B ? ready_bank15_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[253] |-> !OK_A[253])
`assert_clk(!so_or_sz && set_clr_onehot[253] |->  OK_A[253])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[254] <= `CQ reset_value;                        
        else if(ready_254_A)
            OK_A[254] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[254] <= `CQ so_or_sz ? OK_A[254] || set_clr_onehot[254] : OK_A[254] && !set_clr_onehot[254];       //update

assign ready_254_A = OK_onehot_A[254] && (valid_bank15_B ? ready_bank15_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[254] |-> !OK_A[254])
`assert_clk(!so_or_sz && set_clr_onehot[254] |->  OK_A[254])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[255] <= `CQ reset_value;                        
        else if(ready_255_A)
            OK_A[255] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[255] <= `CQ so_or_sz ? OK_A[255] || set_clr_onehot[255] : OK_A[255] && !set_clr_onehot[255];       //update

assign ready_255_A = OK_onehot_A[255] && (valid_bank15_B ? ready_bank15_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[255] |-> !OK_A[255])
`assert_clk(!so_or_sz && set_clr_onehot[255] |->  OK_A[255])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[256] <= `CQ reset_value;                        
        else if(ready_256_A)
            OK_A[256] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[256] <= `CQ so_or_sz ? OK_A[256] || set_clr_onehot[256] : OK_A[256] && !set_clr_onehot[256];       //update

assign ready_256_A = OK_onehot_A[256] && (valid_bank16_B ? ready_bank16_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[256] |-> !OK_A[256])
`assert_clk(!so_or_sz && set_clr_onehot[256] |->  OK_A[256])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[257] <= `CQ reset_value;                        
        else if(ready_257_A)
            OK_A[257] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[257] <= `CQ so_or_sz ? OK_A[257] || set_clr_onehot[257] : OK_A[257] && !set_clr_onehot[257];       //update

assign ready_257_A = OK_onehot_A[257] && (valid_bank16_B ? ready_bank16_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[257] |-> !OK_A[257])
`assert_clk(!so_or_sz && set_clr_onehot[257] |->  OK_A[257])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[258] <= `CQ reset_value;                        
        else if(ready_258_A)
            OK_A[258] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[258] <= `CQ so_or_sz ? OK_A[258] || set_clr_onehot[258] : OK_A[258] && !set_clr_onehot[258];       //update

assign ready_258_A = OK_onehot_A[258] && (valid_bank16_B ? ready_bank16_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[258] |-> !OK_A[258])
`assert_clk(!so_or_sz && set_clr_onehot[258] |->  OK_A[258])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[259] <= `CQ reset_value;                        
        else if(ready_259_A)
            OK_A[259] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[259] <= `CQ so_or_sz ? OK_A[259] || set_clr_onehot[259] : OK_A[259] && !set_clr_onehot[259];       //update

assign ready_259_A = OK_onehot_A[259] && (valid_bank16_B ? ready_bank16_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[259] |-> !OK_A[259])
`assert_clk(!so_or_sz && set_clr_onehot[259] |->  OK_A[259])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[260] <= `CQ reset_value;                        
        else if(ready_260_A)
            OK_A[260] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[260] <= `CQ so_or_sz ? OK_A[260] || set_clr_onehot[260] : OK_A[260] && !set_clr_onehot[260];       //update

assign ready_260_A = OK_onehot_A[260] && (valid_bank16_B ? ready_bank16_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[260] |-> !OK_A[260])
`assert_clk(!so_or_sz && set_clr_onehot[260] |->  OK_A[260])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[261] <= `CQ reset_value;                        
        else if(ready_261_A)
            OK_A[261] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[261] <= `CQ so_or_sz ? OK_A[261] || set_clr_onehot[261] : OK_A[261] && !set_clr_onehot[261];       //update

assign ready_261_A = OK_onehot_A[261] && (valid_bank16_B ? ready_bank16_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[261] |-> !OK_A[261])
`assert_clk(!so_or_sz && set_clr_onehot[261] |->  OK_A[261])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[262] <= `CQ reset_value;                        
        else if(ready_262_A)
            OK_A[262] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[262] <= `CQ so_or_sz ? OK_A[262] || set_clr_onehot[262] : OK_A[262] && !set_clr_onehot[262];       //update

assign ready_262_A = OK_onehot_A[262] && (valid_bank16_B ? ready_bank16_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[262] |-> !OK_A[262])
`assert_clk(!so_or_sz && set_clr_onehot[262] |->  OK_A[262])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[263] <= `CQ reset_value;                        
        else if(ready_263_A)
            OK_A[263] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[263] <= `CQ so_or_sz ? OK_A[263] || set_clr_onehot[263] : OK_A[263] && !set_clr_onehot[263];       //update

assign ready_263_A = OK_onehot_A[263] && (valid_bank16_B ? ready_bank16_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[263] |-> !OK_A[263])
`assert_clk(!so_or_sz && set_clr_onehot[263] |->  OK_A[263])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[264] <= `CQ reset_value;                        
        else if(ready_264_A)
            OK_A[264] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[264] <= `CQ so_or_sz ? OK_A[264] || set_clr_onehot[264] : OK_A[264] && !set_clr_onehot[264];       //update

assign ready_264_A = OK_onehot_A[264] && (valid_bank16_B ? ready_bank16_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[264] |-> !OK_A[264])
`assert_clk(!so_or_sz && set_clr_onehot[264] |->  OK_A[264])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[265] <= `CQ reset_value;                        
        else if(ready_265_A)
            OK_A[265] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[265] <= `CQ so_or_sz ? OK_A[265] || set_clr_onehot[265] : OK_A[265] && !set_clr_onehot[265];       //update

assign ready_265_A = OK_onehot_A[265] && (valid_bank16_B ? ready_bank16_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[265] |-> !OK_A[265])
`assert_clk(!so_or_sz && set_clr_onehot[265] |->  OK_A[265])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[266] <= `CQ reset_value;                        
        else if(ready_266_A)
            OK_A[266] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[266] <= `CQ so_or_sz ? OK_A[266] || set_clr_onehot[266] : OK_A[266] && !set_clr_onehot[266];       //update

assign ready_266_A = OK_onehot_A[266] && (valid_bank16_B ? ready_bank16_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[266] |-> !OK_A[266])
`assert_clk(!so_or_sz && set_clr_onehot[266] |->  OK_A[266])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[267] <= `CQ reset_value;                        
        else if(ready_267_A)
            OK_A[267] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[267] <= `CQ so_or_sz ? OK_A[267] || set_clr_onehot[267] : OK_A[267] && !set_clr_onehot[267];       //update

assign ready_267_A = OK_onehot_A[267] && (valid_bank16_B ? ready_bank16_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[267] |-> !OK_A[267])
`assert_clk(!so_or_sz && set_clr_onehot[267] |->  OK_A[267])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[268] <= `CQ reset_value;                        
        else if(ready_268_A)
            OK_A[268] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[268] <= `CQ so_or_sz ? OK_A[268] || set_clr_onehot[268] : OK_A[268] && !set_clr_onehot[268];       //update

assign ready_268_A = OK_onehot_A[268] && (valid_bank16_B ? ready_bank16_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[268] |-> !OK_A[268])
`assert_clk(!so_or_sz && set_clr_onehot[268] |->  OK_A[268])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[269] <= `CQ reset_value;                        
        else if(ready_269_A)
            OK_A[269] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[269] <= `CQ so_or_sz ? OK_A[269] || set_clr_onehot[269] : OK_A[269] && !set_clr_onehot[269];       //update

assign ready_269_A = OK_onehot_A[269] && (valid_bank16_B ? ready_bank16_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[269] |-> !OK_A[269])
`assert_clk(!so_or_sz && set_clr_onehot[269] |->  OK_A[269])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[270] <= `CQ reset_value;                        
        else if(ready_270_A)
            OK_A[270] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[270] <= `CQ so_or_sz ? OK_A[270] || set_clr_onehot[270] : OK_A[270] && !set_clr_onehot[270];       //update

assign ready_270_A = OK_onehot_A[270] && (valid_bank16_B ? ready_bank16_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[270] |-> !OK_A[270])
`assert_clk(!so_or_sz && set_clr_onehot[270] |->  OK_A[270])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[271] <= `CQ reset_value;                        
        else if(ready_271_A)
            OK_A[271] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[271] <= `CQ so_or_sz ? OK_A[271] || set_clr_onehot[271] : OK_A[271] && !set_clr_onehot[271];       //update

assign ready_271_A = OK_onehot_A[271] && (valid_bank16_B ? ready_bank16_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[271] |-> !OK_A[271])
`assert_clk(!so_or_sz && set_clr_onehot[271] |->  OK_A[271])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[272] <= `CQ reset_value;                        
        else if(ready_272_A)
            OK_A[272] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[272] <= `CQ so_or_sz ? OK_A[272] || set_clr_onehot[272] : OK_A[272] && !set_clr_onehot[272];       //update

assign ready_272_A = OK_onehot_A[272] && (valid_bank17_B ? ready_bank17_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[272] |-> !OK_A[272])
`assert_clk(!so_or_sz && set_clr_onehot[272] |->  OK_A[272])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[273] <= `CQ reset_value;                        
        else if(ready_273_A)
            OK_A[273] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[273] <= `CQ so_or_sz ? OK_A[273] || set_clr_onehot[273] : OK_A[273] && !set_clr_onehot[273];       //update

assign ready_273_A = OK_onehot_A[273] && (valid_bank17_B ? ready_bank17_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[273] |-> !OK_A[273])
`assert_clk(!so_or_sz && set_clr_onehot[273] |->  OK_A[273])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[274] <= `CQ reset_value;                        
        else if(ready_274_A)
            OK_A[274] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[274] <= `CQ so_or_sz ? OK_A[274] || set_clr_onehot[274] : OK_A[274] && !set_clr_onehot[274];       //update

assign ready_274_A = OK_onehot_A[274] && (valid_bank17_B ? ready_bank17_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[274] |-> !OK_A[274])
`assert_clk(!so_or_sz && set_clr_onehot[274] |->  OK_A[274])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[275] <= `CQ reset_value;                        
        else if(ready_275_A)
            OK_A[275] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[275] <= `CQ so_or_sz ? OK_A[275] || set_clr_onehot[275] : OK_A[275] && !set_clr_onehot[275];       //update

assign ready_275_A = OK_onehot_A[275] && (valid_bank17_B ? ready_bank17_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[275] |-> !OK_A[275])
`assert_clk(!so_or_sz && set_clr_onehot[275] |->  OK_A[275])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[276] <= `CQ reset_value;                        
        else if(ready_276_A)
            OK_A[276] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[276] <= `CQ so_or_sz ? OK_A[276] || set_clr_onehot[276] : OK_A[276] && !set_clr_onehot[276];       //update

assign ready_276_A = OK_onehot_A[276] && (valid_bank17_B ? ready_bank17_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[276] |-> !OK_A[276])
`assert_clk(!so_or_sz && set_clr_onehot[276] |->  OK_A[276])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[277] <= `CQ reset_value;                        
        else if(ready_277_A)
            OK_A[277] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[277] <= `CQ so_or_sz ? OK_A[277] || set_clr_onehot[277] : OK_A[277] && !set_clr_onehot[277];       //update

assign ready_277_A = OK_onehot_A[277] && (valid_bank17_B ? ready_bank17_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[277] |-> !OK_A[277])
`assert_clk(!so_or_sz && set_clr_onehot[277] |->  OK_A[277])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[278] <= `CQ reset_value;                        
        else if(ready_278_A)
            OK_A[278] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[278] <= `CQ so_or_sz ? OK_A[278] || set_clr_onehot[278] : OK_A[278] && !set_clr_onehot[278];       //update

assign ready_278_A = OK_onehot_A[278] && (valid_bank17_B ? ready_bank17_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[278] |-> !OK_A[278])
`assert_clk(!so_or_sz && set_clr_onehot[278] |->  OK_A[278])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[279] <= `CQ reset_value;                        
        else if(ready_279_A)
            OK_A[279] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[279] <= `CQ so_or_sz ? OK_A[279] || set_clr_onehot[279] : OK_A[279] && !set_clr_onehot[279];       //update

assign ready_279_A = OK_onehot_A[279] && (valid_bank17_B ? ready_bank17_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[279] |-> !OK_A[279])
`assert_clk(!so_or_sz && set_clr_onehot[279] |->  OK_A[279])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[280] <= `CQ reset_value;                        
        else if(ready_280_A)
            OK_A[280] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[280] <= `CQ so_or_sz ? OK_A[280] || set_clr_onehot[280] : OK_A[280] && !set_clr_onehot[280];       //update

assign ready_280_A = OK_onehot_A[280] && (valid_bank17_B ? ready_bank17_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[280] |-> !OK_A[280])
`assert_clk(!so_or_sz && set_clr_onehot[280] |->  OK_A[280])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[281] <= `CQ reset_value;                        
        else if(ready_281_A)
            OK_A[281] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[281] <= `CQ so_or_sz ? OK_A[281] || set_clr_onehot[281] : OK_A[281] && !set_clr_onehot[281];       //update

assign ready_281_A = OK_onehot_A[281] && (valid_bank17_B ? ready_bank17_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[281] |-> !OK_A[281])
`assert_clk(!so_or_sz && set_clr_onehot[281] |->  OK_A[281])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[282] <= `CQ reset_value;                        
        else if(ready_282_A)
            OK_A[282] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[282] <= `CQ so_or_sz ? OK_A[282] || set_clr_onehot[282] : OK_A[282] && !set_clr_onehot[282];       //update

assign ready_282_A = OK_onehot_A[282] && (valid_bank17_B ? ready_bank17_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[282] |-> !OK_A[282])
`assert_clk(!so_or_sz && set_clr_onehot[282] |->  OK_A[282])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[283] <= `CQ reset_value;                        
        else if(ready_283_A)
            OK_A[283] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[283] <= `CQ so_or_sz ? OK_A[283] || set_clr_onehot[283] : OK_A[283] && !set_clr_onehot[283];       //update

assign ready_283_A = OK_onehot_A[283] && (valid_bank17_B ? ready_bank17_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[283] |-> !OK_A[283])
`assert_clk(!so_or_sz && set_clr_onehot[283] |->  OK_A[283])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[284] <= `CQ reset_value;                        
        else if(ready_284_A)
            OK_A[284] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[284] <= `CQ so_or_sz ? OK_A[284] || set_clr_onehot[284] : OK_A[284] && !set_clr_onehot[284];       //update

assign ready_284_A = OK_onehot_A[284] && (valid_bank17_B ? ready_bank17_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[284] |-> !OK_A[284])
`assert_clk(!so_or_sz && set_clr_onehot[284] |->  OK_A[284])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[285] <= `CQ reset_value;                        
        else if(ready_285_A)
            OK_A[285] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[285] <= `CQ so_or_sz ? OK_A[285] || set_clr_onehot[285] : OK_A[285] && !set_clr_onehot[285];       //update

assign ready_285_A = OK_onehot_A[285] && (valid_bank17_B ? ready_bank17_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[285] |-> !OK_A[285])
`assert_clk(!so_or_sz && set_clr_onehot[285] |->  OK_A[285])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[286] <= `CQ reset_value;                        
        else if(ready_286_A)
            OK_A[286] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[286] <= `CQ so_or_sz ? OK_A[286] || set_clr_onehot[286] : OK_A[286] && !set_clr_onehot[286];       //update

assign ready_286_A = OK_onehot_A[286] && (valid_bank17_B ? ready_bank17_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[286] |-> !OK_A[286])
`assert_clk(!so_or_sz && set_clr_onehot[286] |->  OK_A[286])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[287] <= `CQ reset_value;                        
        else if(ready_287_A)
            OK_A[287] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[287] <= `CQ so_or_sz ? OK_A[287] || set_clr_onehot[287] : OK_A[287] && !set_clr_onehot[287];       //update

assign ready_287_A = OK_onehot_A[287] && (valid_bank17_B ? ready_bank17_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[287] |-> !OK_A[287])
`assert_clk(!so_or_sz && set_clr_onehot[287] |->  OK_A[287])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[288] <= `CQ reset_value;                        
        else if(ready_288_A)
            OK_A[288] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[288] <= `CQ so_or_sz ? OK_A[288] || set_clr_onehot[288] : OK_A[288] && !set_clr_onehot[288];       //update

assign ready_288_A = OK_onehot_A[288] && (valid_bank18_B ? ready_bank18_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[288] |-> !OK_A[288])
`assert_clk(!so_or_sz && set_clr_onehot[288] |->  OK_A[288])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[289] <= `CQ reset_value;                        
        else if(ready_289_A)
            OK_A[289] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[289] <= `CQ so_or_sz ? OK_A[289] || set_clr_onehot[289] : OK_A[289] && !set_clr_onehot[289];       //update

assign ready_289_A = OK_onehot_A[289] && (valid_bank18_B ? ready_bank18_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[289] |-> !OK_A[289])
`assert_clk(!so_or_sz && set_clr_onehot[289] |->  OK_A[289])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[290] <= `CQ reset_value;                        
        else if(ready_290_A)
            OK_A[290] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[290] <= `CQ so_or_sz ? OK_A[290] || set_clr_onehot[290] : OK_A[290] && !set_clr_onehot[290];       //update

assign ready_290_A = OK_onehot_A[290] && (valid_bank18_B ? ready_bank18_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[290] |-> !OK_A[290])
`assert_clk(!so_or_sz && set_clr_onehot[290] |->  OK_A[290])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[291] <= `CQ reset_value;                        
        else if(ready_291_A)
            OK_A[291] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[291] <= `CQ so_or_sz ? OK_A[291] || set_clr_onehot[291] : OK_A[291] && !set_clr_onehot[291];       //update

assign ready_291_A = OK_onehot_A[291] && (valid_bank18_B ? ready_bank18_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[291] |-> !OK_A[291])
`assert_clk(!so_or_sz && set_clr_onehot[291] |->  OK_A[291])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[292] <= `CQ reset_value;                        
        else if(ready_292_A)
            OK_A[292] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[292] <= `CQ so_or_sz ? OK_A[292] || set_clr_onehot[292] : OK_A[292] && !set_clr_onehot[292];       //update

assign ready_292_A = OK_onehot_A[292] && (valid_bank18_B ? ready_bank18_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[292] |-> !OK_A[292])
`assert_clk(!so_or_sz && set_clr_onehot[292] |->  OK_A[292])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[293] <= `CQ reset_value;                        
        else if(ready_293_A)
            OK_A[293] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[293] <= `CQ so_or_sz ? OK_A[293] || set_clr_onehot[293] : OK_A[293] && !set_clr_onehot[293];       //update

assign ready_293_A = OK_onehot_A[293] && (valid_bank18_B ? ready_bank18_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[293] |-> !OK_A[293])
`assert_clk(!so_or_sz && set_clr_onehot[293] |->  OK_A[293])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[294] <= `CQ reset_value;                        
        else if(ready_294_A)
            OK_A[294] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[294] <= `CQ so_or_sz ? OK_A[294] || set_clr_onehot[294] : OK_A[294] && !set_clr_onehot[294];       //update

assign ready_294_A = OK_onehot_A[294] && (valid_bank18_B ? ready_bank18_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[294] |-> !OK_A[294])
`assert_clk(!so_or_sz && set_clr_onehot[294] |->  OK_A[294])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[295] <= `CQ reset_value;                        
        else if(ready_295_A)
            OK_A[295] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[295] <= `CQ so_or_sz ? OK_A[295] || set_clr_onehot[295] : OK_A[295] && !set_clr_onehot[295];       //update

assign ready_295_A = OK_onehot_A[295] && (valid_bank18_B ? ready_bank18_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[295] |-> !OK_A[295])
`assert_clk(!so_or_sz && set_clr_onehot[295] |->  OK_A[295])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[296] <= `CQ reset_value;                        
        else if(ready_296_A)
            OK_A[296] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[296] <= `CQ so_or_sz ? OK_A[296] || set_clr_onehot[296] : OK_A[296] && !set_clr_onehot[296];       //update

assign ready_296_A = OK_onehot_A[296] && (valid_bank18_B ? ready_bank18_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[296] |-> !OK_A[296])
`assert_clk(!so_or_sz && set_clr_onehot[296] |->  OK_A[296])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[297] <= `CQ reset_value;                        
        else if(ready_297_A)
            OK_A[297] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[297] <= `CQ so_or_sz ? OK_A[297] || set_clr_onehot[297] : OK_A[297] && !set_clr_onehot[297];       //update

assign ready_297_A = OK_onehot_A[297] && (valid_bank18_B ? ready_bank18_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[297] |-> !OK_A[297])
`assert_clk(!so_or_sz && set_clr_onehot[297] |->  OK_A[297])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[298] <= `CQ reset_value;                        
        else if(ready_298_A)
            OK_A[298] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[298] <= `CQ so_or_sz ? OK_A[298] || set_clr_onehot[298] : OK_A[298] && !set_clr_onehot[298];       //update

assign ready_298_A = OK_onehot_A[298] && (valid_bank18_B ? ready_bank18_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[298] |-> !OK_A[298])
`assert_clk(!so_or_sz && set_clr_onehot[298] |->  OK_A[298])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[299] <= `CQ reset_value;                        
        else if(ready_299_A)
            OK_A[299] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[299] <= `CQ so_or_sz ? OK_A[299] || set_clr_onehot[299] : OK_A[299] && !set_clr_onehot[299];       //update

assign ready_299_A = OK_onehot_A[299] && (valid_bank18_B ? ready_bank18_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[299] |-> !OK_A[299])
`assert_clk(!so_or_sz && set_clr_onehot[299] |->  OK_A[299])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[300] <= `CQ reset_value;                        
        else if(ready_300_A)
            OK_A[300] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[300] <= `CQ so_or_sz ? OK_A[300] || set_clr_onehot[300] : OK_A[300] && !set_clr_onehot[300];       //update

assign ready_300_A = OK_onehot_A[300] && (valid_bank18_B ? ready_bank18_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[300] |-> !OK_A[300])
`assert_clk(!so_or_sz && set_clr_onehot[300] |->  OK_A[300])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[301] <= `CQ reset_value;                        
        else if(ready_301_A)
            OK_A[301] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[301] <= `CQ so_or_sz ? OK_A[301] || set_clr_onehot[301] : OK_A[301] && !set_clr_onehot[301];       //update

assign ready_301_A = OK_onehot_A[301] && (valid_bank18_B ? ready_bank18_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[301] |-> !OK_A[301])
`assert_clk(!so_or_sz && set_clr_onehot[301] |->  OK_A[301])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[302] <= `CQ reset_value;                        
        else if(ready_302_A)
            OK_A[302] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[302] <= `CQ so_or_sz ? OK_A[302] || set_clr_onehot[302] : OK_A[302] && !set_clr_onehot[302];       //update

assign ready_302_A = OK_onehot_A[302] && (valid_bank18_B ? ready_bank18_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[302] |-> !OK_A[302])
`assert_clk(!so_or_sz && set_clr_onehot[302] |->  OK_A[302])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[303] <= `CQ reset_value;                        
        else if(ready_303_A)
            OK_A[303] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[303] <= `CQ so_or_sz ? OK_A[303] || set_clr_onehot[303] : OK_A[303] && !set_clr_onehot[303];       //update

assign ready_303_A = OK_onehot_A[303] && (valid_bank18_B ? ready_bank18_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[303] |-> !OK_A[303])
`assert_clk(!so_or_sz && set_clr_onehot[303] |->  OK_A[303])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[304] <= `CQ reset_value;                        
        else if(ready_304_A)
            OK_A[304] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[304] <= `CQ so_or_sz ? OK_A[304] || set_clr_onehot[304] : OK_A[304] && !set_clr_onehot[304];       //update

assign ready_304_A = OK_onehot_A[304] && (valid_bank19_B ? ready_bank19_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[304] |-> !OK_A[304])
`assert_clk(!so_or_sz && set_clr_onehot[304] |->  OK_A[304])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[305] <= `CQ reset_value;                        
        else if(ready_305_A)
            OK_A[305] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[305] <= `CQ so_or_sz ? OK_A[305] || set_clr_onehot[305] : OK_A[305] && !set_clr_onehot[305];       //update

assign ready_305_A = OK_onehot_A[305] && (valid_bank19_B ? ready_bank19_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[305] |-> !OK_A[305])
`assert_clk(!so_or_sz && set_clr_onehot[305] |->  OK_A[305])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[306] <= `CQ reset_value;                        
        else if(ready_306_A)
            OK_A[306] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[306] <= `CQ so_or_sz ? OK_A[306] || set_clr_onehot[306] : OK_A[306] && !set_clr_onehot[306];       //update

assign ready_306_A = OK_onehot_A[306] && (valid_bank19_B ? ready_bank19_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[306] |-> !OK_A[306])
`assert_clk(!so_or_sz && set_clr_onehot[306] |->  OK_A[306])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[307] <= `CQ reset_value;                        
        else if(ready_307_A)
            OK_A[307] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[307] <= `CQ so_or_sz ? OK_A[307] || set_clr_onehot[307] : OK_A[307] && !set_clr_onehot[307];       //update

assign ready_307_A = OK_onehot_A[307] && (valid_bank19_B ? ready_bank19_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[307] |-> !OK_A[307])
`assert_clk(!so_or_sz && set_clr_onehot[307] |->  OK_A[307])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[308] <= `CQ reset_value;                        
        else if(ready_308_A)
            OK_A[308] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[308] <= `CQ so_or_sz ? OK_A[308] || set_clr_onehot[308] : OK_A[308] && !set_clr_onehot[308];       //update

assign ready_308_A = OK_onehot_A[308] && (valid_bank19_B ? ready_bank19_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[308] |-> !OK_A[308])
`assert_clk(!so_or_sz && set_clr_onehot[308] |->  OK_A[308])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[309] <= `CQ reset_value;                        
        else if(ready_309_A)
            OK_A[309] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[309] <= `CQ so_or_sz ? OK_A[309] || set_clr_onehot[309] : OK_A[309] && !set_clr_onehot[309];       //update

assign ready_309_A = OK_onehot_A[309] && (valid_bank19_B ? ready_bank19_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[309] |-> !OK_A[309])
`assert_clk(!so_or_sz && set_clr_onehot[309] |->  OK_A[309])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[310] <= `CQ reset_value;                        
        else if(ready_310_A)
            OK_A[310] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[310] <= `CQ so_or_sz ? OK_A[310] || set_clr_onehot[310] : OK_A[310] && !set_clr_onehot[310];       //update

assign ready_310_A = OK_onehot_A[310] && (valid_bank19_B ? ready_bank19_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[310] |-> !OK_A[310])
`assert_clk(!so_or_sz && set_clr_onehot[310] |->  OK_A[310])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[311] <= `CQ reset_value;                        
        else if(ready_311_A)
            OK_A[311] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[311] <= `CQ so_or_sz ? OK_A[311] || set_clr_onehot[311] : OK_A[311] && !set_clr_onehot[311];       //update

assign ready_311_A = OK_onehot_A[311] && (valid_bank19_B ? ready_bank19_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[311] |-> !OK_A[311])
`assert_clk(!so_or_sz && set_clr_onehot[311] |->  OK_A[311])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[312] <= `CQ reset_value;                        
        else if(ready_312_A)
            OK_A[312] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[312] <= `CQ so_or_sz ? OK_A[312] || set_clr_onehot[312] : OK_A[312] && !set_clr_onehot[312];       //update

assign ready_312_A = OK_onehot_A[312] && (valid_bank19_B ? ready_bank19_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[312] |-> !OK_A[312])
`assert_clk(!so_or_sz && set_clr_onehot[312] |->  OK_A[312])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[313] <= `CQ reset_value;                        
        else if(ready_313_A)
            OK_A[313] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[313] <= `CQ so_or_sz ? OK_A[313] || set_clr_onehot[313] : OK_A[313] && !set_clr_onehot[313];       //update

assign ready_313_A = OK_onehot_A[313] && (valid_bank19_B ? ready_bank19_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[313] |-> !OK_A[313])
`assert_clk(!so_or_sz && set_clr_onehot[313] |->  OK_A[313])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[314] <= `CQ reset_value;                        
        else if(ready_314_A)
            OK_A[314] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[314] <= `CQ so_or_sz ? OK_A[314] || set_clr_onehot[314] : OK_A[314] && !set_clr_onehot[314];       //update

assign ready_314_A = OK_onehot_A[314] && (valid_bank19_B ? ready_bank19_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[314] |-> !OK_A[314])
`assert_clk(!so_or_sz && set_clr_onehot[314] |->  OK_A[314])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[315] <= `CQ reset_value;                        
        else if(ready_315_A)
            OK_A[315] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[315] <= `CQ so_or_sz ? OK_A[315] || set_clr_onehot[315] : OK_A[315] && !set_clr_onehot[315];       //update

assign ready_315_A = OK_onehot_A[315] && (valid_bank19_B ? ready_bank19_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[315] |-> !OK_A[315])
`assert_clk(!so_or_sz && set_clr_onehot[315] |->  OK_A[315])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[316] <= `CQ reset_value;                        
        else if(ready_316_A)
            OK_A[316] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[316] <= `CQ so_or_sz ? OK_A[316] || set_clr_onehot[316] : OK_A[316] && !set_clr_onehot[316];       //update

assign ready_316_A = OK_onehot_A[316] && (valid_bank19_B ? ready_bank19_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[316] |-> !OK_A[316])
`assert_clk(!so_or_sz && set_clr_onehot[316] |->  OK_A[316])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[317] <= `CQ reset_value;                        
        else if(ready_317_A)
            OK_A[317] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[317] <= `CQ so_or_sz ? OK_A[317] || set_clr_onehot[317] : OK_A[317] && !set_clr_onehot[317];       //update

assign ready_317_A = OK_onehot_A[317] && (valid_bank19_B ? ready_bank19_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[317] |-> !OK_A[317])
`assert_clk(!so_or_sz && set_clr_onehot[317] |->  OK_A[317])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[318] <= `CQ reset_value;                        
        else if(ready_318_A)
            OK_A[318] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[318] <= `CQ so_or_sz ? OK_A[318] || set_clr_onehot[318] : OK_A[318] && !set_clr_onehot[318];       //update

assign ready_318_A = OK_onehot_A[318] && (valid_bank19_B ? ready_bank19_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[318] |-> !OK_A[318])
`assert_clk(!so_or_sz && set_clr_onehot[318] |->  OK_A[318])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[319] <= `CQ reset_value;                        
        else if(ready_319_A)
            OK_A[319] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[319] <= `CQ so_or_sz ? OK_A[319] || set_clr_onehot[319] : OK_A[319] && !set_clr_onehot[319];       //update

assign ready_319_A = OK_onehot_A[319] && (valid_bank19_B ? ready_bank19_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[319] |-> !OK_A[319])
`assert_clk(!so_or_sz && set_clr_onehot[319] |->  OK_A[319])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[320] <= `CQ reset_value;                        
        else if(ready_320_A)
            OK_A[320] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[320] <= `CQ so_or_sz ? OK_A[320] || set_clr_onehot[320] : OK_A[320] && !set_clr_onehot[320];       //update

assign ready_320_A = OK_onehot_A[320] && (valid_bank20_B ? ready_bank20_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[320] |-> !OK_A[320])
`assert_clk(!so_or_sz && set_clr_onehot[320] |->  OK_A[320])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[321] <= `CQ reset_value;                        
        else if(ready_321_A)
            OK_A[321] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[321] <= `CQ so_or_sz ? OK_A[321] || set_clr_onehot[321] : OK_A[321] && !set_clr_onehot[321];       //update

assign ready_321_A = OK_onehot_A[321] && (valid_bank20_B ? ready_bank20_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[321] |-> !OK_A[321])
`assert_clk(!so_or_sz && set_clr_onehot[321] |->  OK_A[321])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[322] <= `CQ reset_value;                        
        else if(ready_322_A)
            OK_A[322] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[322] <= `CQ so_or_sz ? OK_A[322] || set_clr_onehot[322] : OK_A[322] && !set_clr_onehot[322];       //update

assign ready_322_A = OK_onehot_A[322] && (valid_bank20_B ? ready_bank20_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[322] |-> !OK_A[322])
`assert_clk(!so_or_sz && set_clr_onehot[322] |->  OK_A[322])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[323] <= `CQ reset_value;                        
        else if(ready_323_A)
            OK_A[323] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[323] <= `CQ so_or_sz ? OK_A[323] || set_clr_onehot[323] : OK_A[323] && !set_clr_onehot[323];       //update

assign ready_323_A = OK_onehot_A[323] && (valid_bank20_B ? ready_bank20_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[323] |-> !OK_A[323])
`assert_clk(!so_or_sz && set_clr_onehot[323] |->  OK_A[323])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[324] <= `CQ reset_value;                        
        else if(ready_324_A)
            OK_A[324] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[324] <= `CQ so_or_sz ? OK_A[324] || set_clr_onehot[324] : OK_A[324] && !set_clr_onehot[324];       //update

assign ready_324_A = OK_onehot_A[324] && (valid_bank20_B ? ready_bank20_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[324] |-> !OK_A[324])
`assert_clk(!so_or_sz && set_clr_onehot[324] |->  OK_A[324])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[325] <= `CQ reset_value;                        
        else if(ready_325_A)
            OK_A[325] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[325] <= `CQ so_or_sz ? OK_A[325] || set_clr_onehot[325] : OK_A[325] && !set_clr_onehot[325];       //update

assign ready_325_A = OK_onehot_A[325] && (valid_bank20_B ? ready_bank20_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[325] |-> !OK_A[325])
`assert_clk(!so_or_sz && set_clr_onehot[325] |->  OK_A[325])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[326] <= `CQ reset_value;                        
        else if(ready_326_A)
            OK_A[326] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[326] <= `CQ so_or_sz ? OK_A[326] || set_clr_onehot[326] : OK_A[326] && !set_clr_onehot[326];       //update

assign ready_326_A = OK_onehot_A[326] && (valid_bank20_B ? ready_bank20_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[326] |-> !OK_A[326])
`assert_clk(!so_or_sz && set_clr_onehot[326] |->  OK_A[326])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[327] <= `CQ reset_value;                        
        else if(ready_327_A)
            OK_A[327] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[327] <= `CQ so_or_sz ? OK_A[327] || set_clr_onehot[327] : OK_A[327] && !set_clr_onehot[327];       //update

assign ready_327_A = OK_onehot_A[327] && (valid_bank20_B ? ready_bank20_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[327] |-> !OK_A[327])
`assert_clk(!so_or_sz && set_clr_onehot[327] |->  OK_A[327])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[328] <= `CQ reset_value;                        
        else if(ready_328_A)
            OK_A[328] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[328] <= `CQ so_or_sz ? OK_A[328] || set_clr_onehot[328] : OK_A[328] && !set_clr_onehot[328];       //update

assign ready_328_A = OK_onehot_A[328] && (valid_bank20_B ? ready_bank20_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[328] |-> !OK_A[328])
`assert_clk(!so_or_sz && set_clr_onehot[328] |->  OK_A[328])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[329] <= `CQ reset_value;                        
        else if(ready_329_A)
            OK_A[329] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[329] <= `CQ so_or_sz ? OK_A[329] || set_clr_onehot[329] : OK_A[329] && !set_clr_onehot[329];       //update

assign ready_329_A = OK_onehot_A[329] && (valid_bank20_B ? ready_bank20_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[329] |-> !OK_A[329])
`assert_clk(!so_or_sz && set_clr_onehot[329] |->  OK_A[329])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[330] <= `CQ reset_value;                        
        else if(ready_330_A)
            OK_A[330] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[330] <= `CQ so_or_sz ? OK_A[330] || set_clr_onehot[330] : OK_A[330] && !set_clr_onehot[330];       //update

assign ready_330_A = OK_onehot_A[330] && (valid_bank20_B ? ready_bank20_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[330] |-> !OK_A[330])
`assert_clk(!so_or_sz && set_clr_onehot[330] |->  OK_A[330])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[331] <= `CQ reset_value;                        
        else if(ready_331_A)
            OK_A[331] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[331] <= `CQ so_or_sz ? OK_A[331] || set_clr_onehot[331] : OK_A[331] && !set_clr_onehot[331];       //update

assign ready_331_A = OK_onehot_A[331] && (valid_bank20_B ? ready_bank20_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[331] |-> !OK_A[331])
`assert_clk(!so_or_sz && set_clr_onehot[331] |->  OK_A[331])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[332] <= `CQ reset_value;                        
        else if(ready_332_A)
            OK_A[332] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[332] <= `CQ so_or_sz ? OK_A[332] || set_clr_onehot[332] : OK_A[332] && !set_clr_onehot[332];       //update

assign ready_332_A = OK_onehot_A[332] && (valid_bank20_B ? ready_bank20_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[332] |-> !OK_A[332])
`assert_clk(!so_or_sz && set_clr_onehot[332] |->  OK_A[332])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[333] <= `CQ reset_value;                        
        else if(ready_333_A)
            OK_A[333] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[333] <= `CQ so_or_sz ? OK_A[333] || set_clr_onehot[333] : OK_A[333] && !set_clr_onehot[333];       //update

assign ready_333_A = OK_onehot_A[333] && (valid_bank20_B ? ready_bank20_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[333] |-> !OK_A[333])
`assert_clk(!so_or_sz && set_clr_onehot[333] |->  OK_A[333])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[334] <= `CQ reset_value;                        
        else if(ready_334_A)
            OK_A[334] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[334] <= `CQ so_or_sz ? OK_A[334] || set_clr_onehot[334] : OK_A[334] && !set_clr_onehot[334];       //update

assign ready_334_A = OK_onehot_A[334] && (valid_bank20_B ? ready_bank20_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[334] |-> !OK_A[334])
`assert_clk(!so_or_sz && set_clr_onehot[334] |->  OK_A[334])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[335] <= `CQ reset_value;                        
        else if(ready_335_A)
            OK_A[335] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[335] <= `CQ so_or_sz ? OK_A[335] || set_clr_onehot[335] : OK_A[335] && !set_clr_onehot[335];       //update

assign ready_335_A = OK_onehot_A[335] && (valid_bank20_B ? ready_bank20_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[335] |-> !OK_A[335])
`assert_clk(!so_or_sz && set_clr_onehot[335] |->  OK_A[335])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[336] <= `CQ reset_value;                        
        else if(ready_336_A)
            OK_A[336] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[336] <= `CQ so_or_sz ? OK_A[336] || set_clr_onehot[336] : OK_A[336] && !set_clr_onehot[336];       //update

assign ready_336_A = OK_onehot_A[336] && (valid_bank21_B ? ready_bank21_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[336] |-> !OK_A[336])
`assert_clk(!so_or_sz && set_clr_onehot[336] |->  OK_A[336])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[337] <= `CQ reset_value;                        
        else if(ready_337_A)
            OK_A[337] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[337] <= `CQ so_or_sz ? OK_A[337] || set_clr_onehot[337] : OK_A[337] && !set_clr_onehot[337];       //update

assign ready_337_A = OK_onehot_A[337] && (valid_bank21_B ? ready_bank21_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[337] |-> !OK_A[337])
`assert_clk(!so_or_sz && set_clr_onehot[337] |->  OK_A[337])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[338] <= `CQ reset_value;                        
        else if(ready_338_A)
            OK_A[338] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[338] <= `CQ so_or_sz ? OK_A[338] || set_clr_onehot[338] : OK_A[338] && !set_clr_onehot[338];       //update

assign ready_338_A = OK_onehot_A[338] && (valid_bank21_B ? ready_bank21_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[338] |-> !OK_A[338])
`assert_clk(!so_or_sz && set_clr_onehot[338] |->  OK_A[338])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[339] <= `CQ reset_value;                        
        else if(ready_339_A)
            OK_A[339] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[339] <= `CQ so_or_sz ? OK_A[339] || set_clr_onehot[339] : OK_A[339] && !set_clr_onehot[339];       //update

assign ready_339_A = OK_onehot_A[339] && (valid_bank21_B ? ready_bank21_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[339] |-> !OK_A[339])
`assert_clk(!so_or_sz && set_clr_onehot[339] |->  OK_A[339])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[340] <= `CQ reset_value;                        
        else if(ready_340_A)
            OK_A[340] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[340] <= `CQ so_or_sz ? OK_A[340] || set_clr_onehot[340] : OK_A[340] && !set_clr_onehot[340];       //update

assign ready_340_A = OK_onehot_A[340] && (valid_bank21_B ? ready_bank21_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[340] |-> !OK_A[340])
`assert_clk(!so_or_sz && set_clr_onehot[340] |->  OK_A[340])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[341] <= `CQ reset_value;                        
        else if(ready_341_A)
            OK_A[341] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[341] <= `CQ so_or_sz ? OK_A[341] || set_clr_onehot[341] : OK_A[341] && !set_clr_onehot[341];       //update

assign ready_341_A = OK_onehot_A[341] && (valid_bank21_B ? ready_bank21_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[341] |-> !OK_A[341])
`assert_clk(!so_or_sz && set_clr_onehot[341] |->  OK_A[341])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[342] <= `CQ reset_value;                        
        else if(ready_342_A)
            OK_A[342] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[342] <= `CQ so_or_sz ? OK_A[342] || set_clr_onehot[342] : OK_A[342] && !set_clr_onehot[342];       //update

assign ready_342_A = OK_onehot_A[342] && (valid_bank21_B ? ready_bank21_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[342] |-> !OK_A[342])
`assert_clk(!so_or_sz && set_clr_onehot[342] |->  OK_A[342])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[343] <= `CQ reset_value;                        
        else if(ready_343_A)
            OK_A[343] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[343] <= `CQ so_or_sz ? OK_A[343] || set_clr_onehot[343] : OK_A[343] && !set_clr_onehot[343];       //update

assign ready_343_A = OK_onehot_A[343] && (valid_bank21_B ? ready_bank21_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[343] |-> !OK_A[343])
`assert_clk(!so_or_sz && set_clr_onehot[343] |->  OK_A[343])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[344] <= `CQ reset_value;                        
        else if(ready_344_A)
            OK_A[344] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[344] <= `CQ so_or_sz ? OK_A[344] || set_clr_onehot[344] : OK_A[344] && !set_clr_onehot[344];       //update

assign ready_344_A = OK_onehot_A[344] && (valid_bank21_B ? ready_bank21_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[344] |-> !OK_A[344])
`assert_clk(!so_or_sz && set_clr_onehot[344] |->  OK_A[344])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[345] <= `CQ reset_value;                        
        else if(ready_345_A)
            OK_A[345] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[345] <= `CQ so_or_sz ? OK_A[345] || set_clr_onehot[345] : OK_A[345] && !set_clr_onehot[345];       //update

assign ready_345_A = OK_onehot_A[345] && (valid_bank21_B ? ready_bank21_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[345] |-> !OK_A[345])
`assert_clk(!so_or_sz && set_clr_onehot[345] |->  OK_A[345])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[346] <= `CQ reset_value;                        
        else if(ready_346_A)
            OK_A[346] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[346] <= `CQ so_or_sz ? OK_A[346] || set_clr_onehot[346] : OK_A[346] && !set_clr_onehot[346];       //update

assign ready_346_A = OK_onehot_A[346] && (valid_bank21_B ? ready_bank21_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[346] |-> !OK_A[346])
`assert_clk(!so_or_sz && set_clr_onehot[346] |->  OK_A[346])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[347] <= `CQ reset_value;                        
        else if(ready_347_A)
            OK_A[347] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[347] <= `CQ so_or_sz ? OK_A[347] || set_clr_onehot[347] : OK_A[347] && !set_clr_onehot[347];       //update

assign ready_347_A = OK_onehot_A[347] && (valid_bank21_B ? ready_bank21_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[347] |-> !OK_A[347])
`assert_clk(!so_or_sz && set_clr_onehot[347] |->  OK_A[347])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[348] <= `CQ reset_value;                        
        else if(ready_348_A)
            OK_A[348] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[348] <= `CQ so_or_sz ? OK_A[348] || set_clr_onehot[348] : OK_A[348] && !set_clr_onehot[348];       //update

assign ready_348_A = OK_onehot_A[348] && (valid_bank21_B ? ready_bank21_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[348] |-> !OK_A[348])
`assert_clk(!so_or_sz && set_clr_onehot[348] |->  OK_A[348])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[349] <= `CQ reset_value;                        
        else if(ready_349_A)
            OK_A[349] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[349] <= `CQ so_or_sz ? OK_A[349] || set_clr_onehot[349] : OK_A[349] && !set_clr_onehot[349];       //update

assign ready_349_A = OK_onehot_A[349] && (valid_bank21_B ? ready_bank21_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[349] |-> !OK_A[349])
`assert_clk(!so_or_sz && set_clr_onehot[349] |->  OK_A[349])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[350] <= `CQ reset_value;                        
        else if(ready_350_A)
            OK_A[350] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[350] <= `CQ so_or_sz ? OK_A[350] || set_clr_onehot[350] : OK_A[350] && !set_clr_onehot[350];       //update

assign ready_350_A = OK_onehot_A[350] && (valid_bank21_B ? ready_bank21_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[350] |-> !OK_A[350])
`assert_clk(!so_or_sz && set_clr_onehot[350] |->  OK_A[350])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[351] <= `CQ reset_value;                        
        else if(ready_351_A)
            OK_A[351] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[351] <= `CQ so_or_sz ? OK_A[351] || set_clr_onehot[351] : OK_A[351] && !set_clr_onehot[351];       //update

assign ready_351_A = OK_onehot_A[351] && (valid_bank21_B ? ready_bank21_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[351] |-> !OK_A[351])
`assert_clk(!so_or_sz && set_clr_onehot[351] |->  OK_A[351])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[352] <= `CQ reset_value;                        
        else if(ready_352_A)
            OK_A[352] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[352] <= `CQ so_or_sz ? OK_A[352] || set_clr_onehot[352] : OK_A[352] && !set_clr_onehot[352];       //update

assign ready_352_A = OK_onehot_A[352] && (valid_bank22_B ? ready_bank22_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[352] |-> !OK_A[352])
`assert_clk(!so_or_sz && set_clr_onehot[352] |->  OK_A[352])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[353] <= `CQ reset_value;                        
        else if(ready_353_A)
            OK_A[353] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[353] <= `CQ so_or_sz ? OK_A[353] || set_clr_onehot[353] : OK_A[353] && !set_clr_onehot[353];       //update

assign ready_353_A = OK_onehot_A[353] && (valid_bank22_B ? ready_bank22_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[353] |-> !OK_A[353])
`assert_clk(!so_or_sz && set_clr_onehot[353] |->  OK_A[353])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[354] <= `CQ reset_value;                        
        else if(ready_354_A)
            OK_A[354] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[354] <= `CQ so_or_sz ? OK_A[354] || set_clr_onehot[354] : OK_A[354] && !set_clr_onehot[354];       //update

assign ready_354_A = OK_onehot_A[354] && (valid_bank22_B ? ready_bank22_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[354] |-> !OK_A[354])
`assert_clk(!so_or_sz && set_clr_onehot[354] |->  OK_A[354])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[355] <= `CQ reset_value;                        
        else if(ready_355_A)
            OK_A[355] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[355] <= `CQ so_or_sz ? OK_A[355] || set_clr_onehot[355] : OK_A[355] && !set_clr_onehot[355];       //update

assign ready_355_A = OK_onehot_A[355] && (valid_bank22_B ? ready_bank22_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[355] |-> !OK_A[355])
`assert_clk(!so_or_sz && set_clr_onehot[355] |->  OK_A[355])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[356] <= `CQ reset_value;                        
        else if(ready_356_A)
            OK_A[356] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[356] <= `CQ so_or_sz ? OK_A[356] || set_clr_onehot[356] : OK_A[356] && !set_clr_onehot[356];       //update

assign ready_356_A = OK_onehot_A[356] && (valid_bank22_B ? ready_bank22_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[356] |-> !OK_A[356])
`assert_clk(!so_or_sz && set_clr_onehot[356] |->  OK_A[356])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[357] <= `CQ reset_value;                        
        else if(ready_357_A)
            OK_A[357] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[357] <= `CQ so_or_sz ? OK_A[357] || set_clr_onehot[357] : OK_A[357] && !set_clr_onehot[357];       //update

assign ready_357_A = OK_onehot_A[357] && (valid_bank22_B ? ready_bank22_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[357] |-> !OK_A[357])
`assert_clk(!so_or_sz && set_clr_onehot[357] |->  OK_A[357])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[358] <= `CQ reset_value;                        
        else if(ready_358_A)
            OK_A[358] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[358] <= `CQ so_or_sz ? OK_A[358] || set_clr_onehot[358] : OK_A[358] && !set_clr_onehot[358];       //update

assign ready_358_A = OK_onehot_A[358] && (valid_bank22_B ? ready_bank22_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[358] |-> !OK_A[358])
`assert_clk(!so_or_sz && set_clr_onehot[358] |->  OK_A[358])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[359] <= `CQ reset_value;                        
        else if(ready_359_A)
            OK_A[359] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[359] <= `CQ so_or_sz ? OK_A[359] || set_clr_onehot[359] : OK_A[359] && !set_clr_onehot[359];       //update

assign ready_359_A = OK_onehot_A[359] && (valid_bank22_B ? ready_bank22_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[359] |-> !OK_A[359])
`assert_clk(!so_or_sz && set_clr_onehot[359] |->  OK_A[359])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[360] <= `CQ reset_value;                        
        else if(ready_360_A)
            OK_A[360] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[360] <= `CQ so_or_sz ? OK_A[360] || set_clr_onehot[360] : OK_A[360] && !set_clr_onehot[360];       //update

assign ready_360_A = OK_onehot_A[360] && (valid_bank22_B ? ready_bank22_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[360] |-> !OK_A[360])
`assert_clk(!so_or_sz && set_clr_onehot[360] |->  OK_A[360])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[361] <= `CQ reset_value;                        
        else if(ready_361_A)
            OK_A[361] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[361] <= `CQ so_or_sz ? OK_A[361] || set_clr_onehot[361] : OK_A[361] && !set_clr_onehot[361];       //update

assign ready_361_A = OK_onehot_A[361] && (valid_bank22_B ? ready_bank22_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[361] |-> !OK_A[361])
`assert_clk(!so_or_sz && set_clr_onehot[361] |->  OK_A[361])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[362] <= `CQ reset_value;                        
        else if(ready_362_A)
            OK_A[362] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[362] <= `CQ so_or_sz ? OK_A[362] || set_clr_onehot[362] : OK_A[362] && !set_clr_onehot[362];       //update

assign ready_362_A = OK_onehot_A[362] && (valid_bank22_B ? ready_bank22_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[362] |-> !OK_A[362])
`assert_clk(!so_or_sz && set_clr_onehot[362] |->  OK_A[362])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[363] <= `CQ reset_value;                        
        else if(ready_363_A)
            OK_A[363] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[363] <= `CQ so_or_sz ? OK_A[363] || set_clr_onehot[363] : OK_A[363] && !set_clr_onehot[363];       //update

assign ready_363_A = OK_onehot_A[363] && (valid_bank22_B ? ready_bank22_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[363] |-> !OK_A[363])
`assert_clk(!so_or_sz && set_clr_onehot[363] |->  OK_A[363])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[364] <= `CQ reset_value;                        
        else if(ready_364_A)
            OK_A[364] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[364] <= `CQ so_or_sz ? OK_A[364] || set_clr_onehot[364] : OK_A[364] && !set_clr_onehot[364];       //update

assign ready_364_A = OK_onehot_A[364] && (valid_bank22_B ? ready_bank22_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[364] |-> !OK_A[364])
`assert_clk(!so_or_sz && set_clr_onehot[364] |->  OK_A[364])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[365] <= `CQ reset_value;                        
        else if(ready_365_A)
            OK_A[365] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[365] <= `CQ so_or_sz ? OK_A[365] || set_clr_onehot[365] : OK_A[365] && !set_clr_onehot[365];       //update

assign ready_365_A = OK_onehot_A[365] && (valid_bank22_B ? ready_bank22_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[365] |-> !OK_A[365])
`assert_clk(!so_or_sz && set_clr_onehot[365] |->  OK_A[365])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[366] <= `CQ reset_value;                        
        else if(ready_366_A)
            OK_A[366] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[366] <= `CQ so_or_sz ? OK_A[366] || set_clr_onehot[366] : OK_A[366] && !set_clr_onehot[366];       //update

assign ready_366_A = OK_onehot_A[366] && (valid_bank22_B ? ready_bank22_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[366] |-> !OK_A[366])
`assert_clk(!so_or_sz && set_clr_onehot[366] |->  OK_A[366])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[367] <= `CQ reset_value;                        
        else if(ready_367_A)
            OK_A[367] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[367] <= `CQ so_or_sz ? OK_A[367] || set_clr_onehot[367] : OK_A[367] && !set_clr_onehot[367];       //update

assign ready_367_A = OK_onehot_A[367] && (valid_bank22_B ? ready_bank22_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[367] |-> !OK_A[367])
`assert_clk(!so_or_sz && set_clr_onehot[367] |->  OK_A[367])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[368] <= `CQ reset_value;                        
        else if(ready_368_A)
            OK_A[368] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[368] <= `CQ so_or_sz ? OK_A[368] || set_clr_onehot[368] : OK_A[368] && !set_clr_onehot[368];       //update

assign ready_368_A = OK_onehot_A[368] && (valid_bank23_B ? ready_bank23_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[368] |-> !OK_A[368])
`assert_clk(!so_or_sz && set_clr_onehot[368] |->  OK_A[368])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[369] <= `CQ reset_value;                        
        else if(ready_369_A)
            OK_A[369] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[369] <= `CQ so_or_sz ? OK_A[369] || set_clr_onehot[369] : OK_A[369] && !set_clr_onehot[369];       //update

assign ready_369_A = OK_onehot_A[369] && (valid_bank23_B ? ready_bank23_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[369] |-> !OK_A[369])
`assert_clk(!so_or_sz && set_clr_onehot[369] |->  OK_A[369])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[370] <= `CQ reset_value;                        
        else if(ready_370_A)
            OK_A[370] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[370] <= `CQ so_or_sz ? OK_A[370] || set_clr_onehot[370] : OK_A[370] && !set_clr_onehot[370];       //update

assign ready_370_A = OK_onehot_A[370] && (valid_bank23_B ? ready_bank23_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[370] |-> !OK_A[370])
`assert_clk(!so_or_sz && set_clr_onehot[370] |->  OK_A[370])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[371] <= `CQ reset_value;                        
        else if(ready_371_A)
            OK_A[371] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[371] <= `CQ so_or_sz ? OK_A[371] || set_clr_onehot[371] : OK_A[371] && !set_clr_onehot[371];       //update

assign ready_371_A = OK_onehot_A[371] && (valid_bank23_B ? ready_bank23_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[371] |-> !OK_A[371])
`assert_clk(!so_or_sz && set_clr_onehot[371] |->  OK_A[371])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[372] <= `CQ reset_value;                        
        else if(ready_372_A)
            OK_A[372] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[372] <= `CQ so_or_sz ? OK_A[372] || set_clr_onehot[372] : OK_A[372] && !set_clr_onehot[372];       //update

assign ready_372_A = OK_onehot_A[372] && (valid_bank23_B ? ready_bank23_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[372] |-> !OK_A[372])
`assert_clk(!so_or_sz && set_clr_onehot[372] |->  OK_A[372])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[373] <= `CQ reset_value;                        
        else if(ready_373_A)
            OK_A[373] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[373] <= `CQ so_or_sz ? OK_A[373] || set_clr_onehot[373] : OK_A[373] && !set_clr_onehot[373];       //update

assign ready_373_A = OK_onehot_A[373] && (valid_bank23_B ? ready_bank23_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[373] |-> !OK_A[373])
`assert_clk(!so_or_sz && set_clr_onehot[373] |->  OK_A[373])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[374] <= `CQ reset_value;                        
        else if(ready_374_A)
            OK_A[374] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[374] <= `CQ so_or_sz ? OK_A[374] || set_clr_onehot[374] : OK_A[374] && !set_clr_onehot[374];       //update

assign ready_374_A = OK_onehot_A[374] && (valid_bank23_B ? ready_bank23_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[374] |-> !OK_A[374])
`assert_clk(!so_or_sz && set_clr_onehot[374] |->  OK_A[374])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[375] <= `CQ reset_value;                        
        else if(ready_375_A)
            OK_A[375] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[375] <= `CQ so_or_sz ? OK_A[375] || set_clr_onehot[375] : OK_A[375] && !set_clr_onehot[375];       //update

assign ready_375_A = OK_onehot_A[375] && (valid_bank23_B ? ready_bank23_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[375] |-> !OK_A[375])
`assert_clk(!so_or_sz && set_clr_onehot[375] |->  OK_A[375])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[376] <= `CQ reset_value;                        
        else if(ready_376_A)
            OK_A[376] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[376] <= `CQ so_or_sz ? OK_A[376] || set_clr_onehot[376] : OK_A[376] && !set_clr_onehot[376];       //update

assign ready_376_A = OK_onehot_A[376] && (valid_bank23_B ? ready_bank23_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[376] |-> !OK_A[376])
`assert_clk(!so_or_sz && set_clr_onehot[376] |->  OK_A[376])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[377] <= `CQ reset_value;                        
        else if(ready_377_A)
            OK_A[377] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[377] <= `CQ so_or_sz ? OK_A[377] || set_clr_onehot[377] : OK_A[377] && !set_clr_onehot[377];       //update

assign ready_377_A = OK_onehot_A[377] && (valid_bank23_B ? ready_bank23_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[377] |-> !OK_A[377])
`assert_clk(!so_or_sz && set_clr_onehot[377] |->  OK_A[377])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[378] <= `CQ reset_value;                        
        else if(ready_378_A)
            OK_A[378] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[378] <= `CQ so_or_sz ? OK_A[378] || set_clr_onehot[378] : OK_A[378] && !set_clr_onehot[378];       //update

assign ready_378_A = OK_onehot_A[378] && (valid_bank23_B ? ready_bank23_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[378] |-> !OK_A[378])
`assert_clk(!so_or_sz && set_clr_onehot[378] |->  OK_A[378])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[379] <= `CQ reset_value;                        
        else if(ready_379_A)
            OK_A[379] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[379] <= `CQ so_or_sz ? OK_A[379] || set_clr_onehot[379] : OK_A[379] && !set_clr_onehot[379];       //update

assign ready_379_A = OK_onehot_A[379] && (valid_bank23_B ? ready_bank23_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[379] |-> !OK_A[379])
`assert_clk(!so_or_sz && set_clr_onehot[379] |->  OK_A[379])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[380] <= `CQ reset_value;                        
        else if(ready_380_A)
            OK_A[380] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[380] <= `CQ so_or_sz ? OK_A[380] || set_clr_onehot[380] : OK_A[380] && !set_clr_onehot[380];       //update

assign ready_380_A = OK_onehot_A[380] && (valid_bank23_B ? ready_bank23_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[380] |-> !OK_A[380])
`assert_clk(!so_or_sz && set_clr_onehot[380] |->  OK_A[380])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[381] <= `CQ reset_value;                        
        else if(ready_381_A)
            OK_A[381] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[381] <= `CQ so_or_sz ? OK_A[381] || set_clr_onehot[381] : OK_A[381] && !set_clr_onehot[381];       //update

assign ready_381_A = OK_onehot_A[381] && (valid_bank23_B ? ready_bank23_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[381] |-> !OK_A[381])
`assert_clk(!so_or_sz && set_clr_onehot[381] |->  OK_A[381])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[382] <= `CQ reset_value;                        
        else if(ready_382_A)
            OK_A[382] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[382] <= `CQ so_or_sz ? OK_A[382] || set_clr_onehot[382] : OK_A[382] && !set_clr_onehot[382];       //update

assign ready_382_A = OK_onehot_A[382] && (valid_bank23_B ? ready_bank23_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[382] |-> !OK_A[382])
`assert_clk(!so_or_sz && set_clr_onehot[382] |->  OK_A[382])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[383] <= `CQ reset_value;                        
        else if(ready_383_A)
            OK_A[383] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[383] <= `CQ so_or_sz ? OK_A[383] || set_clr_onehot[383] : OK_A[383] && !set_clr_onehot[383];       //update

assign ready_383_A = OK_onehot_A[383] && (valid_bank23_B ? ready_bank23_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[383] |-> !OK_A[383])
`assert_clk(!so_or_sz && set_clr_onehot[383] |->  OK_A[383])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[384] <= `CQ reset_value;                        
        else if(ready_384_A)
            OK_A[384] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[384] <= `CQ so_or_sz ? OK_A[384] || set_clr_onehot[384] : OK_A[384] && !set_clr_onehot[384];       //update

assign ready_384_A = OK_onehot_A[384] && (valid_bank24_B ? ready_bank24_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[384] |-> !OK_A[384])
`assert_clk(!so_or_sz && set_clr_onehot[384] |->  OK_A[384])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[385] <= `CQ reset_value;                        
        else if(ready_385_A)
            OK_A[385] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[385] <= `CQ so_or_sz ? OK_A[385] || set_clr_onehot[385] : OK_A[385] && !set_clr_onehot[385];       //update

assign ready_385_A = OK_onehot_A[385] && (valid_bank24_B ? ready_bank24_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[385] |-> !OK_A[385])
`assert_clk(!so_or_sz && set_clr_onehot[385] |->  OK_A[385])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[386] <= `CQ reset_value;                        
        else if(ready_386_A)
            OK_A[386] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[386] <= `CQ so_or_sz ? OK_A[386] || set_clr_onehot[386] : OK_A[386] && !set_clr_onehot[386];       //update

assign ready_386_A = OK_onehot_A[386] && (valid_bank24_B ? ready_bank24_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[386] |-> !OK_A[386])
`assert_clk(!so_or_sz && set_clr_onehot[386] |->  OK_A[386])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[387] <= `CQ reset_value;                        
        else if(ready_387_A)
            OK_A[387] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[387] <= `CQ so_or_sz ? OK_A[387] || set_clr_onehot[387] : OK_A[387] && !set_clr_onehot[387];       //update

assign ready_387_A = OK_onehot_A[387] && (valid_bank24_B ? ready_bank24_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[387] |-> !OK_A[387])
`assert_clk(!so_or_sz && set_clr_onehot[387] |->  OK_A[387])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[388] <= `CQ reset_value;                        
        else if(ready_388_A)
            OK_A[388] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[388] <= `CQ so_or_sz ? OK_A[388] || set_clr_onehot[388] : OK_A[388] && !set_clr_onehot[388];       //update

assign ready_388_A = OK_onehot_A[388] && (valid_bank24_B ? ready_bank24_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[388] |-> !OK_A[388])
`assert_clk(!so_or_sz && set_clr_onehot[388] |->  OK_A[388])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[389] <= `CQ reset_value;                        
        else if(ready_389_A)
            OK_A[389] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[389] <= `CQ so_or_sz ? OK_A[389] || set_clr_onehot[389] : OK_A[389] && !set_clr_onehot[389];       //update

assign ready_389_A = OK_onehot_A[389] && (valid_bank24_B ? ready_bank24_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[389] |-> !OK_A[389])
`assert_clk(!so_or_sz && set_clr_onehot[389] |->  OK_A[389])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[390] <= `CQ reset_value;                        
        else if(ready_390_A)
            OK_A[390] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[390] <= `CQ so_or_sz ? OK_A[390] || set_clr_onehot[390] : OK_A[390] && !set_clr_onehot[390];       //update

assign ready_390_A = OK_onehot_A[390] && (valid_bank24_B ? ready_bank24_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[390] |-> !OK_A[390])
`assert_clk(!so_or_sz && set_clr_onehot[390] |->  OK_A[390])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[391] <= `CQ reset_value;                        
        else if(ready_391_A)
            OK_A[391] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[391] <= `CQ so_or_sz ? OK_A[391] || set_clr_onehot[391] : OK_A[391] && !set_clr_onehot[391];       //update

assign ready_391_A = OK_onehot_A[391] && (valid_bank24_B ? ready_bank24_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[391] |-> !OK_A[391])
`assert_clk(!so_or_sz && set_clr_onehot[391] |->  OK_A[391])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[392] <= `CQ reset_value;                        
        else if(ready_392_A)
            OK_A[392] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[392] <= `CQ so_or_sz ? OK_A[392] || set_clr_onehot[392] : OK_A[392] && !set_clr_onehot[392];       //update

assign ready_392_A = OK_onehot_A[392] && (valid_bank24_B ? ready_bank24_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[392] |-> !OK_A[392])
`assert_clk(!so_or_sz && set_clr_onehot[392] |->  OK_A[392])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[393] <= `CQ reset_value;                        
        else if(ready_393_A)
            OK_A[393] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[393] <= `CQ so_or_sz ? OK_A[393] || set_clr_onehot[393] : OK_A[393] && !set_clr_onehot[393];       //update

assign ready_393_A = OK_onehot_A[393] && (valid_bank24_B ? ready_bank24_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[393] |-> !OK_A[393])
`assert_clk(!so_or_sz && set_clr_onehot[393] |->  OK_A[393])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[394] <= `CQ reset_value;                        
        else if(ready_394_A)
            OK_A[394] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[394] <= `CQ so_or_sz ? OK_A[394] || set_clr_onehot[394] : OK_A[394] && !set_clr_onehot[394];       //update

assign ready_394_A = OK_onehot_A[394] && (valid_bank24_B ? ready_bank24_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[394] |-> !OK_A[394])
`assert_clk(!so_or_sz && set_clr_onehot[394] |->  OK_A[394])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[395] <= `CQ reset_value;                        
        else if(ready_395_A)
            OK_A[395] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[395] <= `CQ so_or_sz ? OK_A[395] || set_clr_onehot[395] : OK_A[395] && !set_clr_onehot[395];       //update

assign ready_395_A = OK_onehot_A[395] && (valid_bank24_B ? ready_bank24_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[395] |-> !OK_A[395])
`assert_clk(!so_or_sz && set_clr_onehot[395] |->  OK_A[395])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[396] <= `CQ reset_value;                        
        else if(ready_396_A)
            OK_A[396] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[396] <= `CQ so_or_sz ? OK_A[396] || set_clr_onehot[396] : OK_A[396] && !set_clr_onehot[396];       //update

assign ready_396_A = OK_onehot_A[396] && (valid_bank24_B ? ready_bank24_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[396] |-> !OK_A[396])
`assert_clk(!so_or_sz && set_clr_onehot[396] |->  OK_A[396])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[397] <= `CQ reset_value;                        
        else if(ready_397_A)
            OK_A[397] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[397] <= `CQ so_or_sz ? OK_A[397] || set_clr_onehot[397] : OK_A[397] && !set_clr_onehot[397];       //update

assign ready_397_A = OK_onehot_A[397] && (valid_bank24_B ? ready_bank24_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[397] |-> !OK_A[397])
`assert_clk(!so_or_sz && set_clr_onehot[397] |->  OK_A[397])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[398] <= `CQ reset_value;                        
        else if(ready_398_A)
            OK_A[398] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[398] <= `CQ so_or_sz ? OK_A[398] || set_clr_onehot[398] : OK_A[398] && !set_clr_onehot[398];       //update

assign ready_398_A = OK_onehot_A[398] && (valid_bank24_B ? ready_bank24_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[398] |-> !OK_A[398])
`assert_clk(!so_or_sz && set_clr_onehot[398] |->  OK_A[398])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[399] <= `CQ reset_value;                        
        else if(ready_399_A)
            OK_A[399] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[399] <= `CQ so_or_sz ? OK_A[399] || set_clr_onehot[399] : OK_A[399] && !set_clr_onehot[399];       //update

assign ready_399_A = OK_onehot_A[399] && (valid_bank24_B ? ready_bank24_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[399] |-> !OK_A[399])
`assert_clk(!so_or_sz && set_clr_onehot[399] |->  OK_A[399])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[400] <= `CQ reset_value;                        
        else if(ready_400_A)
            OK_A[400] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[400] <= `CQ so_or_sz ? OK_A[400] || set_clr_onehot[400] : OK_A[400] && !set_clr_onehot[400];       //update

assign ready_400_A = OK_onehot_A[400] && (valid_bank25_B ? ready_bank25_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[400] |-> !OK_A[400])
`assert_clk(!so_or_sz && set_clr_onehot[400] |->  OK_A[400])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[401] <= `CQ reset_value;                        
        else if(ready_401_A)
            OK_A[401] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[401] <= `CQ so_or_sz ? OK_A[401] || set_clr_onehot[401] : OK_A[401] && !set_clr_onehot[401];       //update

assign ready_401_A = OK_onehot_A[401] && (valid_bank25_B ? ready_bank25_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[401] |-> !OK_A[401])
`assert_clk(!so_or_sz && set_clr_onehot[401] |->  OK_A[401])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[402] <= `CQ reset_value;                        
        else if(ready_402_A)
            OK_A[402] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[402] <= `CQ so_or_sz ? OK_A[402] || set_clr_onehot[402] : OK_A[402] && !set_clr_onehot[402];       //update

assign ready_402_A = OK_onehot_A[402] && (valid_bank25_B ? ready_bank25_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[402] |-> !OK_A[402])
`assert_clk(!so_or_sz && set_clr_onehot[402] |->  OK_A[402])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[403] <= `CQ reset_value;                        
        else if(ready_403_A)
            OK_A[403] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[403] <= `CQ so_or_sz ? OK_A[403] || set_clr_onehot[403] : OK_A[403] && !set_clr_onehot[403];       //update

assign ready_403_A = OK_onehot_A[403] && (valid_bank25_B ? ready_bank25_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[403] |-> !OK_A[403])
`assert_clk(!so_or_sz && set_clr_onehot[403] |->  OK_A[403])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[404] <= `CQ reset_value;                        
        else if(ready_404_A)
            OK_A[404] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[404] <= `CQ so_or_sz ? OK_A[404] || set_clr_onehot[404] : OK_A[404] && !set_clr_onehot[404];       //update

assign ready_404_A = OK_onehot_A[404] && (valid_bank25_B ? ready_bank25_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[404] |-> !OK_A[404])
`assert_clk(!so_or_sz && set_clr_onehot[404] |->  OK_A[404])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[405] <= `CQ reset_value;                        
        else if(ready_405_A)
            OK_A[405] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[405] <= `CQ so_or_sz ? OK_A[405] || set_clr_onehot[405] : OK_A[405] && !set_clr_onehot[405];       //update

assign ready_405_A = OK_onehot_A[405] && (valid_bank25_B ? ready_bank25_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[405] |-> !OK_A[405])
`assert_clk(!so_or_sz && set_clr_onehot[405] |->  OK_A[405])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[406] <= `CQ reset_value;                        
        else if(ready_406_A)
            OK_A[406] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[406] <= `CQ so_or_sz ? OK_A[406] || set_clr_onehot[406] : OK_A[406] && !set_clr_onehot[406];       //update

assign ready_406_A = OK_onehot_A[406] && (valid_bank25_B ? ready_bank25_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[406] |-> !OK_A[406])
`assert_clk(!so_or_sz && set_clr_onehot[406] |->  OK_A[406])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[407] <= `CQ reset_value;                        
        else if(ready_407_A)
            OK_A[407] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[407] <= `CQ so_or_sz ? OK_A[407] || set_clr_onehot[407] : OK_A[407] && !set_clr_onehot[407];       //update

assign ready_407_A = OK_onehot_A[407] && (valid_bank25_B ? ready_bank25_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[407] |-> !OK_A[407])
`assert_clk(!so_or_sz && set_clr_onehot[407] |->  OK_A[407])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[408] <= `CQ reset_value;                        
        else if(ready_408_A)
            OK_A[408] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[408] <= `CQ so_or_sz ? OK_A[408] || set_clr_onehot[408] : OK_A[408] && !set_clr_onehot[408];       //update

assign ready_408_A = OK_onehot_A[408] && (valid_bank25_B ? ready_bank25_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[408] |-> !OK_A[408])
`assert_clk(!so_or_sz && set_clr_onehot[408] |->  OK_A[408])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[409] <= `CQ reset_value;                        
        else if(ready_409_A)
            OK_A[409] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[409] <= `CQ so_or_sz ? OK_A[409] || set_clr_onehot[409] : OK_A[409] && !set_clr_onehot[409];       //update

assign ready_409_A = OK_onehot_A[409] && (valid_bank25_B ? ready_bank25_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[409] |-> !OK_A[409])
`assert_clk(!so_or_sz && set_clr_onehot[409] |->  OK_A[409])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[410] <= `CQ reset_value;                        
        else if(ready_410_A)
            OK_A[410] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[410] <= `CQ so_or_sz ? OK_A[410] || set_clr_onehot[410] : OK_A[410] && !set_clr_onehot[410];       //update

assign ready_410_A = OK_onehot_A[410] && (valid_bank25_B ? ready_bank25_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[410] |-> !OK_A[410])
`assert_clk(!so_or_sz && set_clr_onehot[410] |->  OK_A[410])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[411] <= `CQ reset_value;                        
        else if(ready_411_A)
            OK_A[411] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[411] <= `CQ so_or_sz ? OK_A[411] || set_clr_onehot[411] : OK_A[411] && !set_clr_onehot[411];       //update

assign ready_411_A = OK_onehot_A[411] && (valid_bank25_B ? ready_bank25_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[411] |-> !OK_A[411])
`assert_clk(!so_or_sz && set_clr_onehot[411] |->  OK_A[411])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[412] <= `CQ reset_value;                        
        else if(ready_412_A)
            OK_A[412] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[412] <= `CQ so_or_sz ? OK_A[412] || set_clr_onehot[412] : OK_A[412] && !set_clr_onehot[412];       //update

assign ready_412_A = OK_onehot_A[412] && (valid_bank25_B ? ready_bank25_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[412] |-> !OK_A[412])
`assert_clk(!so_or_sz && set_clr_onehot[412] |->  OK_A[412])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[413] <= `CQ reset_value;                        
        else if(ready_413_A)
            OK_A[413] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[413] <= `CQ so_or_sz ? OK_A[413] || set_clr_onehot[413] : OK_A[413] && !set_clr_onehot[413];       //update

assign ready_413_A = OK_onehot_A[413] && (valid_bank25_B ? ready_bank25_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[413] |-> !OK_A[413])
`assert_clk(!so_or_sz && set_clr_onehot[413] |->  OK_A[413])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[414] <= `CQ reset_value;                        
        else if(ready_414_A)
            OK_A[414] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[414] <= `CQ so_or_sz ? OK_A[414] || set_clr_onehot[414] : OK_A[414] && !set_clr_onehot[414];       //update

assign ready_414_A = OK_onehot_A[414] && (valid_bank25_B ? ready_bank25_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[414] |-> !OK_A[414])
`assert_clk(!so_or_sz && set_clr_onehot[414] |->  OK_A[414])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[415] <= `CQ reset_value;                        
        else if(ready_415_A)
            OK_A[415] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[415] <= `CQ so_or_sz ? OK_A[415] || set_clr_onehot[415] : OK_A[415] && !set_clr_onehot[415];       //update

assign ready_415_A = OK_onehot_A[415] && (valid_bank25_B ? ready_bank25_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[415] |-> !OK_A[415])
`assert_clk(!so_or_sz && set_clr_onehot[415] |->  OK_A[415])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[416] <= `CQ reset_value;                        
        else if(ready_416_A)
            OK_A[416] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[416] <= `CQ so_or_sz ? OK_A[416] || set_clr_onehot[416] : OK_A[416] && !set_clr_onehot[416];       //update

assign ready_416_A = OK_onehot_A[416] && (valid_bank26_B ? ready_bank26_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[416] |-> !OK_A[416])
`assert_clk(!so_or_sz && set_clr_onehot[416] |->  OK_A[416])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[417] <= `CQ reset_value;                        
        else if(ready_417_A)
            OK_A[417] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[417] <= `CQ so_or_sz ? OK_A[417] || set_clr_onehot[417] : OK_A[417] && !set_clr_onehot[417];       //update

assign ready_417_A = OK_onehot_A[417] && (valid_bank26_B ? ready_bank26_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[417] |-> !OK_A[417])
`assert_clk(!so_or_sz && set_clr_onehot[417] |->  OK_A[417])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[418] <= `CQ reset_value;                        
        else if(ready_418_A)
            OK_A[418] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[418] <= `CQ so_or_sz ? OK_A[418] || set_clr_onehot[418] : OK_A[418] && !set_clr_onehot[418];       //update

assign ready_418_A = OK_onehot_A[418] && (valid_bank26_B ? ready_bank26_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[418] |-> !OK_A[418])
`assert_clk(!so_or_sz && set_clr_onehot[418] |->  OK_A[418])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[419] <= `CQ reset_value;                        
        else if(ready_419_A)
            OK_A[419] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[419] <= `CQ so_or_sz ? OK_A[419] || set_clr_onehot[419] : OK_A[419] && !set_clr_onehot[419];       //update

assign ready_419_A = OK_onehot_A[419] && (valid_bank26_B ? ready_bank26_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[419] |-> !OK_A[419])
`assert_clk(!so_or_sz && set_clr_onehot[419] |->  OK_A[419])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[420] <= `CQ reset_value;                        
        else if(ready_420_A)
            OK_A[420] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[420] <= `CQ so_or_sz ? OK_A[420] || set_clr_onehot[420] : OK_A[420] && !set_clr_onehot[420];       //update

assign ready_420_A = OK_onehot_A[420] && (valid_bank26_B ? ready_bank26_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[420] |-> !OK_A[420])
`assert_clk(!so_or_sz && set_clr_onehot[420] |->  OK_A[420])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[421] <= `CQ reset_value;                        
        else if(ready_421_A)
            OK_A[421] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[421] <= `CQ so_or_sz ? OK_A[421] || set_clr_onehot[421] : OK_A[421] && !set_clr_onehot[421];       //update

assign ready_421_A = OK_onehot_A[421] && (valid_bank26_B ? ready_bank26_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[421] |-> !OK_A[421])
`assert_clk(!so_or_sz && set_clr_onehot[421] |->  OK_A[421])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[422] <= `CQ reset_value;                        
        else if(ready_422_A)
            OK_A[422] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[422] <= `CQ so_or_sz ? OK_A[422] || set_clr_onehot[422] : OK_A[422] && !set_clr_onehot[422];       //update

assign ready_422_A = OK_onehot_A[422] && (valid_bank26_B ? ready_bank26_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[422] |-> !OK_A[422])
`assert_clk(!so_or_sz && set_clr_onehot[422] |->  OK_A[422])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[423] <= `CQ reset_value;                        
        else if(ready_423_A)
            OK_A[423] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[423] <= `CQ so_or_sz ? OK_A[423] || set_clr_onehot[423] : OK_A[423] && !set_clr_onehot[423];       //update

assign ready_423_A = OK_onehot_A[423] && (valid_bank26_B ? ready_bank26_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[423] |-> !OK_A[423])
`assert_clk(!so_or_sz && set_clr_onehot[423] |->  OK_A[423])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[424] <= `CQ reset_value;                        
        else if(ready_424_A)
            OK_A[424] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[424] <= `CQ so_or_sz ? OK_A[424] || set_clr_onehot[424] : OK_A[424] && !set_clr_onehot[424];       //update

assign ready_424_A = OK_onehot_A[424] && (valid_bank26_B ? ready_bank26_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[424] |-> !OK_A[424])
`assert_clk(!so_or_sz && set_clr_onehot[424] |->  OK_A[424])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[425] <= `CQ reset_value;                        
        else if(ready_425_A)
            OK_A[425] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[425] <= `CQ so_or_sz ? OK_A[425] || set_clr_onehot[425] : OK_A[425] && !set_clr_onehot[425];       //update

assign ready_425_A = OK_onehot_A[425] && (valid_bank26_B ? ready_bank26_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[425] |-> !OK_A[425])
`assert_clk(!so_or_sz && set_clr_onehot[425] |->  OK_A[425])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[426] <= `CQ reset_value;                        
        else if(ready_426_A)
            OK_A[426] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[426] <= `CQ so_or_sz ? OK_A[426] || set_clr_onehot[426] : OK_A[426] && !set_clr_onehot[426];       //update

assign ready_426_A = OK_onehot_A[426] && (valid_bank26_B ? ready_bank26_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[426] |-> !OK_A[426])
`assert_clk(!so_or_sz && set_clr_onehot[426] |->  OK_A[426])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[427] <= `CQ reset_value;                        
        else if(ready_427_A)
            OK_A[427] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[427] <= `CQ so_or_sz ? OK_A[427] || set_clr_onehot[427] : OK_A[427] && !set_clr_onehot[427];       //update

assign ready_427_A = OK_onehot_A[427] && (valid_bank26_B ? ready_bank26_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[427] |-> !OK_A[427])
`assert_clk(!so_or_sz && set_clr_onehot[427] |->  OK_A[427])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[428] <= `CQ reset_value;                        
        else if(ready_428_A)
            OK_A[428] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[428] <= `CQ so_or_sz ? OK_A[428] || set_clr_onehot[428] : OK_A[428] && !set_clr_onehot[428];       //update

assign ready_428_A = OK_onehot_A[428] && (valid_bank26_B ? ready_bank26_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[428] |-> !OK_A[428])
`assert_clk(!so_or_sz && set_clr_onehot[428] |->  OK_A[428])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[429] <= `CQ reset_value;                        
        else if(ready_429_A)
            OK_A[429] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[429] <= `CQ so_or_sz ? OK_A[429] || set_clr_onehot[429] : OK_A[429] && !set_clr_onehot[429];       //update

assign ready_429_A = OK_onehot_A[429] && (valid_bank26_B ? ready_bank26_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[429] |-> !OK_A[429])
`assert_clk(!so_or_sz && set_clr_onehot[429] |->  OK_A[429])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[430] <= `CQ reset_value;                        
        else if(ready_430_A)
            OK_A[430] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[430] <= `CQ so_or_sz ? OK_A[430] || set_clr_onehot[430] : OK_A[430] && !set_clr_onehot[430];       //update

assign ready_430_A = OK_onehot_A[430] && (valid_bank26_B ? ready_bank26_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[430] |-> !OK_A[430])
`assert_clk(!so_or_sz && set_clr_onehot[430] |->  OK_A[430])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[431] <= `CQ reset_value;                        
        else if(ready_431_A)
            OK_A[431] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[431] <= `CQ so_or_sz ? OK_A[431] || set_clr_onehot[431] : OK_A[431] && !set_clr_onehot[431];       //update

assign ready_431_A = OK_onehot_A[431] && (valid_bank26_B ? ready_bank26_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[431] |-> !OK_A[431])
`assert_clk(!so_or_sz && set_clr_onehot[431] |->  OK_A[431])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[432] <= `CQ reset_value;                        
        else if(ready_432_A)
            OK_A[432] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[432] <= `CQ so_or_sz ? OK_A[432] || set_clr_onehot[432] : OK_A[432] && !set_clr_onehot[432];       //update

assign ready_432_A = OK_onehot_A[432] && (valid_bank27_B ? ready_bank27_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[432] |-> !OK_A[432])
`assert_clk(!so_or_sz && set_clr_onehot[432] |->  OK_A[432])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[433] <= `CQ reset_value;                        
        else if(ready_433_A)
            OK_A[433] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[433] <= `CQ so_or_sz ? OK_A[433] || set_clr_onehot[433] : OK_A[433] && !set_clr_onehot[433];       //update

assign ready_433_A = OK_onehot_A[433] && (valid_bank27_B ? ready_bank27_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[433] |-> !OK_A[433])
`assert_clk(!so_or_sz && set_clr_onehot[433] |->  OK_A[433])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[434] <= `CQ reset_value;                        
        else if(ready_434_A)
            OK_A[434] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[434] <= `CQ so_or_sz ? OK_A[434] || set_clr_onehot[434] : OK_A[434] && !set_clr_onehot[434];       //update

assign ready_434_A = OK_onehot_A[434] && (valid_bank27_B ? ready_bank27_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[434] |-> !OK_A[434])
`assert_clk(!so_or_sz && set_clr_onehot[434] |->  OK_A[434])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[435] <= `CQ reset_value;                        
        else if(ready_435_A)
            OK_A[435] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[435] <= `CQ so_or_sz ? OK_A[435] || set_clr_onehot[435] : OK_A[435] && !set_clr_onehot[435];       //update

assign ready_435_A = OK_onehot_A[435] && (valid_bank27_B ? ready_bank27_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[435] |-> !OK_A[435])
`assert_clk(!so_or_sz && set_clr_onehot[435] |->  OK_A[435])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[436] <= `CQ reset_value;                        
        else if(ready_436_A)
            OK_A[436] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[436] <= `CQ so_or_sz ? OK_A[436] || set_clr_onehot[436] : OK_A[436] && !set_clr_onehot[436];       //update

assign ready_436_A = OK_onehot_A[436] && (valid_bank27_B ? ready_bank27_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[436] |-> !OK_A[436])
`assert_clk(!so_or_sz && set_clr_onehot[436] |->  OK_A[436])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[437] <= `CQ reset_value;                        
        else if(ready_437_A)
            OK_A[437] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[437] <= `CQ so_or_sz ? OK_A[437] || set_clr_onehot[437] : OK_A[437] && !set_clr_onehot[437];       //update

assign ready_437_A = OK_onehot_A[437] && (valid_bank27_B ? ready_bank27_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[437] |-> !OK_A[437])
`assert_clk(!so_or_sz && set_clr_onehot[437] |->  OK_A[437])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[438] <= `CQ reset_value;                        
        else if(ready_438_A)
            OK_A[438] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[438] <= `CQ so_or_sz ? OK_A[438] || set_clr_onehot[438] : OK_A[438] && !set_clr_onehot[438];       //update

assign ready_438_A = OK_onehot_A[438] && (valid_bank27_B ? ready_bank27_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[438] |-> !OK_A[438])
`assert_clk(!so_or_sz && set_clr_onehot[438] |->  OK_A[438])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[439] <= `CQ reset_value;                        
        else if(ready_439_A)
            OK_A[439] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[439] <= `CQ so_or_sz ? OK_A[439] || set_clr_onehot[439] : OK_A[439] && !set_clr_onehot[439];       //update

assign ready_439_A = OK_onehot_A[439] && (valid_bank27_B ? ready_bank27_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[439] |-> !OK_A[439])
`assert_clk(!so_or_sz && set_clr_onehot[439] |->  OK_A[439])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[440] <= `CQ reset_value;                        
        else if(ready_440_A)
            OK_A[440] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[440] <= `CQ so_or_sz ? OK_A[440] || set_clr_onehot[440] : OK_A[440] && !set_clr_onehot[440];       //update

assign ready_440_A = OK_onehot_A[440] && (valid_bank27_B ? ready_bank27_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[440] |-> !OK_A[440])
`assert_clk(!so_or_sz && set_clr_onehot[440] |->  OK_A[440])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[441] <= `CQ reset_value;                        
        else if(ready_441_A)
            OK_A[441] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[441] <= `CQ so_or_sz ? OK_A[441] || set_clr_onehot[441] : OK_A[441] && !set_clr_onehot[441];       //update

assign ready_441_A = OK_onehot_A[441] && (valid_bank27_B ? ready_bank27_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[441] |-> !OK_A[441])
`assert_clk(!so_or_sz && set_clr_onehot[441] |->  OK_A[441])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[442] <= `CQ reset_value;                        
        else if(ready_442_A)
            OK_A[442] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[442] <= `CQ so_or_sz ? OK_A[442] || set_clr_onehot[442] : OK_A[442] && !set_clr_onehot[442];       //update

assign ready_442_A = OK_onehot_A[442] && (valid_bank27_B ? ready_bank27_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[442] |-> !OK_A[442])
`assert_clk(!so_or_sz && set_clr_onehot[442] |->  OK_A[442])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[443] <= `CQ reset_value;                        
        else if(ready_443_A)
            OK_A[443] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[443] <= `CQ so_or_sz ? OK_A[443] || set_clr_onehot[443] : OK_A[443] && !set_clr_onehot[443];       //update

assign ready_443_A = OK_onehot_A[443] && (valid_bank27_B ? ready_bank27_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[443] |-> !OK_A[443])
`assert_clk(!so_or_sz && set_clr_onehot[443] |->  OK_A[443])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[444] <= `CQ reset_value;                        
        else if(ready_444_A)
            OK_A[444] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[444] <= `CQ so_or_sz ? OK_A[444] || set_clr_onehot[444] : OK_A[444] && !set_clr_onehot[444];       //update

assign ready_444_A = OK_onehot_A[444] && (valid_bank27_B ? ready_bank27_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[444] |-> !OK_A[444])
`assert_clk(!so_or_sz && set_clr_onehot[444] |->  OK_A[444])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[445] <= `CQ reset_value;                        
        else if(ready_445_A)
            OK_A[445] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[445] <= `CQ so_or_sz ? OK_A[445] || set_clr_onehot[445] : OK_A[445] && !set_clr_onehot[445];       //update

assign ready_445_A = OK_onehot_A[445] && (valid_bank27_B ? ready_bank27_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[445] |-> !OK_A[445])
`assert_clk(!so_or_sz && set_clr_onehot[445] |->  OK_A[445])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[446] <= `CQ reset_value;                        
        else if(ready_446_A)
            OK_A[446] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[446] <= `CQ so_or_sz ? OK_A[446] || set_clr_onehot[446] : OK_A[446] && !set_clr_onehot[446];       //update

assign ready_446_A = OK_onehot_A[446] && (valid_bank27_B ? ready_bank27_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[446] |-> !OK_A[446])
`assert_clk(!so_or_sz && set_clr_onehot[446] |->  OK_A[446])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[447] <= `CQ reset_value;                        
        else if(ready_447_A)
            OK_A[447] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[447] <= `CQ so_or_sz ? OK_A[447] || set_clr_onehot[447] : OK_A[447] && !set_clr_onehot[447];       //update

assign ready_447_A = OK_onehot_A[447] && (valid_bank27_B ? ready_bank27_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[447] |-> !OK_A[447])
`assert_clk(!so_or_sz && set_clr_onehot[447] |->  OK_A[447])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[448] <= `CQ reset_value;                        
        else if(ready_448_A)
            OK_A[448] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[448] <= `CQ so_or_sz ? OK_A[448] || set_clr_onehot[448] : OK_A[448] && !set_clr_onehot[448];       //update

assign ready_448_A = OK_onehot_A[448] && (valid_bank28_B ? ready_bank28_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[448] |-> !OK_A[448])
`assert_clk(!so_or_sz && set_clr_onehot[448] |->  OK_A[448])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[449] <= `CQ reset_value;                        
        else if(ready_449_A)
            OK_A[449] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[449] <= `CQ so_or_sz ? OK_A[449] || set_clr_onehot[449] : OK_A[449] && !set_clr_onehot[449];       //update

assign ready_449_A = OK_onehot_A[449] && (valid_bank28_B ? ready_bank28_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[449] |-> !OK_A[449])
`assert_clk(!so_or_sz && set_clr_onehot[449] |->  OK_A[449])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[450] <= `CQ reset_value;                        
        else if(ready_450_A)
            OK_A[450] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[450] <= `CQ so_or_sz ? OK_A[450] || set_clr_onehot[450] : OK_A[450] && !set_clr_onehot[450];       //update

assign ready_450_A = OK_onehot_A[450] && (valid_bank28_B ? ready_bank28_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[450] |-> !OK_A[450])
`assert_clk(!so_or_sz && set_clr_onehot[450] |->  OK_A[450])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[451] <= `CQ reset_value;                        
        else if(ready_451_A)
            OK_A[451] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[451] <= `CQ so_or_sz ? OK_A[451] || set_clr_onehot[451] : OK_A[451] && !set_clr_onehot[451];       //update

assign ready_451_A = OK_onehot_A[451] && (valid_bank28_B ? ready_bank28_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[451] |-> !OK_A[451])
`assert_clk(!so_or_sz && set_clr_onehot[451] |->  OK_A[451])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[452] <= `CQ reset_value;                        
        else if(ready_452_A)
            OK_A[452] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[452] <= `CQ so_or_sz ? OK_A[452] || set_clr_onehot[452] : OK_A[452] && !set_clr_onehot[452];       //update

assign ready_452_A = OK_onehot_A[452] && (valid_bank28_B ? ready_bank28_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[452] |-> !OK_A[452])
`assert_clk(!so_or_sz && set_clr_onehot[452] |->  OK_A[452])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[453] <= `CQ reset_value;                        
        else if(ready_453_A)
            OK_A[453] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[453] <= `CQ so_or_sz ? OK_A[453] || set_clr_onehot[453] : OK_A[453] && !set_clr_onehot[453];       //update

assign ready_453_A = OK_onehot_A[453] && (valid_bank28_B ? ready_bank28_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[453] |-> !OK_A[453])
`assert_clk(!so_or_sz && set_clr_onehot[453] |->  OK_A[453])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[454] <= `CQ reset_value;                        
        else if(ready_454_A)
            OK_A[454] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[454] <= `CQ so_or_sz ? OK_A[454] || set_clr_onehot[454] : OK_A[454] && !set_clr_onehot[454];       //update

assign ready_454_A = OK_onehot_A[454] && (valid_bank28_B ? ready_bank28_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[454] |-> !OK_A[454])
`assert_clk(!so_or_sz && set_clr_onehot[454] |->  OK_A[454])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[455] <= `CQ reset_value;                        
        else if(ready_455_A)
            OK_A[455] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[455] <= `CQ so_or_sz ? OK_A[455] || set_clr_onehot[455] : OK_A[455] && !set_clr_onehot[455];       //update

assign ready_455_A = OK_onehot_A[455] && (valid_bank28_B ? ready_bank28_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[455] |-> !OK_A[455])
`assert_clk(!so_or_sz && set_clr_onehot[455] |->  OK_A[455])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[456] <= `CQ reset_value;                        
        else if(ready_456_A)
            OK_A[456] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[456] <= `CQ so_or_sz ? OK_A[456] || set_clr_onehot[456] : OK_A[456] && !set_clr_onehot[456];       //update

assign ready_456_A = OK_onehot_A[456] && (valid_bank28_B ? ready_bank28_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[456] |-> !OK_A[456])
`assert_clk(!so_or_sz && set_clr_onehot[456] |->  OK_A[456])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[457] <= `CQ reset_value;                        
        else if(ready_457_A)
            OK_A[457] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[457] <= `CQ so_or_sz ? OK_A[457] || set_clr_onehot[457] : OK_A[457] && !set_clr_onehot[457];       //update

assign ready_457_A = OK_onehot_A[457] && (valid_bank28_B ? ready_bank28_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[457] |-> !OK_A[457])
`assert_clk(!so_or_sz && set_clr_onehot[457] |->  OK_A[457])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[458] <= `CQ reset_value;                        
        else if(ready_458_A)
            OK_A[458] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[458] <= `CQ so_or_sz ? OK_A[458] || set_clr_onehot[458] : OK_A[458] && !set_clr_onehot[458];       //update

assign ready_458_A = OK_onehot_A[458] && (valid_bank28_B ? ready_bank28_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[458] |-> !OK_A[458])
`assert_clk(!so_or_sz && set_clr_onehot[458] |->  OK_A[458])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[459] <= `CQ reset_value;                        
        else if(ready_459_A)
            OK_A[459] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[459] <= `CQ so_or_sz ? OK_A[459] || set_clr_onehot[459] : OK_A[459] && !set_clr_onehot[459];       //update

assign ready_459_A = OK_onehot_A[459] && (valid_bank28_B ? ready_bank28_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[459] |-> !OK_A[459])
`assert_clk(!so_or_sz && set_clr_onehot[459] |->  OK_A[459])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[460] <= `CQ reset_value;                        
        else if(ready_460_A)
            OK_A[460] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[460] <= `CQ so_or_sz ? OK_A[460] || set_clr_onehot[460] : OK_A[460] && !set_clr_onehot[460];       //update

assign ready_460_A = OK_onehot_A[460] && (valid_bank28_B ? ready_bank28_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[460] |-> !OK_A[460])
`assert_clk(!so_or_sz && set_clr_onehot[460] |->  OK_A[460])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[461] <= `CQ reset_value;                        
        else if(ready_461_A)
            OK_A[461] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[461] <= `CQ so_or_sz ? OK_A[461] || set_clr_onehot[461] : OK_A[461] && !set_clr_onehot[461];       //update

assign ready_461_A = OK_onehot_A[461] && (valid_bank28_B ? ready_bank28_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[461] |-> !OK_A[461])
`assert_clk(!so_or_sz && set_clr_onehot[461] |->  OK_A[461])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[462] <= `CQ reset_value;                        
        else if(ready_462_A)
            OK_A[462] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[462] <= `CQ so_or_sz ? OK_A[462] || set_clr_onehot[462] : OK_A[462] && !set_clr_onehot[462];       //update

assign ready_462_A = OK_onehot_A[462] && (valid_bank28_B ? ready_bank28_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[462] |-> !OK_A[462])
`assert_clk(!so_or_sz && set_clr_onehot[462] |->  OK_A[462])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[463] <= `CQ reset_value;                        
        else if(ready_463_A)
            OK_A[463] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[463] <= `CQ so_or_sz ? OK_A[463] || set_clr_onehot[463] : OK_A[463] && !set_clr_onehot[463];       //update

assign ready_463_A = OK_onehot_A[463] && (valid_bank28_B ? ready_bank28_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[463] |-> !OK_A[463])
`assert_clk(!so_or_sz && set_clr_onehot[463] |->  OK_A[463])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[464] <= `CQ reset_value;                        
        else if(ready_464_A)
            OK_A[464] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[464] <= `CQ so_or_sz ? OK_A[464] || set_clr_onehot[464] : OK_A[464] && !set_clr_onehot[464];       //update

assign ready_464_A = OK_onehot_A[464] && (valid_bank29_B ? ready_bank29_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[464] |-> !OK_A[464])
`assert_clk(!so_or_sz && set_clr_onehot[464] |->  OK_A[464])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[465] <= `CQ reset_value;                        
        else if(ready_465_A)
            OK_A[465] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[465] <= `CQ so_or_sz ? OK_A[465] || set_clr_onehot[465] : OK_A[465] && !set_clr_onehot[465];       //update

assign ready_465_A = OK_onehot_A[465] && (valid_bank29_B ? ready_bank29_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[465] |-> !OK_A[465])
`assert_clk(!so_or_sz && set_clr_onehot[465] |->  OK_A[465])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[466] <= `CQ reset_value;                        
        else if(ready_466_A)
            OK_A[466] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[466] <= `CQ so_or_sz ? OK_A[466] || set_clr_onehot[466] : OK_A[466] && !set_clr_onehot[466];       //update

assign ready_466_A = OK_onehot_A[466] && (valid_bank29_B ? ready_bank29_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[466] |-> !OK_A[466])
`assert_clk(!so_or_sz && set_clr_onehot[466] |->  OK_A[466])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[467] <= `CQ reset_value;                        
        else if(ready_467_A)
            OK_A[467] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[467] <= `CQ so_or_sz ? OK_A[467] || set_clr_onehot[467] : OK_A[467] && !set_clr_onehot[467];       //update

assign ready_467_A = OK_onehot_A[467] && (valid_bank29_B ? ready_bank29_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[467] |-> !OK_A[467])
`assert_clk(!so_or_sz && set_clr_onehot[467] |->  OK_A[467])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[468] <= `CQ reset_value;                        
        else if(ready_468_A)
            OK_A[468] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[468] <= `CQ so_or_sz ? OK_A[468] || set_clr_onehot[468] : OK_A[468] && !set_clr_onehot[468];       //update

assign ready_468_A = OK_onehot_A[468] && (valid_bank29_B ? ready_bank29_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[468] |-> !OK_A[468])
`assert_clk(!so_or_sz && set_clr_onehot[468] |->  OK_A[468])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[469] <= `CQ reset_value;                        
        else if(ready_469_A)
            OK_A[469] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[469] <= `CQ so_or_sz ? OK_A[469] || set_clr_onehot[469] : OK_A[469] && !set_clr_onehot[469];       //update

assign ready_469_A = OK_onehot_A[469] && (valid_bank29_B ? ready_bank29_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[469] |-> !OK_A[469])
`assert_clk(!so_or_sz && set_clr_onehot[469] |->  OK_A[469])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[470] <= `CQ reset_value;                        
        else if(ready_470_A)
            OK_A[470] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[470] <= `CQ so_or_sz ? OK_A[470] || set_clr_onehot[470] : OK_A[470] && !set_clr_onehot[470];       //update

assign ready_470_A = OK_onehot_A[470] && (valid_bank29_B ? ready_bank29_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[470] |-> !OK_A[470])
`assert_clk(!so_or_sz && set_clr_onehot[470] |->  OK_A[470])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[471] <= `CQ reset_value;                        
        else if(ready_471_A)
            OK_A[471] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[471] <= `CQ so_or_sz ? OK_A[471] || set_clr_onehot[471] : OK_A[471] && !set_clr_onehot[471];       //update

assign ready_471_A = OK_onehot_A[471] && (valid_bank29_B ? ready_bank29_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[471] |-> !OK_A[471])
`assert_clk(!so_or_sz && set_clr_onehot[471] |->  OK_A[471])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[472] <= `CQ reset_value;                        
        else if(ready_472_A)
            OK_A[472] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[472] <= `CQ so_or_sz ? OK_A[472] || set_clr_onehot[472] : OK_A[472] && !set_clr_onehot[472];       //update

assign ready_472_A = OK_onehot_A[472] && (valid_bank29_B ? ready_bank29_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[472] |-> !OK_A[472])
`assert_clk(!so_or_sz && set_clr_onehot[472] |->  OK_A[472])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[473] <= `CQ reset_value;                        
        else if(ready_473_A)
            OK_A[473] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[473] <= `CQ so_or_sz ? OK_A[473] || set_clr_onehot[473] : OK_A[473] && !set_clr_onehot[473];       //update

assign ready_473_A = OK_onehot_A[473] && (valid_bank29_B ? ready_bank29_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[473] |-> !OK_A[473])
`assert_clk(!so_or_sz && set_clr_onehot[473] |->  OK_A[473])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[474] <= `CQ reset_value;                        
        else if(ready_474_A)
            OK_A[474] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[474] <= `CQ so_or_sz ? OK_A[474] || set_clr_onehot[474] : OK_A[474] && !set_clr_onehot[474];       //update

assign ready_474_A = OK_onehot_A[474] && (valid_bank29_B ? ready_bank29_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[474] |-> !OK_A[474])
`assert_clk(!so_or_sz && set_clr_onehot[474] |->  OK_A[474])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[475] <= `CQ reset_value;                        
        else if(ready_475_A)
            OK_A[475] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[475] <= `CQ so_or_sz ? OK_A[475] || set_clr_onehot[475] : OK_A[475] && !set_clr_onehot[475];       //update

assign ready_475_A = OK_onehot_A[475] && (valid_bank29_B ? ready_bank29_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[475] |-> !OK_A[475])
`assert_clk(!so_or_sz && set_clr_onehot[475] |->  OK_A[475])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[476] <= `CQ reset_value;                        
        else if(ready_476_A)
            OK_A[476] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[476] <= `CQ so_or_sz ? OK_A[476] || set_clr_onehot[476] : OK_A[476] && !set_clr_onehot[476];       //update

assign ready_476_A = OK_onehot_A[476] && (valid_bank29_B ? ready_bank29_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[476] |-> !OK_A[476])
`assert_clk(!so_or_sz && set_clr_onehot[476] |->  OK_A[476])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[477] <= `CQ reset_value;                        
        else if(ready_477_A)
            OK_A[477] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[477] <= `CQ so_or_sz ? OK_A[477] || set_clr_onehot[477] : OK_A[477] && !set_clr_onehot[477];       //update

assign ready_477_A = OK_onehot_A[477] && (valid_bank29_B ? ready_bank29_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[477] |-> !OK_A[477])
`assert_clk(!so_or_sz && set_clr_onehot[477] |->  OK_A[477])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[478] <= `CQ reset_value;                        
        else if(ready_478_A)
            OK_A[478] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[478] <= `CQ so_or_sz ? OK_A[478] || set_clr_onehot[478] : OK_A[478] && !set_clr_onehot[478];       //update

assign ready_478_A = OK_onehot_A[478] && (valid_bank29_B ? ready_bank29_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[478] |-> !OK_A[478])
`assert_clk(!so_or_sz && set_clr_onehot[478] |->  OK_A[478])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[479] <= `CQ reset_value;                        
        else if(ready_479_A)
            OK_A[479] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[479] <= `CQ so_or_sz ? OK_A[479] || set_clr_onehot[479] : OK_A[479] && !set_clr_onehot[479];       //update

assign ready_479_A = OK_onehot_A[479] && (valid_bank29_B ? ready_bank29_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[479] |-> !OK_A[479])
`assert_clk(!so_or_sz && set_clr_onehot[479] |->  OK_A[479])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[480] <= `CQ reset_value;                        
        else if(ready_480_A)
            OK_A[480] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[480] <= `CQ so_or_sz ? OK_A[480] || set_clr_onehot[480] : OK_A[480] && !set_clr_onehot[480];       //update

assign ready_480_A = OK_onehot_A[480] && (valid_bank30_B ? ready_bank30_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[480] |-> !OK_A[480])
`assert_clk(!so_or_sz && set_clr_onehot[480] |->  OK_A[480])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[481] <= `CQ reset_value;                        
        else if(ready_481_A)
            OK_A[481] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[481] <= `CQ so_or_sz ? OK_A[481] || set_clr_onehot[481] : OK_A[481] && !set_clr_onehot[481];       //update

assign ready_481_A = OK_onehot_A[481] && (valid_bank30_B ? ready_bank30_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[481] |-> !OK_A[481])
`assert_clk(!so_or_sz && set_clr_onehot[481] |->  OK_A[481])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[482] <= `CQ reset_value;                        
        else if(ready_482_A)
            OK_A[482] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[482] <= `CQ so_or_sz ? OK_A[482] || set_clr_onehot[482] : OK_A[482] && !set_clr_onehot[482];       //update

assign ready_482_A = OK_onehot_A[482] && (valid_bank30_B ? ready_bank30_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[482] |-> !OK_A[482])
`assert_clk(!so_or_sz && set_clr_onehot[482] |->  OK_A[482])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[483] <= `CQ reset_value;                        
        else if(ready_483_A)
            OK_A[483] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[483] <= `CQ so_or_sz ? OK_A[483] || set_clr_onehot[483] : OK_A[483] && !set_clr_onehot[483];       //update

assign ready_483_A = OK_onehot_A[483] && (valid_bank30_B ? ready_bank30_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[483] |-> !OK_A[483])
`assert_clk(!so_or_sz && set_clr_onehot[483] |->  OK_A[483])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[484] <= `CQ reset_value;                        
        else if(ready_484_A)
            OK_A[484] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[484] <= `CQ so_or_sz ? OK_A[484] || set_clr_onehot[484] : OK_A[484] && !set_clr_onehot[484];       //update

assign ready_484_A = OK_onehot_A[484] && (valid_bank30_B ? ready_bank30_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[484] |-> !OK_A[484])
`assert_clk(!so_or_sz && set_clr_onehot[484] |->  OK_A[484])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[485] <= `CQ reset_value;                        
        else if(ready_485_A)
            OK_A[485] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[485] <= `CQ so_or_sz ? OK_A[485] || set_clr_onehot[485] : OK_A[485] && !set_clr_onehot[485];       //update

assign ready_485_A = OK_onehot_A[485] && (valid_bank30_B ? ready_bank30_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[485] |-> !OK_A[485])
`assert_clk(!so_or_sz && set_clr_onehot[485] |->  OK_A[485])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[486] <= `CQ reset_value;                        
        else if(ready_486_A)
            OK_A[486] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[486] <= `CQ so_or_sz ? OK_A[486] || set_clr_onehot[486] : OK_A[486] && !set_clr_onehot[486];       //update

assign ready_486_A = OK_onehot_A[486] && (valid_bank30_B ? ready_bank30_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[486] |-> !OK_A[486])
`assert_clk(!so_or_sz && set_clr_onehot[486] |->  OK_A[486])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[487] <= `CQ reset_value;                        
        else if(ready_487_A)
            OK_A[487] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[487] <= `CQ so_or_sz ? OK_A[487] || set_clr_onehot[487] : OK_A[487] && !set_clr_onehot[487];       //update

assign ready_487_A = OK_onehot_A[487] && (valid_bank30_B ? ready_bank30_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[487] |-> !OK_A[487])
`assert_clk(!so_or_sz && set_clr_onehot[487] |->  OK_A[487])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[488] <= `CQ reset_value;                        
        else if(ready_488_A)
            OK_A[488] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[488] <= `CQ so_or_sz ? OK_A[488] || set_clr_onehot[488] : OK_A[488] && !set_clr_onehot[488];       //update

assign ready_488_A = OK_onehot_A[488] && (valid_bank30_B ? ready_bank30_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[488] |-> !OK_A[488])
`assert_clk(!so_or_sz && set_clr_onehot[488] |->  OK_A[488])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[489] <= `CQ reset_value;                        
        else if(ready_489_A)
            OK_A[489] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[489] <= `CQ so_or_sz ? OK_A[489] || set_clr_onehot[489] : OK_A[489] && !set_clr_onehot[489];       //update

assign ready_489_A = OK_onehot_A[489] && (valid_bank30_B ? ready_bank30_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[489] |-> !OK_A[489])
`assert_clk(!so_or_sz && set_clr_onehot[489] |->  OK_A[489])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[490] <= `CQ reset_value;                        
        else if(ready_490_A)
            OK_A[490] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[490] <= `CQ so_or_sz ? OK_A[490] || set_clr_onehot[490] : OK_A[490] && !set_clr_onehot[490];       //update

assign ready_490_A = OK_onehot_A[490] && (valid_bank30_B ? ready_bank30_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[490] |-> !OK_A[490])
`assert_clk(!so_or_sz && set_clr_onehot[490] |->  OK_A[490])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[491] <= `CQ reset_value;                        
        else if(ready_491_A)
            OK_A[491] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[491] <= `CQ so_or_sz ? OK_A[491] || set_clr_onehot[491] : OK_A[491] && !set_clr_onehot[491];       //update

assign ready_491_A = OK_onehot_A[491] && (valid_bank30_B ? ready_bank30_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[491] |-> !OK_A[491])
`assert_clk(!so_or_sz && set_clr_onehot[491] |->  OK_A[491])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[492] <= `CQ reset_value;                        
        else if(ready_492_A)
            OK_A[492] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[492] <= `CQ so_or_sz ? OK_A[492] || set_clr_onehot[492] : OK_A[492] && !set_clr_onehot[492];       //update

assign ready_492_A = OK_onehot_A[492] && (valid_bank30_B ? ready_bank30_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[492] |-> !OK_A[492])
`assert_clk(!so_or_sz && set_clr_onehot[492] |->  OK_A[492])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[493] <= `CQ reset_value;                        
        else if(ready_493_A)
            OK_A[493] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[493] <= `CQ so_or_sz ? OK_A[493] || set_clr_onehot[493] : OK_A[493] && !set_clr_onehot[493];       //update

assign ready_493_A = OK_onehot_A[493] && (valid_bank30_B ? ready_bank30_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[493] |-> !OK_A[493])
`assert_clk(!so_or_sz && set_clr_onehot[493] |->  OK_A[493])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[494] <= `CQ reset_value;                        
        else if(ready_494_A)
            OK_A[494] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[494] <= `CQ so_or_sz ? OK_A[494] || set_clr_onehot[494] : OK_A[494] && !set_clr_onehot[494];       //update

assign ready_494_A = OK_onehot_A[494] && (valid_bank30_B ? ready_bank30_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[494] |-> !OK_A[494])
`assert_clk(!so_or_sz && set_clr_onehot[494] |->  OK_A[494])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[495] <= `CQ reset_value;                        
        else if(ready_495_A)
            OK_A[495] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[495] <= `CQ so_or_sz ? OK_A[495] || set_clr_onehot[495] : OK_A[495] && !set_clr_onehot[495];       //update

assign ready_495_A = OK_onehot_A[495] && (valid_bank30_B ? ready_bank30_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[495] |-> !OK_A[495])
`assert_clk(!so_or_sz && set_clr_onehot[495] |->  OK_A[495])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[496] <= `CQ reset_value;                        
        else if(ready_496_A)
            OK_A[496] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[496] <= `CQ so_or_sz ? OK_A[496] || set_clr_onehot[496] : OK_A[496] && !set_clr_onehot[496];       //update

assign ready_496_A = OK_onehot_A[496] && (valid_bank31_B ? ready_bank31_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[496] |-> !OK_A[496])
`assert_clk(!so_or_sz && set_clr_onehot[496] |->  OK_A[496])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[497] <= `CQ reset_value;                        
        else if(ready_497_A)
            OK_A[497] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[497] <= `CQ so_or_sz ? OK_A[497] || set_clr_onehot[497] : OK_A[497] && !set_clr_onehot[497];       //update

assign ready_497_A = OK_onehot_A[497] && (valid_bank31_B ? ready_bank31_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[497] |-> !OK_A[497])
`assert_clk(!so_or_sz && set_clr_onehot[497] |->  OK_A[497])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[498] <= `CQ reset_value;                        
        else if(ready_498_A)
            OK_A[498] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[498] <= `CQ so_or_sz ? OK_A[498] || set_clr_onehot[498] : OK_A[498] && !set_clr_onehot[498];       //update

assign ready_498_A = OK_onehot_A[498] && (valid_bank31_B ? ready_bank31_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[498] |-> !OK_A[498])
`assert_clk(!so_or_sz && set_clr_onehot[498] |->  OK_A[498])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[499] <= `CQ reset_value;                        
        else if(ready_499_A)
            OK_A[499] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[499] <= `CQ so_or_sz ? OK_A[499] || set_clr_onehot[499] : OK_A[499] && !set_clr_onehot[499];       //update

assign ready_499_A = OK_onehot_A[499] && (valid_bank31_B ? ready_bank31_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[499] |-> !OK_A[499])
`assert_clk(!so_or_sz && set_clr_onehot[499] |->  OK_A[499])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[500] <= `CQ reset_value;                        
        else if(ready_500_A)
            OK_A[500] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[500] <= `CQ so_or_sz ? OK_A[500] || set_clr_onehot[500] : OK_A[500] && !set_clr_onehot[500];       //update

assign ready_500_A = OK_onehot_A[500] && (valid_bank31_B ? ready_bank31_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[500] |-> !OK_A[500])
`assert_clk(!so_or_sz && set_clr_onehot[500] |->  OK_A[500])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[501] <= `CQ reset_value;                        
        else if(ready_501_A)
            OK_A[501] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[501] <= `CQ so_or_sz ? OK_A[501] || set_clr_onehot[501] : OK_A[501] && !set_clr_onehot[501];       //update

assign ready_501_A = OK_onehot_A[501] && (valid_bank31_B ? ready_bank31_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[501] |-> !OK_A[501])
`assert_clk(!so_or_sz && set_clr_onehot[501] |->  OK_A[501])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[502] <= `CQ reset_value;                        
        else if(ready_502_A)
            OK_A[502] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[502] <= `CQ so_or_sz ? OK_A[502] || set_clr_onehot[502] : OK_A[502] && !set_clr_onehot[502];       //update

assign ready_502_A = OK_onehot_A[502] && (valid_bank31_B ? ready_bank31_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[502] |-> !OK_A[502])
`assert_clk(!so_or_sz && set_clr_onehot[502] |->  OK_A[502])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[503] <= `CQ reset_value;                        
        else if(ready_503_A)
            OK_A[503] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[503] <= `CQ so_or_sz ? OK_A[503] || set_clr_onehot[503] : OK_A[503] && !set_clr_onehot[503];       //update

assign ready_503_A = OK_onehot_A[503] && (valid_bank31_B ? ready_bank31_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[503] |-> !OK_A[503])
`assert_clk(!so_or_sz && set_clr_onehot[503] |->  OK_A[503])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[504] <= `CQ reset_value;                        
        else if(ready_504_A)
            OK_A[504] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[504] <= `CQ so_or_sz ? OK_A[504] || set_clr_onehot[504] : OK_A[504] && !set_clr_onehot[504];       //update

assign ready_504_A = OK_onehot_A[504] && (valid_bank31_B ? ready_bank31_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[504] |-> !OK_A[504])
`assert_clk(!so_or_sz && set_clr_onehot[504] |->  OK_A[504])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[505] <= `CQ reset_value;                        
        else if(ready_505_A)
            OK_A[505] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[505] <= `CQ so_or_sz ? OK_A[505] || set_clr_onehot[505] : OK_A[505] && !set_clr_onehot[505];       //update

assign ready_505_A = OK_onehot_A[505] && (valid_bank31_B ? ready_bank31_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[505] |-> !OK_A[505])
`assert_clk(!so_or_sz && set_clr_onehot[505] |->  OK_A[505])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[506] <= `CQ reset_value;                        
        else if(ready_506_A)
            OK_A[506] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[506] <= `CQ so_or_sz ? OK_A[506] || set_clr_onehot[506] : OK_A[506] && !set_clr_onehot[506];       //update

assign ready_506_A = OK_onehot_A[506] && (valid_bank31_B ? ready_bank31_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[506] |-> !OK_A[506])
`assert_clk(!so_or_sz && set_clr_onehot[506] |->  OK_A[506])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[507] <= `CQ reset_value;                        
        else if(ready_507_A)
            OK_A[507] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[507] <= `CQ so_or_sz ? OK_A[507] || set_clr_onehot[507] : OK_A[507] && !set_clr_onehot[507];       //update

assign ready_507_A = OK_onehot_A[507] && (valid_bank31_B ? ready_bank31_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[507] |-> !OK_A[507])
`assert_clk(!so_or_sz && set_clr_onehot[507] |->  OK_A[507])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[508] <= `CQ reset_value;                        
        else if(ready_508_A)
            OK_A[508] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[508] <= `CQ so_or_sz ? OK_A[508] || set_clr_onehot[508] : OK_A[508] && !set_clr_onehot[508];       //update

assign ready_508_A = OK_onehot_A[508] && (valid_bank31_B ? ready_bank31_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[508] |-> !OK_A[508])
`assert_clk(!so_or_sz && set_clr_onehot[508] |->  OK_A[508])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[509] <= `CQ reset_value;                        
        else if(ready_509_A)
            OK_A[509] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[509] <= `CQ so_or_sz ? OK_A[509] || set_clr_onehot[509] : OK_A[509] && !set_clr_onehot[509];       //update

assign ready_509_A = OK_onehot_A[509] && (valid_bank31_B ? ready_bank31_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[509] |-> !OK_A[509])
`assert_clk(!so_or_sz && set_clr_onehot[509] |->  OK_A[509])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[510] <= `CQ reset_value;                        
        else if(ready_510_A)
            OK_A[510] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[510] <= `CQ so_or_sz ? OK_A[510] || set_clr_onehot[510] : OK_A[510] && !set_clr_onehot[510];       //update

assign ready_510_A = OK_onehot_A[510] && (valid_bank31_B ? ready_bank31_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[510] |-> !OK_A[510])
`assert_clk(!so_or_sz && set_clr_onehot[510] |->  OK_A[510])
    always@(posedge clk or negedge rstN)
        if(!rstN)
            OK_A[511] <= `CQ reset_value;                        
        else if(ready_511_A)
            OK_A[511] <= `CQ clean_bit;                        //when choosen and can be move to next pipeline, clear to clean_bit
        else 
            OK_A[511] <= `CQ so_or_sz ? OK_A[511] || set_clr_onehot[511] : OK_A[511] && !set_clr_onehot[511];       //update

assign ready_511_A = OK_onehot_A[511] && (valid_bank31_B ? ready_bank31_B : 1);

`assert_clk( so_or_sz && set_clr_onehot[511] |-> !OK_A[511])
`assert_clk(!so_or_sz && set_clr_onehot[511] |->  OK_A[511])
////////////////////////////////////////////////////////////////////////////////////////////////////////
//pipeline A
////////////////////////////////////////////////////////////////////////////////////////////////////////
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_0(
        .in(OK_A[0*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[0*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_0(
        .in(OK_A[0*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[0*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank0_A = so_or_sz ? |OK_A[0*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[0*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank0_A = valid_bank0_B ? ready_bank0_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_1(
        .in(OK_A[1*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[1*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_1(
        .in(OK_A[1*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[1*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank1_A = so_or_sz ? |OK_A[1*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[1*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank1_A = valid_bank1_B ? ready_bank1_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_2(
        .in(OK_A[2*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[2*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_2(
        .in(OK_A[2*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[2*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank2_A = so_or_sz ? |OK_A[2*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[2*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank2_A = valid_bank2_B ? ready_bank2_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_3(
        .in(OK_A[3*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[3*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_3(
        .in(OK_A[3*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[3*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank3_A = so_or_sz ? |OK_A[3*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[3*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank3_A = valid_bank3_B ? ready_bank3_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_4(
        .in(OK_A[4*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[4*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_4(
        .in(OK_A[4*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[4*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank4_A = so_or_sz ? |OK_A[4*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[4*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank4_A = valid_bank4_B ? ready_bank4_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_5(
        .in(OK_A[5*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[5*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_5(
        .in(OK_A[5*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[5*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank5_A = so_or_sz ? |OK_A[5*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[5*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank5_A = valid_bank5_B ? ready_bank5_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_6(
        .in(OK_A[6*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[6*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_6(
        .in(OK_A[6*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[6*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank6_A = so_or_sz ? |OK_A[6*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[6*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank6_A = valid_bank6_B ? ready_bank6_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_7(
        .in(OK_A[7*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[7*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_7(
        .in(OK_A[7*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[7*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank7_A = so_or_sz ? |OK_A[7*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[7*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank7_A = valid_bank7_B ? ready_bank7_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_8(
        .in(OK_A[8*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[8*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_8(
        .in(OK_A[8*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[8*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank8_A = so_or_sz ? |OK_A[8*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[8*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank8_A = valid_bank8_B ? ready_bank8_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_9(
        .in(OK_A[9*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[9*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_9(
        .in(OK_A[9*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[9*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank9_A = so_or_sz ? |OK_A[9*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[9*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank9_A = valid_bank9_B ? ready_bank9_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_10(
        .in(OK_A[10*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[10*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_10(
        .in(OK_A[10*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[10*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank10_A = so_or_sz ? |OK_A[10*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[10*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank10_A = valid_bank10_B ? ready_bank10_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_11(
        .in(OK_A[11*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[11*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_11(
        .in(OK_A[11*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[11*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank11_A = so_or_sz ? |OK_A[11*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[11*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank11_A = valid_bank11_B ? ready_bank11_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_12(
        .in(OK_A[12*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[12*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_12(
        .in(OK_A[12*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[12*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank12_A = so_or_sz ? |OK_A[12*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[12*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank12_A = valid_bank12_B ? ready_bank12_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_13(
        .in(OK_A[13*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[13*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_13(
        .in(OK_A[13*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[13*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank13_A = so_or_sz ? |OK_A[13*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[13*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank13_A = valid_bank13_B ? ready_bank13_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_14(
        .in(OK_A[14*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[14*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_14(
        .in(OK_A[14*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[14*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank14_A = so_or_sz ? |OK_A[14*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[14*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank14_A = valid_bank14_B ? ready_bank14_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_15(
        .in(OK_A[15*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[15*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_15(
        .in(OK_A[15*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[15*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank15_A = so_or_sz ? |OK_A[15*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[15*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank15_A = valid_bank15_B ? ready_bank15_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_16(
        .in(OK_A[16*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[16*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_16(
        .in(OK_A[16*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[16*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank16_A = so_or_sz ? |OK_A[16*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[16*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank16_A = valid_bank16_B ? ready_bank16_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_17(
        .in(OK_A[17*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[17*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_17(
        .in(OK_A[17*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[17*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank17_A = so_or_sz ? |OK_A[17*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[17*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank17_A = valid_bank17_B ? ready_bank17_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_18(
        .in(OK_A[18*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[18*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_18(
        .in(OK_A[18*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[18*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank18_A = so_or_sz ? |OK_A[18*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[18*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank18_A = valid_bank18_B ? ready_bank18_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_19(
        .in(OK_A[19*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[19*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_19(
        .in(OK_A[19*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[19*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank19_A = so_or_sz ? |OK_A[19*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[19*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank19_A = valid_bank19_B ? ready_bank19_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_20(
        .in(OK_A[20*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[20*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_20(
        .in(OK_A[20*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[20*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank20_A = so_or_sz ? |OK_A[20*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[20*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank20_A = valid_bank20_B ? ready_bank20_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_21(
        .in(OK_A[21*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[21*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_21(
        .in(OK_A[21*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[21*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank21_A = so_or_sz ? |OK_A[21*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[21*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank21_A = valid_bank21_B ? ready_bank21_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_22(
        .in(OK_A[22*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[22*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_22(
        .in(OK_A[22*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[22*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank22_A = so_or_sz ? |OK_A[22*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[22*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank22_A = valid_bank22_B ? ready_bank22_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_23(
        .in(OK_A[23*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[23*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_23(
        .in(OK_A[23*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[23*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank23_A = so_or_sz ? |OK_A[23*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[23*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank23_A = valid_bank23_B ? ready_bank23_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_24(
        .in(OK_A[24*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[24*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_24(
        .in(OK_A[24*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[24*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank24_A = so_or_sz ? |OK_A[24*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[24*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank24_A = valid_bank24_B ? ready_bank24_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_25(
        .in(OK_A[25*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[25*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_25(
        .in(OK_A[25*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[25*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank25_A = so_or_sz ? |OK_A[25*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[25*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank25_A = valid_bank25_B ? ready_bank25_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_26(
        .in(OK_A[26*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[26*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_26(
        .in(OK_A[26*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[26*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank26_A = so_or_sz ? |OK_A[26*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[26*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank26_A = valid_bank26_B ? ready_bank26_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_27(
        .in(OK_A[27*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[27*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_27(
        .in(OK_A[27*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[27*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank27_A = so_or_sz ? |OK_A[27*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[27*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank27_A = valid_bank27_B ? ready_bank27_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_28(
        .in(OK_A[28*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[28*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_28(
        .in(OK_A[28*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[28*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank28_A = so_or_sz ? |OK_A[28*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[28*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank28_A = valid_bank28_B ? ready_bank28_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_29(
        .in(OK_A[29*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[29*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_29(
        .in(OK_A[29*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[29*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank29_A = so_or_sz ? |OK_A[29*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[29*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank29_A = valid_bank29_B ? ready_bank29_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_30(
        .in(OK_A[30*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[30*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_30(
        .in(OK_A[30*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[30*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank30_A = so_or_sz ? |OK_A[30*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[30*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank30_A = valid_bank30_B ? ready_bank30_B : 1;       //the bank's onehot bit can move to next pipeline
    so #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) so_bank_31(
        .in(OK_A[31*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_so_onehot_A[31*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

    sz #(
        .DATA(((1<<9)/(1<<(9-4))))
    ) sz_bank_31(
        .in(OK_A[31*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]),
        .out(OK_sz_onehot_A[31*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])
    );

assign valid_bank31_A = so_or_sz ? |OK_A[31*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_A[31*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]);               //the bank has valid OK bit
assign ready_bank31_A = valid_bank31_B ? ready_bank31_B : 1;       //the bank's onehot bit can move to next pipeline

    assign OK_onehot_A = so_or_sz ? OK_so_onehot_A : OK_sz_onehot_A;

////////////////////////////////////////////////////////////////////////////////////////////////////////
//pipeline B
////////////////////////////////////////////////////////////////////////////////////////////////////////
assign valid_B = |valid_bank_d_B;                                                 //has valid OK bit
assign ready_B = valid_B1 ? ready_B1 : 1;
//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[0*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank0_A && ready_bank0_A || valid_bank0_B && ready_bank0_B, valid_bank0_A && ready_bank0_A ? OK_onehot_A[0*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank0_B, valid_bank0_A && ready_bank0_A, valid_bank0_B && ready_bank0_B)
`assert_clk(valid_bank0_B |-> (so_or_sz ? |OK_onehot_B[0*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[0*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[0] = valid_bank0_B;               //all bank compose to one array

assign ready_bank0_B = ready_B && valid_bank_onehot_B[0];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[1*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank1_A && ready_bank1_A || valid_bank1_B && ready_bank1_B, valid_bank1_A && ready_bank1_A ? OK_onehot_A[1*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank1_B, valid_bank1_A && ready_bank1_A, valid_bank1_B && ready_bank1_B)
`assert_clk(valid_bank1_B |-> (so_or_sz ? |OK_onehot_B[1*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[1*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[1] = valid_bank1_B;               //all bank compose to one array

assign ready_bank1_B = ready_B && valid_bank_onehot_B[1];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[2*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank2_A && ready_bank2_A || valid_bank2_B && ready_bank2_B, valid_bank2_A && ready_bank2_A ? OK_onehot_A[2*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank2_B, valid_bank2_A && ready_bank2_A, valid_bank2_B && ready_bank2_B)
`assert_clk(valid_bank2_B |-> (so_or_sz ? |OK_onehot_B[2*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[2*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[2] = valid_bank2_B;               //all bank compose to one array

assign ready_bank2_B = ready_B && valid_bank_onehot_B[2];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[3*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank3_A && ready_bank3_A || valid_bank3_B && ready_bank3_B, valid_bank3_A && ready_bank3_A ? OK_onehot_A[3*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank3_B, valid_bank3_A && ready_bank3_A, valid_bank3_B && ready_bank3_B)
`assert_clk(valid_bank3_B |-> (so_or_sz ? |OK_onehot_B[3*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[3*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[3] = valid_bank3_B;               //all bank compose to one array

assign ready_bank3_B = ready_B && valid_bank_onehot_B[3];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[4*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank4_A && ready_bank4_A || valid_bank4_B && ready_bank4_B, valid_bank4_A && ready_bank4_A ? OK_onehot_A[4*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank4_B, valid_bank4_A && ready_bank4_A, valid_bank4_B && ready_bank4_B)
`assert_clk(valid_bank4_B |-> (so_or_sz ? |OK_onehot_B[4*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[4*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[4] = valid_bank4_B;               //all bank compose to one array

assign ready_bank4_B = ready_B && valid_bank_onehot_B[4];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[5*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank5_A && ready_bank5_A || valid_bank5_B && ready_bank5_B, valid_bank5_A && ready_bank5_A ? OK_onehot_A[5*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank5_B, valid_bank5_A && ready_bank5_A, valid_bank5_B && ready_bank5_B)
`assert_clk(valid_bank5_B |-> (so_or_sz ? |OK_onehot_B[5*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[5*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[5] = valid_bank5_B;               //all bank compose to one array

assign ready_bank5_B = ready_B && valid_bank_onehot_B[5];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[6*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank6_A && ready_bank6_A || valid_bank6_B && ready_bank6_B, valid_bank6_A && ready_bank6_A ? OK_onehot_A[6*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank6_B, valid_bank6_A && ready_bank6_A, valid_bank6_B && ready_bank6_B)
`assert_clk(valid_bank6_B |-> (so_or_sz ? |OK_onehot_B[6*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[6*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[6] = valid_bank6_B;               //all bank compose to one array

assign ready_bank6_B = ready_B && valid_bank_onehot_B[6];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[7*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank7_A && ready_bank7_A || valid_bank7_B && ready_bank7_B, valid_bank7_A && ready_bank7_A ? OK_onehot_A[7*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank7_B, valid_bank7_A && ready_bank7_A, valid_bank7_B && ready_bank7_B)
`assert_clk(valid_bank7_B |-> (so_or_sz ? |OK_onehot_B[7*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[7*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[7] = valid_bank7_B;               //all bank compose to one array

assign ready_bank7_B = ready_B && valid_bank_onehot_B[7];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[8*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank8_A && ready_bank8_A || valid_bank8_B && ready_bank8_B, valid_bank8_A && ready_bank8_A ? OK_onehot_A[8*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank8_B, valid_bank8_A && ready_bank8_A, valid_bank8_B && ready_bank8_B)
`assert_clk(valid_bank8_B |-> (so_or_sz ? |OK_onehot_B[8*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[8*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[8] = valid_bank8_B;               //all bank compose to one array

assign ready_bank8_B = ready_B && valid_bank_onehot_B[8];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[9*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank9_A && ready_bank9_A || valid_bank9_B && ready_bank9_B, valid_bank9_A && ready_bank9_A ? OK_onehot_A[9*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank9_B, valid_bank9_A && ready_bank9_A, valid_bank9_B && ready_bank9_B)
`assert_clk(valid_bank9_B |-> (so_or_sz ? |OK_onehot_B[9*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[9*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[9] = valid_bank9_B;               //all bank compose to one array

assign ready_bank9_B = ready_B && valid_bank_onehot_B[9];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[10*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank10_A && ready_bank10_A || valid_bank10_B && ready_bank10_B, valid_bank10_A && ready_bank10_A ? OK_onehot_A[10*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank10_B, valid_bank10_A && ready_bank10_A, valid_bank10_B && ready_bank10_B)
`assert_clk(valid_bank10_B |-> (so_or_sz ? |OK_onehot_B[10*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[10*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[10] = valid_bank10_B;               //all bank compose to one array

assign ready_bank10_B = ready_B && valid_bank_onehot_B[10];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[11*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank11_A && ready_bank11_A || valid_bank11_B && ready_bank11_B, valid_bank11_A && ready_bank11_A ? OK_onehot_A[11*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank11_B, valid_bank11_A && ready_bank11_A, valid_bank11_B && ready_bank11_B)
`assert_clk(valid_bank11_B |-> (so_or_sz ? |OK_onehot_B[11*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[11*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[11] = valid_bank11_B;               //all bank compose to one array

assign ready_bank11_B = ready_B && valid_bank_onehot_B[11];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[12*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank12_A && ready_bank12_A || valid_bank12_B && ready_bank12_B, valid_bank12_A && ready_bank12_A ? OK_onehot_A[12*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank12_B, valid_bank12_A && ready_bank12_A, valid_bank12_B && ready_bank12_B)
`assert_clk(valid_bank12_B |-> (so_or_sz ? |OK_onehot_B[12*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[12*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[12] = valid_bank12_B;               //all bank compose to one array

assign ready_bank12_B = ready_B && valid_bank_onehot_B[12];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[13*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank13_A && ready_bank13_A || valid_bank13_B && ready_bank13_B, valid_bank13_A && ready_bank13_A ? OK_onehot_A[13*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank13_B, valid_bank13_A && ready_bank13_A, valid_bank13_B && ready_bank13_B)
`assert_clk(valid_bank13_B |-> (so_or_sz ? |OK_onehot_B[13*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[13*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[13] = valid_bank13_B;               //all bank compose to one array

assign ready_bank13_B = ready_B && valid_bank_onehot_B[13];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[14*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank14_A && ready_bank14_A || valid_bank14_B && ready_bank14_B, valid_bank14_A && ready_bank14_A ? OK_onehot_A[14*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank14_B, valid_bank14_A && ready_bank14_A, valid_bank14_B && ready_bank14_B)
`assert_clk(valid_bank14_B |-> (so_or_sz ? |OK_onehot_B[14*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[14*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[14] = valid_bank14_B;               //all bank compose to one array

assign ready_bank14_B = ready_B && valid_bank_onehot_B[14];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[15*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank15_A && ready_bank15_A || valid_bank15_B && ready_bank15_B, valid_bank15_A && ready_bank15_A ? OK_onehot_A[15*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank15_B, valid_bank15_A && ready_bank15_A, valid_bank15_B && ready_bank15_B)
`assert_clk(valid_bank15_B |-> (so_or_sz ? |OK_onehot_B[15*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[15*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[15] = valid_bank15_B;               //all bank compose to one array

assign ready_bank15_B = ready_B && valid_bank_onehot_B[15];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[16*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank16_A && ready_bank16_A || valid_bank16_B && ready_bank16_B, valid_bank16_A && ready_bank16_A ? OK_onehot_A[16*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank16_B, valid_bank16_A && ready_bank16_A, valid_bank16_B && ready_bank16_B)
`assert_clk(valid_bank16_B |-> (so_or_sz ? |OK_onehot_B[16*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[16*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[16] = valid_bank16_B;               //all bank compose to one array

assign ready_bank16_B = ready_B && valid_bank_onehot_B[16];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[17*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank17_A && ready_bank17_A || valid_bank17_B && ready_bank17_B, valid_bank17_A && ready_bank17_A ? OK_onehot_A[17*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank17_B, valid_bank17_A && ready_bank17_A, valid_bank17_B && ready_bank17_B)
`assert_clk(valid_bank17_B |-> (so_or_sz ? |OK_onehot_B[17*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[17*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[17] = valid_bank17_B;               //all bank compose to one array

assign ready_bank17_B = ready_B && valid_bank_onehot_B[17];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[18*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank18_A && ready_bank18_A || valid_bank18_B && ready_bank18_B, valid_bank18_A && ready_bank18_A ? OK_onehot_A[18*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank18_B, valid_bank18_A && ready_bank18_A, valid_bank18_B && ready_bank18_B)
`assert_clk(valid_bank18_B |-> (so_or_sz ? |OK_onehot_B[18*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[18*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[18] = valid_bank18_B;               //all bank compose to one array

assign ready_bank18_B = ready_B && valid_bank_onehot_B[18];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[19*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank19_A && ready_bank19_A || valid_bank19_B && ready_bank19_B, valid_bank19_A && ready_bank19_A ? OK_onehot_A[19*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank19_B, valid_bank19_A && ready_bank19_A, valid_bank19_B && ready_bank19_B)
`assert_clk(valid_bank19_B |-> (so_or_sz ? |OK_onehot_B[19*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[19*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[19] = valid_bank19_B;               //all bank compose to one array

assign ready_bank19_B = ready_B && valid_bank_onehot_B[19];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[20*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank20_A && ready_bank20_A || valid_bank20_B && ready_bank20_B, valid_bank20_A && ready_bank20_A ? OK_onehot_A[20*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank20_B, valid_bank20_A && ready_bank20_A, valid_bank20_B && ready_bank20_B)
`assert_clk(valid_bank20_B |-> (so_or_sz ? |OK_onehot_B[20*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[20*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[20] = valid_bank20_B;               //all bank compose to one array

assign ready_bank20_B = ready_B && valid_bank_onehot_B[20];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[21*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank21_A && ready_bank21_A || valid_bank21_B && ready_bank21_B, valid_bank21_A && ready_bank21_A ? OK_onehot_A[21*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank21_B, valid_bank21_A && ready_bank21_A, valid_bank21_B && ready_bank21_B)
`assert_clk(valid_bank21_B |-> (so_or_sz ? |OK_onehot_B[21*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[21*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[21] = valid_bank21_B;               //all bank compose to one array

assign ready_bank21_B = ready_B && valid_bank_onehot_B[21];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[22*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank22_A && ready_bank22_A || valid_bank22_B && ready_bank22_B, valid_bank22_A && ready_bank22_A ? OK_onehot_A[22*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank22_B, valid_bank22_A && ready_bank22_A, valid_bank22_B && ready_bank22_B)
`assert_clk(valid_bank22_B |-> (so_or_sz ? |OK_onehot_B[22*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[22*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[22] = valid_bank22_B;               //all bank compose to one array

assign ready_bank22_B = ready_B && valid_bank_onehot_B[22];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[23*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank23_A && ready_bank23_A || valid_bank23_B && ready_bank23_B, valid_bank23_A && ready_bank23_A ? OK_onehot_A[23*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank23_B, valid_bank23_A && ready_bank23_A, valid_bank23_B && ready_bank23_B)
`assert_clk(valid_bank23_B |-> (so_or_sz ? |OK_onehot_B[23*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[23*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[23] = valid_bank23_B;               //all bank compose to one array

assign ready_bank23_B = ready_B && valid_bank_onehot_B[23];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[24*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank24_A && ready_bank24_A || valid_bank24_B && ready_bank24_B, valid_bank24_A && ready_bank24_A ? OK_onehot_A[24*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank24_B, valid_bank24_A && ready_bank24_A, valid_bank24_B && ready_bank24_B)
`assert_clk(valid_bank24_B |-> (so_or_sz ? |OK_onehot_B[24*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[24*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[24] = valid_bank24_B;               //all bank compose to one array

assign ready_bank24_B = ready_B && valid_bank_onehot_B[24];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[25*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank25_A && ready_bank25_A || valid_bank25_B && ready_bank25_B, valid_bank25_A && ready_bank25_A ? OK_onehot_A[25*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank25_B, valid_bank25_A && ready_bank25_A, valid_bank25_B && ready_bank25_B)
`assert_clk(valid_bank25_B |-> (so_or_sz ? |OK_onehot_B[25*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[25*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[25] = valid_bank25_B;               //all bank compose to one array

assign ready_bank25_B = ready_B && valid_bank_onehot_B[25];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[26*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank26_A && ready_bank26_A || valid_bank26_B && ready_bank26_B, valid_bank26_A && ready_bank26_A ? OK_onehot_A[26*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank26_B, valid_bank26_A && ready_bank26_A, valid_bank26_B && ready_bank26_B)
`assert_clk(valid_bank26_B |-> (so_or_sz ? |OK_onehot_B[26*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[26*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[26] = valid_bank26_B;               //all bank compose to one array

assign ready_bank26_B = ready_B && valid_bank_onehot_B[26];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[27*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank27_A && ready_bank27_A || valid_bank27_B && ready_bank27_B, valid_bank27_A && ready_bank27_A ? OK_onehot_A[27*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank27_B, valid_bank27_A && ready_bank27_A, valid_bank27_B && ready_bank27_B)
`assert_clk(valid_bank27_B |-> (so_or_sz ? |OK_onehot_B[27*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[27*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[27] = valid_bank27_B;               //all bank compose to one array

assign ready_bank27_B = ready_B && valid_bank_onehot_B[27];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[28*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank28_A && ready_bank28_A || valid_bank28_B && ready_bank28_B, valid_bank28_A && ready_bank28_A ? OK_onehot_A[28*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank28_B, valid_bank28_A && ready_bank28_A, valid_bank28_B && ready_bank28_B)
`assert_clk(valid_bank28_B |-> (so_or_sz ? |OK_onehot_B[28*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[28*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[28] = valid_bank28_B;               //all bank compose to one array

assign ready_bank28_B = ready_B && valid_bank_onehot_B[28];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[29*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank29_A && ready_bank29_A || valid_bank29_B && ready_bank29_B, valid_bank29_A && ready_bank29_A ? OK_onehot_A[29*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank29_B, valid_bank29_A && ready_bank29_A, valid_bank29_B && ready_bank29_B)
`assert_clk(valid_bank29_B |-> (so_or_sz ? |OK_onehot_B[29*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[29*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[29] = valid_bank29_B;               //all bank compose to one array

assign ready_bank29_B = ready_B && valid_bank_onehot_B[29];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[30*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank30_A && ready_bank30_A || valid_bank30_B && ready_bank30_B, valid_bank30_A && ready_bank30_A ? OK_onehot_A[30*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank30_B, valid_bank30_A && ready_bank30_A, valid_bank30_B && ready_bank30_B)
`assert_clk(valid_bank30_B |-> (so_or_sz ? |OK_onehot_B[30*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[30*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[30] = valid_bank30_B;               //all bank compose to one array

assign ready_bank30_B = ready_B && valid_bank_onehot_B[30];      //which bank has been choosen, which bank can move to next pipeline

//for each bank, latch the choosen onehot bit
`Flop(OK_onehot_B[31*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))], valid_bank31_A && ready_bank31_A || valid_bank31_B && ready_bank31_B, valid_bank31_A && ready_bank31_A ? OK_onehot_A[31*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : {((1<<9)/(1<<(9-4))){clean_bit}})
`FlopSC0(valid_bank31_B, valid_bank31_A && ready_bank31_A, valid_bank31_B && ready_bank31_B)
`assert_clk(valid_bank31_B |-> (so_or_sz ? |OK_onehot_B[31*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))] : |(~OK_onehot_B[31*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))])))      
assign valid_bank_d_B[31] = valid_bank31_B;               //all bank compose to one array

assign ready_bank31_B = ready_B && valid_bank_onehot_B[31];      //which bank has been choosen, which bank can move to next pipeline

    so #(
        .DATA((1<<(9-4)))
    ) so_bank(
        .in(valid_bank_d_B),
        .out(valid_bank_onehot_B)
    );

    assign index_lowbit_B =
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[0]}} & c16to(OK_onehot_B[0*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[1]}} & c16to(OK_onehot_B[1*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[2]}} & c16to(OK_onehot_B[2*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[3]}} & c16to(OK_onehot_B[3*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[4]}} & c16to(OK_onehot_B[4*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[5]}} & c16to(OK_onehot_B[5*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[6]}} & c16to(OK_onehot_B[6*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[7]}} & c16to(OK_onehot_B[7*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[8]}} & c16to(OK_onehot_B[8*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[9]}} & c16to(OK_onehot_B[9*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[10]}} & c16to(OK_onehot_B[10*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[11]}} & c16to(OK_onehot_B[11*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[12]}} & c16to(OK_onehot_B[12*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[13]}} & c16to(OK_onehot_B[13*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[14]}} & c16to(OK_onehot_B[14*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[15]}} & c16to(OK_onehot_B[15*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[16]}} & c16to(OK_onehot_B[16*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[17]}} & c16to(OK_onehot_B[17*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[18]}} & c16to(OK_onehot_B[18*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[19]}} & c16to(OK_onehot_B[19*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[20]}} & c16to(OK_onehot_B[20*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[21]}} & c16to(OK_onehot_B[21*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[22]}} & c16to(OK_onehot_B[22*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[23]}} & c16to(OK_onehot_B[23*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[24]}} & c16to(OK_onehot_B[24*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[25]}} & c16to(OK_onehot_B[25*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[26]}} & c16to(OK_onehot_B[26*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[27]}} & c16to(OK_onehot_B[27*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[28]}} & c16to(OK_onehot_B[28*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[29]}} & c16to(OK_onehot_B[29*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[30]}} & c16to(OK_onehot_B[30*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
        {`LOG2(((1<<9)/(1<<(9-4)))){valid_bank_onehot_B[31]}} & c16to(OK_onehot_B[31*((1<<9)/(1<<(9-4)))+:((1<<9)/(1<<(9-4)))]) |
    0;

////////////////////////////////////////////////////////////////////////////////////////////////////////
//pipeline C
////////////////////////////////////////////////////////////////////////////////////////////////////////
`FlopN(index_lowbit_B1, valid_B && ready_B, index_lowbit_B)
assign index_highbit_B1 = c32to(valid_bank_onehot_B1);
assign index_B1 = {index_highbit_B1,index_lowbit_B1};

`Flop(valid_bank_onehot_B1, valid_B && ready_B, valid_bank_onehot_B)
`FlopSC0(valid_B1, valid_B && ready_B, valid_B1 && ready_B1)

endmodule
