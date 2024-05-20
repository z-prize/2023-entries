`include "basic.vh"
`include "env.vh"
//for zprize msm

//when 16=15, CORE_NUM=4, so SBITMULTISIZE = 64 for DDR alignment
///define SBITMULTISIZE       (16*4)

module zprize_F2_F3_combine import zprize_param::*;(
//control
    input                                   mode,       //0 for F3, 1 for F2 
//input 
    input                                   f2_in_valid,
    input [384*4-1:0]       f2_in_data1,   
    input [384*4-1:0]       f2_in_data2,   

    input                                         f3_in_valid,
    input [384*4-1:0]       f3_in_data1,   
    input [384*3-1:0]       f3_in_data2,   

    output                                        out_valid,      //f3 and f2 share final output
    `MAX_FANOUT32 output logic                                   f2_out_valid,      
    `MAX_FANOUT32 output logic                                   f3_out_valid,      
    output[384*4-1:0]       out_data,
//Global
    output                                  idle,
    input                                   clk,
    input                                   rstN
);

//F3 sigal has suffix of pipeline name, like _A
//F2 signal dont has suffix
//
////////////////////////////////////////////////////////////////////////////////////////////////////////
//Declaration
////////////////////////////////////////////////////////////////////////////////////////////////////////

    logic   [`LOG2(39*2+((7+1+1)-1+(7+1+1)+1+1)*2+4+39+1+1)-1:0]    validNum;
    logic                                   idle_golden;
    logic                                   f3_valid_0;
    logic                                   f3_valid_1;
    logic                                   f3_valid_2;
    logic                                   f3_valid_3;
    logic                                   f3_valid_4;
    logic                                   f3_valid_5;
    logic                                   f3_valid_6;
    logic                                   f3_valid_7;
    logic                                   f3_valid_8;
    logic                                   f3_valid_9;
    logic                                   f3_valid_10;
    logic                                   f3_valid_11;
    logic                                   f3_valid_12;
    logic                                   f3_valid_13;
    logic                                   f3_valid_14;
    logic                                   f3_valid_15;
    logic                                   f3_valid_16;
    logic                                   f3_valid_17;
    logic                                   f3_valid_18;
    logic                                   f3_valid_19;
    logic                                   f3_valid_20;
    logic                                   f3_valid_21;
    logic                                   f3_valid_22;
    logic                                   f3_valid_23;
    logic                                   f3_valid_24;
    logic                                   f3_valid_25;
    logic                                   f3_valid_26;
    logic                                   f3_valid_27;
    logic                                   f3_valid_28;
    logic                                   f3_valid_29;
    logic                                   f3_valid_30;
    logic                                   f3_valid_31;
    logic                                   f3_valid_32;
    logic                                   f3_valid_33;
    logic                                   f3_valid_34;
    logic                                   f3_valid_35;
    logic                                   f3_valid_36;
    logic                                   f3_valid_37;
    logic                                   f3_valid_38;
    logic                                   f3_valid_39;
    logic                                   f3_valid_40;
    logic                                   f3_valid_41;
    logic                                   f3_valid_42;
    logic                                   f3_valid_43;
    logic                                   f3_valid_44;
    logic                                   f3_valid_45;
    logic                                   f3_valid_46;
    logic                                   f3_valid_47;
    logic                                   f3_valid_48;
    logic                                   f3_valid_49;
    logic                                   f3_valid_50;
    logic                                   f3_valid_51;
    logic                                   f3_valid_52;
    logic                                   f3_valid_53;
    logic                                   f3_valid_54;
    logic                                   f3_valid_55;
    logic                                   f3_valid_56;
    logic                                   f3_valid_57;
    logic                                   f3_valid_58;
    logic                                   f3_valid_59;
    logic                                   f3_valid_60;
    logic                                   f3_valid_61;
    logic                                   f3_valid_62;
    logic                                   f3_valid_63;
    logic                                   f3_valid_64;
    logic                                   f3_valid_65;
    logic                                   f3_valid_66;
    logic                                   f3_valid_67;
    logic                                   f3_valid_68;
    logic                                   f3_valid_69;
    logic                                   f3_valid_70;
    logic                                   f3_valid_71;
    logic                                   f3_valid_72;
    logic                                   f3_valid_73;
    logic                                   f3_valid_74;
    logic                                   f3_valid_75;
    logic                                   f3_valid_76;
    logic                                   f3_valid_77;
    logic                                   f3_valid_78;
    logic                                   f3_valid_79;
    logic                                   f3_valid_80;
    logic                                   f3_valid_81;
    logic                                   f3_valid_82;
    logic                                   f3_valid_83;
    logic                                   f3_valid_84;
    logic                                   f3_valid_85;
    logic                                   f3_valid_86;
    logic                                   f3_valid_87;
    logic                                   f3_valid_88;
    logic                                   f3_valid_89;
    logic                                   f3_valid_90;
    logic                                   f3_valid_91;
    logic                                   f3_valid_92;
    logic                                   f3_valid_93;
    logic                                   f3_valid_94;
    logic                                   f3_valid_95;
    logic                                   f3_valid_96;
    logic                                   f3_valid_97;
    logic                                   f3_valid_98;
    logic                                   f3_valid_99;
    logic                                   f3_valid_100;
    logic                                   f3_valid_101;
    logic                                   f3_valid_102;
    logic                                   f3_valid_103;
    logic                                   f3_valid_104;
    logic                                   f3_valid_105;
    logic                                   f3_valid_106;
    logic                                   f3_valid_107;
    logic                                   f3_valid_108;
    logic                                   f3_valid_109;
    logic                                   f3_valid_110;
    logic                                   f3_valid_111;
    logic                                   f3_valid_112;
    logic                                   f3_valid_113;
    logic                                   f3_valid_114;
    logic                                   f3_valid_115;
    logic                                   f3_valid_116;
    logic                                   f3_valid_117;
    logic                                   f3_valid_118;
    logic                                   f3_valid_119;
    logic                                   f3_valid_120;

//f2 pipeline means extra mul_mod
    logic                                   f2_valid_0;
    logic                                   f2_valid_1;
    logic                                   f2_valid_2;
    logic                                   f2_valid_3;
    logic                                   f2_valid_4;
    logic                                   f2_valid_5;
    logic                                   f2_valid_6;
    logic                                   f2_valid_7;
    logic                                   f2_valid_8;
    logic                                   f2_valid_9;
    logic                                   f2_valid_10;
    logic                                   f2_valid_11;
    logic                                   f2_valid_12;
    logic                                   f2_valid_13;
    logic                                   f2_valid_14;
    logic                                   f2_valid_15;
    logic                                   f2_valid_16;
    logic                                   f2_valid_17;
    logic                                   f2_valid_18;
    logic                                   f2_valid_19;
    logic                                   f2_valid_20;
    logic                                   f2_valid_21;
    logic                                   f2_valid_22;
    logic                                   f2_valid_23;
    logic                                   f2_valid_24;
    logic                                   f2_valid_25;
    logic                                   f2_valid_26;
    logic                                   f2_valid_27;
    logic                                   f2_valid_28;
    logic                                   f2_valid_29;
    logic                                   f2_valid_30;
    logic                                   f2_valid_31;
    logic                                   f2_valid_32;
    logic                                   f2_valid_33;
    logic                                   f2_valid_34;
    logic                                   f2_valid_35;
    logic                                   f2_valid_36;
    logic                                   f2_valid_37;
    logic                                   f2_valid_38;
    logic                                   f2_valid_39;
    logic                                   f2_valid_40;

    logic                                   valid_A;
    logic                                   valid_E;
    logic                                   valid_J;
    logic                                   valid_M;
    logic                                   valid_B;
    logic                                   valid_C;

    logic   [377-1:0]       X1_A;
    logic   [377-1:0]       X2_A;
    logic   [377-1:0]       Y1_A;
    logic   [377-1:0]       Y2_A;
    logic   [377-1:0]       T1_A;
    logic   [377-1:0]       T2_A;
    logic   [377-1:0]       Z1_A;

    logic   [377-1:0]       X1;
    logic   [377-1:0]       X2;
    logic   [377-1:0]       Y1;
    logic   [377-1:0]       Y2;
    logic   [377-1:0]       T1;
    logic   [377-1:0]       T2;
    logic   [377-1:0]       Z1;
    logic   [377-1:0]       Z2;

    logic   [377-1:0]       Y1sX1_E;
    logic   [377-1:0]       Y2sX2_E;
    logic   [377-1:0]       X2aY2_E_last;
    logic   [377-1:0]       X1aY1_E_last;
    logic   [377-1:0]       X2aY2_E;
    logic   [377-1:0]       X1aY1_E;

    logic   [377-1:0]       a_J;
    logic   [377-1:0]       b_J;

    logic   [377-1:0]       f_M;
    logic   [377-1:0]       g_M_last;
    logic   [377-1:0]       g_M;

    logic   [377-1:0]       c_B;
    logic   [377-1:0]       d_C;
    logic   [377-1:0]       c_J;
    logic   [377-1:0]       d_J;

    logic   [377-1:0]       e_M;
    logic   [377-1:0]       h_M_last;
    logic   [377-1:0]       h_M;
    
    logic   [377-1:0]       e_N_local;
    logic   [377-1:0]       f_N_local;
    logic   [377-1:0]       g_N_local;
    logic   [377-1:0]       h_N_local;
    logic   [377-1:0]       e_N_remote;
    logic   [377-1:0]       f_N_remote;
    logic   [377-1:0]       g_N_remote;
    logic   [377-1:0]       h_N_remote;
    logic   [377-1:0]       e_N;
    logic   [377-1:0]       f_N;
    logic   [377-1:0]       g_N;
    logic   [377-1:0]       h_N;

    logic   [377-1:0]       X3_U;
    logic   [377-1:0]       Y3_U;
    logic   [377-1:0]       Z3_U;
    logic   [377-1:0]       T3_U;

    logic   [377-1:0]       X3_W_remote;
    logic   [377-1:0]       Y3_W_remote;
    logic   [377-1:0]       X3_W;
    logic   [377-1:0]       Y3_W;
    logic   [377-1:0]       Z3_W;
    logic   [377-1:0]       T3_W;

    logic   [377-1:0]       Z1_Y1sX1_E; //Z1 combine with Y1sX1_E
    logic   [377-1:0]       Z2_Y2sX2_E;

    logic   [377-1:0]       k_X1aY1_E;
    logic   [377-1:0]       T1_X2aY2_E;

    logic   [377-1:0]       Z1Z2;
    logic   [377-1:0]       kT1;

    logic   [377-1:0]       Z1Z2_r;
    logic   [377-1:0]       kT1_r;

    logic   [377-1:0]       kT1_T1_A;
    logic   [377-1:0]       Z1Z2_Z1_A;

    logic   [377-1:0]       X1_pipe;
    logic   [377-1:0]       X2_pipe;
    logic   [377-1:0]       Y1_pipe;
    logic   [377-1:0]       Y2_pipe;
    logic   [377-1:0]       T2_pipe;

    logic   [377-1:0]       X1_P_X1_A;
    logic   [377-1:0]       X2_P_X2_A;
    logic   [377-1:0]       Y1_P_Y1_A;
    logic   [377-1:0]       Y2_P_Y2_A;
    logic   [377-1:0]       T2_P_T2_A;
////////////////////////////////////////////////////////////////////////////////////////////////////////
//Implementation
////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////
//validNum
////////////////////////////////////////////////////////////////////////////////////////////////////////
`Flop(validNum, (f2_in_valid || f3_in_valid) != out_valid, out_valid ? validNum - 1 : validNum + 1)
assign idle = validNum == 0;
////////////////////////////////////////////////////////////////////////////////////////////////////////
//valid
////////////////////////////////////////////////////////////////////////////////////////////////////////
assign f3_valid_0 = 
                f3_in_valid ||                          //F3 new input 
                f2_valid_40;   //F2's Z1Z2 kT1

`Flop(f3_valid_1, 1, f3_valid_0)
`Flop(f3_valid_2, 1, f3_valid_1)
`Flop(f3_valid_3, 1, f3_valid_2)
`Flop(f3_valid_4, 1, f3_valid_3)
`Flop(f3_valid_5, 1, f3_valid_4)
`Flop(f3_valid_6, 1, f3_valid_5)
`Flop(f3_valid_7, 1, f3_valid_6)
`Flop(f3_valid_8, 1, f3_valid_7)
`Flop(f3_valid_9, 1, f3_valid_8)
`Flop(f3_valid_10, 1, f3_valid_9)
`Flop(f3_valid_11, 1, f3_valid_10)
`Flop(f3_valid_12, 1, f3_valid_11)
`Flop(f3_valid_13, 1, f3_valid_12)
`Flop(f3_valid_14, 1, f3_valid_13)
`Flop(f3_valid_15, 1, f3_valid_14)
`Flop(f3_valid_16, 1, f3_valid_15)
`Flop(f3_valid_17, 1, f3_valid_16)
`Flop(f3_valid_18, 1, f3_valid_17)
`Flop(f3_valid_19, 1, f3_valid_18)
`Flop(f3_valid_20, 1, f3_valid_19)
`Flop(f3_valid_21, 1, f3_valid_20)
`Flop(f3_valid_22, 1, f3_valid_21)
`Flop(f3_valid_23, 1, f3_valid_22)
`Flop(f3_valid_24, 1, f3_valid_23)
`Flop(f3_valid_25, 1, f3_valid_24)
`Flop(f3_valid_26, 1, f3_valid_25)
`Flop(f3_valid_27, 1, f3_valid_26)
`Flop(f3_valid_28, 1, f3_valid_27)
`Flop(f3_valid_29, 1, f3_valid_28)
`Flop(f3_valid_30, 1, f3_valid_29)
`Flop(f3_valid_31, 1, f3_valid_30)
`Flop(f3_valid_32, 1, f3_valid_31)
`Flop(f3_valid_33, 1, f3_valid_32)
`Flop(f3_valid_34, 1, f3_valid_33)
`Flop(f3_valid_35, 1, f3_valid_34)
`Flop(f3_valid_36, 1, f3_valid_35)
`Flop(f3_valid_37, 1, f3_valid_36)
`Flop(f3_valid_38, 1, f3_valid_37)
`Flop(f3_valid_39, 1, f3_valid_38)
`Flop(f3_valid_40, 1, f3_valid_39)
`Flop(f3_valid_41, 1, f3_valid_40)
`Flop(f3_valid_42, 1, f3_valid_41)
`Flop(f3_valid_43, 1, f3_valid_42)
`Flop(f3_valid_44, 1, f3_valid_43)
`Flop(f3_valid_45, 1, f3_valid_44)
`Flop(f3_valid_46, 1, f3_valid_45)
`Flop(f3_valid_47, 1, f3_valid_46)
`Flop(f3_valid_48, 1, f3_valid_47)
`Flop(f3_valid_49, 1, f3_valid_48)
`Flop(f3_valid_50, 1, f3_valid_49)
`Flop(f3_valid_51, 1, f3_valid_50)
`Flop(f3_valid_52, 1, f3_valid_51)
`Flop(f3_valid_53, 1, f3_valid_52)
`Flop(f3_valid_54, 1, f3_valid_53)
`Flop(f3_valid_55, 1, f3_valid_54)
`Flop(f3_valid_56, 1, f3_valid_55)
`Flop(f3_valid_57, 1, f3_valid_56)
`Flop(f3_valid_58, 1, f3_valid_57)
`Flop(f3_valid_59, 1, f3_valid_58)
`Flop(f3_valid_60, 1, f3_valid_59)
`Flop(f3_valid_61, 1, f3_valid_60)
`Flop(f3_valid_62, 1, f3_valid_61)
`Flop(f3_valid_63, 1, f3_valid_62)
`Flop(f3_valid_64, 1, f3_valid_63)
`Flop(f3_valid_65, 1, f3_valid_64)
`Flop(f3_valid_66, 1, f3_valid_65)
`Flop(f3_valid_67, 1, f3_valid_66)
`Flop(f3_valid_68, 1, f3_valid_67)
`Flop(f3_valid_69, 1, f3_valid_68)
`Flop(f3_valid_70, 1, f3_valid_69)
`Flop(f3_valid_71, 1, f3_valid_70)
`Flop(f3_valid_72, 1, f3_valid_71)
`Flop(f3_valid_73, 1, f3_valid_72)
`Flop(f3_valid_74, 1, f3_valid_73)
`Flop(f3_valid_75, 1, f3_valid_74)
`Flop(f3_valid_76, 1, f3_valid_75)
`Flop(f3_valid_77, 1, f3_valid_76)
`Flop(f3_valid_78, 1, f3_valid_77)
`Flop(f3_valid_79, 1, f3_valid_78)
`Flop(f3_valid_80, 1, f3_valid_79)
`Flop(f3_valid_81, 1, f3_valid_80)
`Flop(f3_valid_82, 1, f3_valid_81)
`Flop(f3_valid_83, 1, f3_valid_82)
`Flop(f3_valid_84, 1, f3_valid_83)
`Flop(f3_valid_85, 1, f3_valid_84)
`Flop(f3_valid_86, 1, f3_valid_85)
`Flop(f3_valid_87, 1, f3_valid_86)
`Flop(f3_valid_88, 1, f3_valid_87)
`Flop(f3_valid_89, 1, f3_valid_88)
`Flop(f3_valid_90, 1, f3_valid_89)
`Flop(f3_valid_91, 1, f3_valid_90)
`Flop(f3_valid_92, 1, f3_valid_91)
`Flop(f3_valid_93, 1, f3_valid_92)
`Flop(f3_valid_94, 1, f3_valid_93)
`Flop(f3_valid_95, 1, f3_valid_94)
`Flop(f3_valid_96, 1, f3_valid_95)
`Flop(f3_valid_97, 1, f3_valid_96)
`Flop(f3_valid_98, 1, f3_valid_97)
`Flop(f3_valid_99, 1, f3_valid_98)
`Flop(f3_valid_100, 1, f3_valid_99)
`Flop(f3_valid_101, 1, f3_valid_100)
`Flop(f3_valid_102, 1, f3_valid_101)
`Flop(f3_valid_103, 1, f3_valid_102)
`Flop(f3_valid_104, 1, f3_valid_103)
`Flop(f3_valid_105, 1, f3_valid_104)
`Flop(f3_valid_106, 1, f3_valid_105)
`Flop(f3_valid_107, 1, f3_valid_106)
`Flop(f3_valid_108, 1, f3_valid_107)
`Flop(f3_valid_109, 1, f3_valid_108)
`Flop(f3_valid_110, 1, f3_valid_109)
`Flop(f3_valid_111, 1, f3_valid_110)
`Flop(f3_valid_112, 1, f3_valid_111)
`Flop(f3_valid_113, 1, f3_valid_112)
`Flop(f3_valid_114, 1, f3_valid_113)
`Flop(f3_valid_115, 1, f3_valid_114)
`Flop(f3_valid_116, 1, f3_valid_115)
`Flop(f3_valid_117, 1, f3_valid_116)
`Flop(f3_valid_118, 1, f3_valid_117)
`Flop(f3_valid_119, 1, f3_valid_118)
`Flop(f3_valid_120, 1, f3_valid_119)

assign out_valid = f3_valid_120;
`Flop(f2_out_valid, 1, mode == 1 && f3_valid_119)
`Flop(f3_out_valid, 1, mode == 0 && f3_valid_119)

//after extra mul_mod+1 cycle, f2_valid combine into f3_valid_0
assign f2_valid_0 = mode ? f2_in_valid : 0;
`Flop(f2_valid_1, 1, f2_valid_0)
`Flop(f2_valid_2, 1, f2_valid_1)
`Flop(f2_valid_3, 1, f2_valid_2)
`Flop(f2_valid_4, 1, f2_valid_3)
`Flop(f2_valid_5, 1, f2_valid_4)
`Flop(f2_valid_6, 1, f2_valid_5)
`Flop(f2_valid_7, 1, f2_valid_6)
`Flop(f2_valid_8, 1, f2_valid_7)
`Flop(f2_valid_9, 1, f2_valid_8)
`Flop(f2_valid_10, 1, f2_valid_9)
`Flop(f2_valid_11, 1, f2_valid_10)
`Flop(f2_valid_12, 1, f2_valid_11)
`Flop(f2_valid_13, 1, f2_valid_12)
`Flop(f2_valid_14, 1, f2_valid_13)
`Flop(f2_valid_15, 1, f2_valid_14)
`Flop(f2_valid_16, 1, f2_valid_15)
`Flop(f2_valid_17, 1, f2_valid_16)
`Flop(f2_valid_18, 1, f2_valid_17)
`Flop(f2_valid_19, 1, f2_valid_18)
`Flop(f2_valid_20, 1, f2_valid_19)
`Flop(f2_valid_21, 1, f2_valid_20)
`Flop(f2_valid_22, 1, f2_valid_21)
`Flop(f2_valid_23, 1, f2_valid_22)
`Flop(f2_valid_24, 1, f2_valid_23)
`Flop(f2_valid_25, 1, f2_valid_24)
`Flop(f2_valid_26, 1, f2_valid_25)
`Flop(f2_valid_27, 1, f2_valid_26)
`Flop(f2_valid_28, 1, f2_valid_27)
`Flop(f2_valid_29, 1, f2_valid_28)
`Flop(f2_valid_30, 1, f2_valid_29)
`Flop(f2_valid_31, 1, f2_valid_30)
`Flop(f2_valid_32, 1, f2_valid_31)
`Flop(f2_valid_33, 1, f2_valid_32)
`Flop(f2_valid_34, 1, f2_valid_33)
`Flop(f2_valid_35, 1, f2_valid_34)
`Flop(f2_valid_36, 1, f2_valid_35)
`Flop(f2_valid_37, 1, f2_valid_36)
`Flop(f2_valid_38, 1, f2_valid_37)
`Flop(f2_valid_39, 1, f2_valid_38)
`Flop(f2_valid_40, 1, f2_valid_39)

assign valid_A = f3_valid_0;
assign valid_E =        f2_in_valid ||                   //F2 new input
                        f3_valid_19;   //F3's Y1sX1...
`assert_clk({f2_in_valid,f3_valid_19}!=2'b11)  //the two must be exclusive with eachother                
assign valid_J = f3_valid_58;
assign valid_M = f3_valid_77;
assign valid_B = f3_valid_39;
assign valid_C = f3_valid_10;

assign idle_golden = 
        !f3_valid_0 &&
        !f3_valid_1 &&
        !f3_valid_2 &&
        !f3_valid_3 &&
        !f3_valid_4 &&
        !f3_valid_5 &&
        !f3_valid_6 &&
        !f3_valid_7 &&
        !f3_valid_8 &&
        !f3_valid_9 &&
        !f3_valid_10 &&
        !f3_valid_11 &&
        !f3_valid_12 &&
        !f3_valid_13 &&
        !f3_valid_14 &&
        !f3_valid_15 &&
        !f3_valid_16 &&
        !f3_valid_17 &&
        !f3_valid_18 &&
        !f3_valid_19 &&
        !f3_valid_20 &&
        !f3_valid_21 &&
        !f3_valid_22 &&
        !f3_valid_23 &&
        !f3_valid_24 &&
        !f3_valid_25 &&
        !f3_valid_26 &&
        !f3_valid_27 &&
        !f3_valid_28 &&
        !f3_valid_29 &&
        !f3_valid_30 &&
        !f3_valid_31 &&
        !f3_valid_32 &&
        !f3_valid_33 &&
        !f3_valid_34 &&
        !f3_valid_35 &&
        !f3_valid_36 &&
        !f3_valid_37 &&
        !f3_valid_38 &&
        !f3_valid_39 &&
        !f3_valid_40 &&
        !f3_valid_41 &&
        !f3_valid_42 &&
        !f3_valid_43 &&
        !f3_valid_44 &&
        !f3_valid_45 &&
        !f3_valid_46 &&
        !f3_valid_47 &&
        !f3_valid_48 &&
        !f3_valid_49 &&
        !f3_valid_50 &&
        !f3_valid_51 &&
        !f3_valid_52 &&
        !f3_valid_53 &&
        !f3_valid_54 &&
        !f3_valid_55 &&
        !f3_valid_56 &&
        !f3_valid_57 &&
        !f3_valid_58 &&
        !f3_valid_59 &&
        !f3_valid_60 &&
        !f3_valid_61 &&
        !f3_valid_62 &&
        !f3_valid_63 &&
        !f3_valid_64 &&
        !f3_valid_65 &&
        !f3_valid_66 &&
        !f3_valid_67 &&
        !f3_valid_68 &&
        !f3_valid_69 &&
        !f3_valid_70 &&
        !f3_valid_71 &&
        !f3_valid_72 &&
        !f3_valid_73 &&
        !f3_valid_74 &&
        !f3_valid_75 &&
        !f3_valid_76 &&
        !f3_valid_77 &&
        !f3_valid_78 &&
        !f3_valid_79 &&
        !f3_valid_80 &&
        !f3_valid_81 &&
        !f3_valid_82 &&
        !f3_valid_83 &&
        !f3_valid_84 &&
        !f3_valid_85 &&
        !f3_valid_86 &&
        !f3_valid_87 &&
        !f3_valid_88 &&
        !f3_valid_89 &&
        !f3_valid_90 &&
        !f3_valid_91 &&
        !f3_valid_92 &&
        !f3_valid_93 &&
        !f3_valid_94 &&
        !f3_valid_95 &&
        !f3_valid_96 &&
        !f3_valid_97 &&
        !f3_valid_98 &&
        !f3_valid_99 &&
        !f3_valid_100 &&
        !f3_valid_101 &&
        !f3_valid_102 &&
        !f3_valid_103 &&
        !f3_valid_104 &&
        !f3_valid_105 &&
        !f3_valid_106 &&
        !f3_valid_107 &&
        !f3_valid_108 &&
        !f3_valid_109 &&
        !f3_valid_110 &&
        !f3_valid_111 &&
        !f3_valid_112 &&
        !f3_valid_113 &&
        !f3_valid_114 &&
        !f3_valid_115 &&
        !f3_valid_116 &&
        !f3_valid_117 &&
        !f3_valid_118 &&
        !f3_valid_119 &&
        !f3_valid_120 &&
        !f2_valid_0 &&
        !f2_valid_1 &&
        !f2_valid_2 &&
        !f2_valid_3 &&
        !f2_valid_4 &&
        !f2_valid_5 &&
        !f2_valid_6 &&
        !f2_valid_7 &&
        !f2_valid_8 &&
        !f2_valid_9 &&
        !f2_valid_10 &&
        !f2_valid_11 &&
        !f2_valid_12 &&
        !f2_valid_13 &&
        !f2_valid_14 &&
        !f2_valid_15 &&
        !f2_valid_16 &&
        !f2_valid_17 &&
        !f2_valid_18 &&
        !f2_valid_19 &&
        !f2_valid_20 &&
        !f2_valid_21 &&
        !f2_valid_22 &&
        !f2_valid_23 &&
        !f2_valid_24 &&
        !f2_valid_25 &&
        !f2_valid_26 &&
        !f2_valid_27 &&
        !f2_valid_28 &&
        !f2_valid_29 &&
        !f2_valid_30 &&
        !f2_valid_31 &&
        !f2_valid_32 &&
        !f2_valid_33 &&
        !f2_valid_34 &&
        !f2_valid_35 &&
        !f2_valid_36 &&
        !f2_valid_37 &&
        !f2_valid_38 &&
        !f2_valid_39 &&
        !f2_valid_40 &&
        1;

////////////////////////////////////////////////////////////////////////////////////////////////////////
//Implementation
////////////////////////////////////////////////////////////////////////////////////////////////////////
//pipeline A
assign X1_A = f3_in_data1[384*0+:377];
assign Y1_A = f3_in_data1[384*1+:377];
assign T1_A = f3_in_data1[384*2+:377];
assign Z1_A = f3_in_data1[384*3+:377];

assign X2_A = f3_in_data2[384*0+:377];
assign Y2_A = f3_in_data2[384*1+:377];
assign T2_A = f3_in_data2[384*2+:377];

assign X1 = f2_in_data1[384*0+:377];
assign Y1 = f2_in_data1[384*1+:377];
assign T1 = f2_in_data1[384*2+:377];
assign Z1 = f2_in_data1[384*3+:377];

assign X2 = f2_in_data2[384*0+:377];
assign Y2 = f2_in_data2[384*1+:377];
assign T2 = f2_in_data2[384*2+:377];
assign Z2 = f2_in_data2[384*3+:377]; //only for F2

//F2's E to F3's A, latency is 39+1, so that the F2 and F3 don't conflict at E
`PIPE(39+1,377,X1_pipe,X1)
`PIPE(39+1,377,Y1_pipe,Y1)
`PIPE(39+1,377,X2_pipe,X2)
`PIPE(39+1,377,Y2_pipe,Y2)
`PIPE(39+1,377,T2_pipe,T2)

assign X1_P_X1_A = mode ? X1_pipe : X1_A;
assign Y1_P_Y1_A = mode ? Y1_pipe : Y1_A;
assign X2_P_X2_A = mode ? X2_pipe : X2_A;
assign Y2_P_Y2_A = mode ? Y2_pipe : Y2_A;
assign T2_P_T2_A = mode ? T2_pipe : T2_A;

`ADDMOD(A,X1aY1_E_last,Y1_P_Y1_A,X1_P_X1_A)
`SUBMOD(A,Y1sX1_E     ,Y1_P_Y1_A,X1_P_X1_A)
`ADDMOD(A,X2aY2_E_last,Y2_P_Y2_A,X2_P_X2_A)
`SUBMOD(A,Y2sX2_E     ,Y2_P_Y2_A,X2_P_X2_A)

`PIPE(((7+1+1)-1+(7+1+1)+1+1)-((7+1+1)-1+(7+1+1)+1),377,X1aY1_E,X1aY1_E_last)
`PIPE(((7+1+1)-1+(7+1+1)+1+1)-((7+1+1)-1+(7+1+1)+1),377,X2aY2_E,X2aY2_E_last)

//assign Z1Z2_T1_A = mode ? Z1Z2 : T1_A;
//assign kT1_Z1_A  = mode ? kT1  : Z1_A;

assign kT1_T1_A   = mode ? kT1   : T1_A;
assign Z1Z2_Z1_A  = mode ? Z1Z2  : Z1_A;

`MULMOD(A,c_B,kT1_T1_A,T2_P_T2_A)
`DUOMOD(A,d_C,Z1Z2_Z1_A)

//pipeline E

assign Z1_Y1sX1_E = mode&f2_valid_0 ? Z1       : Y1sX1_E;
assign Z2_Y2sX2_E = mode&f2_valid_0 ? Z2       : Y2sX2_E;
assign k_X1aY1_E  = mode&f2_valid_0 ? k_zprize : X1aY1_E;
assign T1_X2aY2_E = mode&f2_valid_0 ? T1       : X2aY2_E;

`MULMOD(E,a_J,Z1_Y1sX1_E,Z2_Y2sX2_E)
`MULMOD(E,b_J, k_X1aY1_E,T1_X2aY2_E)

//pipeline J
//add 1 pipeline after F2's E, so that the F2 and F3 don't conflict at E
`FlopN(Z1Z2, 1, a_J)
`FlopN(kT1 , 1, b_J)

`SUBMOD(J,e_M,b_J ,a_J)
`ADDMOD(J,h_M_last,a_J,b_J)
`PIPE(((7+1+1)-1+(7+1+1)+1+1)-((7+1+1)-1+(7+1+1)+1),377,h_M,h_M_last)

`PIPE(((7+1+1)-1+(7+1+1)+1+1),377,c_J,c_B)
`PIPE(((7+1+1)-1+(7+1+1)+1+1)+39-((7+1+1)+1),377,d_J,d_C)

`ADDMOD(J,g_M_last,d_J,c_J)
`SUBMOD(J,f_M,d_J ,c_J)
`PIPE(((7+1+1)-1+(7+1+1)+1+1)-((7+1+1)-1+(7+1+1)+1),377,g_M,g_M_last)

// e_M --> e_N_local --||--> e_N_remote
// f_M --> f_N_local --||--> f_N_remote
`PIPE(1,377,e_N_local ,e_M)
`PIPE(1,377,f_N_local ,f_M)
`PIPE(1,377,g_N_local ,g_M)
`PIPE(1,377,h_N_local ,h_M)
`PIPE(1,377,e_N_remote,e_N_local) // other SLR
`PIPE(1,377,f_N_remote,f_N_local) // other SLR
`PIPE(1,377,g_N_remote,g_N_local) // pick  SLR
`PIPE(1,377,h_N_remote,h_N_local) // pick  SLR

`PIPE(2,377,e_N,e_M)
`PIPE(2,377,f_N,f_M)
`PIPE(2,377,g_N,g_M)
`PIPE(2,377,h_N,h_M)

//pipeline J
`MULMOD(M,X3_U,f_N_remote,e_N_remote) // other SLR
`MULMOD(M,Z3_U,f_N,g_N)
`MULMOD(M,Y3_U,g_N_remote,h_N_remote) // pick  SLR
`MULMOD(M,T3_U,e_N,h_N)

//`PIPE(2,377,X3_W,X3_U)
//`PIPE(2,377,Y3_W,Y3_U)
// X3_W <--||-- X3_W_remote <-- X3_U
`PIPE(1,377,X3_W_remote,X3_U) // other SLR
`PIPE(1,377,X3_W,X3_W_remote)
`PIPE(2,377,Z3_W,Z3_U)
`PIPE(1,377,Y3_W_remote,Y3_U) // pick  SLR
`PIPE(1,377,Y3_W,Y3_W_remote)
`PIPE(2,377,T3_W,T3_U)

assign out_data[384*0+:384] = {{384-377{1'b0}},X3_W};
assign out_data[384*1+:384] = {{384-377{1'b0}},Y3_W};
assign out_data[384*2+:384] = {{384-377{1'b0}},T3_W};
assign out_data[384*3+:384] = {{384-377{1'b0}},Z3_W};

endmodule
