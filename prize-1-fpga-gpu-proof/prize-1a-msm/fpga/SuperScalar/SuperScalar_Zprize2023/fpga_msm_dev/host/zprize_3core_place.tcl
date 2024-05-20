 		############################
 		## zprize_msm_inst --> /pfm_top_wrapper/pfm_top_i/pfm_dynamic_inst/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm
 		## zprize_msm_inst --> level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm
 		
 		# ------------------------------------------------------------------------------
 		
 		create_pblock pblock_0
 		add_cells_to_pblock [get_pblocks pblock_0] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/pointRamFifo0 ]]
 		
 		add_cells_to_pblock [get_pblocks pblock_0] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/pointShiftFifo0 ]]
 		add_cells_to_pblock [get_pblocks pblock_0] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_pointPipe0]]
 		add_cells_to_pblock [get_pblocks pblock_0] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_pointPipe0_0 ]]
 		add_cells_to_pblock [get_pblocks pblock_0] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_pointPipe0_1]]
 		add_cells_to_pblock [get_pblocks pblock_0] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_scalarPipe0]]
 		add_cells_to_pblock [get_pblocks pblock_0] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core0.zprize0]]
 		add_cells_to_pblock [get_pblocks pblock_0] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/inst_axi_read_master0]]
 		add_cells_to_pblock [get_pblocks pblock_0] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/inst_axi_write_master0]]
 		add_cells_to_pblock [get_pblocks pblock_0] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/inst_rd_xpm_fifo_sync0[0]]]
 		add_cells_to_pblock [get_pblocks pblock_0] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/inst_wr_xpm_fifo_sync0]]
 		remove_cells_from_pblock [get_pblocks pblock_0] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core0.zprize0/zprize_calc_bucket/f2_f3/mul_mod_X3_U]]
 		remove_cells_from_pblock [get_pblocks pblock_0] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core0.zprize0/zprize_calc_bucket/f2_f3/hpe_N_remote]]
 		remove_cells_from_pblock [get_pblocks pblock_0] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core0.zprize0/zprize_calc_bucket/f2_f3/hpf_N_remote]]
 		remove_cells_from_pblock [get_pblocks pblock_0] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core0.zprize0/zprize_calc_bucket/f2_f3/hpX3_W_remote]]
 	# 	resize_pblock [get_pblocks pblock_0] -add {CLOCKREGION_X0Y0:CLOCKREGION_X7Y3}
 	 	resize_pblock [get_pblocks pblock_0] -add {CLOCKREGION_X0Y0:CLOCKREGION_X3Y3}
 	 	resize_pblock [get_pblocks pblock_0] -add {CLOCKREGION_X4Y0:CLOCKREGION_X4Y2}
 	 	# resize_pblock [get_pblocks pblock_0] -add {CLOCKREGION_X4Y0:CLOCKREGION_X4Y1}
 	 	resize_pblock [get_pblocks pblock_0] -add {CLOCKREGION_X5Y0:CLOCKREGION_X7Y3}
 		
 		create_pblock pblock_1
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/pointRamFifo1 ]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/pointShiftFifo1 ]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/scalarRamFifo ]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/scalarShiftFifo ]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_pointPipe0_2 ]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_pointPipe0_3 ]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_pointPipe1 ]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_pointPipe1_0 ]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_pointPipe1_1 ]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_pointPipe2_2 ]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_pointPipe2_3 ]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_pointPipe3_4 ]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_pointPipe3_5 ]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/pointRamBufferFifo* ]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/zprize_msm_point_mux ]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_pointPipe1a ]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_pointPipe1b]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_scalarPipe1 ]]
 		#add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core1.zprize1]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/inst_axi_read_master1 ]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/inst_axi_read_master2 ]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/inst_rd_xpm_fifo_sync1[0] ]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/inst_rd_xpm_fifo_sync2[0] ]]
 	  # fixed s_axi place	
    #########################################################
    add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/inst_krnl_vadd_control_s_axi ]]
    #########################################################
 		resize_pblock [get_pblocks pblock_1] -add {CLOCKREGION_X0Y4:CLOCKREGION_X3Y7}
 		
 		create_pblock pblock_2
 		add_cells_to_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/pointRamFifo2 ]]
 		add_cells_to_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/pointShiftFifo2 ]]
 		add_cells_to_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_pointPipe2_0 ]]
 		add_cells_to_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_pointPipe2_1 ]]
 		add_cells_to_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_pointPipe3_2 ]]
 		add_cells_to_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_pointPipe3_3 ]]
 		add_cells_to_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_pointPipe2 ]]
 		add_cells_to_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_pointPipe2a]]
 		add_cells_to_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_scalarPipe2 ]]
 		add_cells_to_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core2.zprize2]]
 		add_cells_to_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/inst_axi_read_master3]]
 		add_cells_to_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/inst_axi_write_master3]]
 		add_cells_to_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/inst_rd_xpm_fifo_sync3[0]]]
 		add_cells_to_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/inst_wr_xpm_fifo_sync3 ]]
 		remove_cells_from_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core2.zprize2/zprize_calc_bucket/f2_f3/mul_mod_X3_U]]
 		remove_cells_from_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core2.zprize2/zprize_calc_bucket/f2_f3/hpe_N_remote]]
 		remove_cells_from_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core2.zprize2/zprize_calc_bucket/f2_f3/hpf_N_remote]]
 		remove_cells_from_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core2.zprize2/zprize_calc_bucket/f2_f3/hpX3_W_remote]]
 		remove_cells_from_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core2.zprize2/zprize_calc_bucket/f2_f3/mul_mod_Y3_U]]
 		remove_cells_from_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core2.zprize2/zprize_calc_bucket/f2_f3/hpg_N_remote]]
 		remove_cells_from_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core2.zprize2/zprize_calc_bucket/f2_f3/hph_N_remote]]
 		remove_cells_from_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core2.zprize2/zprize_calc_bucket/f2_f3/hpY3_W_remote]]
 		resize_pblock [get_pblocks pblock_2] -add {CLOCKREGION_X0Y8:CLOCKREGION_X7Y11}
 		resize_pblock [get_pblocks pblock_2] -remove {CLOCKREGION_X4Y9:CLOCKREGION_X4Y9}
 		resize_pblock [get_pblocks pblock_2] -remove {CLOCKREGION_X4Y10:CLOCKREGION_X4Y10}
 		resize_pblock [get_pblocks pblock_2] -remove {CLOCKREGION_X4Y11:CLOCKREGION_X4Y11}
 		
 		create_pblock pblock_3
 		add_cells_to_pblock [get_pblocks pblock_3] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/pointRamFifo3]]
 		add_cells_to_pblock [get_pblocks pblock_3] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/pointShiftFifo3 ]]
 		add_cells_to_pblock [get_pblocks pblock_3] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_pointPipe3_0 ]]
 		add_cells_to_pblock [get_pblocks pblock_3] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_pointPipe3_1 ]]
 		add_cells_to_pblock [get_pblocks pblock_3] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core3.zprize3 ]]
 		add_cells_to_pblock [get_pblocks pblock_3] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_scalarPipe3]]
 		add_cells_to_pblock [get_pblocks pblock_3] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/u_pointPipe3]]
 		add_cells_to_pblock [get_pblocks pblock_3] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/inst_axi_read_master4]]
 		add_cells_to_pblock [get_pblocks pblock_3] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/inst_axi_write_master4]]
 		add_cells_to_pblock [get_pblocks pblock_3] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/inst_rd_xpm_fifo_sync4[0]]]
 		add_cells_to_pblock [get_pblocks pblock_3] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/inst_wr_xpm_fifo_sync4 ]]
 		remove_cells_from_pblock [get_pblocks pblock_3] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core3.zprize3/zprize_calc_bucket/f2_f3/mul_mod_X3_U]]
 		remove_cells_from_pblock [get_pblocks pblock_3] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core3.zprize3/zprize_calc_bucket/f2_f3/hpe_N_remote]]
 		remove_cells_from_pblock [get_pblocks pblock_3] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core3.zprize3/zprize_calc_bucket/f2_f3/hpf_N_remote]]
 		remove_cells_from_pblock [get_pblocks pblock_3] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core3.zprize3/zprize_calc_bucket/f2_f3/hpX3_W_remote]]
 		resize_pblock [get_pblocks pblock_3] -add {CLOCKREGION_X0Y12:CLOCKREGION_X7Y15}
 		
 		
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core0.zprize0/zprize_calc_bucket/f2_f3/mul_mod_X3_U]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core0.zprize0/zprize_calc_bucket/f2_f3/hpe_N_remote]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core0.zprize0/zprize_calc_bucket/f2_f3/hpf_N_remote]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core0.zprize0/zprize_calc_bucket/f2_f3/hpX3_W_remote]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core2.zprize2/zprize_calc_bucket/f2_f3/mul_mod_X3_U]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core2.zprize2/zprize_calc_bucket/f2_f3/hpe_N_remote]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core2.zprize2/zprize_calc_bucket/f2_f3/hpf_N_remote]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core2.zprize2/zprize_calc_bucket/f2_f3/hpX3_W_remote]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core2.zprize2/zprize_calc_bucket/f2_f3/mul_mod_Y3_U]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core2.zprize2/zprize_calc_bucket/f2_f3/hpg_N_remote]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core2.zprize2/zprize_calc_bucket/f2_f3/hph_N_remote]]
 		add_cells_to_pblock [get_pblocks pblock_1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core2.zprize2/zprize_calc_bucket/f2_f3/hpY3_W_remote]]
 		add_cells_to_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core3.zprize3/zprize_calc_bucket/f2_f3/mul_mod_X3_U]]
 		add_cells_to_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core3.zprize3/zprize_calc_bucket/f2_f3/hpe_N_remote]]
 		add_cells_to_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core3.zprize3/zprize_calc_bucket/f2_f3/hpf_N_remote]]
 		add_cells_to_pblock [get_pblocks pblock_2] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/msm/core3.zprize3/zprize_calc_bucket/f2_f3/hpX3_W_remote]]
 		
 		# ------------------------------------------------------------------------------
 		# ------------------------------------------------------------------------------
    #
    
  		create_pblock pblock_mem00_interconnect
  		create_pblock pblock_mem01_interconnect
  		create_pblock pblock_mem02_interconnect
  		resize_pblock [get_pblocks pblock_mem00_interconnect] -add {CLOCKREGION_X3Y1:CLOCKREGION_X4Y1}
  		resize_pblock [get_pblocks pblock_mem00_interconnect] -add {CLOCKREGION_X2Y2:CLOCKREGION_X4Y2}
  		resize_pblock [get_pblocks pblock_mem00_interconnect] -add {CLOCKREGION_X1Y3:CLOCKREGION_X3Y3}
  		resize_pblock [get_pblocks pblock_mem00_interconnect] -add {CLOCKREGION_X4Y3:CLOCKREGION_X4Y3}
     
  		resize_pblock [get_pblocks pblock_mem01_interconnect] -add {CLOCKREGION_X4Y9:CLOCKREGION_X4Y9}
  		resize_pblock [get_pblocks pblock_mem01_interconnect] -add {CLOCKREGION_X4Y10:CLOCKREGION_X4Y10}
  		resize_pblock [get_pblocks pblock_mem01_interconnect] -add {CLOCKREGION_X4Y11:CLOCKREGION_X4Y11}
  		
     	resize_pblock [get_pblocks pblock_mem02_interconnect] -add {CLOCKREGION_X3Y12:CLOCKREGION_X5Y12}
  		resize_pblock [get_pblocks pblock_mem02_interconnect] -add {CLOCKREGION_X3Y13:CLOCKREGION_X5Y13}
  		resize_pblock [get_pblocks pblock_mem02_interconnect] -add {CLOCKREGION_X3Y14:CLOCKREGION_X5Y14}
    
     	remove_cells_from_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -quiet [list  level0_i/level1/level1_i/ulp/memory_subsystem/inst/interconnect/interconnect_ddr4_mem00]]
  	 	remove_cells_from_pblock [get_pblocks pblock_dynamic_SLR2] [get_cells -quiet [list  level0_i/level1/level1_i/ulp/memory_subsystem/inst/interconnect/interconnect_ddr4_mem01]]
     	remove_cells_from_pblock [get_pblocks pblock_dynamic_SLR3] [get_cells -quiet [list  level0_i/level1/level1_i/ulp/memory_subsystem/inst/interconnect/interconnect_ddr4_mem02]]
     	add_cells_to_pblock [get_pblocks pblock_mem00_interconnect] [get_cells -quiet [list level0_i/level1/level1_i/ulp/memory_subsystem/inst/interconnect/interconnect_ddr4_mem00]]
     	add_cells_to_pblock [get_pblocks pblock_mem01_interconnect] [get_cells -quiet [list level0_i/level1/level1_i/ulp/memory_subsystem/inst/interconnect/interconnect_ddr4_mem01]]
     	add_cells_to_pblock [get_pblocks pblock_mem02_interconnect] [get_cells -quiet [list level0_i/level1/level1_i/ulp/memory_subsystem/inst/interconnect/interconnect_ddr4_mem02]]

    remove_cells_from_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -quiet [list  level0_i/level1/level1_i/ulp/ip_psr_aresetn_kernel_00_slr0]]
    add_cells_to_pblock [get_pblocks pblock_dynamic_SLR1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/ip_psr_aresetn_kernel_00_slr0]]
    # remove_cells_from_pblock [get_pblocks pblock_dynamic_region] [get_cells -quiet [list  level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/BUFG_rst_n_inst]]
    # add_cells_to_pblock [get_pblocks pblock_dynamic_SLR1] [get_cells -quiet [list level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/BUFG_rst_n_inst]]
      # set_false_path -through [get_cells level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/BUFG_rst_n_inst]
      # set_max_delay 6 -from level0_i/level1/level1_i/ulp/ip_psr_aresetn_kernel_00_slr0/U0/ACTIVE_LOW_PR_OUT_DFF[0].FDRE_PER_N/C
      set_max_delay 8 -through [get_cells level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/BUFG_rst_n_inst]
    # create_clock -period 5.000 -name 200M_xdc_clk [get_nets level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/ap_clk]
      #
      set_max_delay 8 -through [get_nets level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/areset]
      # set_false_path -through [get_cells level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/BUFG_areset_inst]
      # set_max_delay 6 -through [get_cells level0_i/level1/level1_i/ulp/krnl_msm_rtl_1/inst/inst_krnl_vadd_rtl_int/BUFG_areset_inst]

