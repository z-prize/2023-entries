/**********************************************************************************************************************
*	$Id$
**********************************************************************************************************************/
/*!	@file
*	@brief
*	Common definitions for all Verilog files.
**********************************************************************************************************************/
`include "env.vh"
`define		CQ	

`define		DIV_CEIL(a,b)	((a)<(b)?1:((a)%(b)+(a))/(b))
`define		DIV_UP(a,b)	(a<b ? 1 :  a%b ? (a/b+1) : (a/b) )
`define		LOG2(DATA)  ( \
			(DATA) > 33554432 ? 26 : \
			(DATA) > 16777216 ? 25 : \
			(DATA) > 8388608 ? 24 : \
			(DATA) > 4194304 ? 23 : \
			(DATA) > 2097152 ? 22 : \
			(DATA) > 1048576 ? 21 : \
			(DATA) > 524288 ? 20 : \
			(DATA) > 262144 ? 19 : \
			(DATA) > 131072 ? 18 : \
			(DATA) > 65536 ? 17 : \
			(DATA) > 32768 ? 16 : \
			(DATA) > 16384 ? 15 : \
			(DATA) > 8192  ? 14 : \
			(DATA) > 4096  ? 13 : \
			(DATA) > 2048  ? 12 : \
			(DATA) > 1024  ? 11 : \
			(DATA) > 512   ? 10 : \
			(DATA) > 256   ? 9  : \
			(DATA) > 128   ? 8  : \
			(DATA) > 64    ? 7  : \
			(DATA) > 32    ? 6  : \
			(DATA) > 16    ? 5  : \
			(DATA) > 8     ? 4  : \
			(DATA) > 4     ? 3  : \
			(DATA) > 2     ? 2  : \
			(DATA) > 1     ? 1  : \
			0)










/**********************************************************************************************************************
*	EndOfFile
**/
