#include <ap_int.h>
#include <cstdint>

typedef ap_uint<384> ap_uint384_t;
typedef ap_uint<385> ap_uint385_t;
typedef ap_uint<768> ap_uint768_t;

typedef struct argument_element_t {
	uint64_t x1[6];
	uint64_t y1[6];
	uint64_t z1[6];
	uint64_t x2[6];
	uint64_t y2[6];
} argument_element_t;

typedef struct result_element_t {
	uint64_t x3[6];
	uint64_t y3[6];
	uint64_t z3[6];
} result_element_t;

const ap_uint384_t P("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab", 16);
const ap_uint384_t rinv("14fec701e8fb0ce9ed5e64273c4f538b1797ab1458a88de9343ea97914956dc87fe11274d898fafbf4d38259380b4820", 16);

ap_uint768_t mult384e(ap_uint384_t a, ap_uint384_t b) {
	#pragma HLS pipeline
	#pragma HLS allocation operation instances=mult limit=9

	ap_uint<128> a2 = a.range(383, 256);
	ap_uint<128> a1 = a.range(255, 128);
	ap_uint<128> a0 = a.range(127, 0);
	ap_uint<128> b2 = b.range(383, 256);
	ap_uint<128> b1 = b.range(255, 128);
	ap_uint<128> b0 = b.range(127, 0);

	ap_uint<256> a2b2 = a2 * b2;
	ap_uint<256> a2b1 = a2 * b1;
	ap_uint<256> a2b0 = a2 * b0;
	ap_uint<256> a1b2 = a1 * b2;
	ap_uint<256> a1b1 = a1 * b1;
	ap_uint<256> a1b0 = a1 * b0;
	ap_uint<256> a0b2 = a0 * b2;
	ap_uint<256> a0b1 = a0 * b1;
	ap_uint<256> a0b0 = a0 * b0;

	ap_uint<257> a2b1_a1b2 = a2b1 + a1b2;
	ap_uint<258> a2b0_a1b1_a0b2 = a2b0 + a1b1 + a0b2;
	ap_uint<257> a1b0_a0b1 = a1b0 + a0b1;

	ap_uint768_t t1("0", 16);
	t1.range(767, 512) = a2b2;
	ap_uint768_t t2("0", 16);
	t2.range(640, 384) = a2b1_a1b2;
	ap_uint768_t t3("0", 16);
	t3.range(513, 256) = a2b0_a1b1_a0b2;
	ap_uint768_t t4("0", 16);
	t4.range(384, 128) = a1b0_a0b1;
	ap_uint768_t t5("0", 16);
	t5.range(255, 0) = a0b0;

	return t1 + t2 + t3 + t4 + t5;
}

ap_uint384_t modMult384(ap_uint384_t a, ap_uint384_t b) {
	#pragma HLS pipeline
	#pragma HLS allocation function instances=mult384e limit=3
	const ap_uint384_t X("ceb06106feaafc9468b316fee268cf5819ecca0e8eb2db4c16ef2ef0c8e30b48286adb92d9d113e889f3fffcfffcfffd", 16);
	ap_uint768_t ab = mult384e(a, b);
	ap_uint384_t t = (mult384e(mult384e(ab, X), P) + ab).range(767, 384);
	if (t > P) return t - P;
	else return t;
}

ap_uint384_t modAdd384(ap_uint384_t a, ap_uint384_t b) {
	#pragma HLS pipeline II=1
	ap_uint385_t t = a + b;
	if (t > P) return t - P;
	else return t;
}

ap_uint384_t modSub384(ap_uint384_t a, ap_uint384_t b) {
	#pragma HLS pipeline II=1
	if (a > b) return a - b;
	else return a + P - b;
}

void array2ap(uint64_t a[6], ap_uint384_t &b) {
	#pragma HLS inline
	for (int i = 0; i < 6; i++) {
		b.range((i + 1) * 64 - 1, i * 64) = a[i];
	}
}

void ap2array(ap_uint384_t &a, uint64_t b[6]) {
	#pragma HLS inline
	for (int i = 0; i < 6; i++) {
		b[i] = a.range((i + 1) * 64 - 1, i * 64);
	}
}

void addMixed(argument_element_t &arge, result_element_t &rese) {
	#pragma HLS inline
	#pragma HLS allocation function instances=modMult384 limit=1

	ap_uint384_t x1;
	array2ap(arge.x1, x1);
	ap_uint384_t y1;
	array2ap(arge.y1, y1);
	ap_uint384_t z1;
	array2ap(arge.z1, z1);
	ap_uint384_t x2;
	array2ap(arge.x2, x2);
	ap_uint384_t y2;
	array2ap(arge.y2, y2);

	ap_uint384_t z1z1 = modMult384(z1, z1);
	ap_uint384_t y2z1 = modMult384(y2, z1);

	ap_uint384_t s2 = modMult384(y2z1, z1z1);
	ap_uint384_t u2 = modMult384(x2, z1z1);

	ap_uint384_t h = modSub384(u2, x1) ;
	ap_uint384_t hh = modMult384(h, h);

	ap_uint384_t hh2 = modAdd384(hh, hh);
	ap_uint384_t i = modAdd384(hh2, hh2);

	ap_uint384_t j = modMult384(h, i) ;

	ap_uint384_t rtmp = modSub384(s2, y1);
	ap_uint384_t r = modAdd384(rtmp, rtmp);

	ap_uint384_t v = modMult384(x1, i) ;

	ap_uint384_t rr = modMult384(r, r);
	ap_uint384_t v2 = modAdd384(v, v);
	ap_uint384_t jv2 = modAdd384(j, v2);
	ap_uint384_t x3 = modSub384(rr, jv2);

	ap_uint384_t v_minus_x3 = modSub384(v, x3);
	ap_uint384_t y1j = modMult384(y1, j);
	ap_uint384_t rvx3 = modMult384(r, v_minus_x3);
	ap_uint384_t y1j2 = modAdd384(y1j, y1j);
	ap_uint384_t y3 = modSub384(rvx3, y1j2);

	ap_uint384_t z1h = modAdd384(z1, h);
	ap_uint384_t z1h2 = modMult384(z1h, z1h) ;
	ap_uint384_t z1z1hh = modAdd384(z1z1, hh);
	ap_uint384_t z3 = modSub384(z1h2, z1z1hh);

	ap2array(x3, rese.x3);
	ap2array(y3, rese.y3);
	ap2array(z3, rese.z3);
}

void addMixedKernel(argument_element_t *a, result_element_t *r, unsigned int length) {
	#pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem
	#pragma HLS INTERFACE m_axi port=r offset=slave bundle=gmem

	#pragma HLS INTERFACE s_axilite port=a	  bundle=control
	#pragma HLS INTERFACE s_axilite port=r	  bundle=control
	#pragma HLS INTERFACE s_axilite port=length bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control

	for (unsigned int i = 0; i < length; i++) {
		#pragma HLS pipeline II=30
		addMixed(a[i], r[i]);
	}
}


//Functionality to setup OpenCL context and trigger the Kernel
uint64_t addmixed_fpga (
    std::vector<argument_element_t, aligned_allocator<argument_element_t>> &arguments,
    std::vector<result_element_t, aligned_allocator<result_element_t>> &results, //Output Matrix
    std::string &binaryFile   //Binary file string
) {
    cl_int err;

    //The get_xil_devices will return vector of Xilinx Devices
    size_t i;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    cl::Platform platform;
    for (i = 0; i < platforms.size(); i++) {
        platform                  = platforms[i];
        std::string platform_name = platform.getInfo<CL_PLATFORM_NAME>();
        if (platform_name == "Xilinx") {
            break;
        }
    }
    if (i == platforms.size()) {
        assert(0 && "Unable to find Xilinx OpenCL devices");
    }

    // Get ACCELERATOR devices
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
    auto device = devices[0];

    //Creating Context and Command Queue for selected Device
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(
        err,
        cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

    //read_binary() command will find the OpenCL binary file created using the
    //xocc compiler load into OpenCL Binary and return a pointer to file buffer
    //and it can contain many functions which can be executed on the
    //device.
    std::ifstream xclbin(binaryFile.c_str(), std::ifstream::binary);
    xclbin.seekg(0, xclbin.end);
    unsigned int nb = xclbin.tellg();
    xclbin.seekg(0, xclbin.beg);
    char *buf = new char[nb];
    xclbin.read(buf, nb);
    cl::Program::Binaries bins;
    bins.push_back({buf, nb});

    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    //This call will extract a kernel out of the program we loaded in the
    //previous line. A kernel is an OpenCL function that is executed on the
    //FPGA. This function is defined in the src/mmult.cl file.
    OCL_CHECK(err, cl::Kernel kernel(program, "addMixedKernel", &err));

    //These commands will allocate memory on the FPGA. The cl::Buffer
    //objects can be used to reference the memory locations on the device.
    //The cl::Buffer object cannot be referenced directly and must be passed
    //to other OpenCL functions.
    OCL_CHECK(err,
              cl::Buffer buffer_arg(context,
                                    CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                    sizeof(argument_element_t) * arguments.size(),
                                    arguments.data(),
                                    &err));
    OCL_CHECK(err,
              cl::Buffer buffer_res(context,
                                       CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                       sizeof(result_element_t) * results.size(),
                                       results.data(),
                                       &err));

    //Set the kernel arguments
    int narg = 0;
    OCL_CHECK(err, err = kernel.setArg(narg++, buffer_arg));
    OCL_CHECK(err, err = kernel.setArg(narg++, buffer_res));
    OCL_CHECK(err, err = kernel.setArg(narg++, (unsigned int)(arguments.size())));

    //These commands will load the source_in1 and source_in2 vectors from the host
    //application into the buffer_in1 and buffer_in2 cl::Buffer objects. The data
    //will be be transferred from system memory over PCIe to the FPGA on-board
    //DDR memory.
    OCL_CHECK(err,
              err = q.enqueueMigrateMemObjects({buffer_arg},
                                               0 /* 0 means from host*/));

    cl::Event event;
    uint64_t kernel_duration = 0;

    //Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(kernel, NULL, &event));

    //The result of the previous kernel execution will need to be retrieved in
    //order to view the results. This call will write the data from the
    //buffer_output cl_mem object to the source_fpga_results vector
    OCL_CHECK(err,
              err = q.enqueueMigrateMemObjects({buffer_res},
                                               CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q.finish());

    kernel_duration = get_duration_ns(event);

    return kernel_duration;
}

void addmixed_op(
    const argument_element_t     *arguments_in,
    const result_element_t     *result_out,

){
    printf(" -- addmixed_op -- \n");
    std::string binaryFile = "zPrize/binding/addmixed-bindings/c/hlsdesign/addmixed.xclbin";
    printf(" -- load file-- \n");


    // transfer data into vector
    /**/

    std::vector<argument_element_t, aligned_allocator<argument_element_t>> arguments;
    std::vector<result_element_t, aligned_allocator<result_element_t>> results;

    for (int i = 0; i < SLICE_NUM; i++) {
        x1_vec[i] = (*x1_in).a[i];
        //printf(" -- x1_in[%d] = %016lx  \n", i, x1_in->a[i]);
        //printf(" -- x1_vec[%d] = %016lx  \n", i, x1_vec[i]);
        y1_vec[i] = (*y1_in).a[i];
        z1_vec[i] = (*z1_in).a[i];
        x2_vec[i] = (*x2_in).a[i];
        y2_vec[i] = (*y2_in).a[i];
    }

    
    int kernel_duration = addmixed_fpga(arguments, results, binaryFile);


    printf("finsih the addmixed fpga \n");


}
