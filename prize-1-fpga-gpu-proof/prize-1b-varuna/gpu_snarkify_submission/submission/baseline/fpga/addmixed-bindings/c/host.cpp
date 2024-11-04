#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1


#include <CL/cl2.hpp>
#include <CL/cl_ext_xilinx.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstdint>

#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

template <typename T>
struct aligned_allocator
{
    using value_type = T;
    T *allocate(std::size_t num)
    {
        void *ptr = nullptr;
        if (posix_memalign(&ptr, 4096, num * sizeof(T)))
            throw std::bad_alloc();
        return reinterpret_cast<T *>(ptr);
    }
    void deallocate(T *p, std::size_t num)
    {
        free(p);
    }
};

uint64_t get_duration_ns(const cl::Event &event) {
    cl_int err;
    uint64_t nstimestart, nstimeend;
    OCL_CHECK(err,
              err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START,
                                                     &nstimestart));
    OCL_CHECK(err,
              err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END,
                                                     &nstimeend));
    return (nstimeend - nstimestart);
}

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



int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];

    //When creating a buffer with user pointer, under the hood user ptr is
    //used if and only if it is properly aligned (page aligned). When not
    //aligned, runtime has no choice but to create its own host side buffer
    //that backs user ptr. This in turn implies that all operations that move
    //data to/from device incur an extra memcpy to move data to/from runtime's
    //own host buffer from/to user pointer. So it is recommended to use this
    //allocator if user wish to Create Buffer/Memory Object to align user buffer
    //to the page boundary. It will ensure that user buffer will be used when
    //user create Buffer/Mem Object.
    const int round = 100000;

    std::vector<argument_element_t, aligned_allocator<argument_element_t>> arguments(round);
    std::vector<result_element_t, aligned_allocator<result_element_t>> results(round);
    std::vector<result_element_t, aligned_allocator<result_element_t>> ref_results(round);

    //Create the test data
    for (int i = 0; i < round; i++) {
        arguments[i].x1[5] = 0x11BDC38C466F1AB0ULL; arguments[i].x1[4] = 0x59FAB75B43335171ULL; arguments[i].x1[3] = 0xC98C7B21ABA8F77BULL; arguments[i].x1[2] = 0x39186D6BCEB521C9ULL; arguments[i].x1[1] = 0xA18AFEE0406537F8ULL; arguments[i].x1[0] = 0xF46B6B0FB5C55D2AULL;
        arguments[i].y1[5] = 0x0F601759D2452CA2ULL; arguments[i].y1[4] = 0xC06131143BDB94EAULL; arguments[i].y1[3] = 0x076A1EDD95152E14ULL; arguments[i].y1[2] = 0x61109085E445742FULL; arguments[i].y1[1] = 0x1CADC7A6A1C7DAF2ULL; arguments[i].y1[0] = 0x6E6B01CA791189CAULL;
        arguments[i].z1[5] = 0x06914F774734A80DULL; arguments[i].z1[4] = 0xBE03B8B58CBA2F78ULL; arguments[i].z1[3] = 0x0D2AB77BAADB6339ULL; arguments[i].z1[2] = 0xEAD3C6DBF084E791ULL; arguments[i].z1[1] = 0xA7DCDACD2432F6E5ULL; arguments[i].z1[0] = 0x70D8EFC67EDD2B23ULL;
        arguments[i].x2[5] = 0x0ED6E4DA4A89F038ULL; arguments[i].x2[4] = 0xAC0EC400307280B7ULL; arguments[i].x2[3] = 0x2F81A7E303F58CB6ULL; arguments[i].x2[2] = 0x9201981CF15E73E9ULL; arguments[i].x2[1] = 0xECF64ABD58C27900ULL; arguments[i].x2[0] = 0xE9C0CC09A093AC06ULL;
        arguments[i].y2[5] = 0x104930B0C05A4754ULL; arguments[i].y2[4] = 0x048FB035E802EE27ULL; arguments[i].y2[3] = 0xB8D101804F303AFCULL; arguments[i].y2[2] = 0xA9024CBE739AEEDAULL; arguments[i].y2[1] = 0xE3446D65EAC37B72ULL; arguments[i].y2[0] = 0x956BF7E33F6E6AE7ULL;
        ref_results[i].x3[5] = 0x0AC6DB2DCD9173AFULL; ref_results[i].x3[4] = 0xF151BF789CB2CAF0ULL; ref_results[i].x3[3] = 0x03AA63258E571E93ULL; ref_results[i].x3[2] = 0x9CC06211B064ECEBULL; ref_results[i].x3[1] = 0xBB65A61EF9ABEBF3ULL; ref_results[i].x3[0] = 0x73A2038F5EC3577BULL;
        ref_results[i].y3[5] = 0x0AF0A0A817C4DCE6ULL; ref_results[i].y3[4] = 0x179399628D89305DULL; ref_results[i].y3[3] = 0x449D47B2D5FC487AULL; ref_results[i].y3[2] = 0xE25AF562CF3E967BULL; ref_results[i].y3[1] = 0x3BBFAB4AB2ECA28FULL; ref_results[i].y3[0] = 0x8E9C973A5B33FE2DULL;
        ref_results[i].z3[5] = 0x03A595DA6944D6EBULL; ref_results[i].z3[4] = 0x723A5D8B8FF56A16ULL; ref_results[i].z3[3] = 0x11F1336B8718C957ULL; ref_results[i].z3[2] = 0x585E6CA79A9C2122ULL; ref_results[i].z3[1] = 0x95F0DE6AF9B88178ULL; ref_results[i].z3[0] = 0x72CC2D0F71E74BE3ULL;
    }

    uint64_t kernel_duration = 0;

    //Compute FPGA Results
    kernel_duration = addmixed_fpga(arguments, results, binaryFile);

    //Compare the results of FPGA to CPU
    bool match = true;
    for (int i = 0; i < round; i++) {
        for (int j = 0; j < 6; j++) {
            if (results[i].x3[j] != ref_results[i].x3[j]) {
                std::cout << "Error: Result mismatch" << std::endl;
                std::cout << "x3" << " i = " << i << " j = " << j
                        << " FPGA result = " << results[i].x3[j]
                        << " REF result = " << ref_results[i].x3[j]
                        << std::endl;
                match = false;
                goto out;
            }
        }

        for (int j = 0; j < 6; j++) {
            if (results[i].y3[j] != ref_results[i].y3[j]) {
                std::cout << "Error: Result mismatch" << std::endl;
                std::cout << "y3" << " i = " << i << " j = " << j
                        << " FPGA result = " << results[i].y3[j]
                        << " REF result = " << ref_results[i].y3[j]
                        << std::endl;
                match = false;
                goto out;
            }
        }

        for (int j = 0; j < 6; j++) {
            if (results[i].z3[j] != ref_results[i].z3[j]) {
                std::cout << "Error: Result mismatch" << std::endl;
                std::cout << "z3" << " i = " << i << " j = " << j
                        << " FPGA result = " << results[i].z3[j]
                        << " REF result = " << ref_results[i].z3[j]
                        << std::endl;
                match = false;
                goto out;
            }
        }
    }
out:
    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;

    std::cout << "Wall Clock Time (Kernel execution): " << kernel_duration << " ns"
              << std::endl;

    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}
