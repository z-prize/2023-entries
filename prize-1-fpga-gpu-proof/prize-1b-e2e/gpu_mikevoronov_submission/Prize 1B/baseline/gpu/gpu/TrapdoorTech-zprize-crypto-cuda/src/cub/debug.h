#pragma once

#include "types.h"

//TODO: convert to storage
#if 0
static void printElem(const char* name, const Fr_long* array, const int N) {
    std::cout << "first " << name << " ";
    for (int z = 0; z < Fr_LIMBS / 2; ++z)
        std::cout << array[0].val[z] << " ";
    std::cout << "\n";
    
    if (N > 1) 
    {
        std::cout << "last " << name << " ";
        for (int z = 0; z < Fr_LIMBS / 2; ++z)
            std::cout << array[N - 1].val[z] << " ";
        std::cout << "\n";
    }
}

static void printElem(const char* name, const point_jacobian* array, const int N) {
    std::cout << "first " << name << "\n";
    unsigned long long *val = (unsigned long long *)(array[0].val[0].val);
    printf(" x: %llu %llu %llu %llu %llu %llu\n", val[0], val[1], val[2], val[3], val[4], val[5]);
    val = (unsigned long long *)(array[0].val[1].val);
    printf(" y: %llu %llu %llu %llu %llu %llu\n", val[0], val[1], val[2], val[3], val[4], val[5]);
    val = (unsigned long long *)(array[0].val[2].val);
    printf(" z: %llu %llu %llu %llu %llu %llu\n\n", val[0], val[1], val[2], val[3], val[4], val[5]);
    
    
    if (N > 1) 
    {
        std::cout << "last " << name << " ";
        unsigned long long *val = (unsigned long long *)(array[N - 1].val[0].val);
        printf(" x: %llu %llu %llu %llu %llu %llu\n", val[0], val[1], val[2], val[3], val[4], val[5]);
        val = (unsigned long long *)(array[N - 1].val[1].val);
        printf(" y: %llu %llu %llu %llu %llu %llu\n", val[0], val[1], val[2], val[3], val[4], val[5]);
        val = (unsigned long long *)(array[N - 1].val[2].val);
        printf(" z: %llu %llu %llu %llu %llu %llu\n\n", val[0], val[1], val[2], val[3], val[4], val[5]);
    }
}

static void printElem(const char* name, const point_xyzz* array, const int N) {
    std::cout << "first " << name << "\n";
    unsigned long long *val = (unsigned long long *)(array[0].val[0].val);
    printf(" x: %llu %llu %llu %llu %llu %llu\n", val[0], val[1], val[2], val[3], val[4], val[5]);
    val = (unsigned long long *)(array[0].val[1].val);
    printf(" y: %llu %llu %llu %llu %llu %llu\n", val[0], val[1], val[2], val[3], val[4], val[5]);
    val = (unsigned long long *)(array[0].val[2].val);
    printf(" z: %llu %llu %llu %llu %llu %llu\n", val[0], val[1], val[2], val[3], val[4], val[5]);
    val = (unsigned long long *)(array[0].val[3].val);
    printf(" zz: %llu %llu %llu %llu %llu %llu\n\n", val[0], val[1], val[2], val[3], val[4], val[5]);
    
    if (N > 1) 
    {
        std::cout << "last " << name << " ";
        unsigned long long *val = (unsigned long long *)(array[N - 1].val[0].val);
        printf(" x: %llu %llu %llu %llu %llu %llu\n", val[0], val[1], val[2], val[3], val[4], val[5]);
        val = (unsigned long long *)(array[N - 1].val[1].val);
        printf(" y: %llu %llu %llu %llu %llu %llu\n", val[0], val[1], val[2], val[3], val[4], val[5]);
        val = (unsigned long long *)(array[N - 1].val[2].val);
        printf(" z: %llu %llu %llu %llu %llu %llu\n", val[0], val[1], val[2], val[3], val[4], val[5]);
        val = (unsigned long long *)(array[N - 1].val[3].val);
        printf(" zz: %llu %llu %llu %llu %llu %llu\n\n", val[0], val[1], val[2], val[3], val[4], val[5]);
    } 
}

static void printElem(const char* name, const affine* array, const int N) {
    //printf("sizeof = %d\n", sizeof(affine));
    std::cout << "first " << name << "\n";
    unsigned long long *val = (unsigned long long *)(array[0].val[0].val);
    printf(" x: %llu %llu %llu %llu %llu %llu\n", val[0], val[1], val[2], val[3], val[4], val[5]);
    val = (unsigned long long *)(array[0].val[1].val);
    printf(" y: %llu %llu %llu %llu %llu %llu\n", val[0], val[1], val[2], val[3], val[4], val[5]);
    
    if (N > 1) 
    {
        std::cout << "last " << name << " ";
        unsigned long long *val = (unsigned long long *)(array[N - 1].val[0].val);
        printf(" x: %llu %llu %llu %llu %llu %llu\n", val[0], val[1], val[2], val[3], val[4], val[5]);
        val = (unsigned long long *)(array[N - 1].val[1].val);
        printf(" y: %llu %llu %llu %llu %llu %llu\n", val[0], val[1], val[2], val[3], val[4], val[5]);        
    }
}
#endif