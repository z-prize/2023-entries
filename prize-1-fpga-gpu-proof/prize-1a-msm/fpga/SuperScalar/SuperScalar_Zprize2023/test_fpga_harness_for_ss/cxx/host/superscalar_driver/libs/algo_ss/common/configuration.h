#ifndef _COMMON_CONFIGURATION_H_
#define _COMMON_CONFIGURATION_H_

#include <iostream>
#include <string>
#include <map>

namespace common
{
    enum CURVE_CLUSTER_TYPE
    {
        CURVE_CLUSTER_TYPE_DEFAULT = 0x0,
        CURVE_CLUSTER_TYPE_BLS12_377 = 0x1,
        CURVE_CLUSTER_TYPE_BLS12_381 = 0x2,
    };

    enum CURVE_TYPE
    {
        CURVE_TYPE_DEFAULT = 0x0,
        CURVE_TYPE_SW = 0x1,
        CURVE_TYPE_EXED = 0x2,
    };

    enum CURVE_INPUT_DATA_TYPE
    {
        CURVE_INPUT_DATA_TYPE_DEFAULT = 0x0,
        // CURVE_INPUT_DATA_TYPE_U32 = 0x1,
        // CURVE_INPUT_DATA_TYPE_U64 = 0x2,
        CURVE_INPUT_DATA_TYPE_U64 = 0x1,
    };

    enum FIELD_RADIX_TYPE
    {
        RADIX_TYPE_64BIT = 0,
        // RADIX_TYPE_32BIT,
    };

    enum FP_TYPE
    {
        FP_384_64BIT = 0,
        // FP_384_32BIT,
    };

    enum FR_TYPE
    {
        FR_256_64BIT = 0,
        // FR_256_32BIT,
    };

    struct SUPPORTED_CURVE_CLUSTER_TYPE
    {
        std::map<CURVE_CLUSTER_TYPE, std::string> name;
        std::map<CURVE_CLUSTER_TYPE, bool> is_support;

        SUPPORTED_CURVE_CLUSTER_TYPE() {
            name[CURVE_CLUSTER_TYPE_BLS12_377] = "BLS12_377";
            name[CURVE_CLUSTER_TYPE_BLS12_381] = "BLS12_381";

            is_support[CURVE_CLUSTER_TYPE_BLS12_377] = true;
            is_support[CURVE_CLUSTER_TYPE_BLS12_381] = false;
        };

        void print_status() const {
            std::cout << "\n        " <<  "Curve Cluster Type:"<< std::endl;
            for (const auto& entry : name) {
                std::cout << "            - " << entry.second << ":     " << (is_support.at(entry.first) ? "Supported" : "Not Supported") << std::endl;
            }
        }
    };

    struct SUPPORTED_CURVE_TYPE
    {
        std::map<CURVE_TYPE, std::string> name;
        std::map<CURVE_TYPE, bool> is_support;

        SUPPORTED_CURVE_TYPE() {
            name[CURVE_TYPE_SW] = "Short Weierstrass Curve";
            name[CURVE_TYPE_EXED] = "Extened Twisted Edwards Curve";

            is_support[CURVE_TYPE_SW] = true;
            is_support[CURVE_TYPE_EXED] = true;
        };

        void print_status() const {
            std::cout << "\n        " <<  "Curve Type:"<< std::endl;
            for (const auto& entry : name) {
                std::cout << "            - " << entry.second << ":     " << (is_support.at(entry.first) ? "Supported" : "Not Supported") << std::endl;
            }
        }
    };

    struct SUPPORTED_FIELD_RADIX_TYPE
    {
        std::map<FIELD_RADIX_TYPE, std::string> name;
        std::map<FIELD_RADIX_TYPE, bool> is_support;

        SUPPORTED_FIELD_RADIX_TYPE() {
            name[RADIX_TYPE_64BIT] = "Radix 64bit";
            //name[RADIX_TYPE_32BIT] = "Radix 32bit";

            is_support[RADIX_TYPE_64BIT] = true;
            //is_support[RADIX_TYPE_32BIT] = true;
        };

        void print_status() const {
            std::cout << "\n        " <<  "Field Radix Type:"<< std::endl;
            for (const auto& entry : name) {
                std::cout << "            - " << entry.second << ":     " << (is_support.at(entry.first) ? "Supported" : "Not Supported") << std::endl;
            }
        }
    };

    struct SUPPORTED_FP_TYPE
    {
        std::map<FP_TYPE, std::string> name;
        std::map<FP_TYPE, bool> is_support;

        SUPPORTED_FP_TYPE() {
            name[FP_384_64BIT] = "FP384 64bit";
            //name[FP_384_32BIT] = "FP384 32bit";

            is_support[FP_384_64BIT] = true;
            //is_support[FP_384_32BIT] = true;
        };

        void print_status() const {
            std::cout << "\n        " <<  "FP Type:"<< std::endl;
            for (const auto& entry : name) {
                std::cout << "            - " << entry.second << ":     " << (is_support.at(entry.first) ? "Supported" : "Not Supported") << std::endl;
            }
        }
    };

    struct SUPPORTED_FR_TYPE
    {
        std::map<FR_TYPE, std::string> name;
        std::map<FR_TYPE, bool> is_support;

        SUPPORTED_FR_TYPE() {
            name[FR_256_64BIT] = "FR256 64bit";
            //name[FR_256_32BIT] = "FR256 32bit";

            is_support[FR_256_64BIT] = true;
            //is_support[FR_256_32BIT] = true;
        };

        void print_status() const {
            std::cout << "\n        " <<  "FR Type:"<< std::endl;
            for (const auto& entry : name) {
                std::cout << "            - " << entry.second << ":     " << (is_support.at(entry.first) ? "Supported" : "Not Supported") << std::endl;
            }
        }
    };

    struct CONFIGURATION
    {
        CURVE_CLUSTER_TYPE curve_cluster_type;
        CURVE_TYPE curve_type;
        FIELD_RADIX_TYPE radix_type;
        bool dump_input_bases_flag;
        char dump_input_bases_filepath[128];
        bool dump_input_scalars_flag;
        char dump_input_scalars_filepath[128];
        bool dump_groups_result_flag;
        char dump_groups_result_filepath[128];
        bool lookup_flag;
        bool debug_info_flag;
        bool print_input_bases_flag;
        bool print_input_scalars_flag;
        bool print_groups_result_flag;
        bool profiling_flag;

    };

}


#endif




