#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>
#undef CL_VERSION_1_2
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <iostream>

#define BUFSIZE 4096

cl_int error = 0;   // Used to handle error codes
cl_uint nPlatforms;
cl_uint nDevices;
cl_platform_id platforms[16];
cl_device_id devices[16];
cl_bool bVal;
cl_uint uiVal;
cl_ulong ulVal;




int main (int argc, char** argv)
{
    // Platform
    error = clGetPlatformIDs (16, platforms, &nPlatforms);
    if (error != CL_SUCCESS) {
        std::cerr << "Error querying platforms: " << error << std::endl;
        exit(error);
    }
    std::cout << "\nFound " << nPlatforms << " OpenCL platform(s)\n" << std::endl;
    char cStr[BUFSIZE];
    for (cl_uint p = 0; p < nPlatforms; ++p)
    {
        error = clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, BUFSIZE, cStr, NULL);
        std::cout << "Platform[" << p << "]:  " << cStr << std::endl;
        error = clGetPlatformInfo(platforms[p], CL_PLATFORM_VERSION, BUFSIZE, cStr, NULL);
        std::cout << "   VERSION:     " << cStr << std::endl;
        error = clGetPlatformInfo(platforms[p], CL_PLATFORM_VENDOR, BUFSIZE, cStr, NULL);
        std::cout << "   VENDOR:      " << cStr << std::endl;
        error = clGetPlatformInfo(platforms[p], CL_PLATFORM_PROFILE, BUFSIZE, cStr, NULL);
        std::cout << "   PROFILE:     " << cStr << std::endl;
        //error = clGetPlatformInfo(platforms[p], CL_PLATFORM_EXTENSIONS, BUFSIZE, cStr, NULL);
        //std::cout << "   CL_PLATFORM_EXTENSIONS:  " << cStr << "\n" std::endl;
        std::cout << std::endl;
        error = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 16, devices, &nDevices);
        if (error != CL_SUCCESS) {
            std::cout << "Error querying devices: " << error << std::endl;
            exit(error);
        }
        std::cout << "   Found " << nDevices << " OpenCL devices(s)\n" << std::endl;
        for (cl_uint d = 0; d < nDevices; ++d)
        {
            error = clGetDeviceInfo(devices[d], CL_DEVICE_NAME, BUFSIZE, cStr, NULL);
            std::cout << "   Device[" << d << "]:  " << cStr << std::endl;
            error = clGetDeviceInfo(devices[d], CL_DEVICE_VERSION, BUFSIZE, cStr, NULL);
            std::cout << "      DEVICE VERSION:     " << cStr << std::endl;
            error = clGetDeviceInfo(devices[d], CL_DRIVER_VERSION, BUFSIZE, cStr, NULL);
            std::cout << "      DRIVER VERSION:     " << cStr << std::endl;
            error = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uiVal), &uiVal, NULL);
            std::cout << "      COMPUTE UNITS:      " << uiVal << std::endl;
            error = clGetDeviceInfo(devices[d], CL_DEVICE_IMAGE_SUPPORT, sizeof(bVal), &bVal, NULL);
            std::cout << "      IMAGE SUPPORT:      " << (bVal?"Yes":"No") << std::endl;
            error = clGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(ulVal), &ulVal, NULL);
            std::cout << "      GLOBAL MEM SIZE:    " << ulVal << std::endl;
            error = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(ulVal), &ulVal, NULL);
            std::cout << "      MAX MEM ALLOC SIZE: " << ulVal << std::endl;
            error = clGetDeviceInfo(devices[d], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(ulVal), &ulVal, NULL);
            std::cout << "      LOCAL MEM SIZE:     " << ulVal << std::endl;

        }

        
        std::cout << "*********************************************************************\n" << std::endl; 
    } 

    return 0;
}
