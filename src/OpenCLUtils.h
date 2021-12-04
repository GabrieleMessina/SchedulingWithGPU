/* ************************************************************************* *\
               INTEL CORPORATION PROPRIETARY INFORMATION
     This software is supplied under the terms of a license agreement or 
     nondisclosure agreement with Intel Corporation and may not be copied 
     or disclosed except in accordance with the terms of that agreement. 
        Copyright (C) 2014 Intel Corporation. All Rights Reserved.
\* ************************************************************************* */

#ifndef OPENCLUTILS_DOT_H
#define OPENCLUTILS_DOT_H

#include <stdio.h>
#include <stdlib.h>
#include <C:\Users\gabry\Universita\GPGPU\IntelSWTools\sw_dev_tools\OpenCL\sdk\include\CL\cl.h>
#include <string>


// Util for error checking:
//#undef __OCL_NO_ERROR_CHECKING
#define __OCL_NO_ERROR_CHECKING

#ifdef __OCL_NO_ERROR_CHECKING
#define CheckCLError(__errNum__, __failMsg__, __passMsg__)	\
	assert (CL_SUCCESS == __errNum__);
#else
#define CheckCLError(__errNum__, __failMsg__, __passMsg__)	\
if (CL_SUCCESS != __errNum__)								\
{															\
		char __msgBuf__[256];								\
		sprintf (__msgBuf__, "CL Error num %d: %s at line %d, file %s in function %s().\n", __errNum__, __failMsg__, __LINE__, __FILE__, __FUNCTION__);	\
		printf (__msgBuf__);								\
		getchar();											\
		printf("Failed on OpenCLError\n");					\
		assert (CL_SUCCESS != __errNum__);					\
		exit(0);											\
} else if (__passMsg__)										\
{															\
	printf("CL Success: %s\n", __passMsg__);				\
}				
#endif


// Util for OpenCL build log:
void BuildFailLog( cl_program program,
                  cl_device_id device_id )
{
    size_t paramValueSizeRet = 0;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &paramValueSizeRet);

    char* buildLogMsgBuf = (char *)malloc(sizeof(char) * paramValueSizeRet + 1);
	if( buildLogMsgBuf )
	{
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, paramValueSizeRet, buildLogMsgBuf, &paramValueSizeRet);
		buildLogMsgBuf[paramValueSizeRet] = '\0';	//mark end of message string

		printf("\nOpenCL C Program Build Log:\n");
		puts(buildLogMsgBuf);
		fflush(stdout);

		free(buildLogMsgBuf);
	}
}


void InitializeOpenCL(char* pDeviceStr, char* pVendorStr, cl_device_id* pDeviceID, cl_context* pContextHdl, cl_command_queue* pCmdQHdl, bool& bCPUDevice)
{	
	// OpenCL System Initialization

	*pDeviceID		= NULL;
	*pContextHdl	= NULL;

	// First query for all of the available platforms 
	// Choose the one that matches the command line request
	cl_uint			numPlatforms= 0;
	cl_int			ciErrNum	= clGetPlatformIDs(0, NULL, &numPlatforms);
	CheckCLError (ciErrNum, "No platforms Found.", "OpenCL platforms found.");

	char				pPlatformVendor[256];
	char                pDevVersion[256];
	char                pLangVersion[256];
	cl_platform_id		platformID	= NULL;
	
	cl_device_id		deviceID;
	cl_context			contextHdl;
	cl_command_queue	cmdQueueHdl;

	bCPUDevice = false;
	if (0 < numPlatforms) 
	{
		cl_platform_id		*pPlatformIDs = (cl_platform_id*)malloc(sizeof(cl_platform_id)*numPlatforms);
		if (pPlatformIDs == 0) 
		{
			printf("Error: could allocate space for platform IDs\n");
			exit(-1);
		}
 		ciErrNum = clGetPlatformIDs(numPlatforms, pPlatformIDs, NULL);
		CheckCLError (ciErrNum, "Could not get platform IDs.", "Got platform IDs.");
		unsigned i;

		for (i = 0; i < numPlatforms; ++i) 
		{
			ciErrNum = clGetPlatformInfo(pPlatformIDs[i],CL_PLATFORM_VENDOR,sizeof(pPlatformVendor),pPlatformVendor,NULL);
			CheckCLError (ciErrNum, "Could not get platform info.", "Got platform info.");
			
			platformID = pPlatformIDs[i];

			if ((!strcmp(pPlatformVendor, "Intel Corporation")|| !strcmp(pPlatformVendor, "Intel(R) Corporation")) && !strcmp(pVendorStr, "intel") && !strcmp(pDeviceStr, "gpu") )
			{
				if(CL_SUCCESS == clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, NULL))
					break;
			}
			if ((!strcmp(pPlatformVendor, "Intel Corporation")|| !strcmp(pPlatformVendor, "Intel(R) Corporation")) && !strcmp(pVendorStr, "intel") && !strcmp(pDeviceStr, "cpu"))
			{
				if(CL_SUCCESS == clGetDeviceIDs(platformID, CL_DEVICE_TYPE_CPU, 1, &deviceID, NULL)) {
					bCPUDevice = true;
					break;
				}
			}
			if (!strcmp(pPlatformVendor, "Advanced Micro Devices, Inc.") && !strcmp(pVendorStr, "amd") && !strcmp(pDeviceStr, "gpu") )
			{
				if(CL_SUCCESS == clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, NULL))
					break;
			}
			if (!strcmp(pPlatformVendor, "NVIDIA Corporation") && !strcmp(pVendorStr, "nvidia") && !strcmp(pDeviceStr, "gpu") )
			{
				if(CL_SUCCESS == clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, NULL))
					break;
			}
		}

		if(i == numPlatforms) 
		{
			printf("Error didn't find platform that matches requested platform: %s\n", pVendorStr);
			free(pPlatformIDs);
			exit(-1);
		}
		free(pPlatformIDs);

		ciErrNum = clGetDeviceInfo(deviceID, CL_DEVICE_VERSION, sizeof(pDevVersion), pDevVersion, NULL);
		if (CL_SUCCESS != ciErrNum) {
			printf("Error: couldn't get CL_DEVICE_VERSION!\n");
			exit(-1);
		}
		ciErrNum = clGetDeviceInfo(deviceID, CL_DEVICE_OPENCL_C_VERSION, sizeof(pLangVersion), pLangVersion, NULL);
		if (CL_SUCCESS != ciErrNum) {
			printf("Error: couldn't get CL_DEVICE_OPENCL_C_VERSION!\n");
			exit(-1);
		}

		// Courtesy Aaron Kunze
		// The format of the version string is defined in the spec as 
		// "OpenCL <major>.<minor> <vendor-specific>" 
		std::string dev_version(pDevVersion);
		std::string lang_version(pLangVersion);
		dev_version = dev_version.substr(std::string("OpenCL ").length()); 
		dev_version = dev_version.substr(0, dev_version.find('.')); 

		// The format of the version string is defined in the spec as 
		// "OpenCL C <major>.<minor> <vendor-specific>" 
		lang_version = lang_version.substr(std::string("OpenCL C ").length()); 
		lang_version = lang_version.substr(0, lang_version.find('.')); 

		if (!(stoi(dev_version) >= 2 && stoi(lang_version) >= 2)) {
			printf("Device does not support OpenCL 2.0 needed for this sample! CL_DEVICE_VERSION: %s, CL_DEVICE_OPENCL_C_VERSION, %s\n", pDevVersion, pLangVersion);
			exit(-1);
		}
	}
	else 
	{
		printf("numPlatforms is %d\n", numPlatforms);
		exit(-1);
	}
	
    // Create the OpenCL context
    contextHdl = clCreateContext(0, 1, &deviceID, NULL, NULL, &ciErrNum);
	CheckCLError (ciErrNum, "Could not create CL context.", "Created CL context.");

    // Create a command-queue
    cmdQueueHdl = clCreateCommandQueue(contextHdl, deviceID, 0, &ciErrNum);
	CheckCLError (ciErrNum, "Could not create CL command queue.", "Created CL command queue.");

	// The recommended minimum size of the device queue is 128K - enough for our purposes, since the algorithm enqueues only one kernel at a time
	cl_queue_properties qprop[] = {CL_QUEUE_SIZE, 128*1024, CL_QUEUE_PROPERTIES, (cl_command_queue_properties)(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT | CL_QUEUE_PROFILING_ENABLE), 0}; // CL_QUEUE_ON_DEVICE –
    cl_command_queue my_device_q = clCreateCommandQueueWithProperties(contextHdl, deviceID, qprop, &ciErrNum);
	CheckCLError (ciErrNum, "Could not create device side queue.", "Device side queue created.");

	int default_queue_size = 0;
	ciErrNum = clGetCommandQueueInfo(my_device_q, CL_QUEUE_SIZE, sizeof(int), &default_queue_size, 0);
	CheckCLError (ciErrNum, "Could not get device side queue info.", "Device side queue info fetched successfully.");
	printf("CL_QUEUE_SIZE is %d\n", default_queue_size);

	// Output parameters:
	*pDeviceID		= deviceID;
	*pContextHdl	= contextHdl;
	*pCmdQHdl		= cmdQueueHdl;

}

void CreateOCLProgramFromSourceFile(char const *pSrcFilePath, cl_context hClContext, cl_program *pCLProgram )
{
	    FILE* fp = fopen(pSrcFilePath, "rb");
        if (!fp) 
		{
			printf("Failed to find OpenCL source program: %s\n", pSrcFilePath);
			//Cleanup (-1, true, "Failed to open CL Source file.\n");
			exit(0);
		}

        fseek(fp, 0, SEEK_END);
        long size = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        char* buf = (char*)malloc((size + 1)*sizeof(char));
		if (buf == 0)
		{
			printf("Failed to allocated buffer of sufficient size fo the file: %s\n", pSrcFilePath);
			exit(-1);
		}
        buf[size] = '\0';
        size_t records_read = fread(buf, size, 1, fp);
		if (records_read != 1) {
			printf("Failed to read the file: %s\n", pSrcFilePath);
			free(buf);
			exit(-1);
		}
        int err = fclose(fp);
		if (err != 0) {
			printf("Failed to close the file: %s\n", pSrcFilePath);
			free(buf);
			exit(-1);
		}

		size_t szKernelLength = size;
		cl_int ciErrNum;
		*pCLProgram = clCreateProgramWithSource(hClContext, 1, (const char **) &buf, &szKernelLength, &ciErrNum);
        CheckCLError (ciErrNum, "Failed to create program.", "Created program.");

        free(buf);
}

void CompileOpenCLProgram(bool bCPUDevice, cl_device_id oclDeviceID, cl_context oclContextHdl, const char* pSourceFileStr, cl_program* pOclProgramHdl)
{
	cl_int		ciErrNum;
	cl_program	oclProgramHdl;

	*pOclProgramHdl = NULL;

	CreateOCLProgramFromSourceFile(pSourceFileStr, oclContextHdl, &oclProgramHdl);

	if (bCPUDevice) {
		ciErrNum = clBuildProgram(oclProgramHdl, 0, NULL, "-cl-std=CL2.0 -cl-mad-enable -DCPU_DEVICE=1", NULL, NULL);
	} else {
		ciErrNum = clBuildProgram(oclProgramHdl, 0, NULL, "-cl-std=CL2.0 -cl-mad-enable", NULL, NULL);
	}
	if (ciErrNum != CL_SUCCESS)
	{
		printf("ERROR: Failed to build program... ciErrNum = %d\n", ciErrNum);
		BuildFailLog(oclProgramHdl, oclDeviceID);
	}
	CheckCLError (ciErrNum, "Program building failed.", "Built Program");
	if (ciErrNum != CL_SUCCESS)
	{
		printf("Enter any key to exit.\n");
		getchar();
		exit(0);
	}

	// Output parameters:
	*pOclProgramHdl = oclProgramHdl;
}

// Util for OpenCL info queries:
void QueryPrintOpenCLDeviceInfo(cl_device_id deviceID, cl_context contextHdl)
{
	cl_uint		uMaxComputeUnits			= 0;
	cl_uint		uMaxWorkItemDim				= 0;
	size_t		uMaxWorkItemSizes[3];
	cl_uint		uMaxNumSamplers				= 0;
	cl_uint		uMinBaseAddrAlignSizeBits	= 0;		// CL_DEVICE_MEM_BASE_ADDR_ALIGN
	cl_uint		uMinBaseAddrAlignSizeBytes	= 0;		// CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE
	size_t		uNumBytes					= 0;
	char		pDeviceVendorString[512];		// CL_DEVICE_VENDOR
	char		pDeviceNameString[512];			// CL_DEVICE_NAME
	char		pDriverVersionString[512];		// CL_DRIVER_VERSION
	char		pDeviceProfileString[512];		// CL_DEVICE_PROFILE
	char		pDeviceVersionString[512];		// CL_DEVICE_VERSION
	char		pOpenCLCVersionString[512];		// CL_DEVICE_OPENCL_C_VERSION
	cl_int		ciErrNum;

	// Device Property Queries:
	ciErrNum = clGetDeviceInfo(deviceID, CL_DEVICE_VENDOR, sizeof(pDeviceVendorString), &pDeviceVendorString, &uNumBytes);
	CheckCLError (ciErrNum, "clGetDeviceInfo() query failed.", "clGetDeviceinfo() query success");
	ciErrNum = clGetDeviceInfo(deviceID, CL_DEVICE_NAME, sizeof(pDeviceNameString), &pDeviceNameString, &uNumBytes);
	CheckCLError (ciErrNum, "clGetDeviceInfo() query failed.", "clGetDeviceinfo() query success");

	printf("Using platform: %s and device: %s.\n", pDeviceVendorString, pDeviceNameString);
	printf ("OpenCL Device info:\n");
	printf ("CL_DEVICE_VENDOR			:%s\n", pDeviceVendorString);
	printf ("CL_DEVICE_NAME				:%s\n", pDeviceNameString);

	ciErrNum = clGetDeviceInfo(deviceID, CL_DRIVER_VERSION, sizeof(pDriverVersionString), &pDriverVersionString, &uNumBytes);
	CheckCLError (ciErrNum, "clGetDeviceInfo() query failed.", "clGetDeviceinfo() query success");
	printf ("CL_DRIVER_VERSION			:%s\n", pDriverVersionString);

	ciErrNum = clGetDeviceInfo(deviceID, CL_DEVICE_PROFILE, sizeof(pDeviceProfileString), &pDeviceProfileString, &uNumBytes);
	CheckCLError (ciErrNum, "clGetDeviceInfo() query failed.", "clGetDeviceinfo() query success");
	printf ("CL_DEVICE_PROFILE			:%s\n", pDeviceProfileString);

	ciErrNum = clGetDeviceInfo(deviceID, CL_DEVICE_VERSION, sizeof(pDeviceVersionString), &pDeviceVersionString, &uNumBytes);
	CheckCLError (ciErrNum, "clGetDeviceInfo() query failed.", "clGetDeviceinfo() query success");
	printf ("CL_DEVICE_VERSION			:%s\n", pDeviceVersionString);
	
	ciErrNum = clGetDeviceInfo(deviceID, CL_DEVICE_OPENCL_C_VERSION, sizeof(pOpenCLCVersionString), &pOpenCLCVersionString, &uNumBytes);
	CheckCLError (ciErrNum, "clGetDeviceInfo() query failed.", "clGetDeviceinfo() query success");
	printf ("CL_DEVICE_OPENCL_C_VERSION		:%s\n", pOpenCLCVersionString);


	ciErrNum = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &uMaxComputeUnits, &uNumBytes);
	CheckCLError (ciErrNum, "clGetDeviceInfo() query failed.", "clGetDeviceinfo() query success");
	printf ("CL_DEVICE_MAX_COMPUTE_UNITS		:%8d\n", uMaxComputeUnits);

	ciErrNum = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &uMaxWorkItemDim, &uNumBytes);
	CheckCLError (ciErrNum, "clGetDeviceInfo() query failed.", "clGetDeviceinfo() query success");
	printf ("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS	:%8d\n", uMaxWorkItemDim);

	ciErrNum = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(uMaxWorkItemSizes), &uMaxWorkItemSizes, &uNumBytes);
	CheckCLError (ciErrNum, "clGetDeviceInfo() query failed.", "clGetDeviceinfo() query success");
	printf ("CL_DEVICE_MAX_WORK_ITEM_SIZES		:    (%5d, %5d, %5d)%\n", 
					uMaxWorkItemSizes[0],uMaxWorkItemSizes[1], uMaxWorkItemSizes[2]);
	
	size_t	uMaxWorkGroupSize;
	ciErrNum = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &uMaxWorkGroupSize, &uNumBytes);
	CheckCLError (ciErrNum, "clGetDeviceInfo() query failed.", "clGetDeviceinfo() query success");
	printf ("CL_DEVICE_MAX_WORK_GROUP_SIZE		:%8d\n", uMaxWorkGroupSize);

	ciErrNum = clGetDeviceInfo(deviceID, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &uMinBaseAddrAlignSizeBits, &uNumBytes);
	CheckCLError (ciErrNum, "clGetDeviceInfo() query failed.", "clGetDeviceinfo() query success");
	printf ("CL_DEVICE_MEM_BASE_ADDR_ALIGN		:%8d\n", uMinBaseAddrAlignSizeBits);

	ciErrNum = clGetDeviceInfo(deviceID, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, sizeof(cl_uint), &uMinBaseAddrAlignSizeBytes, &uNumBytes);
	CheckCLError (ciErrNum, "clGetDeviceInfo() query failed.", "clGetDeviceinfo() query success");
	printf ("CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE	:%8d\n", uMinBaseAddrAlignSizeBytes);

	cl_uint	uMaxDeviceFrequency;
	ciErrNum = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &uMaxDeviceFrequency, &uNumBytes);
	CheckCLError (ciErrNum, "clGetDeviceInfo() query failed.", "clGetDeviceinfo() query success");
	printf ("CL_DEVICE_MAX_CLOCK_FREQUENCY		:%8d\n", uMaxDeviceFrequency);

	cl_uint	uMaxImage2DWidth;
	ciErrNum = clGetDeviceInfo(deviceID, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(cl_uint), &uMaxImage2DWidth, &uNumBytes);
	CheckCLError (ciErrNum, "clGetDeviceInfo() query failed.", "clGetDeviceinfo() query success");
	printf ("CL_DEVICE_IMAGE2D_MAX_WIDTH		:%8d\n", uMaxImage2DWidth);

	cl_ulong	uLocalMemSize;
	float		fLocalMemSize;
	ciErrNum = clGetDeviceInfo(deviceID, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &uLocalMemSize, &uNumBytes);
	CheckCLError (ciErrNum, "clGetDeviceInfo() query failed.", "clGetDeviceinfo() query success");
	fLocalMemSize = (float) uLocalMemSize;
	printf ("CL_DEVICE_LOCAL_MEM_SIZE		:%12.1f\n", fLocalMemSize);

	cl_long	uMaxMemAllocSize;
	float fMaxMemAllocSize;  
	ciErrNum = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_long), &uMaxMemAllocSize, &uNumBytes);
	CheckCLError (ciErrNum, "clGetDeviceInfo() query failed.", "clGetDeviceinfo() query success");
	fMaxMemAllocSize = (float) uMaxMemAllocSize;
	printf ("CL_DEVICE_MAX_MEM_ALLOC_SIZE		:%12.1f\n", fMaxMemAllocSize);
	
#define MAX_NUM_FORMATS 500
	cl_uint numFormats;
	cl_image_format myFormats[MAX_NUM_FORMATS];

	ciErrNum = clGetSupportedImageFormats(contextHdl, CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE2D, 255, myFormats, &numFormats);
	CheckCLError (ciErrNum, "clGetSupportedImageFormats() query failed.", "clGetSupportedImageFormats() query success");
}

#endif
