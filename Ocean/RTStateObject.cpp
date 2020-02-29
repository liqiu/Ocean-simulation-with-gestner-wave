#include "RTStateObject.h"
#include "ocean.h"

#include <sutil/sutil.h>
#include <sutil/Exception.h>

#include <optix_stubs.h>
#include <cuda_runtime.h>

#include <string>
#include <iostream>
#include <iomanip>


template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;


static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
        << message << "\n";
}


void RTStateObject::initialize()
{
    createContext();
    createPTXModule();
    createProgramGroups();
    createPipeline();
    createSBT();
}

void RTStateObject::createContext()
{
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    CUcontext cuCtx = 0;  // zero means take the current context
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &mContext));
}

void RTStateObject::createPTXModule()
{
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    mPipelineCompileOptions.usesMotionBlur = false;
    mPipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    mPipelineCompileOptions.numPayloadValues = 3;
    mPipelineCompileOptions.numAttributeValues = 3;
    mPipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    mPipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    const std::string ptx = sutil::getPtxString(NULL, "Ocean");
    char log[2048];
    size_t sizeof_log = sizeof(log);

    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        mContext,
        &module_compile_options,
        &mPipelineCompileOptions,
        ptx.c_str(),
        ptx.size(),
        log,
        &sizeof_log,
        &mModule
    ));
}

void RTStateObject::createProgramGroups()
{   
    OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

    OptixProgramGroupDesc raygen_prog_group_desc = {}; //
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = mModule;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        mContext,
        &raygen_prog_group_desc,
        1,   // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &mRaygenProgGroup
    ));

    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = mModule;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        mContext,
        &miss_prog_group_desc,
        1,   // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &mMissProgGroup
    ));

    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH = mModule;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        mContext,
        &hitgroup_prog_group_desc,
        1,   // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &mHitgroupProgGroup
    ));
}

void RTStateObject::createPipeline()
{
    
    OptixProgramGroup program_groups[] = { mRaygenProgGroup, mMissProgGroup, mHitgroupProgGroup };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 5;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    pipeline_link_options.overrideUsesMotionBlur = false;

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(
        mContext,
        &mPipelineCompileOptions,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        log,
        &sizeof_log,
        &mPipeline
    ));

}

void RTStateObject::createSBT()
{
    CUdeviceptr  raygen_record;
    const size_t raygen_record_size = sizeof(RayGenSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));

    RayGenSbtRecord rg_sbt;
    rg_sbt.data = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(mRaygenProgGroup, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(raygen_record),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr miss_record;
    size_t      miss_record_size = sizeof(MissSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
    MissSbtRecord ms_sbt;
    ms_sbt.data = { 0.3f, 0.1f, 0.2f };
    OPTIX_CHECK(optixSbtRecordPackHeader(mMissProgGroup, &ms_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(miss_record),
        &ms_sbt,
        miss_record_size,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr hitgroup_record;
    size_t      hitgroup_record_size = sizeof(HitGroupSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
    HitGroupSbtRecord hg_sbt;
    hg_sbt.data = { 1.5f };
    OPTIX_CHECK(optixSbtRecordPackHeader(mHitgroupProgGroup, &hg_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(hitgroup_record),
        &hg_sbt,
        hitgroup_record_size,
        cudaMemcpyHostToDevice
    ));

    mSbt.raygenRecord = raygen_record;
    mSbt.missRecordBase = miss_record;
    mSbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    mSbt.missRecordCount = 1;
    mSbt.hitgroupRecordBase = hitgroup_record;
    mSbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    mSbt.hitgroupRecordCount = 1;
}
