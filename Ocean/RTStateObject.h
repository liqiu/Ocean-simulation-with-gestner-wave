#pragma once

#include <optix.h>


class RTStateObject
{
public:

    void initialize();

    OptixDeviceContext getContext() { return mContext; }
    OptixPipeline getPipeline() { return mPipeline; }
    const OptixShaderBindingTable* getSbt() { return &mSbt; }

private:
    void createContext();
    void createPTXModule();
    void createProgramGroups();
    void createPipeline();
    void createSBT();

    OptixDeviceContext mContext;

    OptixModule mModule;

    OptixPipelineCompileOptions mPipelineCompileOptions;

    OptixProgramGroup mRaygenProgGroup;
    OptixProgramGroup mMissProgGroup;
    OptixProgramGroup mHitgroupProgGroup;

    OptixPipeline mPipeline;

    OptixShaderBindingTable mSbt;
};
