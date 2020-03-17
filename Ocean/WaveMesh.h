#pragma once


#include "Wave.h"
#include "MeshKernel.h"

#include <sutil/Matrix.h>

#include <optix.h>

#include <vector>
#include <memory>


namespace sutil
{
	struct MeshGroup;
}


struct ProjectedGrid;


class WaveMesh
{
public:
	WaveMesh(int windowWidth, int windowHeight, int samplesPerPixel = 1);
	~WaveMesh();

	void generateMesh(float t);
	void updateMesh(float t);
	void addWave(std::shared_ptr<Wave> pWave) { mWaves.push_back(*pWave); }
	void buildAccelerationStructure(OptixDeviceContext context);
	void updateAccelerationStructure(OptixDeviceContext context);

	OptixTraversableHandle getTraversableHandle() { return mGasHandle; }
	std::shared_ptr<sutil::MeshGroup> getMesh() { return mpMesh; }

	void setTransform(const sutil::Matrix4x4& transform);
	sutil::Matrix4x4 getTransform() const { return mTransform; }

	void updateCamera(const float3& eye, const float3& U, const float3& V, const float3& W);
private:

	std::vector<Wave> mWaves;
	CUdeviceptr mdWaves;

	int mWindowWidth;
	int mWindowHeight;
	int mSamplesPerPixel;

	MeshBuffer mMeshBuffer;

	OptixTraversableHandle mGasHandle = 0;
	CUdeviceptr mdGasOutputBuffer = 0;
	CUdeviceptr mdTempBufferGas = 0;

	std::shared_ptr<sutil::MeshGroup> mpMesh;
	sutil::Matrix4x4 mTransform;

	std::shared_ptr<ProjectedGrid> mpProjectedGrid;

	int mNumVerts;
	int mNumTriangles;
};
