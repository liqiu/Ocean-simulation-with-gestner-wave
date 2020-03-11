#pragma


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

class WaveMesh
{
public:
	WaveMesh();
	~WaveMesh();

	void generateMesh(float t);
	void updateMesh(float t);
	void addWave(std::shared_ptr<Wave> pWave) { mWaves.push_back(*pWave); }
	void buildAccelerationStructure(OptixDeviceContext context);
	void updateAccelerationStructure(OptixDeviceContext context);

	OptixTraversableHandle getTraversableHandle() { return mGasHandle; }
	std::shared_ptr<sutil::MeshGroup> getMesh() { return mpMesh; }

	void setTransform(const sutil::Matrix4x4& transform) { mTransform = transform; }
	sutil::Matrix4x4 getTransform() const { return mTransform; }
private:
	std::vector<Wave> mWaves;
	CUdeviceptr mdWaves;

	uint16_t mSamplesX = 1920;
	uint16_t mSamplesZ = 1920;
	float mLength = 2000.f;

	MeshBuffer mMeshBuffer;

	OptixTraversableHandle mGasHandle = 0;
	CUdeviceptr mdGasOutputBuffer = 0;
	CUdeviceptr mdTempBufferGas = 0;

	std::shared_ptr<sutil::MeshGroup> mpMesh;
	sutil::Matrix4x4 mTransform;
};
