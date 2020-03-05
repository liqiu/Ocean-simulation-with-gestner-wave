#pragma

#include "Wave.h"
#include "MeshKernel.h"

#include <optix.h>

#include <vector>
#include <memory>


class Mesh
{
public:
	~Mesh();

	void generateMesh(float t);
	void updateMesh(float t);
	void addWave(std::shared_ptr<Wave> pWave) { mWaves.push_back(*pWave); }
	void buildAccelerationStructure(OptixDeviceContext context);
	void updateAccelerationStructure(OptixDeviceContext context);

	OptixTraversableHandle getTraversableHandle() { return mGasHandle; }
private:
	std::vector<Wave> mWaves;
	CUdeviceptr mdWaves;

	uint16_t mSamplesX = 96;
	uint16_t mSamplesZ = 96;
	float mLength = 1.f;

	CUdeviceptr mdVertices;
	CUdeviceptr mdIndices;

	OptixTraversableHandle mGasHandle = 0;
	CUdeviceptr mdGasOutputBuffer = 0;
	CUdeviceptr mdTempBufferGas = 0;
};
