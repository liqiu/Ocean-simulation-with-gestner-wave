#pragma

#include "Wave.h"
#include "MeshKernel.h"

#include <optix.h>

#include <vector>
#include <memory>


class Mesh
{
public:

	void generateMesh(float t);
	void updateMesh(float t);
	void addWave(std::shared_ptr<Wave> pWave) { mpWaves.push_back(pWave); }
	void buidAccelerationStructure(OptixDeviceContext context);

	OptixTraversableHandle getTraversableHandle() { return mGasHandle; }
private:
	std::vector<std::shared_ptr<Wave>> mpWaves;

	uint16_t mSamplesX = 96;
	uint16_t mSamplesZ = 96;
	float mLength = 1.f;

	CUdeviceptr mdVertices;
	CUdeviceptr mdIndices;
	size_t mVerticesSize;
	size_t mIndicesSize;

	OptixTraversableHandle mGasHandle;
	CUdeviceptr            mdGasOutputBuffer;
};
