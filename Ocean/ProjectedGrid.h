#pragma once


#include "Plane.h"

#include <sutil/vec_math.h>
#include <sutil/Matrix.h>
#include <sutil/Camera.h>

#include <glm/vec4.hpp>

#include <vector_types.h>
#include <vector_functions.h>


struct ProjectedGrid
{
	void init();

	bool calculateRangeMatrix(glm::mat4& out_rangeMatrix);
	glm::vec4 calculateWorldPos(glm::vec2 uv, const glm::mat4& m);
	void calculateCorners(const glm::mat4& projectorMatrix);

	int samplesU;
	int samplesV;

	float elevation;
	float maxDisplacement = 15;

	float znear = 0.1f;
	float zfar = 3000.f;
	float minHeightProjector = 1;
	float waveMaxElevation;

	Plane upper_bound;
	Plane lower_bound;
	Plane plane;

	glm::vec3 up = glm::vec3(0, 1, 0);

	sutil::Camera* pRenderingCamera;
	sutil::Camera projecting_camera;

	float4 corners[4];
	float du, dv;
};