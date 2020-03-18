#include "ProjectedGrid.h"

#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>

#include <iostream>


static bool planeIntersectLine(glm::vec3& pout, const Plane& pp, const glm::vec3& pv1, const glm::vec3& pv2)
{
	glm::vec3 direction, normal;
	float dot, temp;

	normal = pp.normal;
	direction = pv2 - pv1;
	dot = glm::dot(normal, direction);
	if (abs(dot) < 1e-6) return false;
	temp = (pp.d + glm::dot(normal, pv1)) / dot;
	pout.x = pv1.x - temp * direction.x;
	pout.y = pv1.y - temp * direction.y;
	pout.z = pv1.z - temp * direction.z;
	return true;
}

#define log_xor || log_xor_helper() ||

struct log_xor_helper {
	bool value;
};

template<typename LEFT>
log_xor_helper& operator ||(const LEFT& left, log_xor_helper&xor) {
	xor .value = (bool)left;
	return xor;
}

template<typename RIGHT>
bool operator ||(const log_xor_helper&xor, const RIGHT& right) {
	return xor .value ^ (bool)right;
}


void ProjectedGrid::init()
{
	waveMaxElevation = elevation + maxDisplacement;

	upper_bound.normal = up;
	upper_bound.d = -glm::dot(up, glm::vec3(0, elevation + maxDisplacement, 0));

	lower_bound.normal = up;
	lower_bound.d = -glm::dot(up, glm::vec3(0, elevation - maxDisplacement, 0));

	plane.normal = up;
	plane.d = -glm::dot(up, glm::vec3(0, elevation, 0));
}

bool ProjectedGrid::calculateRangeMatrix(glm::mat4& out_rangeMatrix)
{
	float		x_min, y_min, x_max, y_max;
	glm::vec3 frustum[8], proj_points[24];		// frustum to check the camera against

	int n_points = 0;
	int cube[] = { 0,1,	0,2,	2,3,	1,3,
		0,4,	2,6,	3,7,	1,5,
		4,6,	4,5,	5,7,	6,7 };	// which frustum points are connected together?

	// transform frustum points to worldspace (should be done to the rendering_camera because it's the interesting one)

	glm::mat4 invViewProj = pRenderingCamera->getInvViewProj(znear, zfar);
	glm::vec4 tmp[24];
	tmp[0] = invViewProj * glm::vec4(-1, -1, -1, 1);
	tmp[1] = invViewProj * glm::vec4(+1, -1, -1, 1);
	tmp[2] = invViewProj * glm::vec4(-1, +1, -1, 1);
	tmp[3] = invViewProj * glm::vec4(+1, +1, -1, 1);
	tmp[4] = invViewProj * glm::vec4(-1, -1, +1, 1);
	tmp[5] = invViewProj * glm::vec4(+1, -1, +1, 1);
	tmp[6] = invViewProj * glm::vec4(-1, +1, +1, 1);
	tmp[7] = invViewProj * glm::vec4(+1, +1, +1, 1);

	//std::cout << "Frustum:" << std::endl;
	for (int i = 0; i < 8; i++) {
		frustum[i] = glm::vec3(tmp[i]) / tmp[i].w;
		//std::cout << frustum[i].x << "\t" << frustum[i].y << "\t" << frustum[i].z << std::endl;
	}

	// check intersections with upper_bound and lower_bound	
	for (int i = 0; i < 12; i++) {
		int src = cube[i * 2], dst = cube[i * 2 + 1];
		if ((upper_bound.normal.x * frustum[src].x + upper_bound.normal.y * frustum[src].y + upper_bound.normal.z * frustum[src].z + upper_bound.d * 1) / 
			(upper_bound.normal.x * frustum[dst].x + upper_bound.normal.y * frustum[dst].y + upper_bound.normal.z * frustum[dst].z + upper_bound.d * 1) < 0) {
			planeIntersectLine(proj_points[n_points++], upper_bound, frustum[src], frustum[dst]);
		}
		if ((lower_bound.normal.x * frustum[src].x + lower_bound.normal.y * frustum[src].y + lower_bound.normal.z * frustum[src].z + lower_bound.d * 1) /
			(lower_bound.normal.x * frustum[dst].x + lower_bound.normal.y * frustum[dst].y + lower_bound.normal.z * frustum[dst].z + lower_bound.d * 1) < 0) {
			planeIntersectLine(proj_points[n_points++], lower_bound, frustum[src], frustum[dst]);
		}
	}
	// check if any of the frustums vertices lie between the upper_bound and lower_bound planes
	{
		for (int i = 0; i < 8; i++) {
			if ((upper_bound.normal.x * frustum[i].x + upper_bound.normal.y * frustum[i].y + upper_bound.normal.z * frustum[i].z + upper_bound.d * 1) / 
				(lower_bound.normal.x * frustum[i].x + lower_bound.normal.y * frustum[i].y + lower_bound.normal.z * frustum[i].z + lower_bound.d * 1) < 0) {
				proj_points[n_points++] = frustum[i];
			}
		}
	}

	//
	// create the camera the grid will be projected from
	//
	projecting_camera = sutil::Camera(pRenderingCamera->eye(), pRenderingCamera->lookat(),
		pRenderingCamera->up(), pRenderingCamera->fovY(), pRenderingCamera->aspectRatio());
	// make sure the camera isn't too close to the plane
	float height_in_plane = (upper_bound.normal.x * projecting_camera.eye().x +
		upper_bound.normal.y * projecting_camera.eye().y +
		upper_bound.normal.z * projecting_camera.eye().z + upper_bound.d);

	bool keep_it_simple = false;
	bool underwater = false;

	if (height_in_plane < 0.0f) underwater = true;

	if (keep_it_simple)
	{
	}
	else
	{
		glm::vec3 aimpoint, aimpoint2;

		if (height_in_plane < waveMaxElevation)
		{
			if (underwater)
				projecting_camera.setEye(make_float3(projecting_camera.eye().x, projecting_camera.eye().y, projecting_camera.eye().z) +
					make_float3(lower_bound.normal.x, lower_bound.normal.y, lower_bound.normal.z) * waveMaxElevation - 2 * (height_in_plane - waveMaxElevation));
			else
				projecting_camera.setEye(make_float3(projecting_camera.eye().x, projecting_camera.eye().y, projecting_camera.eye().z) + 
					make_float3(lower_bound.normal.x, lower_bound.normal.y, lower_bound.normal.z) * waveMaxElevation - (height_in_plane - waveMaxElevation));
		}

		// aim the projector at the point where the camera view-vector intersects the plane
		// if the camera is aimed away from the plane, mirror it's view-vector against the plane
		float3 U, V, W;
		pRenderingCamera->UVWFrame(U, V, W);
		glm::vec3 renderingCameraPos = glm::vec3(pRenderingCamera->eye().x, pRenderingCamera->eye().y, pRenderingCamera->eye().z);
		if (glm::dot(lower_bound.normal, glm::vec3(W.x, W.y, W.z)) < 0.0f log_xor 
			(glm::dot(plane.normal, renderingCameraPos) + plane.d) < 0.0f)
		{
			planeIntersectLine(aimpoint, plane, renderingCameraPos, (renderingCameraPos + zfar * glm::vec3(W.x, W.y, W.z)));
		}
		else
		{
			glm::vec3 flipped;
			flipped = glm::vec3(W.x, W.y, W.z) - 2.f * up * glm::dot(glm::vec3(W.x, W.y, W.z), up);
			planeIntersectLine(aimpoint, plane, renderingCameraPos, (renderingCameraPos + zfar * flipped));
		}

		// force the point the camera is looking at in a plane, and have the projector look at it
		// works well against horizon, even when camera is looking upwards
		// doesn't work straight down/up
		float af = fabs(glm::dot(plane.normal, glm::normalize(glm::vec3(W.x, W.y, W.z))));
		//af = 1 - (1-af)*(1-af)*(1-af)*(1-af)*(1-af);
		//aimpoint2 = (rendering_camera->position + rendering_camera->zfar * rendering_camera->forward);
		aimpoint2 = renderingCameraPos + glm::vec3(W.x, W.y, W.z);
		aimpoint2 = aimpoint2 - up * (glm::dot(aimpoint2, up) + plane.d);

		// fade between aimpoint & aimpoint2 depending on view angle

		aimpoint = aimpoint * af + aimpoint2 * (1.0f - af);
		//aimpoint = aimpoint2;

		projecting_camera.setLookat(make_float3(aimpoint.x, aimpoint.y, aimpoint.z));
	}



	//printf("n_points %i\n", n_points);
	{
		for (int i = 0; i < n_points; i++) {
			// project the point onto the surface plane
			proj_points[i] = proj_points[i] - up * (glm::dot(proj_points[i], up) + plane.d);
			//printf("%f  %f  %f\n", proj_points[i].x, proj_points[i].y, proj_points[i].z);
		}
	}

	{
		glm::mat4 viewMat = projecting_camera.getView();
		glm::mat4 projMat = projecting_camera.getProj(znear, zfar);
		for (int i = 0; i < n_points; i++) {
			tmp[i] = viewMat * glm::vec4(proj_points[i], 1);
			//printf("%f  %f  %f\n", proj_points[i].x, proj_points[i].y, proj_points[i].z);
			tmp[i] = projMat * tmp[i];
			proj_points[i] = tmp[i] / tmp[i].w;
		}
	}

	// debughonk

	/*	for(int i=0; i<n_points; i++){
	sprintf( debugdata, "%s%f  %f  %f\n",debugdata,proj_points[i].x,proj_points[i].y,proj_points[i].z);
	}*/

	// get max/min x & y-values to determine how big the "projection window" must be
	if (n_points > 0) {
		x_min = proj_points[0].x;
		x_max = proj_points[0].x;
		y_min = proj_points[0].y;
		y_max = proj_points[0].y;
		for (int i = 1; i < n_points; i++) {
			if (proj_points[i].x > x_max) x_max = proj_points[i].x;
			if (proj_points[i].x < x_min) x_min = proj_points[i].x;
			if (proj_points[i].y > y_max) y_max = proj_points[i].y;
			if (proj_points[i].y < y_min) y_min = proj_points[i].y;
		}


		//printf("x = [%f..%f] y = [%f..%f]\n", x_min, x_max, y_min, y_max);

		//printf("height_in_plane: %f\n", height_in_plane);

		//sprintf( debugdata,	"%slimit_y_upper = %f\n",debugdata,limit_y_upper);
		//		sprintf( debugdata, "%sy1 = [%f] y2 = [%f]\n",debugdata,y1,y2);

		// build the packing matrix that spreads the grid across the "projection window"
		glm::mat4 pack(x_max - x_min, 0, 0, x_min,
			0, y_max - y_min, 0, y_min,
			0, 0, 1, 0,
			0, 0, 0, 1);
		pack = glm::transpose(pack);
		out_rangeMatrix = projecting_camera.getInvViewProj(znear, zfar) * pack;

		return true;
	}
	return false;
}

glm::vec4 ProjectedGrid::calculateWorldPos(glm::vec2 uv, const glm::mat4& m)
{
	glm::vec4 origin(uv.x, uv.y, -1.f, 1.f);
	glm::vec4 direction(uv.x, uv.y, 1.f, 1.f);

	origin = m * origin;
	direction = m * direction;
	direction -= origin;

	float t = (origin.w * (-plane.d) - origin.y) / (direction.y - direction.w * (-plane.d));

	return origin + direction * t;
}


void ProjectedGrid::calculateCorners(const glm::mat4& projectorMatrix)
{
	glm::vec4 c[4];
	c[0] = calculateWorldPos(glm::vec2(0.f, 0.f), projectorMatrix);
	c[1] = calculateWorldPos(glm::vec2(1.f, 0.f), projectorMatrix);
	c[2] = calculateWorldPos(glm::vec2(0.f, 1.f), projectorMatrix);
	c[3] = calculateWorldPos(glm::vec2(1.f, 1.f), projectorMatrix);

	glm::vec4 tmp[4];
	for (size_t i = 0; i < 4; i++)
	{
		tmp[i] = c[i] / c[i].w;
		corners[i] = make_float4(c[i].x, c[i].y, c[i].z, c[i].w);
	}

	du = 1.f / (samplesU - 1);
	dv = 1.f / (samplesV - 1);
}
