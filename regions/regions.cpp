#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <array>
#include <cstdio>
#include <mpi.h>
#include <ospray/ospray.h>
#include <ospray/ospcommon/vec.h>
#include <ospray/ospcommon/box.h>

using namespace ospcommon;

struct Particle {
	vec3f pos;
	int color_id;

	Particle(float x, float y, float z)
		: pos(vec3f{x, y, z}), color_id(0)
		{}
};

void write_ppm(const std::string &file_name, const int width, const int height,
		const uint32_t *img);
// Compute an X x Y x Z grid to have num bricks,
// only gives a nice grid for numbers with even factors since
// we don't search for factors of the number, we just try dividing by two
vec3i compute_grid(int num);

int main(int argc, char **argv) {
	int provided = 0;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	assert(provided == MPI_THREAD_MULTIPLE);

	int world_size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (ospLoadModule("mpi") != OSP_NO_ERROR) {
		throw std::runtime_error("Failed to load OSPRay MPI module");
	}

	OSPDevice device = ospNewDevice("mpi_distributed");
	ospDeviceSet1i(device, "masterRank", 0);
	ospDeviceSetStatusFunc(device, [](const char *msg) { std::cout << msg << "\n"; });
	ospDeviceCommit(device);
	ospSetCurrentDevice(device);

	// Setup a transfer function for the volume
	OSPTransferFunction transfer_fcn = ospNewTransferFunction("piecewise_linear");
	{
		const std::vector<vec3f> colors = {
			vec3f(0, 0, 0.56),
			vec3f(0, 0, 1),
			vec3f(0, 1, 1),
			vec3f(0.5, 1, 0.5),
			vec3f(1, 1, 0),
			vec3f(1, 0, 0),
			vec3f(0.5, 0, 0)
		};
		const std::vector<float> opacities = {0.05f, 0.05f};
		OSPData colors_data = ospNewData(colors.size(), OSP_FLOAT3, colors.data());
		ospCommit(colors_data);
		OSPData opacity_data = ospNewData(opacities.size(), OSP_FLOAT, opacities.data());
		ospCommit(opacity_data);

		ospSetData(transfer_fcn, "colors", colors_data);
		ospSetData(transfer_fcn, "opacities", opacity_data);
		ospSetVec2f(transfer_fcn, "valueRange", osp::vec2f{0, static_cast<float>(world_size - 1)});
		ospCommit(transfer_fcn);
	}

	// Setup our piece of the volume data, each rank has some brick of
	// volume data within the [0, 1] box
	OSPVolume volume = ospNewVolume("block_bricked_volume");
	const vec3i volume_dims(64);
	const vec3i grid = compute_grid(world_size);
	const vec3i brick_id(rank % grid.x,
			(rank / grid.x) % grid.y, rank / (grid.x * grid.y));

	// We use the grid_origin to translate the bricks to the right location
	// in the space.
	const vec3f grid_origin = vec3f(brick_id) * vec3f(volume_dims);

	ospSetString(volume, "voxelType", "uchar");
	ospSetVec3i(volume, "dimensions", (osp::vec3i&)volume_dims);
	ospSetVec3f(volume, "gridOrigin", (osp::vec3f&)grid_origin);
	ospSetObject(volume, "transferFunction", transfer_fcn);

	std::vector<unsigned char> volume_data(volume_dims.x * volume_dims.y * volume_dims.z,
			static_cast<unsigned char>(rank));
	ospSetRegion(volume, volume_data.data(), osp::vec3i{0, 0, 0}, (osp::vec3i&)volume_dims);
	ospCommit(volume);

	OSPModel model = ospNewModel();
	ospAddVolume(model, volume);

	// For correct compositing we must specify a list of regions that bound the
	// data owned by this rank. These region bounds will be used for sort-last
	// compositing when rendering.
	const box3f bounds(grid_origin, grid_origin + vec3f(volume_dims));
	OSPData region_data = ospNewData(2, OSP_FLOAT3, &bounds);
	ospSetData(model, "regions", region_data);
	ospCommit(model);

	// Position the camera based on the world bounds, which go from
	// [0, 0, 0] to the upper corner of the last brick
	const vec3f world_diagonal = vec3f((world_size - 1) % grid.x,
			((world_size - 1) / grid.x) % grid.y,
			(world_size - 1) / (grid.x * grid.y))
		* vec3f(volume_dims) + vec3f(volume_dims);

	const vec3f cam_pos = world_diagonal * vec3f(1.5);
	const vec3f cam_up(0, 1, 0);
	const vec3f cam_at = world_diagonal * vec3f(0.5);
	const vec3f cam_dir = cam_at - cam_pos;

	// Setup the camera we'll render the scene from
	const osp::vec2i img_size{1024, 1024};
	OSPCamera camera = ospNewCamera("perspective");
	ospSet1f(camera, "aspect", 1.0);
	ospSet3fv(camera, "pos", &cam_pos.x);
	ospSet3fv(camera, "up", &cam_up.x);
	ospSet3fv(camera, "dir", &cam_dir.x);
	ospCommit(camera);

	// For distributed rendering we must use the MPI raycaster
	OSPRenderer renderer = ospNewRenderer("mpi_raycast");
	// Setup the parameters for the renderer
	ospSet1i(renderer, "spp", 1);
	ospSet1f(renderer, "bgColor", 1.f);
	ospSetObject(renderer, "model", model);
	ospSetObject(renderer, "camera", camera);
	ospCommit(renderer);

	// Create a framebuffer to render the image too
	OSPFrameBuffer framebuffer = ospNewFrameBuffer(img_size, OSP_FB_SRGBA,
			OSP_FB_COLOR | OSP_FB_ACCUM);
	ospFrameBufferClear(framebuffer, OSP_FB_COLOR);

	// Render the image and save it out
	for (size_t i = 0; i < 16; ++i) {
		ospRenderFrame(framebuffer, renderer, OSP_FB_COLOR);
	}

	if (rank == 0) {
		const uint32_t *img = static_cast<const uint32_t*>(ospMapFrameBuffer(framebuffer, OSP_FB_COLOR));
		write_ppm("regions_sample.ppm", img_size.x, img_size.y, img);
		std::cout << "Image saved to 'regions_sample.ppm'\n";
		ospUnmapFrameBuffer(img, framebuffer);
	}

	// Clean up all our objects
	ospFreeFrameBuffer(framebuffer);
	ospRelease(renderer);
	ospRelease(camera);
	ospRelease(model);
	ospRelease(volume);

	MPI_Finalize();

	return 0;
}
void write_ppm(const std::string &file_name, const int width, const int height,
		const uint32_t *img)
{
	FILE *file = fopen(file_name.c_str(), "wb");
	if (!file) {
		throw std::runtime_error("Failed to open file for PPM output");
	}

	fprintf(file, "P6\n%i %i\n255\n", width, height);
	std::vector<uint8_t> out(3 * width, 0);
	for (int y = 0; y < height; ++y) {
		const uint8_t *in = reinterpret_cast<const uint8_t*>(&img[(height - 1 - y) * width]);
		for (int x = 0; x < width; ++x) {
			out[3 * x] = in[4 * x];
			out[3 * x + 1] = in[4 * x + 1];
			out[3 * x + 2] = in[4 * x + 2];
		}
		fwrite(out.data(), out.size(), sizeof(uint8_t), file);
	}
	fprintf(file, "\n");
	fclose(file);
}
bool compute_divisor(int x, int &divisor) {
	int upper_bound = std::sqrt(x);
	for (int i = 2; i <= upper_bound; ++i) {
		if (x % i == 0) {
			divisor = i;
			return true;
		}
	}
	return false;
}
vec3i compute_grid(int num){
	vec3i grid(1);
	int axis = 0;
	int divisor = 0;
	while (compute_divisor(num, divisor)) {
		grid[axis] *= divisor;
		num /= divisor;
		axis = (axis + 1) % 3;
	}
	if (num != 1) {
		grid[axis] *= num;
	}
	return grid;
}

