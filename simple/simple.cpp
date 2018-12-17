#include <vector>
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

using namespace ospcommon;

struct Particle {
	vec3f pos;
	int color_id;

	Particle(float x, float y, float z)
		: pos(vec3f{x, y, z}), color_id(0)
		{}
};

bool compute_divisor(int x, int &divisor);
// Compute an X x Y x Z grid to have num bricks,
// only gives a nice grid for numbers with even factors since
// we don't search for factors of the number, we just try dividing by two
vec3i compute_grid(int num);

void write_ppm(const std::string &file_name, const int width, const int height,
		const uint32_t *img);

int main(int argc, char **argv) {
	int provided = 0;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

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

	const vec3f cam_pos(0, 0, 14);
	const vec3f cam_up(0, 1, 0);
	const vec3f cam_at(0, 0, 0);
	const vec3f cam_dir = cam_at - cam_pos;

	// We want to use the same random seed on all procs, to
	// generate the same set of particles
	std::mt19937 rng(0);
	std::vector<Particle> atoms;
	const float radius = 0.25f;
	std::uniform_real_distribution<float> pos(-5.0 + radius, 5.0 - radius);

	// Randomly generate some spheres
	for (size_t i = 0; i < 50 * world_size; ++i) {
		atoms.push_back(Particle(pos(rng), pos(rng), pos(rng)));
	}

	const std::array<float, 3> atom_color = {
		static_cast<float>(rank) / world_size,
		static_cast<float>(rank) / world_size,
		static_cast<float>(rank) / world_size
	};

	// Make the OSPData which will refer to our particle and color data.
	// The OSP_DATA_SHARED_BUFFER flag tells OSPRay not to share our buffer,
	// instead of taking a copy.
	OSPData sphere_data = ospNewData(atoms.size() * sizeof(Particle), OSP_CHAR,
			atoms.data(), OSP_DATA_SHARED_BUFFER);
	ospCommit(sphere_data);
	OSPData color_data = ospNewData(1, OSP_FLOAT3,
			atom_color.data(), OSP_DATA_SHARED_BUFFER);
	ospCommit(color_data);

	// Create the sphere geometry that we'll use to represent our particles
	OSPGeometry spheres = ospNewGeometry("spheres");
	ospSetData(spheres, "spheres", sphere_data);
	ospSetData(spheres, "color", color_data);
	ospSet1f(spheres, "radius", radius);
	ospSet1i(spheres, "bytes_per_sphere", sizeof(Particle));
	ospSet1i(spheres, "offset_colorID", sizeof(osp::vec3f));
	ospCommit(spheres);

	// Create the model we'll place all our scene geometry into, representing
	// our owned piece of the world to render. We only own some sub-brick for
	// rendering, but have all the particles on the node for ghost zones.
	const vec3i grid = compute_grid(world_size);
	const vec3f brick_size = vec3f(10.0) / vec3f(grid);
    const vec3i brick_id(rank % grid.x, (rank / grid.x) % grid.y, rank / (grid.x * grid.y));
	const vec3f brick_lower = brick_size * vec3f(brick_id) - vec3f(5.0);
	const vec3f brick_upper = brick_lower + brick_size;

	OSPModel model = ospNewModel();
	ospAddGeometry(model, spheres);
	ospSet1i(model, "id", rank);
	// We need to clip off the ghost particles which we don't own
	// on this process, so we must override the region bounds
	ospSetVec3f(model, "region.lower", (osp::vec3f&)brick_lower);
	ospSetVec3f(model, "region.upper", (osp::vec3f&)brick_upper);
	ospCommit(model);

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
	OSPFrameBuffer framebuffer = ospNewFrameBuffer(img_size, OSP_FB_SRGBA, OSP_FB_COLOR);
	ospFrameBufferClear(framebuffer, OSP_FB_COLOR);

	// Render the image and save it out
	ospRenderFrame(framebuffer, renderer, OSP_FB_COLOR);

	if (rank == 0) {
		const uint32_t *img = static_cast<const uint32_t*>(ospMapFrameBuffer(framebuffer, OSP_FB_COLOR));
		write_ppm("simple_particles.ppm", img_size.x, img_size.y, img);
		std::cout << "Image saved to 'simple_particles.ppm'\n";
		ospUnmapFrameBuffer(img, framebuffer);
	}

	// Clean up all our objects
	ospRelease(framebuffer);
	ospRelease(renderer);
	ospRelease(camera);
	ospRelease(model);
	ospRelease(spheres);
	ospShutdown();

	MPI_Finalize();

	return 0;
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
vec3i compute_grid(int num) {
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

