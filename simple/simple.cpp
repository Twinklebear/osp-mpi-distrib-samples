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

struct Particle {
	osp::vec3f pos;
	int color_id;

	Particle(float x, float y, float z)
		: pos(osp::vec3f{x, y, z}), color_id(0)
	{}
};

void write_ppm(const std::string &file_name, const int width, const int height,
		const uint32_t *img);

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

	const std::array<float, 3> cam_pos = {0.0, 0.0, 9.0};
	const std::array<float, 3> cam_up = {0.0, 1.0, 0.0};
	const std::array<float, 3> cam_at = {0.0, 0.0, 0.0};
	std::array<float, 3> cam_dir;
	for (size_t i = 0; i < 3; ++i) {
		cam_dir[i] = cam_at[i] - cam_pos[i];
	}

	std::random_device rd;
	std::mt19937 rng(rd());
	std::vector<Particle> atoms;
	std::uniform_real_distribution<float> pos(-3.0, 3.0);

	// Randomly generate some spheres
	for (size_t i = 0; i < 50; ++i) {
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

	// For distributed rendering we must use the MPI raycaster
	OSPRenderer renderer = ospNewRenderer("mpi_raycast");

	// Create the sphere geometry that we'll use to represent our particles
	OSPGeometry spheres = ospNewGeometry("spheres");
	ospSetData(spheres, "spheres", sphere_data);
	ospSetData(spheres, "color", color_data);
	ospSet1f(spheres, "radius", 0.25f);
	ospSet1i(spheres, "bytes_per_sphere", sizeof(Particle));
	ospSet1i(spheres, "offset_colorID", sizeof(osp::vec3f));
	ospCommit(spheres);

	// Create the model we'll place all our scene geometry into, representing
	// the world to be rendered. If we don't need sort-last compositing to be
	// performed (i.e. all our objects are opaque), we don't need to specify
	// any regions to the model.
	OSPModel model = ospNewModel();
	ospAddGeometry(model, spheres);
	ospCommit(model);

	// Setup the camera we'll render the scene from
	const osp::vec2i img_size{1024, 1024};
	OSPCamera camera = ospNewCamera("perspective");
	ospSet1f(camera, "aspect", 1.0);
	ospSet3fv(camera, "pos", cam_pos.data());
	ospSet3fv(camera, "up", cam_up.data());
	ospSet3fv(camera, "dir", cam_dir.data());
	ospCommit(camera);

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
	ospFreeFrameBuffer(framebuffer);
	ospRelease(renderer);
	ospRelease(camera);
	ospRelease(model);
	ospRelease(spheres);

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

