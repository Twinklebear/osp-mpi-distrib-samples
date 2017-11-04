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

void ospray_rendering_work(MPI_Comm partition_comm);
void write_ppm(const std::string &file_name, const int width, const int height,
		const uint32_t *img);

int main(int argc, char **argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); 
	MPI_Comm_size(MPI_COMM_WORLD, &world_size); 
	if (ospLoadModule("mpi") != OSP_NO_ERROR) {
		throw std::runtime_error("Failed to load OSPRay MPI module");
	}

	// Partition the world so we have 1 OSPRay rank on each node
	MPI_Comm node_comm;
	MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
			MPI_INFO_NULL, &node_comm);
	int node_size, node_rank;
	MPI_Comm_rank(node_comm, &node_rank);
	MPI_Comm_size(node_comm, &node_size);

	int is_ospray_rank = 0;
	if (node_size == world_size) {
		if (world_rank == 0) {
			std::cout << "Single node run detected, configuring to run OSPRay"
				<< " on even numbered ranks\n";
		}
		is_ospray_rank = node_rank % 2 == 0 ? 1 : 0;
	} else {
		if (world_rank == 0) {
			std::cout << "Multi-node run detected with " << node_size << "ranks per-node,"
				<< " configuring to run OSPRay on one rank per-node\n";
		}
		is_ospray_rank = node_rank == 0 ? 1 : 0;
	}

	// The first rank on each node will be the OSPRay rank
	MPI_Comm partition_comm;
	MPI_Comm_split(MPI_COMM_WORLD, is_ospray_rank, world_rank, &partition_comm);
	if (is_ospray_rank) {
		ospray_rendering_work(partition_comm);
	}
	
	MPI_Comm_free(&partition_comm);
	MPI_Comm_free(&node_comm);

	MPI_Finalize();

	return 0;
}
void ospray_rendering_work(MPI_Comm partition_comm) {
	int world_size, world_rank;
	MPI_Comm_size(partition_comm, &world_size);
	MPI_Comm_rank(partition_comm, &world_rank);

	if (world_rank == 0) {
		std::cout << "OSPRay partition has " << world_size << " ranks\n";
	}
	{
		int global_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &global_rank); 
		std::cout << "global_rank " << global_rank << " is " << world_rank
			<< " in OSPRay partition" << std::endl;
	}

	OSPDevice device = ospNewDevice("mpi_distributed");
	ospDeviceSet1i(device, "masterRank", 0);
	ospDeviceSetVoidPtr(device, "world_communicator", static_cast<void*>(&partition_comm));
	ospDeviceSetStatusFunc(device, [](const char *msg) { std::cout << msg << "\n"; });
	ospDeviceCommit(device);
	ospSetCurrentDevice(device);

	const vec3f cam_pos(0, 0, 9);
	const vec3f cam_up(0, 1, 0);
	const vec3f cam_at(0, 0, 0);
	const vec3f cam_dir = cam_at - cam_pos;

	std::random_device rd;
	std::mt19937 rng(rd());
	std::vector<Particle> atoms;
	std::uniform_real_distribution<float> pos(-3.0, 3.0);

	// Randomly generate some spheres
	for (size_t i = 0; i < 50; ++i) {
		atoms.push_back(Particle(pos(rng), pos(rng), pos(rng)));
	}

	const std::array<float, 3> atom_color = {
		static_cast<float>(world_rank) / world_size,
		static_cast<float>(world_rank) / world_size,
		static_cast<float>(world_rank) / world_size
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
	ospSet3fv(camera, "pos", &cam_pos.x);
	ospSet3fv(camera, "up", &cam_up.x);
	ospSet3fv(camera, "dir", &cam_dir.x);
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

	if (world_rank == 0) {
		const uint32_t *img = static_cast<const uint32_t*>(ospMapFrameBuffer(framebuffer, OSP_FB_COLOR));
		write_ppm("partition_particles.ppm", img_size.x, img_size.y, img);
		std::cout << "Image saved to 'partition_particles.ppm'\n";
		ospUnmapFrameBuffer(img, framebuffer);
	}

	// Clean up all our objects
	ospFreeFrameBuffer(framebuffer);
	ospRelease(renderer);
	ospRelease(camera);
	ospRelease(model);
	ospRelease(spheres);
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

