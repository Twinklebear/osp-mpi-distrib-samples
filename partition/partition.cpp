#include <unistd.h>
#include <vector>
#include <cassert>
#include <iostream>
#include <cstring>
#include <cmath>
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

	Particle() : pos(vec3f{0, 0, 0}), color_id(0) {}
	Particle(float x, float y, float z, int color_id)
		: pos(vec3f{x, y, z}), color_id(color_id)
	{}
};

void ospray_rendering_work(MPI_Comm partition_comm, std::vector<Particle> &collected_particles,
		const int node_size);
void write_ppm(const std::string &file_name, const int width, const int height,
		const uint32_t *img);
vec3f hsv_to_rgb(const float hue, const float sat, const float val) {
	const float c = val * sat;
	const int h_prime = static_cast<int>(hue / 60.0);
	const float x = c * (1.0 - std::abs(h_prime % 2 - 1.0));
	vec3f rgb{0, 0, 0};
	if (h_prime >= 0 && h_prime <= 1) {
		rgb.x = c;
		rgb.y = x;
	} else if (h_prime > 1 && h_prime <= 2) {
		rgb.x = x;
		rgb.y = c;
	} else if (h_prime > 2 && h_prime <= 3) {
		rgb.y = c;
		rgb.z = x;
	} else if (h_prime > 3 && h_prime <= 4) {
		rgb.y = x;
		rgb.z = c;
	} else if (h_prime > 4 && h_prime <= 5) {
		rgb.x = x;
		rgb.z = c;
	} else if (h_prime > 5 && h_prime < 6) {
		rgb.x = c;
		rgb.z = x;
	}
	const float m = val - c;
	return rgb + vec3f{m, m, m};
}

int main(int argc, char **argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); 
	MPI_Comm_size(MPI_COMM_WORLD, &world_size); 
	if (ospLoadModule("mpi") != OSP_NO_ERROR) {
		throw std::runtime_error("Failed to load OSPRay MPI module");
	}

	// Determine how many ranks we have on each node, so we can assign one
	// to be the OSPRay rank
	MPI_Comm node_comm;
	MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
			MPI_INFO_NULL, &node_comm);
	int node_size, node_rank;
	MPI_Comm_rank(node_comm, &node_rank);
	MPI_Comm_size(node_comm, &node_size);

	std::random_device rd;
	std::mt19937 rng(rd());
	std::vector<Particle> atoms;
	std::uniform_real_distribution<float> pos(-3.0, 3.0);
	const size_t atoms_per_rank = 50;
	// Randomly generate some spheres on each rank
	for (size_t i = 0; i < atoms_per_rank; ++i) {
		atoms.push_back(Particle(pos(rng), pos(rng), pos(rng), node_rank));
	}

	// The first rank on each node will be the OSPRay rank
	int is_ospray_rank = 0;
	if (world_rank == 0) {
		std::cout << "App run with " << node_size << " ranks per-node,"
			<< " configuring to run OSPRay on one rank per-node\n";
	}
	is_ospray_rank = node_rank == 0 ? 1 : 0;

	// Collect all particles to the rank responsible for rendering with OSPRay.
	// Here node_rank 0 collects data from the other nodes on the rank
	atoms.resize(node_size * atoms_per_rank);
	for (int i = 1; i < node_size; ++i) {
		if (is_ospray_rank) {
			MPI_Recv(&atoms[i * atoms_per_rank], atoms_per_rank * sizeof(Particle),
					MPI_BYTE, i, 0, node_comm, MPI_STATUS_IGNORE);

		} else {
			MPI_Send(&atoms[0], atoms_per_rank * sizeof(Particle),
					MPI_BYTE, 0, 0, node_comm);
		}
	}

	// Partition the world so we have 1 OSPRay rank on each node
	MPI_Comm partition_comm;
	MPI_Comm_split(MPI_COMM_WORLD, is_ospray_rank, world_rank, &partition_comm);
	if (is_ospray_rank) {
		ospray_rendering_work(partition_comm, atoms, node_size);
	}
	
	MPI_Comm_free(&partition_comm);
	MPI_Comm_free(&node_comm);

	MPI_Finalize();

	return 0;
}
void ospray_rendering_work(MPI_Comm partition_comm, std::vector<Particle> &collected_particles,
		const int node_size)
{
	int world_size, world_rank;
	MPI_Comm_size(partition_comm, &world_size);
	MPI_Comm_rank(partition_comm, &world_rank);

	if (world_rank == 0) {
		std::cout << "OSPRay partition has " << world_size << " ranks\n";
	}
	{
		char host_name[1024] = {0};
		gethostname(host_name, 1023);
		int global_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &global_rank); 
		std::cout << "global_rank " << global_rank << " on host '"
			<< host_name << "' is " << world_rank
			<< " in OSPRay partition" << std::endl;
	}

	OSPDevice device = ospNewDevice("mpi_distributed");
	ospDeviceSet1i(device, "masterRank", 0);
	ospDeviceSetVoidPtr(device, "worldCommunicator", static_cast<void*>(&partition_comm));
	ospDeviceSetStatusFunc(device, [](const char *msg) { std::cout << msg << "\n"; });
	ospDeviceCommit(device);
	ospSetCurrentDevice(device);

	const vec3f cam_pos(0, 0, 9);
	const vec3f cam_up(0, 1, 0);
	const vec3f cam_at(0, 0, 0);
	const vec3f cam_dir = cam_at - cam_pos;

	// Generate color with hue based on our OSPRay world rank and value
	// based on the node rank which sent us the data (the particle's color id) 
	std::vector<vec3f> atom_colors;
	const float hue = 360.0 * static_cast<float>(world_rank) / world_size;
	for (size_t i = 0; i < node_size; ++i) {
		atom_colors.push_back(hsv_to_rgb(hue, 1, static_cast<float>(i + 1) / node_size));
	}

	// Make the OSPData which will refer to our particle and color data.
	// The OSP_DATA_SHARED_BUFFER flag tells OSPRay not to share our buffer,
	// instead of taking a copy.
	OSPData sphere_data = ospNewData(collected_particles.size() * sizeof(Particle), OSP_CHAR,
			collected_particles.data(), OSP_DATA_SHARED_BUFFER);
	ospCommit(sphere_data);
	OSPData color_data = ospNewData(atom_colors.size(), OSP_FLOAT3,
			atom_colors.data(), OSP_DATA_SHARED_BUFFER);
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

