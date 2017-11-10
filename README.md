# OSPRay MPI Distributed Device Mini-samples

Small set of samples for the HPC DevCon 2017 hands on with
the OSPRay distributed device. For more advanced examples
see the [mpi module apps directory](https://github.com/ospray/ospray/tree/master/modules/mpi/apps).

## Building

First, you'll need to grab the dependences for the application, and make sure
MPI is installed on the machine.

- [OSPRay](https://github.com/ospray/ospray/releases)
- [Embree](https://github.com/embree/embree/releases)
- [TBB](https://github.com/01org/tbb/releases)

On Windows you can use the [Windows Subsystem for Linux](https://msdn.microsoft.com/en-us/commandline/wsl/install-win10?f=255&MSPPError=-2147217396) to install an Ubuntu system on your
machine. I recommend building OSPRay from source to make sure it's linked against
the MPI installed on your machine. Follow the instructions for compiling
OSPRay with the [MPI module enabled](https://github.com/ospray/OSPRay#parallel-rendering-with-mpi) first.

Once you've unzipped the dependencies and built OSPRay you can make a build directory
in the repo and run cmake and make the samples. If you built OSPRay you'll also
need to point it to your Embree and TBB libraries.

```
mkdir build
cd build
cmake .. -Dospray_DIR=<path to ospray>/lib/cmake/ospray-<version> \
	-Dembree_DIR=<path to embree> \
	-DTBB_ROOT=<path to TBB>
make
```

## Running

Make sure Embree and TBB are in your `LD_LIBRARY_PATH`, either manually
or by sourcing the `embree-vars.sh` and `tbbvars.sh` scripts included with
the libraries. Then you can run the samples with MPI and view the output images.

```
mpirun -np <N> ./simple/simple
mpirun -np <N> ./regions/regions
```

Running the simple example will render some particle data, with 4 ranks
you should see something like this:
![simple example, 4 ranks](https://i.imgur.com/3QaakE6.png)

Running the regions example will render mixed particle and volume data,
with 4 ranks you should something like this:

![regions example, 4 ranks](https://i.imgur.com/F1CxiSm.png)

