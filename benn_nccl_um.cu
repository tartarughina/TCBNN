//
// File: benn_nccl_um.cu
// A ResNet-18 BNN for ImageNet on multi-nodes GPU cluster through NCCL and MPI
// Author: Riccardo Strina, Politecnico di Milano, Italy
// Taking inspiration from the work of Ang Li, Scientist, Pacific Northwest
// National Laboratory(PNNL), U.S.

#include "data.h"
#include "kernel.cuh"
#include "param.h"
#include "utility.h"
#include <assert.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <mpi.h>
#include <nccl.h>
#include <stdio.h>
#include <string>
#include <sys/time.h>
#include <vector>

using namespace cooperative_groups;
using namespace std;

#define OVERSUB_FACTOR_1 1.5
#define OVERSUB_FACTOR_2 2.0

#define TO_BYTE(X) static_cast<size_t>(X) * 1024 * 1024

#ifdef NEWFMT

__global__ void resnet128(InConv128LayerParam *bconv1,
                          Conv128LayerParam *l1b1c1, Conv128LayerParam *l1b1c2,
                          Conv128LayerParam *l1b2c1, Conv128LayerParam *l1b2c2,
                          Conv128LayerParam *l2b1c1, Conv128LayerParam *l2b1c2,
                          Conv128LayerParam *l2b2c1, Conv128LayerParam *l2b2c2,
                          Conv128LayerParam *l3b1c1, Conv128LayerParam *l3b1c2,
                          Conv128LayerParam *l3b2c1, Conv128LayerParam *l3b2c2,
                          Conv128LayerParam *l4b1c1, Conv128LayerParam *l4b1c2,
                          Conv128LayerParam *l4b2c1, Conv128LayerParam *l4b2c2,
                          Fc128LayerParam *bfc1, Out128LayerParam *bout) {
  // SET_KERNEL_TIMER;
  grid_group grid = this_grid();
  //========= Conv1 ============
  InConv128LayerFMT(bconv1);
  grid.sync();
  // TICK_KERNEL_TIMER(bconv1);
  //========= L1B1 ============
  Conv128LayerFMT(l1b1c1);
  grid.sync();
  // TICK_KERNEL_TIMER(l1b1c1);
  Conv128LayerFMT(l1b1c2);
  grid.sync();
  // TICK_KERNEL_TIMER(l1b1c2);
  //========= L1B2 ============
  Conv128LayerFMT(l1b2c1);
  grid.sync();
  // TICK_KERNEL_TIMER(l1b2c1);
  Conv128LayerFMT(l1b2c2);
  grid.sync();
  // TICK_KERNEL_TIMER(l1b2c2);
  //========= L2B1 ============
  Conv128LayerFMT(l2b1c1);
  grid.sync();
  // TICK_KERNEL_TIMER(l2b1c1);
  Conv128LayerFMT(l2b1c2);
  grid.sync();
  // TICK_KERNEL_TIMER(l2b1c2);
  //========= L2B2 ============
  Conv128LayerFMT(l2b2c1);
  grid.sync();
  // TICK_KERNEL_TIMER(l2b2c1);
  Conv128LayerFMT(l2b2c2);
  grid.sync();
  // TICK_KERNEL_TIMER(l2b2c2);
  //========= L3B1 ============
  Conv128LayerFMT(l3b1c1);
  grid.sync();
  // TICK_KERNEL_TIMER(l3b1c1);
  Conv128LayerFMT(l3b1c2);
  grid.sync();
  // TICK_KERNEL_TIMER(l3b1c2);
  //========= L3B2 ============
  Conv128LayerFMT(l3b2c1);
  grid.sync();
  // TICK_KERNEL_TIMER(l3b2c1);
  Conv128LayerFMT(l3b2c2);
  grid.sync();
  // TICK_KERNEL_TIMER(l3b2c2);
  //========= L4B1 ============
  Conv128LayerFMT(l4b1c1);
  grid.sync();
  // TICK_KERNEL_TIMER(l4b1c1);
  Conv128LayerFMT(l4b1c2);
  grid.sync();
  // TICK_KERNEL_TIMER(l4b1c2);
  //========= L4B2 ============
  Conv128LayerFMT(l4b2c1);
  grid.sync();
  // TICK_KERNEL_TIMER(l4b2c1);
  Conv128LayerFMT(l4b2c2);
  grid.sync();
  // TICK_KERNEL_TIMER(l4b2c2);
  //========= Fc1 ============
  Fc128LayerFMT(bfc1);
  grid.sync();
  // TICK_KERNEL_TIMER(bfc1);
  //========== Output ===========
  Out128LayerFMT(bout);
  // grid.sync();
  // TICK_KERNEL_TIMER(bout);
}

#else

__global__ void resnet128(InConv128LayerParam *bconv1,
                          Conv128LayerParam *l1b1c1, Conv128LayerParam *l1b1c2,
                          Conv128LayerParam *l1b2c1, Conv128LayerParam *l1b2c2,
                          Conv128LayerParam *l2b1c1, Conv128LayerParam *l2b1c2,
                          Conv128LayerParam *l2b2c1, Conv128LayerParam *l2b2c2,
                          Conv128LayerParam *l3b1c1, Conv128LayerParam *l3b1c2,
                          Conv128LayerParam *l3b2c1, Conv128LayerParam *l3b2c2,
                          Conv128LayerParam *l4b1c1, Conv128LayerParam *l4b1c2,
                          Conv128LayerParam *l4b2c1, Conv128LayerParam *l4b2c2,
                          Fc128LayerParam *bfc1, Out128LayerParam *bout) {
  grid_group grid = this_grid();
  //========= Conv1 ============
  InConv128Layer(bconv1);
  grid.sync();
  //========= L1B1 ============
  Conv128Layer(l1b1c1);
  grid.sync();
  Conv128Layer(l1b1c2);
  grid.sync();
  //========= L1B2 ============
  Conv128Layer(l1b2c1);
  grid.sync();
  Conv128Layer(l1b2c2);
  grid.sync();
  //========= L2B1 ============
  Conv128Layer(l2b1c1);
  grid.sync();
  Conv128Layer(l2b1c2);
  grid.sync();
  //========= L2B2 ============
  Conv128Layer(l2b2c1);
  grid.sync();
  Conv128Layer(l2b2c2);
  grid.sync();
  //========= L3B1 ============
  Conv128Layer(l3b1c1);
  grid.sync();
  Conv128Layer(l3b1c2);
  grid.sync();
  //========= L3B2 ============
  Conv128Layer(l3b2c1);
  grid.sync();
  Conv128Layer(l3b2c2);
  grid.sync();
  //========= L4B1 ============
  Conv128Layer(l4b1c1);
  grid.sync();
  Conv128Layer(l4b1c2);
  grid.sync();
  //========= L4B2 ============
  Conv128Layer(l4b2c1);
  grid.sync();
  Conv128Layer(l4b2c2);
  grid.sync();
  //========= Fc1 ============
  Fc128Layer(bfc1);
  grid.sync();
  //========== Output ===========
  Out128Layer(bout);
}

#endif

static void usage(const char *pname) {

  const char *bname = rindex(pname, '/');
  if (!bname) {
    bname = pname;
  } else {
    bname++;
  }

  fprintf(stdout,
          "Usage: %s [options]\n"
          "options:\n"
          "\t-u|--unified_mem\n"
          "\t\tUse unified memory for data arrays. (default: false) \n"
          "\t-t|--um_tuning\n"
          "\t\tEnable unified memory tuning. (default: false) \n"
          "\t-b|--batch\n"
          "\t\tSet batch size. (default: 64) \n"
          "\t-o|--oversub\n"
          "\t\tEnable oversubscribing GPUs. (default: 0, options: 1->1.5x "
          "2->2x) \n",
          // "\t-v|--verbose\n"
          // "\t\tEnable verbose logs. (default: false)\n",
          bname);
  exit(EXIT_SUCCESS);
}

int main(int argc, char *argv[]) {
  int n_gpu, i_gpu, rank;
  bool unified_mem = false;
  bool um_tuning = false;
  unsigned batch = 64;
  int oversub = 0;
  void *oversub_ptr = nullptr;
  // bool verbose = false;

  MPI_Comm local_comm;

  CHECK_MPI(MPI_Init(&argc, &argv));
  CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &n_gpu));

  static struct option long_options[] = {{"unified_mem", no_argument, 0, 'u'},
                                         {"um_tuning", no_argument, 0, 't'},
                                         // {"verbose", no_argument, 0, 'v'},
                                         {"oversub", required_argument, 0, 'o'},
                                         {"batch", required_argument, 0, 'b'},
                                         {"help", no_argument, 0, 'h'},
                                         {0, 0, 0, 0}};

  while (1) {
    int option_index = 0;
    int ch = getopt_long(argc, argv, "utb:o:h", long_options, &option_index);
    if (ch == -1)
      break;

    switch (ch) {
    case 0:
      break;
    case 'u':
      unified_mem = true;
      break;
    case 't':
      um_tuning = true;
      break;
    case 'b':
      batch = atoi(optarg);
      break;
    // case 'v':
    //   verbose = true;
    //   break;
    case 'o':
      oversub = atoi(optarg);
      if (oversub != 1 && oversub != 2) {
        fprintf(stderr, "Invalid value for oversub: %d\n", oversub);
        exit(EXIT_FAILURE);
      }
      break;
    case 'h':
      usage(argv[0]);
      break;
    case '?':
      exit(EXIT_FAILURE);
    default:
      fprintf(stderr, "unknown option: %c\n", ch);
      usage(argv[0]);
      exit(EXIT_FAILURE);
    }
  }

  CHECK_MPI(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                MPI_INFO_NULL, &local_comm));
  CHECK_MPI(MPI_Comm_rank(local_comm, &i_gpu));

  CUDA_SAFE_CALL(cudaSetDevice(i_gpu));

  if (oversub) {
    if (!unified_mem) {
      fprintf(stderr, "Oversubscribing GPUs requires unified memory\n");
      exit(-1);
    }

    size_t buffer_size, free_mem, total_mem;
    double factor;

    CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem, &total_mem));

    if (oversub == 1) {
      factor = OVERSUB_FACTOR_1;
    } else if (oversub == 2) {
      factor = OVERSUB_FACTOR_2;
    } else {
      fprintf(stderr, "Invalid oversub option: %d\n", oversub);
      exit(EXIT_FAILURE);
    }

    switch (batch) {
    case 1024:
      buffer_size = free_mem - (TO_BYTE(8300)) / factor;
      break;
    case 2048:
      buffer_size = free_mem - (TO_BYTE(15200)) / factor;
      break;
    case 4096:
      buffer_size = free_mem - (TO_BYTE(29000)) / factor;
      break;
    default:
      fprintf(stderr, "Unsupported batch size for oversub\n");
      exit(-1);
    }

    SAFE_ALOC_GPU(oversub_ptr, buffer_size);
  }

  vector<float> init_times;
  vector<float> comp_times;
  vector<float> comm_times;

  ncclUniqueId id;

  ncclComm_t comm;

  if (rank == 0) {
    ncclGetUniqueId(&id);
  }

  CHECK_MPI(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  CHECK_NCCL(ncclCommInitRank(&comm, n_gpu, id, i_gpu));

  MPI_Barrier(MPI_COMM_WORLD);
  cudaEvent_t init_start, init_stop, comp_start, comp_stop, comm_start,
      comm_stop;

  CUDA_SAFE_CALL(cudaEventCreate(&init_start));
  CUDA_SAFE_CALL(cudaEventCreate(&init_stop));
  CUDA_SAFE_CALL(cudaEventCreate(&comp_start));
  CUDA_SAFE_CALL(cudaEventCreate(&comp_stop));
  CUDA_SAFE_CALL(cudaEventCreate(&comm_start));
  CUDA_SAFE_CALL(cudaEventCreate(&comm_stop));

  int dev = i_gpu;
  const unsigned output_size = 1000;
  const unsigned image_height = 224;
  const unsigned image_width = 224;
  const unsigned image_channel = 3;

  //=============== Get Input and Label =================
  float *images = nullptr;
  unsigned *image_labels = nullptr;

  CUDA_SAFE_CALL(cudaEventRecord(init_start));

  if (unified_mem) {
    SAFE_ALOC_UM(images, batch * image_height * image_width * image_channel *
                             sizeof(float));
    SAFE_ALOC_UM(image_labels, batch * sizeof(unsigned));

    if (um_tuning) {
      cudaMemAdvise(images,
                    batch * image_height * image_width * image_channel *
                        sizeof(float),
                    cudaMemAdviseSetPreferredLocation, dev);
      cudaMemAdvise(images,
                    batch * image_height * image_width * image_channel *
                        sizeof(float),
                    cudaMemAdviseSetReadMostly, dev);
      cudaMemAdvise(image_labels, batch * sizeof(unsigned),
                    cudaMemAdviseSetPreferredLocation, dev);
      cudaMemAdvise(image_labels, batch * sizeof(unsigned),
                    cudaMemAdviseSetReadMostly, dev);
    }
  } else {
    images = (float *)malloc(batch * image_height * image_width *
                             image_channel * sizeof(float));
    image_labels = (unsigned *)malloc(batch * sizeof(unsigned));
  }

  read_ImageNet_normalized("./polaris_imagenet_files.txt", images, image_labels,
                           batch);

  if (unified_mem && um_tuning) {
    cudaMemPrefetchAsync(images,
                         batch * image_height * image_width * image_channel *
                             sizeof(float),
                         dev);
    cudaMemPrefetchAsync(image_labels, batch * sizeof(unsigned), dev);
  }

  //================ Get Weight =================
  FILE *config_file = fopen("./resnet_imagenet.csv", "r");

  if (!config_file) {
    fprintf(stderr, "Error: Could not open config file.\n");
    exit(EXIT_FAILURE);
  }

  if (rank == 0)
    printf("Starting ResNet-128 with batch: %u\n", batch);

  //================ Set Network =================
  // Layer-0
  InConv128LayerParam *bconv1 = nullptr;
  if (unified_mem) {
    SAFE_ALOC_UM(bconv1, sizeof(InConv128LayerParam));
    new (bconv1) InConv128LayerParam("Conv1", image_height, image_width, 7, 7,
                                     3, 64, batch, 4, 4, true, 1, 1, true,
                                     unified_mem, um_tuning); // save residual
  } else {
    bconv1 =
        new InConv128LayerParam("Conv1", image_height, image_width, 7, 7, 3, 64,
                                batch, 4, 4, true, 1, 1, true); // save residual
  }
  InConv128LayerParam *bconv1_gpu = bconv1->initialize(images, config_file);

  // Layer-1, basic-block-1, conv1
  Conv128LayerParam *l1b1c1 = nullptr;
  if (unified_mem) {
    SAFE_ALOC_UM(l1b1c1, sizeof(Conv128LayerParam));
    new (l1b1c1)
        Conv128LayerParam("L1B1C1", bconv1->output_height, bconv1->output_width,
                          3, 3, 64, 64, batch, 1, 1, true, 1, 1, false, false,
                          false, 0, false, unified_mem, um_tuning);
  } else {
    l1b1c1 = new Conv128LayerParam("L1B1C1", bconv1->output_height,
                                   bconv1->output_width, 3, 3, 64, 64, batch);
  }
  Conv128LayerParam *l1b1c1_gpu =
      l1b1c1->initialize(config_file, bconv1->get_output_gpu());

  // Layer-1, basic-block-1, conv2
  Conv128LayerParam *l1b1c2 = nullptr;
  if (unified_mem) {
    SAFE_ALOC_UM(l1b1c2, sizeof(Conv128LayerParam));
    new (l1b1c2)
        Conv128LayerParam("L1B1C2", l1b1c1->output_height, l1b1c1->output_width,
                          3, 3, 64, 64, batch, 1, 1, true, 1, 1, false, true,
                          true, 64, false, unified_mem, um_tuning);
  } else {
    l1b1c2 = new Conv128LayerParam("L1B1C2", l1b1c1->output_height,
                                   l1b1c1->output_width, 3, 3, 64, 64, batch, 1,
                                   1, true, 1, 1, false, true, true, 64);
  }
  Conv128LayerParam *l1b1c2_gpu = l1b1c2->initialize(
      config_file, l1b1c1->get_output_gpu(), bconv1->get_output_residual_gpu());

  // Layer-1, basic-block-2, conv1
  Conv128LayerParam *l1b2c1 = nullptr;
  if (unified_mem) {
    SAFE_ALOC_UM(l1b2c1, sizeof(Conv128LayerParam));
    new (l1b2c1)
        Conv128LayerParam("L1B2C1", l1b1c2->output_height, l1b1c2->output_width,
                          3, 3, 64, 64, batch, 1, 1, true, 1, 1, false, false,
                          false, 0, false, unified_mem, um_tuning);
  } else {
    l1b2c1 = new Conv128LayerParam("L1B2C1", l1b1c2->output_height,
                                   l1b1c2->output_width, 3, 3, 64, 64, batch);
  }

  Conv128LayerParam *l1b2c1_gpu =
      l1b2c1->initialize(config_file, l1b1c2->get_output_gpu());

  // Layer-1, basic-block-2, conv2
  Conv128LayerParam *l1b2c2 = nullptr;
  if (unified_mem) {
    SAFE_ALOC_UM(l1b2c2, sizeof(Conv128LayerParam));
    new (l1b2c2)
        Conv128LayerParam("L1B2C2", l1b2c1->output_height, l1b2c1->output_width,
                          3, 3, 64, 64, batch, 1, 1, true, 1, 1, false, true,
                          true, 128, false, unified_mem, um_tuning);
  } else {
    l1b2c2 = new Conv128LayerParam("L1B2C2", l1b2c1->output_height,
                                   l1b2c1->output_width, 3, 3, 64, 64, batch, 1,
                                   1, true, 1, 1, false, true, true, 128);
  }

  Conv128LayerParam *l1b2c2_gpu = l1b2c2->initialize(
      config_file, l1b2c1->get_output_gpu(), l1b1c2->get_output_residual_gpu());

  //=============
  // Layer-2, basic-block-1, conv1
  Conv128LayerParam *l2b1c1 = nullptr;
  if (unified_mem) {
    SAFE_ALOC_UM(l2b1c1, sizeof(Conv128LayerParam));
    new (l2b1c1)
        Conv128LayerParam("L2B1C1", l1b2c2->output_height, l1b2c2->output_width,
                          3, 3, 64, 128, batch, 2, 2, true, 1, 1, false, false,
                          false, 0, false, unified_mem, um_tuning);
  } else {
    l2b1c1 =
        new Conv128LayerParam("L2B1C1", l1b2c2->output_height,
                              l1b2c2->output_width, 3, 3, 64, 128, batch, 2, 2);
  }

  Conv128LayerParam *l2b1c1_gpu =
      l2b1c1->initialize(config_file, l1b2c2->get_output_gpu());

  // Layer-2, basic-block-1, conv2
  Conv128LayerParam *l2b1c2 = nullptr;
  if (unified_mem) {
    SAFE_ALOC_UM(l2b1c2, sizeof(Conv128LayerParam));
    new (l2b1c2)
        Conv128LayerParam("L2B1C2", l2b1c1->output_height, l2b1c1->output_width,
                          3, 3, 128, 128, batch, 1, 1, true, 1, 1, false, true,
                          true, 128, true, unified_mem, um_tuning);
  } else {
    l2b1c2 = new Conv128LayerParam(
        "L2B1C2", l2b1c1->output_height, l2b1c1->output_width, 3, 3, 128, 128,
        batch, 1, 1, true, 1, 1, false, true, true, 128, true);
  }
  Conv128LayerParam *l2b1c2_gpu = l2b1c2->initialize(
      config_file, l2b1c1->get_output_gpu(), l1b2c2->get_output_residual_gpu());

  // Layer-2, basic-block-2, conv1
  Conv128LayerParam *l2b2c1 = nullptr;
  if (unified_mem) {
    SAFE_ALOC_UM(l2b2c1, sizeof(Conv128LayerParam));
    new (l2b2c1)
        Conv128LayerParam("L2B2C1", l2b1c2->output_height, l2b1c2->output_width,
                          3, 3, 128, 128, batch, 1, 1, true, 1, 1, false, false,
                          false, 0, false, unified_mem, um_tuning);
  } else {
    l2b2c1 = new Conv128LayerParam("L2B2C1", l2b1c2->output_height,
                                   l2b1c2->output_width, 3, 3, 128, 128, batch,
                                   1, 1);
  }

  Conv128LayerParam *l2b2c1_gpu =
      l2b2c1->initialize(config_file, l2b1c2->get_output_gpu());

  // Layer-2, basic-block-2, conv2
  Conv128LayerParam *l2b2c2 = nullptr;
  if (unified_mem) {
    SAFE_ALOC_UM(l2b2c2, sizeof(Conv128LayerParam));
    new (l2b2c2)
        Conv128LayerParam("L2B2C2", l2b2c1->output_height, l2b2c1->output_width,
                          3, 3, 128, 128, batch, 1, 1, true, 1, 1, false, true,
                          true, 128, false, unified_mem, um_tuning);
  } else {
    l2b2c2 = new Conv128LayerParam("L2B2C2", l2b2c1->output_height,
                                   l2b2c1->output_width, 3, 3, 128, 128, batch,
                                   1, 1, true, 1, 1, false, true, true, 128);
  }
  Conv128LayerParam *l2b2c2_gpu = l2b2c2->initialize(
      config_file, l2b2c1->get_output_gpu(), l2b1c2->get_output_residual_gpu());

  //=============
  // Layer-3, basic-block-1, conv1
  Conv128LayerParam *l3b1c1 = nullptr;
  if (unified_mem) {
    SAFE_ALOC_UM(l3b1c1, sizeof(Conv128LayerParam));
    new (l3b1c1)
        Conv128LayerParam("L3B1C1", l2b2c2->output_height, l2b2c2->output_width,
                          3, 3, 128, 256, batch, 2, 2, true, 1, 1, false, false,
                          false, 0, false, unified_mem, um_tuning);
  } else {
    l3b1c1 = new Conv128LayerParam("L3B1C1", l2b2c2->output_height,
                                   l2b2c2->output_width, 3, 3, 128, 256, batch,
                                   2, 2);
  }

  Conv128LayerParam *l3b1c1_gpu =
      l3b1c1->initialize(config_file, l2b2c2->get_output_gpu());

  // Layer-3, basic-block-1, conv2
  Conv128LayerParam *l3b1c2 = nullptr;
  if (unified_mem) {
    SAFE_ALOC_UM(l3b1c2, sizeof(Conv128LayerParam));
    new (l3b1c2)
        Conv128LayerParam("L3B1C2", l3b1c1->output_height, l3b1c1->output_width,
                          3, 3, 256, 256, batch, 1, 1, true, 1, 1, false, true,
                          true, 128, true, unified_mem, um_tuning);
  } else {
    l3b1c2 = new Conv128LayerParam(
        "L3B1C2", l3b1c1->output_height, l3b1c1->output_width, 3, 3, 256, 256,
        batch, 1, 1, true, 1, 1, false, true, true, 128, true);
  }
  Conv128LayerParam *l3b1c2_gpu = l3b1c2->initialize(
      config_file, l3b1c1->get_output_gpu(), l2b2c2->get_output_residual_gpu());

  // Layer-3, basic-block-2, conv1
  Conv128LayerParam *l3b2c1 = nullptr;
  if (unified_mem) {
    SAFE_ALOC_UM(l3b2c1, sizeof(Conv128LayerParam));
    new (l3b2c1)
        Conv128LayerParam("L3B2C1", l3b1c2->output_height, l3b1c2->output_width,
                          3, 3, 256, 256, batch, 1, 1, true, 1, 1, false, false,
                          false, 0, false, unified_mem, um_tuning);
  } else {
    l3b2c1 = new Conv128LayerParam("L3B2C1", l3b1c2->output_height,
                                   l3b1c2->output_width, 3, 3, 256, 256, batch,
                                   1, 1);
  }

  Conv128LayerParam *l3b2c1_gpu =
      l3b2c1->initialize(config_file, l3b1c2->get_output_gpu());

  // Layer-3, basic-block-2, conv2
  Conv128LayerParam *l3b2c2 = nullptr;
  if (unified_mem) {
    SAFE_ALOC_UM(l3b2c2, sizeof(Conv128LayerParam));
    new (l3b2c2)
        Conv128LayerParam("L3B2C2", l3b2c1->output_height, l3b2c1->output_width,
                          3, 3, 256, 256, batch, 1, 1, true, 1, 1, false, true,
                          true, 256, false, unified_mem, um_tuning);
  } else {
    l3b2c2 = new Conv128LayerParam("L3B2C2", l3b2c1->output_height,
                                   l3b2c1->output_width, 3, 3, 256, 256, batch,
                                   1, 1, true, 1, 1, false, true, true, 256);
  }
  Conv128LayerParam *l3b2c2_gpu = l3b2c2->initialize(
      config_file, l3b2c1->get_output_gpu(), l3b1c2->get_output_residual_gpu());

  //=============
  // Layer-4, basic-block-1, conv1
  Conv128LayerParam *l4b1c1 = nullptr;
  if (unified_mem) {
    SAFE_ALOC_UM(l4b1c1, sizeof(Conv128LayerParam));
    new (l4b1c1)
        Conv128LayerParam("L4B1C1", l3b2c2->output_height, l3b2c2->output_width,
                          3, 3, 256, 512, batch, 2, 2, true, 1, 1, false, false,
                          false, 0, false, unified_mem, um_tuning);
  } else {
    l4b1c1 = new Conv128LayerParam("L4B1C1", l3b2c2->output_height,
                                   l3b2c2->output_width, 3, 3, 256, 512, batch,
                                   2, 2);
  }

  Conv128LayerParam *l4b1c1_gpu =
      l4b1c1->initialize(config_file, l3b2c2->get_output_gpu());

  // Layer-4, basic-block-1, conv2
  Conv128LayerParam *l4b1c2 = nullptr;
  if (unified_mem) {
    SAFE_ALOC_UM(l4b1c2, sizeof(Conv128LayerParam));
    new (l4b1c2)
        Conv128LayerParam("L4B1C2", l4b1c1->output_height, l4b1c1->output_width,
                          3, 3, 512, 512, batch, 1, 1, true, 1, 1, false, true,
                          true, 256, true, unified_mem, um_tuning);
  } else {
    l4b1c2 = new Conv128LayerParam(
        "L4B1C2", l4b1c1->output_height, l4b1c1->output_width, 3, 3, 512, 512,
        batch, 1, 1, true, 1, 1, false, true, true, 256, true);
  }
  Conv128LayerParam *l4b1c2_gpu = l4b1c2->initialize(
      config_file, l4b1c1->get_output_gpu(), l3b2c2->get_output_residual_gpu());

  // Layer-4, basic-block-2, conv1
  Conv128LayerParam *l4b2c1 = nullptr;
  if (unified_mem) {
    SAFE_ALOC_UM(l4b2c1, sizeof(Conv128LayerParam));
    new (l4b2c1)
        Conv128LayerParam("L4B2C1", l4b1c2->output_height, l4b1c2->output_width,
                          3, 3, 512, 512, batch, 1, 1, true, 1, 1, false, false,
                          false, 0, false, unified_mem, um_tuning);
  } else {
    l4b2c1 = new Conv128LayerParam("L4B2C1", l4b1c2->output_height,
                                   l4b1c2->output_width, 3, 3, 512, 512, batch,
                                   1, 1);
  }

  Conv128LayerParam *l4b2c1_gpu =
      l4b2c1->initialize(config_file, l4b1c2->get_output_gpu());

  // Layer-4, basic-block-2, conv2
  Conv128LayerParam *l4b2c2 = nullptr;
  if (unified_mem) {
    SAFE_ALOC_UM(l4b2c2, sizeof(Conv128LayerParam));
    new (l4b2c2)
        Conv128LayerParam("L4B2C2", l4b2c1->output_height, l4b2c1->output_width,
                          3, 3, 512, 512, batch, 1, 1, true, 1, 1, true, false,
                          true, 512, false, unified_mem, um_tuning);
  } else {
    l4b2c2 = new Conv128LayerParam("L4B2C2", l4b2c1->output_height,
                                   l4b2c1->output_width, 3, 3, 512, 512, batch,
                                   1, 1, true, 1, 1, true, false, true, 512);
  }
  Conv128LayerParam *l4b2c2_gpu = l4b2c2->initialize(
      config_file, l4b2c1->get_output_gpu(), l4b1c2->get_output_residual_gpu());

  //=============
  // Layer-5
  Fc128LayerParam *bfc1 = nullptr;
  if (unified_mem) {
    SAFE_ALOC_UM(bfc1, sizeof(Fc128LayerParam));
    new (bfc1) Fc128LayerParam(
        "Fc1", batch, (l4b2c2->output_height) * (l4b2c2->output_width) * 512,
        512, unified_mem, um_tuning);
  } else {
    bfc1 = new Fc128LayerParam(
        "Fc1", batch, (l4b2c2->output_height) * (l4b2c2->output_width) * 512,
        512);
  }

  Fc128LayerParam *bfc1_gpu =
      bfc1->initialize(config_file, l4b2c2->get_output_gpu());
  // Out Layer
  Out128LayerParam *bout = nullptr;
  if (unified_mem) {
    SAFE_ALOC_UM(bout, sizeof(Out128LayerParam));
    new (bout) Out128LayerParam("Fout", batch, 512, output_size, unified_mem,
                                um_tuning);
  } else {
    bout = new Out128LayerParam("Fout", batch, 512, output_size);
  }

  Out128LayerParam *bout_gpu =
      bout->initialize(config_file, bfc1->get_output_gpu());

  CUDA_SAFE_CALL(cudaEventRecord(init_stop));

  //============= Memory Allocation ===============

  // size_t free_mem, total_mem;
  // cudaMemGetInfo(&free_mem, &total_mem);
  // double used_mem = (total_mem - free_mem) / (1024 * 1024);
  // printf("Allocated memory with %d batch %f\n", batch, used_mem);
  // exit(1);

  //================ Setup Kernel =================
  int numThreads = 1024;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  int numBlocksPerSm;
  int shared_memory = 512 * sizeof(int) * 32;
  cudaFuncSetAttribute(resnet128, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       shared_memory);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, resnet128,
                                                numThreads, shared_memory);

  // cudaFuncSetAttribute(resnet128,
  // cudaFuncAttributePreferredSharedMemoryCarveout,0);

  void *args[] = {&bconv1_gpu, &l1b1c1_gpu, &l1b1c2_gpu, &l1b2c1_gpu,
                  &l1b2c2_gpu, &l2b1c1_gpu, &l2b1c2_gpu, &l2b2c1_gpu,
                  &l2b2c2_gpu, &l3b1c1_gpu, &l3b1c2_gpu, &l3b2c1_gpu,
                  &l3b2c2_gpu, &l4b1c1_gpu, &l4b1c2_gpu, &l4b2c1_gpu,
                  &l4b2c2_gpu, &bfc1_gpu,   &bout_gpu};

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);
  // START_TIMER;

  cudaEventRecord(comp_start);

  cudaLaunchCooperativeKernel((void *)resnet128,
                              numBlocksPerSm * deviceProp.multiProcessorCount,
                              numThreads, args, shared_memory);

  cudaEventRecord(comp_stop);
  if (rank == 0)
    printf("Completed ResNet-128... Collecting results\n");
  // if (i_gpu == 0) cudaMemset(bout->get_output_gpu(), 0,
  // bout->output_bytes()); if (i_gpu == 1)
  // cudaMemset(bout->get_output_gpu(), 0, bout->output_bytes());

  cudaEventRecord(comm_start);

  // To be replaced with NCCL_Reduce as in scaleup
  // MPI_Reduce(bout->get_output_gpu(), bout->get_output_gpu(),
  //            bout->output_size(), MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
  // MPI_Barrier(MPI_COMM_WORLD);
  CHECK_NCCL(ncclReduce(bout->get_output_gpu(), bout->get_output_gpu(),
                        bout->output_size(), ncclFloat, ncclMax, 0, comm, 0));

  cudaEventRecord(comm_stop);

  // STOP_TIMER;
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  float init_time, comp_time, comm_time;

  cudaEventElapsedTime(&init_time, init_start, init_stop);
  cudaEventElapsedTime(&comp_time, comp_start, comp_stop);
  cudaEventElapsedTime(&comm_time, comm_start, comm_stop);

  if (rank == 0) {
    init_times.resize(n_gpu, 0);
    comm_times.resize(n_gpu, 0);
    comp_times.resize(n_gpu, 0);
  }

  CHECK_MPI(MPI_Gather(&init_time, 1, MPI_FLOAT, init_times.data(), 1,
                       MPI_FLOAT, 0, MPI_COMM_WORLD));
  CHECK_MPI(MPI_Gather(&comp_time, 1, MPI_FLOAT, comp_times.data(), 1,
                       MPI_FLOAT, 0, MPI_COMM_WORLD));
  CHECK_MPI(MPI_Gather(&comm_time, 1, MPI_FLOAT, comm_times.data(), 1,
                       MPI_FLOAT, 0, MPI_COMM_WORLD));

  CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

  //================ Output =================
  // if (rank == 0) {
  //   float *output = bout->download_output();
  //   validate_prediction(output, image_labels, output_size, batch);
  // }

  // float *out = l1b2c1->download_full_output();
  // // float* out = l1b1c2->download_full_output();
  // // for (int i=0; i<512; i++)
  // for (int i = 4096; i < 4096 + 512; i++) {
  //   printf("%.f ", out[i]);
  //   if ((i + 1) % 32 == 0)
  //     printf("\n");
  // }
  // printf("\n===%f===\n", bout->bn_scale[0]);

  if (rank == 0) {
    double avg_init_time = 0.0;
    double avg_comp_time = 0.0;
    double avg_comm_time = 0.0;

    for (int k = 0; k < n_gpu; k++) {
      avg_init_time += init_times[k];
      avg_comp_time += comp_times[k];
      avg_comm_time += comm_times[k];
    }

    avg_init_time /= static_cast<double>(n_gpu);
    avg_comp_time /= static_cast<double>(n_gpu);
    avg_comm_time /= static_cast<double>(n_gpu);

    printf("%s %s\n", unified_mem ? "Unified Memory" : "Normal Memory",
           um_tuning ? "Tuning" : "No Tuning");
    printf("\nBatch: %u, Init_time:%.3lf[ms], Comp_time:%.3lf[ms], "
           "Comm_time:%.3lf[ms]\n",
           batch, avg_init_time, avg_comp_time, avg_comm_time);
  }

  if (unified_mem) {
    bconv1_gpu->~InConv128LayerParam();
    SAFE_FREE_UM(bconv1);
    l1b1c1->~Conv128LayerParam();
    SAFE_FREE_UM(l1b1c1);
    l1b1c2->~Conv128LayerParam();
    SAFE_FREE_UM(l1b1c2);
    l1b2c1->~Conv128LayerParam();
    SAFE_FREE_UM(l1b2c1);
    l1b2c2->~Conv128LayerParam();
    SAFE_FREE_UM(l1b2c2);
    l2b1c1->~Conv128LayerParam();
    SAFE_FREE_UM(l2b1c1);
    l2b1c2->~Conv128LayerParam();
    SAFE_FREE_UM(l2b1c2);
    l2b2c1->~Conv128LayerParam();
    SAFE_FREE_UM(l2b2c1);
    l2b2c2->~Conv128LayerParam();
    SAFE_FREE_UM(l2b2c2);
    l3b1c1->~Conv128LayerParam();
    SAFE_FREE_UM(l3b1c1);
    l3b1c2->~Conv128LayerParam();
    SAFE_FREE_UM(l3b1c2);
    l3b2c1->~Conv128LayerParam();
    SAFE_FREE_UM(l3b2c1);
    l3b2c2->~Conv128LayerParam();
    SAFE_FREE_UM(l3b2c2);
    l4b1c1->~Conv128LayerParam();
    SAFE_FREE_UM(l4b1c1);
    l4b1c2->~Conv128LayerParam();
    SAFE_FREE_UM(l4b1c2);
    l4b2c1->~Conv128LayerParam();
    SAFE_FREE_UM(l4b2c1);
    l4b2c2->~Conv128LayerParam();
    SAFE_FREE_UM(l4b2c2);
    bfc1_gpu->~Fc128LayerParam();
    SAFE_FREE_UM(bfc1);
    bout_gpu->~Out128LayerParam();
    SAFE_FREE_UM(bout);

    SAFE_FREE_UM(image_labels);
    SAFE_FREE_UM(images);

    if (oversub) {
      SAFE_FREE_GPU(oversub_ptr);
    }
  } else {
    delete bconv1;
    delete l1b1c1;
    delete l1b1c2;
    delete l1b2c1;
    delete l1b2c2;

    delete l2b1c1;
    delete l2b1c2;
    delete l2b2c1;
    delete l2b2c2;

    delete l3b1c1;
    delete l3b1c2;
    delete l3b2c1;
    delete l3b2c2;

    delete l4b1c1;
    delete l4b1c2;
    delete l4b2c1;
    delete l4b2c2;

    delete bfc1;
    delete bout;
  }

  ncclCommDestroy(comm);
  MPI_Finalize();

  return 0;
}
