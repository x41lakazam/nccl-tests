#include "cuda_runtime.h"
#include "common.h"

const int data_size = 128;
const int num_nodes = 16;
const int ppn = 8;
const int num_ranks = num_nodes * ppn;

int pipeline_parallelism(cudaStream_t stream, ncclComm_t comm, int rank, int size, void *send_buf, void *recv_buf) {
    int peer = rank % 32 < 16 ? rank + 16 : rank - 16;

    NCCLCHECK(ncclSend((char *)send_buf, size, ncclFloat, peer, comm, stream));
    NCCLCHECK(ncclRecv((char *)recv_buf, size, ncclFloat, peer, comm, stream));
    return 0;
}

int experts_parallelism(cudaStream_t stream, ncclComm_t comm, int rank, int size, void *send_buf, void *recv_buf) {

    int count = size / 4;
    int commFirstRank = 16*(rank / 16);
    int peer;

    for (int off=0; off < 16; off++) {
        peer = commFirstRank + off;
        NCCLCHECK(ncclSend(send_buf, count, ncclFloat, peer, comm, stream));
        NCCLCHECK(ncclRecv(recv_buf, count, ncclFloat, peer, comm, stream));
    }
    return 0;
}

testResult_t InitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, 33*rep + rank, 1, 0));
    for (int j=0; j<nranks; j++) {
      size_t partcount = sendcount/nranks;
      TESTCHECK(InitData((char*)args->expected[i] + j*partcount*wordSize(type), partcount, rank*partcount, type, ncclSum, 33*rep + j, 1, 0));
    }
    CUDACHECK(cudaDeviceSynchronize());
  }
  // We don't support in-place alltoall
  args->reportErrors = in_place ? 0 : 1;
  return testSuccess;
}

void GetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) { // TODO
  double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks-1))/((double)(nranks));
  *busBw = baseBw * factor;
}

void GetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, int nranks) {
    *sendcount = (count/nranks)*nranks;
    *recvcount = (count/nranks)*nranks;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
    *paramcount = count/nranks;
}

testResult_t RunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
    /* The env var EXPERTS_REDUCTIONS_OP controls the collective simulating experts reductions
     * should be 0 for Reduce Scatter and 1 for All Gather
     * The env var PARALLEL_OP controls the additional coll, 0=experts parallelism, 1=pipeline parallelism
     */
    char *env;
    int rank, size, experts_op, parallel_op;
    ncclResult_t state;
    NCCLCHECK(ncclCommCount(comm, &size));
    NCCLCHECK(ncclCommUserRank(comm, &rank));

    env = getenv("EXPERTS_REDUCTIONS_OP");
    experts_op = env ? atoi(env) : 0;

    env = getenv("PARALLEL_OP");
    parallel_op = env ? atoi(env) : 0;

    if (size < 32){
        printf("This test is meant to be ran with at least 32 nodes.");
        return testNcclError;
    }

    ncclComm_t expertsComm;
    NCCLCHECK(ncclCommSplit(comm, rank % 32, 0, &expertsComm, NULL));
    do {
        NCCLCHECK(ncclCommGetAsyncError(comm, &state));
    } while(state == ncclInProgress);

    // Run desired scenario
    NCCLCHECK(ncclGroupStart());
    // Run experts RS/AG
    switch (experts_op){
        case 0:
            NCCLCHECK(ncclReduceScatter((char *)sendbuff, (char *) recvbuff, size, ncclFloat, ncclSum, expertsComm, stream));
            break;
        case 1:
            NCCLCHECK(ncclAllGather((char *)sendbuff, (char *) recvbuff, size/num_ranks, ncclFloat, expertsComm, stream));
            break;
        default:
            printf("Invalid experts_op value, should be 0 for RS or 1 for AG, but received %d\n", experts_op);
            return testNcclError;
    }

    // Run another collective in parallel
    switch (parallel_op){
        case 0:
            experts_parallelism(stream, comm, rank, size, sendbuff, recvbuff);
            break;
        case 1:
            pipeline_parallelism(stream, comm, rank, size, sendbuff, recvbuff);
            break;
        default:
            printf("Invalid parallel op value, should be 0 for experts parallelism and 1 for pipeline parallelism");
            return testNcclError;
    }
    
    NCCLCHECK(ncclGroupEnd());

    ncclCommDestroy(expertsComm);
    return testSuccess;
}


struct testColl moeBenchmarkTest = {
  "MoeBenchmark",
  GetCollByteCount,
  InitData,
  GetBw,
  RunColl
};

void GetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  GetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t RunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &moeBenchmarkTest;
  ncclDataType_t *run_types;
  const char **run_typenames;
  int type_count;

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = test_typenum;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  for (int i=0; i<type_count; i++) {
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "none", -1));
  }
  return testSuccess;
}

struct testEngine moeBenchmarkEngine = {
  GetBuffSize,
  RunTest
};

#pragma weak ncclTestEngine=moeBenchmarkEngine
