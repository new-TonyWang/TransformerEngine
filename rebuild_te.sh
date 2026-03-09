export NVTE_CUDA_ARCHS="103a"
export CUDACXX=/usr/local/cuda-13.0/bin/nvcc
export NCCL_INCLUDE_PATH=/mnt/afs/conda/miniconda3/envs/wty_metis/lib/python3.12/site-packages/nvidia/nccl/include/
NCCL_INC=/mnt/afs/conda/miniconda3/envs/wty_metis/lib/python3.12/site-packages/nvidia/nccl/include/
CUDNN_INC=/mnt/afs/conda/miniconda3/envs/wty_metis/lib/python3.12/site-packages/nvidia/cudnn/include/
export CXXFLAGS="-I${NCCL_INC} -I${CUDNN_INC}"
export CUDAFLAGS="-I${NCCL_INC} -I${CUDNN_INC}"
pip install -e . --no-build-isolation

