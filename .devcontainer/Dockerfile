FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget bzip2 ca-certificates build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy

ENV PATH="$CONDA_DIR/bin:$PATH"

# Copy the project files into the container
WORKDIR /workspace
COPY . .

# Create Conda environment from environment.yml
RUN conda env create -f environment.yml && conda clean -afy

# Install additional packages after creating the environment
RUN conda run -n JFBAN conda install -c conda-forge graphviz -y && \
    apt-get update && apt-get install -y libgl1

# Replace <ENV_NAME> with actual name from your environment.yml
SHELL ["conda", "run", "--no-capture-output", "-n", "JFBAN", "/bin/bash", "-c"]

# Install package directly from GitHub
RUN pip install git+https://github.com/jellyfishtracking101/JF_Deepreg.git && \
    pip install git+https://github.com/flavell-lab/euler_gpu.git

# Set environment active by default
ENV PATH="/opt/conda/envs/<ENV_NAME>/bin:$PATH"
ENV CONDA_DEFAULT_ENV=<ENV_NAME>

# Final workspace setup
WORKDIR /workspace
CMD ["/bin/bash"]
