#!/bin/bash
if [[ $# -ne 1 ]]; then
    echo "Usage: start_docker_with_tf_gpu CellSegmentation_directory"
    exit 1
fi
docker run --gpus all -it --rm -e U=$UID --name U-NET -v $1:/CellSegmentation tensorflow/tensorflow:latest-gpu bash -c -c "/CellSegmentation/devcontainer/init.sh;su dev"
