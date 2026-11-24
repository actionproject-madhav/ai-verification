No model is sealed into the image by default. `enclave/Dockerfile` copies this folder to `/model`;
build with `--build-arg MODEL_SRC=<your model folder>` to seal a private model in its place.
