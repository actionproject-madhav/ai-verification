# The lineage-check enclave's container: the package and the service on CPU-only PyTorch. The fingerprint reads
# weights only, so the enclave needs no GPU. Build from the repository root:
#   docker build -f enclave/Dockerfile -t lineage-check .
# Pin the resulting digest in tinfoil-config.yml, as SASH's inference demo pins its image.
# The reference encoder for the exact check (lineage_check.derive): llama.cpp's libggml-base, built four ways with
# this base image's own compilers, so that the libraries load in the image that runs them.
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS ggml
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake clang git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY tools/build_ggml.sh /build_ggml.sh
RUN bash /build_ggml.sh /src > /paths.txt && mkdir -p /opt/ggml \
    && for p in $(cat /paths.txt); do b=$(basename "$(dirname "$(dirname "$p")")"); mkdir -p /opt/ggml/$b/bin; cp -L "$p" /opt/ggml/$b/bin/; done

FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
COPY enclave/requirements.txt /opt/lineage-check/enclave/requirements.txt
RUN pip install --no-cache-dir -r /opt/lineage-check/enclave/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=ggml /opt/ggml /opt/ggml
ENV LINEAGE_GGML=/opt/ggml/build-plain/bin/libggml-base.so,/opt/ggml/build-fma-chains/bin/libggml-base.so,/opt/ggml/build-fma/bin/libggml-base.so,/opt/ggml/build-clang-fma/bin/libggml-base.so \
    LINEAGE_GGML_REF=dfdfca22bce36f66b4cfa3e6c96f0c4d92dac0fe
COPY pyproject.toml README.md /opt/lineage-check/
COPY lineage_check /opt/lineage-check/lineage_check
COPY enclave/lineage_service.py enclave/concern-demo.json /opt/lineage-check/enclave/
COPY watchlists /opt/lineage-check/watchlists
RUN pip install --no-cache-dir --no-deps /opt/lineage-check
# A private model can be sealed into the image: build with --build-arg MODEL_SRC=<a directory inside the build
# context> and run with MODEL_DIR=/model. The launch policy then pins the model's bytes together with the code.
ARG MODEL_SRC=enclave/no-model
COPY ${MODEL_SRC} /model
ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
ENTRYPOINT ["python3", "/opt/lineage-check/enclave/lineage_service.py"]
