# OCI twin of mu2efastsim.def for container-only sites (ESnet East IRI on AWS EKS, Kubernetes, docker/podman).
# Build context = umbrella checkout with submodules (git-tracked files only; CI uses actions/checkout).
# docker build -f container/Dockerfile --build-arg VARIANT=cpu|gpu -t mu2e-fastsim:cpu .
FROM python:3.13-slim-bookworm

ARG VARIANT=cpu
ARG TRIM="nvidia-nccl-cu12 nvidia-nvshmem-cu12"   # cuFFT and cuDNN stay: the JAX CUDA plugin checks both at start
LABEL org.opencontainers.image.title="mu2e-fastsim" \
      org.opencontainers.image.source="https://github.com/RobMina/mu2e-fastsim" \
      org.opencontainers.image.description="Mu2e Geant4-free fast simulation and CRV fast reconstruction (JAX); public code and public Offline files only" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV MU2E_SEARCH_PATH=/opt/offline_data \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JAX_ENABLE_X64=1 \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* && python -m pip install -q --upgrade pip

# Dependencies first (cached layer); gpu: pip CUDA 12 wheels minus the ones no kernel loads
RUN if [ "$VARIANT" = cpu ]; then \
      python -m pip install -q "jax==0.11.2" numpy pandas awkward uproot matplotlib pytest; \
    else \
      python -m pip install -q "jax[cuda12]==0.11.2" numpy pandas awkward uproot matplotlib pytest \
      && for w in $TRIM; do python -m pip uninstall -q -y "$w" || true; done; \
    fi

COPY container/fetch_offline_data.sh /opt/fetch_offline_data.sh
RUN sh /opt/fetch_offline_data.sh /opt/offline_data

COPY . /opt/mu2e-fastsim
RUN for p in core crv tracker; do python -m pip install -q --no-deps /opt/mu2e-fastsim/$p; done \
    && rm -rf /opt/mu2e-fastsim/.git /opt/mu2e-fastsim/*/.git /opt/mu2e-fastsim/site.env \
    && find / -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true \
    && printf '#!/bin/sh\nexec python -m crvfastsim "$@"\n' > /usr/local/bin/crvfastsim && chmod +x /usr/local/bin/crvfastsim \
    && echo "variant $VARIANT" >> /opt/mu2e-fastsim/VERSION \
    && chmod -R a+rX /opt \
    && python -c "import mu2efastsim, crvfastsim, crvfastsim.cli, crvfastsim.reco.chain_jax, trkfastsim, jax; print('jax', jax.__version__)"

WORKDIR /opt/mu2e-fastsim
CMD ["python", "-c", "import jax; print(jax.devices())"]
