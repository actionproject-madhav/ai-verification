# lineage-check enclave

The lineage check as an enclave service, built for Tinfoil in the shape of SASH's verified inference demo
(`SG-AI-Safety-Hub/inference-demo`). So far it has run as a local twin and in Azure confidential containers, not
on Tinfoil. It reuses that demo's statement format, digest schemes, attested key and
key-binding code, so a lineage statement and an inference statement about the same model can be linked by SASH's
own tools.

| file | what it is |
| --- | --- |
| `lineage_service.py` | the service: hashes the mounted model at boot, answers `POST /v1/challenge {nonce}` with one signed statement |
| `../lineage_check/fourpillars.py` | inference-demo's identity and statement code (Apache-2.0), vendored so the image is self-contained |
| `verify_statement.py` | checks a statement with inference-demo's own `open_envelope`, `identifiers`, `dirhash_sha256`, `jcs_sha256` |
| `tinfoil-config.yml` | the enclave: CPU only, attested Ed25519 key, the audited model and the listed models as Modelwrap packs |
| `Dockerfile`, `requirements.txt` | the container, with pinned versions |
| `concern-demo.json` | the demo list: an independent model with the same tokenizer, and TinyLlama's 3T base |
| `azure/template.json` | the same service as an Azure confidential container group, with Microsoft's attestation sidecar |
| `azure/template-light.json` | the traffic-light mode in the same kind of enclave, on a copy of a listed model |

## The statement

An in-toto Statement v1 in a DSSE envelope, signed with the enclave's attested key. Its `scannedModel` has the
inference statement's component shape: `dirhash-sha256/1` of the mounted files, with `hf-revision/1` and
`modelwrap/<schema>` aliases. The predicate carries the verdict (match or no match, and which entry), the
refusal-ablation bit, the verifier's nonce, the concern list's `jcs-sha256/1` digest and the method. No scores,
tokens or weights leave the enclave. In Tinfoil, each statement carries a v3 attestation document fetched with a
nonce derived from the verifier's challenge, which lists the signing key. This describes the match mode. The
traffic-light mode signs a `LightStatement` instead: the light, the matched entry, the rule that decided and the
digests of both lists (see the Azure run below).

## Local twin run (18 Nov 2026)

TinyLlama-1.1B-Chat-v1.0 at the revision SASH's inference demo serves, scanned against `concern-demo.json`:

```
llama-160m:        distance 0.983 (threshold 0.5)   independent, same tokenizer: no match
tinyllama-1.1b-3t: distance 0.001 (threshold 0.5)   its base: match
replay of the same nonce: refused
```

`verify_statement.py --upstream <path to a clone of inference-demo>` on the result (`demo/enclave/`):

```
PASS  signature verifies under the statement key (inference-demo's own open_envelope)
PASS  predicate type urn:lineage-check:LineageStatement:v0.2
PASS  the signing key is the statement key it names
PASS  issued for this verifier's nonce (no replay)
PASS  concern list digest matches (inference-demo's own jcs_sha256 and the vendored copy agree)
NOTE  keyBinding null: a local twin, signed with a dev key no hardware report binds
PASS  scanned bytes recomputed: dirhash-sha256/1 d1e29a1ad2ef057d... (inference-demo's own dirhash_sha256)
PASS  same model as the inference statement's served base TinyLlama/TinyLlama-1.1B-Chat-v1.0:
      shared dirhash-sha256/1, hf-revision/1
lineage: match -> tinyllama-1.1b-3t
```

The dirhash equals the one in inference-demo's `fixtures/crosswalk.json`, so the lineage statement names
exactly the bytes SASH's inference enclave serves. The twin signs with a dev key (`keyBinding: null`), as the
inference demo's own local twin does. The scan took 8 minutes, mostly fetching the listed models' layers over
the network; in the enclave they are mounted packs.

## Run in an Azure confidential container group (23 Nov 2026)

The same service ran inside an AMD SEV-SNP guest on Azure Container Instances. A launch policy pins both
container images, their commands and every environment variable. The service fetched TinyLlama at the pinned
revision inside the guest, hashed it, made its signing key there, and answered one challenge in 29 seconds.
`azure/template.json` is the deployment and `../demo/azure/` holds the evidence.

```
python enclave/verify_statement.py demo/azure/response.json --nonce $(cat demo/azure/nonce.txt) \
    --concern enclave/concern-demo.json --policy demo/azure/policy.b64 --model-dir <TinyLlama snapshot>

PASS  signature verifies under the statement key
PASS  predicate type urn:lineage-check:LineageStatement:v0.2
PASS  the signing key is the statement key it names
PASS  issued for this verifier's nonce (no replay)
PASS  concern list digest matches
PASS  AMD SEV-SNP report verifies down to AMD's root (ARK-Genoa), debugging off
PASS  the report names this statement key and this nonce (report_data)
PASS  launched under the expected policy: host_data bf4e895a4ca88639...
PASS  the base system is one Microsoft signed: ContainerPlat-AMD-UVM, version 104, measurement dc3f5a934489232a...
PASS  scanned bytes recomputed: dirhash-sha256/1 d1e29a1ad2ef057d...
lineage: match -> tinyllama-1.1b-3t  (512 tokens, 4 layers, azure-confidential-containers)
```

| What is checked | What the check rests on |
| --- | --- |
| the report is genuine | AMD's root key, pinned from AMD's key server (`lineage_check/snp.py`) |
| the base system | Microsoft's signature over the launch measurement, its chain pinned to Microsoft's supply-chain root |
| the code that ran | `host_data` is the SHA-256 of the launch policy in `demo/azure/policy.b64` |
| the signing key | `report_data` is the SHA-256 of the statement key and the nonce; the key exists only inside the guest |
| the model | `dirhash-sha256/1` of the fetched files, equal to the digest in inference-demo's crosswalk |

A wrong policy, a wrong nonce or a changed report each fail the verifier.

Limits of this run. The policy leaves the containers' logs readable by the operator; they carry no secrets, and a
production policy turns them off (`--disable-stdio`). The two listed models are read from the Hub over TLS at
pinned revisions, where Tinfoil mounts them as packs. The container group had a public address for the challenge
and was deleted after the run.

To repeat it: build the image, put its digest in `azure/template.json`, generate the policy with
`az confcom acipolicygen -a template.json`, and deploy with `az deployment group create`.

## The traffic light in the same enclave (23 Nov 2026)

`azure/template-light.json` runs the service with `MODE=light` on `unsloth/GLM-5.3-GGUF`, the most-downloaded GGUF
copy of a government-named model (334,000 downloads), with `MODEL_FILE=UD-Q4_K_XL`: a 4-bit file in eleven shards.
The model has 753 billion parameters. The enclave read the shard headers and a few matrices of two layers by byte
range, compared them with the three listed models, and answered in 99 seconds. An earlier run of the same service on
an 8-bit safetensors copy published under another name (`dealignai/GLM-5.3-CYBERSECURITY-FP8`) took 122 seconds;
its evidence is in the repository's history.

```
python enclave/verify_statement.py demo/azure-light/response.json --nonce $(cat demo/azure-light/nonce.txt) \
    --watchlist watchlists/government-named.json --known watchlists/known-demo.json --policy demo/azure-light/policy.b64

PASS  signature verifies under the statement key
PASS  predicate type urn:lineage-check:LightStatement:v0.4
PASS  the signing key is the statement key it names
PASS  issued for this verifier's nonce (no replay)
PASS  watchlist digest matches
PASS  registry digest matches
PASS  AMD SEV-SNP report verifies down to AMD's root (ARK-Genoa), debugging off
PASS  the report names this statement key and this nonce (report_data)
PASS  launched under the expected policy: host_data 962c3fe280825a6c...
PASS  the base system is one Microsoft signed: ContainerPlat-AMD-UVM, version 104, measurement dc3f5a934489232a...
light: RED -> glm-5.3 [matches a listed model]  (unsloth/GLM-5.3-GGUF@346b3591..., file UD-Q4_K_XL/GLM-5.3-UD-Q4_K_XL-00001-of-00011.gguf)
```

The statement carries the light, the matched entry, the rule that decided and the file that was read. It carries
no distances. Because the model was read by byte range, the statement names it by Hub revision and file name, and
so it trusts the Hub to serve the right bytes for that revision. An enclave that serves a model holds all of its bytes and names it by hash, as the
first run does.

Limits of this run. It is one challenge, decided by the first rule, a match by name. The image was built before
the last change to the cut test (a looser agreement bar, and a cut called only on an aligned copy), which this
run did not exercise. Matching tokens by geometry,
the cut test, Green and Amber have run with the same code outside the enclave only. As in the first run, the
policy leaves the containers' logs readable by the operator.

## What a Tinfoil deployment still needs

1. A Tinfoil account: `tinfoil model wrap` the two listed models, fill in their `mpk` references.
2. `docker build -f enclave/Dockerfile .` in CI, and pin the image digest in `tinfoil-config.yml`. The Azure runs above
   built this Dockerfile; no image has been published for Tinfoil yet.
3. Pin `CONCERN_LIST_DIGEST`, and point the listed models at their mounted packs (`path`).
4. A configuration for the traffic-light mode. `tinfoil-config.yml` and the steps above cover the match mode; the
   traffic-light mode has run on Azure only.
