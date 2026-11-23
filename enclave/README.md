# lineage-check enclave

The lineage check as a Tinfoil enclave service, built in the shape of SASH's verified inference demo
(`SG-AI-Safety-Hub/inference-demo`). It reuses that demo's statement format, digest schemes, attested key and
key-binding code, so a lineage statement and an inference statement about the same model can be linked by SASH's
own tools.

| file | what it is |
| --- | --- |
| `lineage_service.py` | the service: hashes the mounted model at boot, answers `POST /v1/challenge {nonce}` with one signed statement |
| `../lineage_check/fourpillars.py` | inference-demo's identity and statement code (Apache-2.0), vendored so the image is self-contained |
| `verify_statement.py` | checks a statement with inference-demo's own `open_envelope`, `identifiers`, `dirhash_sha256`, `jcs_sha256` |
| `tinfoil-config.yml` | the enclave: CPU only, attested Ed25519 key, the audited model and the listed models as Modelwrap packs |
| `Dockerfile`, `requirements.txt` | the container, pinned to the versions every result in `results/` used |
| `concern-demo.json` | the demo list: an independent model with the same tokenizer, and TinyLlama's 3T base |

## The statement

An in-toto Statement v1 in a DSSE envelope, signed with the enclave's attested key. Its `scannedModel` has the
inference statement's component shape: `dirhash-sha256/1` of the mounted files, with `hf-revision/1` and
`modelwrap/<schema>` aliases. The predicate carries the verdict (match or no match, and which entry), the
refusal-ablation bit, the verifier's nonce, the concern list's `jcs-sha256/1` digest and the method. No scores,
tokens or weights leave the enclave. In Tinfoil, each statement carries a v3 attestation document fetched with a
nonce derived from the verifier's challenge, which lists the signing key.

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

## What deployment still needs

1. A Tinfoil account: `tinfoil model wrap` the two listed models, fill in their `mpk` references.
2. `docker build -f enclave/Dockerfile .` in CI, and pin the image digest in `tinfoil-config.yml`. The image has not been built yet.
3. Pin `CONCERN_LIST_DIGEST`, and point the listed models at their mounted packs (`path`).
