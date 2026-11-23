"""GGUF files: how far each quantisation level moves a model, and the GGUF repositories around the listed models.

  ladder     every GGUF file of a few public models against the safetensors original it was converted from (what
             quantisation alone does) and against the registry base (what the light sees, with the cut test)
  watchlist  the repositories the watchlist scan could not read (no safetensors), read again as GGUF and compared
             with every listed model, on the same tokens as that scan

Usage: python exp_gguf.py ladder <out.json> [workers]
       python exp_gguf.py watchlist <watchlist.json> <results/watchlist.json> <out.jsonl> [workers]
       python exp_gguf.py summary <out.jsonl> <summary.json>
"""
import collections, json, multiprocessing as mp, os, shutil, sys, time
import torch
from huggingface_hub import HfApi
from lineage_check import fingerprint as fp4, gguf as G, light as LT, remote as R, universal as U

torch.set_grad_enabled(False)
NONCE = os.environ.get("WATCH_NONCE", "watchlist-2026-11")          # the nonce of exp_watchlist.py
LADDER = [  # the safetensors original, its GGUF conversion, the known base both descend from
    ("unsloth/Llama-3.1-8B-Instruct", "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "unsloth/Llama-3.1-8B"),
    ("Qwen/Qwen2.5-7B-Instruct", "bartowski/Qwen2.5-7B-Instruct-GGUF", "Qwen/Qwen2.5-7B"),
    ("mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated", "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF", "unsloth/Llama-3.1-8B")]
LADDER_NONCE = "gguf-ladder-2026-11"


def formats(src, names):
    """How the tensors that were read are stored: {"Q4_K": 9, "Q6_K": 5}."""
    return dict(collections.Counter(src.index[n][1][len(G.PREFIX):] for n in names if n in src.index))


def sets(files):
    """Every GGUF file of a repository, shards counted once: {name without the shard number: [files]}."""
    out = {}
    for f in files:
        if f.lower().endswith(".gguf") and "mmproj" not in f.lower() and "imatrix" not in f.lower():
            out.setdefault(G._SHARD.sub("", f), []).append(f)
    return out


def rung(job):
    """One GGUF file against the original it was converted from and against the known base."""
    original, repo, base, key, files, size = job
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    t0, row = time.time(), {"original": original, "repo": repo, "base": base, "file": key}
    try:
        first, known, src = R.Remote(original), R.Remote(base), R.Remote(repo, file=key)
        h = json.load(open(os.path.join(src.dir, files[0].replace("/", "--") + ".header.json")))
        numbers = sum(int(torch.tensor(t[1]).prod()) for t in h["tensors"].values())
        row["bits_per_weight"] = round(8 * size / numbers, 2)
        row["tensors_decoded"], row["tensors"] = len(src.index), len(h["tensors"])
        cache = {}
        d = LT.distances(src, first, LADDER_NONCE, cache=cache)
        if d is None:
            raise ValueError("no matrix in a format the reader decodes")
        used = cache[next(k for k in cache if k[0] == id(src))]["used"]
        row["formats"] = formats(src, used + [U.layout(src)[0]])
        b = LT.distances(src, known, LADDER_NONCE, cache=cache)
        found = U.edit_directions(src, known, LT.layers_for(src, LADDER_NONCE, LT.CUT_LAYERS), LT.layers_for(known, LADDER_NONCE, LT.CUT_LAYERS))
        if found:
            lead = torch.stack([f[1][:, 0] for f in found])
            row["writers"] = {"layer": [f[0] for f in found], "share": [round(f[2], 3) for f in found],
                              "size": [round(f[3], 3) for f in found], "own": [round(f[4], 3) for f in found],
                              "agreement": [round(abs(float(lead[i] @ lead[i + 1])), 3) for i in range(0, len(found) - 1, 2)
                                            if found[i][0] == found[i + 1][0]]}
        cut = LT.direction_cut(src, known, LADDER_NONCE)
        row.update(to_original=[round(max(x, 0), 4) for x in d], to_base=[round(max(x, 0), 4) for x in b],
                   cut_test={True: "cut", False: "compared, no cut", None: "cannot be compared"}[cut],
                   light="red" if cut else "green" if b[1] < LT.GREEN and cut is False else "amber")
        shutil.rmtree(src.dir, ignore_errors=True)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:160]}"
    row["seconds"] = round(time.time() - t0)
    return row


def ladder(out, workers=6):
    api, jobs = HfApi(), []
    for original, repo, base in LADDER:
        size = {s.rfilename: s.size for s in api.model_info(repo, files_metadata=True).siblings}
        jobs += [(original, repo, base, key, files, sum(size[f] for f in files)) for key, files in sorted(sets(size).items())]
    rows = []
    with mp.get_context("spawn").Pool(int(workers)) as pool:
        for row in pool.imap_unordered(rung, jobs):
            rows.append(row)
            print(json.dumps(row), flush=True)
            json.dump(sorted(rows, key=lambda r: (r["repo"], -r.get("bits_per_weight", 0))), open(out, "w"), indent=1)


ANCHORS = {}


def _start(anchors):
    torch.set_grad_enabled(False)
    torch.set_num_threads(2)
    ANCHORS.update(anchors)


def read(c):
    """One repository the first scan skipped: the GGUF file it holds, and its distance to every listed model."""
    t0, row = time.time(), {k: c[k] for k in ("repo", "listed", "relation", "downloads")}
    try:
        src = R.Remote(c["repo"])
        if not src.gguf:
            row["unread"] = "no GGUF file"
        else:
            row["file"], row["shards"], row["tensors_decoded"] = src.gguf[0], len(src.gguf), len(src.index)
            try:
                vocab = src.vocab
            except Exception:
                vocab = None
            if vocab is None or not src.index:
                row["unread"] = "no token list" if src.index else "no tensor in a format the reader decodes"
            else:
                dist, kept, used = {}, {}, []
                for mid, (fa, strings) in ANCHORS.items():
                    fq = U.fingerprint(src, vocab, strings)
                    r = U.compare(fa, fq)
                    dist[mid], kept[mid], used = (None if r is None else round(r["d"], 4)), len(fq["keep"]), fq["used"] or used
                ok = {k: v for k, v in dist.items() if v is not None}
                best = min(ok, key=ok.get) if ok else None
                row.update(distances=dist, tokens_found=kept, formats=formats(src, used),
                           matched=best if best and ok[best] < LT.RED else None)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:160]}"
    row["seconds"] = round(time.time() - t0)
    shutil.rmtree(os.path.join(R.CACHE, c["repo"].replace("/", "--")), ignore_errors=True)
    return row


def watchlist(wl_path, scan, out, workers=4):
    wl = json.load(open(wl_path))
    todo = [r for r in json.load(open(scan))["rows"] if r.get("skip")]
    done = {json.loads(l)["repo"] for l in open(out)} if os.path.exists(out) else set()
    anchors = {}
    for m in wl["models"]:
        src = R.Remote(m["repo"])
        strings = fp4.select_strings(src.vocab, NONCE, 256)
        anchors[m["id"]] = (U.fingerprint(src, src.vocab, strings), strings)
        print("listed", m["id"], "layers", anchors[m["id"]][0]["layer_ids"], flush=True)
    with mp.get_context("spawn").Pool(workers, _start, (anchors,)) as pool, open(out, "a") as f:
        for row in pool.imap_unordered(read, [c for c in todo if c["repo"] not in done]):
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{row['repo'][:60]:60s} {row['relation']:9s} -> {str(row.get('matched')):8s} "
                  f"{row.get('distances') or row.get('unread') or row.get('error')}", flush=True)


def summary(path, out):
    rows = [json.loads(l) for l in open(path)]
    read_ = [r for r in rows if "distances" in r]
    s = {"repositories": len(rows), "read": len(read_), "matched": sum(bool(r["matched"]) for r in read_),
         "not_read": dict(collections.Counter(r.get("unread") or "error" for r in rows if "distances" not in r)),
         "matched_by_relation": dict(collections.Counter(r["relation"] for r in read_ if r["matched"])),
         "not_matched_by_relation": dict(collections.Counter(r["relation"] for r in read_ if not r["matched"])),
         "furthest_match": max((min(v for v in r["distances"].values() if v is not None) for r in read_ if r["matched"]), default=None),
         "closest_non_match": min((min(v for v in r["distances"].values() if v is not None) for r in read_
                                   if not r["matched"] and any(v is not None for v in r["distances"].values())), default=None),
         "matched_repos": sorted(r["repo"] for r in read_ if r["matched"])}
    json.dump(s, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in s.items() if k != "matched_repos"}, indent=1))


if __name__ == "__main__":
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == "ladder":
        ladder(*args[:2])
    elif mode == "watchlist":
        watchlist(args[0], args[1], args[2], *[int(x) for x in args[3:4]])
    else:
        summary(args[0], args[1])
