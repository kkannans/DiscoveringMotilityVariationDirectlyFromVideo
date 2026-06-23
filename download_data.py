"""
download_data.py — fetch the dataset, model weights, and cached artifacts from Dropbox and
unpack them into the repo's paths.py layout (./dataset + ./outputs/...), so figures and
comparisons reproduce without retraining.

Usage:
    python download_data.py                  # download + extract everything in data_manifest.json
    python download_data.py --only weights   # one bucket
    python download_data.py --skip-existing  # don't re-download buckets already present

Each manifest entry has a Dropbox `url` (use dl=1), a `size_mb` (sanity print), and a `marker`
file used to detect "already extracted". Tarballs contain repo-relative paths (dataset/...,
outputs/...), so they extract at the repo root.
"""
import argparse
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data_manifest.json"


def _download(url, dest):
    if "REPLACE_ME" in url or "REPLACE" in url:
        sys.exit(f"ERROR: manifest URL not filled in: {url}\n"
                 f"Edit data_manifest.json with your Dropbox links (dl=1). See REPRODUCIBILITY.md.")
    print(f"  downloading -> {dest.name}")
    last = [-1]
    def _hook(blk, bs, total):
        if total > 0:
            pct = min(100, blk * bs * 100 // total)
            if pct != last[0] and pct % 5 == 0:   # print at most every 5%
                last[0] = pct
                print(f"\r    {pct:3d}%", end="", flush=True)
    urllib.request.urlretrieve(url, dest, _hook)
    print()


def _extract(tar_path, root):
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(root)  # members are repo-relative (dataset/..., outputs/...)


def _postprocess():
    """Populate paths.py-derived locations not in the tarballs + checkpoint name aliases."""
    out = ROOT / "outputs"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    # datainfo: committed splits + test_assignments → outputs/datainfo (paths.DATAINFO_DIR)
    di = out / "datainfo"; di.mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "datainfo").glob("data_split_covering_*.json"):
        shutil.copy(f, di / f.name)
    ta = out / "intermediates" / "test_assignments.json"
    if ta.exists() and not (di / "test_assignments.json").exists():
        shutil.copy(ta, di / "test_assignments.json")
    # checkpoint aliases: scripts use either *_residual (get_output_dir) or *_pre_residual (fig scripts)
    models = out / "models"
    if models.exists():
        for d in list(models.glob("*_pre_residual")):
            alias = models / d.name.replace("_pre_residual", "_residual")
            if not alias.exists():
                alias.symlink_to(d.name)  # relative symlink within outputs/models


def main():
    ap = argparse.ArgumentParser(description="Download + unpack data from Dropbox")
    ap.add_argument("--only", nargs="+", help="Only these buckets (names from manifest)")
    ap.add_argument("--skip-existing", action="store_true", help="Skip buckets whose marker exists")
    args = ap.parse_args()

    entries = json.loads(MANIFEST.read_text())["files"]
    if args.only:
        entries = [e for e in entries if e["name"] in args.only]

    with tempfile.TemporaryDirectory(prefix="kkannans-dl-") as tmp:
        for e in entries:
            marker = ROOT / e["marker"]
            if args.skip_existing and marker.exists():
                print(f"[{e['name']}] present ({marker.relative_to(ROOT)}) — skip")
                continue
            print(f"[{e['name']}] ~{e['size_mb']} MB")
            tar_path = Path(tmp) / f"{e['name']}.tar.gz"
            _download(e["url"], tar_path)
            _extract(tar_path, ROOT)
            tar_path.unlink(missing_ok=True)
            ok = "OK" if marker.exists() else "WARN: marker missing after extract"
            print(f"  extracted [{ok}]")

    _postprocess()
    # quick census
    nmp4 = len(list((ROOT / "dataset").glob("*.mp4"))) if (ROOT / "dataset").exists() else 0
    nw = len(list((ROOT / "outputs/models").glob("*_pre_residual/best_model.pth")))
    nnpz = len(list((ROOT / "outputs/intermediates/per_organoid_residuals").glob("*.npz"))) \
        if (ROOT / "outputs/intermediates/per_organoid_residuals").exists() else 0
    print(f"\nDone. dataset={nmp4} mp4s, weights={nw} checkpoints, residuals={nnpz} npz.")
    print("Now run REPRODUCIBILITY.md §Runbook Step 2.")


if __name__ == "__main__":
    main()
