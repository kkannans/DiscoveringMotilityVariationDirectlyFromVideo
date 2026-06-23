# Prediction-surprise explorer (static site)

An interactive, server-less page: each of the 108 organoids is a point on a population scatter
(mean speed vs prediction surprise); clicking one plays its precomputed clip
[ ground truth | model prediction | prediction error ] and reports where it sits in the
population (percentile). No model or GPU runs in the browser.

Files: `index.html`, `app.js`, `style.css`, `data.json`, `clips/<batch>.mp4` (108 clips, ~16 MB).

## Rebuild the assets
```bash
bash stage_cache.sh          # or: python download_data.py   (populate ./outputs, ./dataset)
srun -p gpu-debug --gres=gpu:1 --cpus-per-task=4 --mem=64G \
  bash -lc '/users/k/k/kkannans/alife-2026/.venv/bin/python app/build_site_assets.py'
# → app/site/clips/*.mp4 + app/site/data.json
```

## Preview locally
```bash
cd app/site && python -m http.server 8754
# open http://localhost:8754  (port-forward if on a cluster)
```

## Deploy to GitHub Pages
The clips are committed (a `.gitignore` exception re-includes `app/site/clips/*.mp4`).

**Option A — `gh-pages` branch (site at the branch root):**
```bash
git add -f app/site && git commit -m "site assets"
git subtree push --prefix app/site origin gh-pages
```
Then: repo **Settings → Pages → Branch: gh-pages / (root)**.

**Option B — `/docs` on the main branch:** copy `app/site/*` into `docs/` and set
**Settings → Pages → Branch: <branch> / docs**.

Live URL: `https://<user>.github.io/<repo>/`. Embed anywhere with
`<iframe src="https://<user>.github.io/<repo>/" width="100%" height="720"></iframe>`.
