# California house price predictor

Python trains a model and exports it to JSON. A plain HTML/CSS/JS page runs it
in the browser: pick a spot on the map, tune the neighborhood, and get a live
price estimate. No server needed, so it hosts free on GitHub Pages.

## Run locally
```bash
pip install -r requirements.txt
python train.py                  # creates docs/model.json
cd docs && python -m http.server 8000   # open http://localhost:8000
```

## Publish on GitHub Pages
1. Commit everything, including `docs/model.json`, and push.
2. Repo **Settings, Pages**: Source = *Deploy from a branch*, Branch = `main`, Folder = `/docs`.
3. Your site appears at `https://<username>.github.io/<repo-name>/` after a minute.

## Structure
```
train.py          trains the model, writes docs/model.json
docs/index.html   page
docs/style.css    styles
docs/script.js    map, sliders, and the in-browser model
```

## Notes
- Data is the 1990 California census, so prices reflect that era and stop at $500,000.
- Add a screenshot to this README after you run it.
