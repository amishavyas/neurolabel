# neurolabel

neurolabel provides a small, installable Python interface for loading and working with labeled neuroimaging atlases while keeping atlas metadata and scientific assumptions explicit.

```bash
python3 -m pip install -e .
```

```python
import neurolabel

atlas = neurolabel.load_atlas("neurosynth_k50")
print(atlas.parcel_ids)
```

See the [documentation](docs/) and [contribution guide](CONTRIBUTING.md).
