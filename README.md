# MMF-PRT

## Run

Make sure the project structure is like this:

```bash
MMF-PRT/
└── code/
    └── MMF-PRT.py
└── dataset/
    └── dataset.db
```

The script uses the default database path:

```python
DB_PATH = '../dataset/dataset.db'
```

So run the script in the same directory as `MMF-PRT.py`.

### Windows

```bash
python -m venv venv
venv\Scripts\activate
pip install numpy pandas scikit-learn networkx sentence-transformers
pip install torch torchvision torchaudio
pip install torch-geometric
python MMF-PRT.py
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
pip install numpy pandas scikit-learn networkx sentence-transformers
pip install torch torchvision torchaudio
pip install torch-geometric
python MMF-PRT.py
```

## Notes

- Make sure `dataset/dataset.db` exists before running the script.

- The dataset is not included in this repository due to file size limitations.
  You can download it from the following Google Drive link:

  `https://drive.google.com/file/d/10weiIjX_a_BhlxCHnsO8unlaFlqXG7-d/view?usp=drive_link`
