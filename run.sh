python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 train/late-fusion.py
python3 train/mid-fusion.py
python3 train/early-fusion.py
python3 train/mid-fusion-vit.py
python3 train/early-fusion-vit.py
