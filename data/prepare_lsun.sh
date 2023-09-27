git clone https://github.com/fyu/lsun.git
cd lsun
pip install lmdb
python3 download.py -c bedroom
unzip bedroom_val_lmdb.zip
python3 lsun_prepare bedroom_val_lmdb # data dir for putting lsun
