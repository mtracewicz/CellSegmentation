#!/bin/bash
/usr/bin/python3 -m pip install --upgrade pip
pip install -r /CellSegmentation/requirements.txt 
groupadd -g $U dev
useradd -u $U -g $U -ms /bin/bash dev
chown -R dev:dev /CellSegmentation
