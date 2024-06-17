#!/bin/bash

#https://unix.stackexchange.com/questions/87908/how-do-you-empty-the-buffers-and-cache-on-a-linux-system
# free && sync && echo 3 > /proc/sys/vm/drop_caches && free

CONFIG_FILE="configs/rtmdet/golfpose/rtmdet_tiny_8xb32-300e_golfpose.py"

python tools/train.py $CONFIG_FILE --amp

