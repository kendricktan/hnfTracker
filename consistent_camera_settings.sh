#!/usr/bin/env bash

# sudo v4l2-ctl --list-ctrls --device=/dev/video1



sudo v4l2-ctl --set-ctrl exposure_auto=1 --device=/dev/video1
sudo v4l2-ctl --set-ctrl exposure_absolute=20 --device=/dev/video1
