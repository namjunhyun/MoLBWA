#!/bin/bash
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV
export DISPLAY=:0
exec /home/nooyjs/MoLBWA-gaze-hri/run_gui_dual.sh
