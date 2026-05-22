#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import ctypes
import pysurvive.pysurvive_generated
from pysurvive.pysurvive_generated import *

SURVIVE_BUTTON_MENU = 6
LP_c_char = ctypes.POINTER(ctypes.c_char)
LP_LP_c_char = ctypes.POINTER(LP_c_char)

def button_process(so, eventType, buttonId, axisIds, axisVals):
    if buttonId == SURVIVE_BUTTON_MENU:
        survive_reset_lighthouse_positions(so.ctx)

args = sys.argv[1:]
argc = len(args)
argv = (LP_c_char * (argc + 1))()
for i, arg in enumerate(args):
    enc_arg = arg.encode('utf-8')
    argv[i] = ctypes.create_string_buffer(enc_arg)

survive_verify_FLT_size(ctypes.sizeof(ctypes.c_double))
ctx = survive_init_internal(argc, argv, 0, ctypes.cast(None, log_process_func))

survive_startup(ctx)

survive_install_button_fn(ctx, button_process_func(button_process))

try:
    while survive_poll(ctx) == 0:
        time.sleep(0.01)
except KeyboardInterrupt:
    pass

survive_close(ctx)
