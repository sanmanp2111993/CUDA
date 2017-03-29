#!/usr/bin/python

import os
import subprocess

if os.path.isfile("parallel_optimized"):
	subprocess.call("rm parallel_optimized",shell = True)
if os.path.isfile("original_image.px"):
	subprocess.call("rm original_image.px",shell = True)
if os.path.isfile("output_image.jpg"):	
	subprocess.call("rm output_image.jpg",shell = True)
if os.path.isfile("output_image.px"):
	subprocess.call("rm output_image.px",shell = True)
if os.path.isfile('compress_output_image.jpg'):
        subprocess.call("rm compress_output_image.jpg",shell=True)
if os.path.isfile('compress_output_image.px'):
        subprocess.call("rm compress_output_image.px",shell=True)




