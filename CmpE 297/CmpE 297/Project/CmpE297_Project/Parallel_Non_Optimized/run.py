#!/usr/bin/python

import os
import subprocess

subprocess.call("java -jar Image2PX.jar original_image.jpg original_image.px",shell = True)
subprocess.call("nvcc parallel_non_optimized.cu -o parallel_non_optimized",shell = True)
subprocess.call("./parallel_non_optimized original_image.px output_image.px compress_output_image.px",shell = True)
subprocess.call("java -jar PX2Image.jar output_image.px jpg output_image.jpg",shell = True)
subprocess.call("java -jar PX2Image.jar compress_output_image.px jpg compress_output_image.jpg",shell=True)

