#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : computedose.py
# Time       ：2024/3/14 14:41
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# version    ：python 3.10.11
# Description：
"""
import argparse


def read_file(file_name):
    with open(file_name, 'r') as file:
        blocks = {}
        current_block = None
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith('[ZValue') and line.endswith(']'):
                block_name = line.split('=')[-1].strip().rstrip(']')
                current_block = block_name
                blocks[current_block] = {}
            elif current_block:
                key, value = line.split('=')
                blocks[current_block][key.strip()] = value.strip()
    return blocks


def write_log_file(blocks, pixelsize, output_file):
    with open(output_file, 'w') as file:
        file.write("TiltAngle\tDoseRate\tExposureTime\tDose\n")
        sorted_blocks = sorted(blocks.items(), key=lambda x: float(x[1]['TiltAngle']))
        for block_name, block_data in sorted_blocks:
            tilt_angle = float(block_data.get('TiltAngle', 0))
            dose_rate = float(block_data.get('DoseRate', 0))
            exposure_time = float(block_data.get('ExposureTime', 0))
            dose = dose_rate * exposure_time / pixelsize / pixelsize
            file.write(f"{tilt_angle}\t{dose_rate}\t{exposure_time}\t{dose}\n")


def write_doseperview_file(blocks, pixelsize, output_file):
    with open(output_file, 'w') as file:
        sorted_blocks = sorted(blocks.items(), key=lambda x: float(x[1]['TiltAngle']))
        for block_name, block_data in sorted_blocks:
            dose_rate = float(block_data.get('DoseRate', 0))
            exposure_time = float(block_data.get('ExposureTime', 0))
            dose = dose_rate * exposure_time / pixelsize / pixelsize
            file.write(f"{dose}\n")


# example: python process_file.py your_file.txt --pixelsize 1.68 --log output_log.txt --doseperview output_doseperview.txt
read_file('20240226_R_0003.mdoc')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a file containing blocks of data')
    parser.add_argument('file', type=str, help='File name to process')
    parser.add_argument('--pixelsize', type=float, help='in Angstrom')
    parser.add_argument('--log', type=str, help='Log file name', default='output_log.txt')
    parser.add_argument('--doseperview', type=str, help='Dose per view file name', default='output_doseperview.txt')
    args = parser.parse_args()

    file_name = args.file
    blocks = read_file(file_name)

    write_log_file(blocks, args.pixelsize, args.log)
    write_doseperview_file(blocks, args.pixelsize, args.doseperview)
