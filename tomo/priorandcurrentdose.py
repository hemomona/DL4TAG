#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : priorandcurrentdose.py
# Time       ：2024/3/14 20:47
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# version    ：python 3.10.11
# Description：
"""
import argparse


def read_file(file_name, pixelsize):
    with open(file_name, 'r') as file:
        blocks = {}
        current_block = None
        total_dose = 0  # 该值为前几张累积dose + 当前dose ！！！

        for line in file:
            line = line.strip()
            if not line:
                continue

            if line.startswith('[ZValue') and line.endswith(']'):
                if current_block:  # 如果是第二个到倒数第二个block，再赋给current_block block_name前，它就是"前一个"
                    dose_rate = float(blocks[current_block]['DoseRate'])
                    exposure_time = float(blocks[current_block]['ExposureTime'])
                    curr_dose = dose_rate * exposure_time / pixelsize / pixelsize
                    total_dose += curr_dose
                    blocks[current_block]['CurrDose'] = curr_dose
                    blocks[current_block]['TotalDose'] = total_dose
                # 包括了第一个到倒数第二个block的情况
                block_name = line.split('=')[-1].strip().rstrip(']')
                current_block = block_name
                blocks[current_block] = {}
            elif current_block:
                key, value = line.split('=')
                blocks[current_block][key.strip()] = value.strip()
        # 最后一个block
        dose_rate = float(blocks[current_block]['DoseRate'])
        exposure_time = float(blocks[current_block]['ExposureTime'])
        curr_dose = dose_rate * exposure_time / pixelsize / pixelsize
        total_dose += curr_dose
        blocks[current_block]['CurrDose'] = curr_dose
        blocks[current_block]['TotalDose'] = total_dose
    return blocks


def write_log_file(blocks, output_file):
    with open(output_file, 'w') as file:
        file.write("TiltAngle\tDoseRate\tExposureTime\tCurrDose\tPriorDose\n")
        sorted_blocks = sorted(blocks.items(), key=lambda x: float(x[1]['TiltAngle']))
        for block_name, block_data in sorted_blocks:
            tilt_angle = float(block_data.get('TiltAngle', 0))
            dose_rate = float(block_data.get('DoseRate', 0))
            exposure_time = float(block_data.get('ExposureTime', 0))
            curr_dose = float(block_data.get('CurrDose', 0))
            prior_dose = float(block_data.get('TotalDose', 0)) - curr_dose
            file.write(f"{tilt_angle}\t{dose_rate}\t{exposure_time}\t{curr_dose}\t{prior_dose}\n")


def write_doseperview_file(blocks, output_file):
    with open(output_file, 'w') as file:
        sorted_blocks = sorted(blocks.items(), key=lambda x: float(x[1]['TiltAngle']))
        for block_name, block_data in sorted_blocks:
            curr_dose = float(block_data.get('CurrDose', 0))
            prior_dose = float(block_data.get('TotalDose', 0)) - curr_dose
            file.write(f"{prior_dose}\t{curr_dose}\n")


# example: python process_file.py your_file.txt --pixelsize 1.68 --log output_log.txt --doseperview output_doseperview.txt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a file containing blocks of data')
    parser.add_argument('file', type=str, help='File name to process')
    parser.add_argument('--pixelsize', type=float, help='in Angstrom', default=1.68)
    parser.add_argument('--log', type=str, help='Log file name', default='log.txt')
    parser.add_argument('--doseperview', type=str, help='Dose per view file name', default='priorandcurrdose.txt')
    args = parser.parse_args()

    file_name = args.file
    blocks = read_file(file_name, args.pixelsize)

    write_log_file(blocks, args.log)
    write_doseperview_file(blocks, args.doseperview)
