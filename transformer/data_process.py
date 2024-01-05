import pandas as pd
import re
import argparse
import os


def parse_line_updates(line):
    # Updated regular expression to match exponential format for 'lr'
    pattern = r'\|\s*epoch\s*(\d+):.*loss=(\d+\.\d+), nll_loss=(\d+\.\d+), ppl=(\d+\.\d+), wps=(\d+), ups=(\d+), wpb=(\d+\.\d+), bsz=(\d+\.\d+), num_updates=(\d+), lr=([\d\.]+e?[-\+]?\d*), gnorm=(\d+\.\d+), clip=(\d+\.\d+), oom=(\d+\.\d+), wall=(\d+), train_wall=(\d+)'
    match = re.search(pattern, line)
    if match:
        return {
            'epoch': int(match.group(1)),
            'loss': float(match.group(2)),
            'nll_loss': float(match.group(3)),
            'ppl': float(match.group(4)),
            'wps': int(match.group(5)),
            'ups': int(match.group(6)),
            'wpb': float(match.group(7)),
            'bsz': float(match.group(8)),
            'num_updates': int(match.group(9)),
            'lr': float(match.group(10)),  # Converts exponential format to float
            'gnorm': float(match.group(11)),
            'clip': float(match.group(12)),
            'oom': float(match.group(13)),
            'wall': int(match.group(14)),
            'train_wall': int(match.group(15))
        }
    return None

def parse_line_epochs(line):
    # Regular expression for the new format, supporting 'lr' in exponential format
    pattern = r'\|\s*epoch\s*(\d+)\s*\|\s*loss\s*(\d+\.\d+)\s*\|\s*nll_loss\s*(\d+\.\d+)\s*\|\s*ppl\s*(\d+\.\d+)\s*\|\s*wps\s*(\d+)\s*\|\s*ups\s*(\d+)\s*\|\s*wpb\s*(\d+\.\d+)\s*\|\s*bsz\s*(\d+\.\d+)\s*\|\s*num_updates\s*(\d+)\s*\|\s*lr\s*([\d\.]+e?[-\+]?\d*)\s*\|\s*gnorm\s*(\d+\.\d+)\s*\|\s*clip\s*(\d+\.\d+)\s*\|\s*oom\s*(\d+\.\d+)\s*\|\s*wall\s*(\d+)\s*\|\s*train_wall\s*(\d+)'
    match = re.search(pattern, line)
    if match:
        return {
            'epoch': int(match.group(1)),
            'loss': float(match.group(2)),
            'nll_loss': float(match.group(3)),
            'ppl': float(match.group(4)),
            'wps': int(match.group(5)),
            'ups': int(match.group(6)),
            'wpb': float(match.group(7)),
            'bsz': float(match.group(8)),
            'num_updates': int(match.group(9)),
            'lr': float(match.group(10)),
            'gnorm': float(match.group(11)),
            'clip': float(match.group(12)),
            'oom': float(match.group(13)),
            'wall': int(match.group(14)),
            'train_wall': int(match.group(15))
        }
    return None

def parse_line_validation_stats(line):
    # Regular expression for parsing validation stats
    pattern = r'\|\s*epoch\s*(\d+)\s*\|\s*valid on \'valid\' subset\s*\|\s*loss\s*(\d+\.\d+)\s*\|\s*nll_loss\s*(\d+\.\d+)\s*\|\s*ppl\s*(\d+\.\d+)\s*\|\s*num_updates\s*(\d+)'
    match = re.search(pattern, line)
    if match:
        return {
            'epoch': int(match.group(1)),
            'loss': float(match.group(2)),
            'nll_loss': float(match.group(3)),
            'ppl': float(match.group(4)),
            'num_updates': int(match.group(5))
        }
    return None

def parse_file(file_path):
    updates_data = []
    epoch_data = []
    val_data = []
    with open(file_path, 'r') as file:
        for line in file:
            parsed_line_updates = parse_line_updates(line)
            if parsed_line_updates:
                updates_data.append(parsed_line_updates)
            parsed_line_epochs = parse_line_epochs(line)
            if parsed_line_epochs:
                epoch_data.append(parsed_line_epochs)
            parsed_line_val = parse_line_validation_stats(line)
            if parsed_line_val:
                val_data.append(parsed_line_val)

    return pd.DataFrame(updates_data), pd.DataFrame(epoch_data), pd.DataFrame(val_data)


if __name__ == '__main__':
    
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-dir', type=str, default='')
    parser.add_argument('--file-name', type=str, default='')
    parser.add_argument('--output-dir', type=str, default='')

    args = parser.parse_args()

    if args.file_dir == '' or args.file_name == '' or args.output_dir == '':
        print('Please specify.')
        exit(0)

    file_path = os.path.join(args.file_dir, args.file_name)
    
    # create output path of not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # parse file
    updates_data, epoch_data, val_data = parse_file(file_path)

    # save to csv
    updates_data.to_csv(os.path.join(args.output_dir, 'updates_stats.csv'))
    epoch_data.to_csv(os.path.join(args.output_dir, 'epoch_stats.csv'))
    val_data.to_csv(os.path.join(args.output_dir, 'val_stats.csv'))

    