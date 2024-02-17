import argparse
import os.path
import os
from test_vectorization import main
import time

if __name__ == '__main__':
    start_time_point = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default='new_d', help="The test image path.")
    parser.add_argument('--model', '-m', type=str, default='pretrain_clean_line_drawings', help="The trained model.")
    parser.add_argument('--sample', '-s', type=int, default=1, help="The number of outputs.")
    args = parser.parse_args()

    assert args.input != ''
    assert args.sample > 0
    print(args.input)
    for filename in os.listdir(args.input):
        if 'png' in filename:
            main(args.model, args.input+'/'+filename, args.sample)
    end_time_point = time.time()
    print(end_time_point-start_time_point)


    #main(args.model, args.input, args.sample)K,