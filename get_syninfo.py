from twasp_helper import request_features_from_stanford, request_features_from_berkeley
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        default=None,
                        type=str,
                        required=True,
                        help="The dataset. Should be one of \'CTB5\', \'CTB6\', \'CTB7\', \'CTB9\', "
                             "\'UD1\' and \'UD2\'.")

    parser.add_argument("--toolkit",
                        default=None,
                        type=str,
                        required=True,
                        help="The toolkit to be used. Should be one of \'SCT\' and \'BNP\'.")

    parser.add_argument("--overwrite",
                        action='store_true',
                        help="Whether to overwrite existing data.")

    args = parser.parse_args()

    print(vars(args))

    input_dir = os.path.join('./data/', args.dataset)

    if args.overwrite:
        print('All existing files will be overwrote')

    for flag in ['train', 'dev', 'test']:
        input_file = os.path.join(input_dir, flag + '.tsv')
        if not os.path.exists(input_file):
            print('File does not exits: %s' % str(input_file))
            continue

        if args.toolkit == 'SCT':
            out_file = os.path.join(input_dir, flag + '.stanford.json')
            if os.path.exists(out_file) and not args.overwrite:
                print('File already exists: %s' % str(out_file))
                continue
            request_features_from_stanford(input_file)

        elif args.toolkit == 'BNP':
            out_file = os.path.join(input_dir, flag + '.berkeley.json')
            if os.path.exists(out_file) and not args.overwrite:
                print('File already exists: %s' % str(out_file))
                continue
            request_features_from_berkeley(input_file)
        else:
            raise ValueError('Invalid type of toolkit name: %s. Should be one of \'SCT\' and \'BNP\'.' % args.toolkit)


