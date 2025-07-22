#SBATCH --account=def-cmaddis
#SBATCH --time=0-02:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=256M

import argparse
import os
import subprocess


def main(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_dir = timestamp
    csv_output = timestamp + '.csv'

    if os.path.isdir(final_dir):
        print('{} already exists; return'.format(final_dir))
        return

    tmp_dir = '{}.tmp'.format(final_dir)

    args_list = ['python', args.py_file, csv_output] + args.optional_args

    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)

    try:
        subprocess.run(args_list, cwd=tmp_dir, check=True)
        os.rename(tmp_dir, final_dir)
        print('run complete; renamed {} to {}'.format(tmp_dir, final_dir))
    except subprocess.CalledProcessError as e:
        print('error: run failed with code {}'.format(e.returncode))
        print('tmp output preserved in'.format(tmp_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('py_file', help='adjoint sampling file name')
    parser.add_argument('optional_args', nargs='*',
            help='additional arguments for adjoint sampling program')

    args = parser.parse_args()

    main(args)
