from argparse import ArgumentParser


def main(args):
    print(f'Hello World! We have some args, like the alpha {args["alpha"]}')


def parse_all_args():
    parser = ArgumentParser()

    parser.add_argument('alpha',
                        '-a',
                        type=float,
                        default=1)
    parser.add_argument('rho',
                        '-r',
                        type=float,
                        default=1)
    parser.add_argument('dataset',
                        '-d',
                        type=str,
                        choices=['voc'],
                        default='voc')
    parser.add_argument('expansion',
                        '-t',
                        type=int,
                        default=6)

    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_all_args()
    main(arguments)
