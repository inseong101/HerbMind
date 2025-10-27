import sys

from herbmind.cli import build_parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args(['test', *sys.argv[1:]])
    args.func(args)
