import argparse

# Create the parser and add arguments
parser = argparse.ArgumentParser()
# Cast the input to int 
parser.add_argument('--f', '--first', type=int, help="first number",
                    required=False, default= 2)
parser.add_argument('--s', '--second', type=int, help="second number",
                    required=False, default= 3)

# Parse and print the results
args = parser.parse_args()

def sum(a, b):
    return a+b

print('Output: ',sum(args.f, args.s))


# usage:
# >>> python exp.py --f 3 --s 4
# >>> python exp.py --first 3 --second 4
# >>> Output: 7