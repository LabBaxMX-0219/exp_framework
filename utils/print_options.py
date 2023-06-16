import sys

def print_options(logger):
    logger.debug("input cmd is:\n\t..."+" ".join(["python"]+sys.argv)+"\n")