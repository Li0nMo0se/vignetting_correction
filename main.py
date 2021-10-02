from argparse import ArgumentParser
from vignetting import correct_vignetting
import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = ArgumentParser(description="Apply vignetting correction over an image")
    parser.add_argument("--input", type=str, help="Input filename", required=True)
    parser.add_argument("--output", type=str, help="Output filename")
    parser.add_argument("--sigma", type=float, help="Specify the sigma smoothness value for the log entropy \
        computation")

    args = parser.parse_args()

    input_filename = args.input
    input_image = plt.imread(input_filename)
    output_image = correct_vignetting(input_image, args.sigma)

    if args.output is None:
        filename_split = input_filename.split('.')
        # Add `-output` before the extension and after the filename
        # i.e `filename.jpeg` -> `filename-output.jpeg`
        output_filename = '.'.join(filename_split[:-1]) + "-output" + '.' + filename_split[-1]
    else:
        output_filename = args.output
    plt.imsave(output_filename, output_image)


