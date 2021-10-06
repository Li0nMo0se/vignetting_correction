# Vignetting correction

## Paper

The implementation is based on Revisiting [Image Vignetting Correction by Constrained Minimization of Log-Intensity Entropy](https://www.researchgate.net/publication/300786398_Revisiting_Image_Vignetting_Correction_by_Constrained_Minimization_of_Log-Intensity_Entropy).

It has been implemented in Python.

## Notebook

The notebook [vignetting.ipynb](https://github.com/Li0nMo0se/vignetting_correction/blob/main/vignetting.ipynb) contains all the details and explanation about why does it work and how we implemented the approach.

## Python

### Package vignetting

### Installation
```shell
pip install vignetting --extra-index-url=https://pypi.org/simple/
```

### Usage

```python
from vignetting import correct_vignetting

sigma_smooth = SIGMA_VALUE # (default: 2.25 if not given)
correct_vignetting(input_image, sigma_smooth)
```

### Main script

Download the [main.py](https://github.com/Li0nMo0se/vignetting_correction/blob/main/main.py) from the [Github repository](https://github.com/Li0nMo0se/vignetting_correction/)

```
usage: main.py [-h] --input INPUT [--output OUTPUT] [--sigma SIGMA]

Apply vignetting correction over an image

optional arguments:
  -h, --help       show this help message and exit
  --input INPUT    Input filename
  --output OUTPUT  Output filename
  --sigma SIGMA    Specify the sigma smoothness value for the log entropy computation
```

## Results

<table>
  <thead>
    <tr>
      <th>Input</th>
      <th>Output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><img src="https://raw.githubusercontent.com/Li0nMo0se/vignetting_correction/main/img/sample-1.jpeg"></td>
      <td><img src="https://raw.githubusercontent.com/Li0nMo0se/vignetting_correction/main/img/sample-1-output.jpeg"></td>
    </tr>
  </tbody>
</table>
