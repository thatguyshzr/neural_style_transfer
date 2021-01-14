# Neural Style Transfer

Basically, the model learns the style and content of the painting and manipulates the given image accordingly.

The code is based on [this colab notebook].

![alt text](https://github.com/thatguyshzr/neural_style_transfer/blob/main/output_images/content_style_images.png)
![alt text](https://github.com/thatguyshzr/neural_style_transfer/blob/main/output_images/vulture.png)

## Running the code
### Dependencies
* `matplotlib==3.2.1`
* `numpy==1.19.5`
* `Pillow==8.0.1`
* `tensorflow==2.4.0`

### Setting up
Open command prompt where you want to clone the code.

> git clone https://github.com/thatguyshzr/neural_style_transfer.git <br> cd neural_style_transfer <br> pip install  -r requirements.txt

## Command-line Execution
Run the example (use 1000 iterations for best result): `python .\run_nst.py --input '.\input_images\turtle.jpg' --style '.\input_images\wave.jpg' --iter 100`

Arguments: `python run_nst.py [-h] --input INPUT --style STYLE [--iter ITER]`

Optional arguments:

| | |
|-------------|--------|
|`--h`, `--help`|show help message and exit |
|`--in`, `--input`|input image|
|`--s`, `--style`|style image|
|`--it`, `--iter`|number of iterations (default: 100)|

---

[this colab notebook]:https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiDvojLlpvuAhXXcn0KHQzpBvYQFjAAegQIBRAC&url=https%3A%2F%2Fcolab.research.google.com%2Fgithub%2Ftensorflow%2Fmodels%2Fblob%2Fmaster%2Fresearch%2Fnst_blogpost%2F4_Neural_Style_Transfer_with_Eager_Execution.ipynb&usg=AOvVaw3xlFLR_ihslRjdBhdFT60x
