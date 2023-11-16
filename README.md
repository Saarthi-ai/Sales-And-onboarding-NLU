<h1 align="center">saarthi-train: Saarthi's Python library for training text models</h1>

<div align="center">
    <img alt="python" src="https://img.shields.io/badge/Python-3.11.4-blue">
    <img alt="pytorch" src="https://img.shields.io/badge/PyTorch-2.0.1-orange">
    <img alt="lightning" src="https://img.shields.io/badge/PyTorch_Lightning-2.0.8-violet">
</div>
</br>
A Python library written in PyTorch Lightning to train text and sequence classification models.


## Setup

There are 2 ways to use this repository:
1. Install the requirements and use the codebase as is.
2. Install the code as a python package.

### Installing the requirements

> *Note: This is the recommended way if you're using this code for the first time.*

```bash
pip install -r requirements.txt
```

### Installing as a package

To execute the following command, make sure that you're in the saarthi-train directory:
```bash
pip install -e .
```

## Usage

The library can be used for 3 purposes:
1. Training
2. Local inference
3. Containerized inference

Let's take a look at each of these one by one.

### Training

The `train.py` file inside the `saarthi_train` directory is the main entrypoint for training runs.

The function of interest in the file is the `train` method. It's the main function that encapsulates the entire training process for your models.

#### The training configuration

The function requires a training configuration to be provided to it. The configuration is just a dictionary, and looks something like this:

```python
{
    'input_path': '<path to input files>',
    'root_output_path': '<path to output directory>',
    'max_seq_len': 26,
    'precision': '16-mixed',
    'task': 'intent_classification',
    'teacher_name': 'xlm-roberta',
    'teacher_epochs': 3,
    'teacher_batch_size': 16,
    'teacher_dropout': 0.1,
    'teacher_clip_grad_norm': None,
    'teacher_early_stopping_patience': 2,
    'enable_teacher_checkpointing': True,
    'teacher_output_folder': 'teacher',
    'torchscript': False,
    'quantization': None,
    'teacher_exists': False,
    'distillation': True,
    'teacher_layer_to_distil': 6,
    'student_lstm_dim': 600,
    'student_lstm_layers': 1,
    'student_lstm_dropout': 0.3,
    'student_word_emb_dim': 300,
    'student_epochs': 200,
    'student_batch_size': 512,
    'student_output_folder': 'student',
    'fast_dev_run': True,
    'production': False
}
```

Here's a small explanation of each required field:
- `input_path` (`str`): Path to the directory where your training data is present.
- `root_output_path` (`str`): Path to the directory where you want to save your model outputs.
- `max_seq_len` (`int`): Maximum sequence length of the model.
- `precision` (`str|int`): Precision to be used during training. Can be one of the following: `64`, `32`, `16`, `'16-mixed'`, `'bf16-mixed'`
- `task` (`str`): Name of the training task. Currently supported: `'intent_classification'`, `'text_classification'`, `'ner'`, `'pos'`. The supported values can be changed in the `saarthi_train/utils/__init__.py` file.
- `teacher_name` (`str`): Name of the transformer model to be used as the teacher. Currently supported: `'xlm-roberta'`, `'bert'`, `'muril'`. Supported values can be changed in the `saarthi_train/utils/general.py` file.
- `teacher_epochs` (`int`): Number of training epochs for the teacher model.
- `teacher_batch_size` (`int`): Batch size to be used for teacher fine-tuning.
- `teacher_dropout` (`float`): Dropout probability value for teacher fine-tuning.
- `teacher_clip_grad_norm` (`float|None`): Maximum value of gradient norm above which the gradients will be clipped. Supports any float value, `None` for no clipping.
- `teacher_early_stopping_patience` (`int`): Number of epochs to wait before early stopping stops the training due to no improvement.
- `enable_teacher_checkpointing` (`bool`): Whether to save the fine-tuned teacher model on disk.
- `teacher_output_folder` (`str`): Name of the output folder in which teacher model artifacts will be saved.
- `torchscript` (`bool`): Whether to serialize the models in TorchScript format.
- `quantization` (`str|None`): Which quantization to run on the saved models. Currently supported values are `'int8'`, `'float16'`.`None` for no quantization. Supported values can be changed in the `saarthi_train/utils/general.py` file.
- `teacher_exists` (`bool`): Whether an instanced of a fine-tuned teacher already exists in the output directory or not. If `True`, the code simply imports the saved model and skips teacher fine-tuning.
- `distillation` (`bool`): Whether to distil the fine-tuned teacher model into a smaller LSTM student model or not. Currently, only LSTM architecture is supported for the student model. The distillation procedure is based on this paper: [XtremeDistil: Multi-stage Distillation for Massive Multilingual Models](https://www.microsoft.com/en-us/research/uploads/prod/2020/04/XtremeDistil_ACL_2020.pdf)
- `teacher_layer_to_distil` (`int`): The layer of the teacher transformer model that needs to be used for the student model's learning (refer to the provided XtremeDistil paper if this does not make sense).
- `student_lstm_dim` (`int`): Hidden dimension of the LSTM.
- `student_lstm_layers` (`int`): Number of LSTM layers to use for the student model.
- `student_lstm_dropout` (`float`): Dropout to be used between the LSTM layers (only works if there are more than 1 LSTM layers).
- `student_word_emb_dim` (`int`): Dimension of the student model's word embeddings.
- `student_epochs` (`int`): Number of training epochs to be used for the student model.
- `student_batch_size` (`int`): Batch size for student model training.
- `student_output_folder` (`str`): Name of the directory in which the student model outputs need to be saved.
- `fast_dev_run` (`bool`): If `True`, the code will execute a single training epoch on a small subset of the training data. Useful to test the code end to end without having to actually train a model end to end.
- `production` (`bool`): If `True`, the code will delete all the artifacts (including training logs and unused checkpoints) that are not necessary for a production deployment.

A default set of values is already present in `train.py` towards the end of the file. You can refer to it in case you feel lost.

**A small note about the teacher configurations:** The training configuration for the teacher models follow the optimization strategies mentioned in the paper: ["On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines"](https://arxiv.org/abs/2006.04884). Refer to page number 15 for the appropriate hyperparameters to be used for each model type.

The optimizers and learning rate schedules are configured in the code itself (`saarthi_train/utils/optim_strategies.py`). The code automatically picks these parameters based on the name of the teacher provided in the config.

The only value to be adjusted manually in the training configuration is the `teacher_clip_grad_norm` parameter. Here are the values you need to enter as per the architecture of the teacher you will use:

| Model architecture | `teacher_clip_grad_norm` |
| ------------------ | ------------------------ |
| BERT               | 1.0                      |
| RoBERTa            | None                     |
| ALBERT             | 1.0                      |

#### Loggers

The train method also requires a `loggers` argument. It's basically a dictionary of loggers that looks something like this:
```python
loggers = {
    'teacher': [teacher_tensorboard_logger],
    'student_stage_1': [student_tensorboard_logger],
    'student_stage_2': [student_tensorboard_logger],
    'student_stage_3': [student_tensorboard_logger],
    'student_stage_4': [student_tensorboard_logger],
    'student_stage_5': [student_tensorboard_logger]
}
```

Each stage of training requires logger object(s) compatible with PyTorch Lightning. You can initialize any of the loggers that PyTorch Lightning offers, or one of the custom ones defined in `saarthi_train/utils/loggers.py`.

You can also leave the entry for a logger blank in case you do not want to log any training runs (which is not recommended).

A complete example of how to use the `train` method is given in the `saarthi_train/train.py` file itself towards the end.

#### Running the train.py script

The train.py script already contains the code to run the training locally. You can adjust the configuration and loggers as you please, and execute the script using the following command:
```bash
python -m saarthi_train.train
```
**In order to run this command, you need to be in the saarthi-train directory.**

Since the code is structured like a python library, simply using the `python train.py` won't work. The namespaces won't align properly. Hence, the above command is necessary.

In case you have installed the library itself, you can import the `train` function from `saarthi_train.train` and run it in your own scripts.

> *Note: By default, the entire training happens on a GPU machine. In case you want to modify it for some other device, you will have to go through the code and change the `device` parameter on each Trainer object. These changes will need to be made in the `saarthi_train/train.py` and `saarthi_train/models/distillation_modules.py` files.*

### Local inference

In order to use local inference, you need to have the saved model in some directory in your local file system.

#### TorchScript conversion

Currently, inference is supported using TorchScript only.

If your model is not already serialized in the TorchScript format, you can use the `saarthi_train/quantize.py` script provided to convert it into TorchScript. Just comment out the quantization bits from the script.

Following the same trend as the `train.py` file, the way you run this script is:
```bash
python -m saarthi_train.quantize
```

By default, all the inference code expects your model to be in the `saarthi-train/model` folder (it probably does not exist for you just yet, so you will have to create that directory manually).

If you want to change this, you can do so by going to the inference code files and changing the import directory in the `init()` function call.

Required artifacts for inference are: `labels.json`, `training_config.json`, `distil_vocab.json` (in case of the student model) and `model.pt`.

You can use local inference in 3 ways:
1. By hosting an API.
2. By using the `run_model_on_df.py` script if you have a tabular file that you want to run inference on.
3. By importing the `init()` and `run()` methods from the `infer.py` file and using them in your own script.

Let's look at each of the options one by one.

> *Note: You don't have to make any changes to the inference code to run the text or sequence classification models separately. Given the model artifacts the code will automatically determine which inference method to use based on the task supplied in the training config.*

#### Hosting the API

The code for hosting the API is already present in the `saarthi_train/api.py` file.

The API is written in [FastAPI](https://fastapi.tiangolo.com/) and can be executed as such:
```bash
uvicorn saarthi_train.api:app
```
**To execute this command, you need to be in the `saarthi-train` directory.**

By default, it will host your API on localhost:8000 and you can query the model using the `/predict` POST endpoint.

You can change the host and the port through command line arguments to uvicorn. See more [here](https://www.uvicorn.org/#command-line-options).


The request body for text classification models will look something like this

```json
{
    "text": "<input to model goes here>"
}
```

Whereas, for an NER model, the request body will look something like this:

```json
{
    "text": "<input to model goes here>",
    "lang": "<language in which the input text is in. all lowercase full names, for ex: english, hindi, marathi, etc.>"
}
```

You can take a look at the `api.py` script to learn more.

#### Using the `run_model_on_df.py` script

> *Note: This script works only for text classification models. Support for sequence classification models has not been added yet.*

This convenience script exists to run inference using your model on some tabular data.

The data must have a column called `"text"` in order for this script to work (or you can change it in the code).

Given the correct labels for each text sample, the script will also calculate the model's accuracy on each output.

The script can be executed as follows:
```bash
python -m saarthi_train.run_model_on_df
```

**To execute this command, you need to be in the saarthi-train directory.**

The model directory and filepath for the tabular data can be changed in the code. It's not a complete command line solution as of now.

The script does not save the model predictions by default. You can change that behaviour by saving the `final_df` DataFrame object in the script.

#### Writing your own code using the `init()` and `run()` methods

The `infer.py` file in the `saarthi_train` directory contains two methods that are used for inference:
1. `init()`
2. `run()`


##### `init()`

The `init` method takes two arguments:
- `output_path`: The directory in which the training artifacts are saved.
- `model_type`: Type of the model. Can be either `'teacher'` or `'student'`.

Calling this method will initialize all the dependencies for inference and turn them into global variables so that they're accessible by the `run()` method.

Example usage: `init('path/to/model/folder', 'student')`

##### `run()`

This is the method that's going to give you the final model output.

If the model is a text classification one, you only need to provide the input text to it.
For ex: `run('hello')`

If the model is an NER model, you need to give it both the input text and the langauge of the input text.
For ex: `run('hello', lang='english')`

To write your own code, simply import the `init()` and `run()` methods from the `infer.py` file and use them in your custom code.

Both the `api.py` and `run_model_on_df.py` files use these two methods to run inference as well.


### Containerized inference

For most deployments, building a docker image is the way to go.

Based on the type of model, the minimal dependencies for inference change a little. Because of that 2 dockerfiles have been provided in the `saarthi-train` directory.

For the text classification models, `nlu.dockerfile` will be used to build the image, and for NER models, `ner.dockerfile` will be used.

To build the image:
```bash
docker build -f nlu.dockerfile -t <name of the image>:<version number> .
```

Or

```bash
docker build -f ner.dockerfile -t <name of the image>:<version number> .
```

**To run this command, you need to be in the `saarthi-train` directory.**


Once the image is built, you can run the image as a container locally too.

```bash
docker run -p 8000:8000 <name of the image>:<version number>
```

The docker image hosts an API on port 8000. So if the container is to be deployed, port 8000 of the container will need to be exposed. The API is the same as the one used in local inference, so the endpoints will be the same.

For more details on how the API is hosted inside the image, you can refer to the respective dockerfiles.

## Stress testing deployments

A `locustfile.py` has also been included in the `saarthi-train` directory for stress testing purposes.

The script is written in [locust](https://locust.io/) python library (included in the `requirements.txt` file).

In order to start a stress test, use the following command:
```bash
locust
```
This will start host a locust UI instance on the machine's localhost.

**To run this command, you need to be in the `saarthi-train` directory.**

You can refer to [this page](https://docs.locust.io/en/stable/quickstart.html) for more details on how to use Locust.

## Issues with the current code

- The ner_requirements.txt has an additional requirement that is unnecessary for inference (balanced-loss). This issue happens because of the way the code files are structured and that library happens to be imported in one of the code files that the Python interpreter checks for.
- There are a lot of print statements in the code for logging. Those need to be replaced with a standardized logger.
- No unit tests for the deterministic parts of the code.
- No error handling.

## Improvements

Here are the planned improvements for the codebase:
- Separate the inference codebase into a different repo (this will solve the first point in the issues section).
- Define training modules and steps using a config.yaml file
- Look into packed_sequences for RNN student models.
- Standardize output directory structures.
- Implement model interpretability report generation using Captum.
- Implement joint model.
- Try GRU student.
- Implement transformer student.
    - Finish implementation with huggingface model.
    - Add conversion to torchscript.
    - Add quantization.
    - Add native torch implementation to utilise their fastpath for transformers.
- Experiment with other teacher layers for distillation.
- Implement teacher testing before distillation.
- Change backend to pyarrow for all dataframes.
- [Reweighting samples](https://arxiv.org/abs/1803.09050), and in general how you can use a small high quality dataset with a large noisy one.
- [Selective Classification](https://arxiv.org/pdf/2002.10319.pdf): Where model learns to not say anything if it is not sure.
- Add deployment metadata to the health check GET endpoint.
    - model name
    - model version
    - date created
    - accuracy scores
    - etc.

## Handy tools that could be used for automation
- Auto-tag generator for Github repos: https://github.com/mateimicu/auto-tag
- Python build tutorial using Github actions: https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python
