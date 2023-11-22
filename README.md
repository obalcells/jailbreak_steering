# Jailbreak Steering

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then create a `.env` file with the following variables (see `.env.example`):

```
HF_TOKEN=huggingface_token_with_access_to_llama2
```

## Usage

### Suffix generation


From the root directory, run:
```bash
python3 -m jailbreak_steering.suffix_gen.run_suffix_gen
```

### Vector generation

Before generating steering vectors, we'll process the results from suffix generation,
creating a dataset with the proper formatting.
```bash
python3 -m jailbreak_steering.vector_gen.process_suffix_gen
```

This will output datasets in `jailbreak_steering/vector_gen/instruction_datasets/`.

To generate steering vectors, run:
```bash
python3 -m jailbreak_steering.vector_gen.run_vector_gen
```

## Tests

To run tests, run:
```bash
pytest
```