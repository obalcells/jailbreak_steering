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

### Suffix Generation

```bash
python3 -m jailbreak_steering.suffix_generation.jailbreak
```
