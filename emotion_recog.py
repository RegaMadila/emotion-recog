{
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-q_lhAiuOmjC",
        "outputId": "0ea02d27-1bd7-454b-d0e4-93bcfc2489d6"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: transformers in /usr/local/lib/python3.12/dist-packages (4.57.3)\n",
            "Requirement already satisfied: torch in /usr/local/lib/python3.12/dist-packages (2.9.0+cu126)\n",
            "Requirement already satisfied: pandas in /usr/local/lib/python3.12/dist-packages (2.2.2)\n",
            "Requirement already satisfied: scikit-learn in /usr/local/lib/python3.12/dist-packages (1.6.1)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from transformers) (3.20.0)\n",
            "Requirement already satisfied: huggingface-hub<1.0,>=0.34.0 in /usr/local/lib/python3.12/dist-packages (from transformers) (0.36.0)\n",
            "Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.12/dist-packages (from transformers) (2.0.2)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.12/dist-packages (from transformers) (25.0)\n",
            "Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.12/dist-packages (from transformers) (6.0.3)\n",
            "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.12/dist-packages (from transformers) (2025.11.3)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.12/dist-packages (from transformers) (2.32.4)\n",
            "Requirement already satisfied: tokenizers<=0.23.0,>=0.22.0 in /usr/local/lib/python3.12/dist-packages (from transformers) (0.22.1)\n",
            "Requirement already satisfied: safetensors>=0.4.3 in /usr/local/lib/python3.12/dist-packages (from transformers) (0.7.0)\n",
            "Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.12/dist-packages (from transformers) (4.67.1)\n",
            "Requirement already satisfied: typing-extensions>=4.10.0 in /usr/local/lib/python3.12/dist-packages (from torch) (4.15.0)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.12/dist-packages (from torch) (75.2.0)\n",
            "Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch) (1.14.0)\n",
            "Requirement already satisfied: networkx>=2.5.1 in /usr/local/lib/python3.12/dist-packages (from torch) (3.6.1)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch) (3.1.6)\n",
            "Requirement already satisfied: fsspec>=0.8.5 in /usr/local/lib/python3.12/dist-packages (from torch) (2025.3.0)\n",
            "Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch) (12.6.77)\n",
            "Requirement already satisfied: nvidia-cuda-runtime-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch) (12.6.77)\n",
            "Requirement already satisfied: nvidia-cuda-cupti-cu12==12.6.80 in /usr/local/lib/python3.12/dist-packages (from torch) (12.6.80)\n",
            "Requirement already satisfied: nvidia-cudnn-cu12==9.10.2.21 in /usr/local/lib/python3.12/dist-packages (from torch) (9.10.2.21)\n",
            "Requirement already satisfied: nvidia-cublas-cu12==12.6.4.1 in /usr/local/lib/python3.12/dist-packages (from torch) (12.6.4.1)\n",
            "Requirement already satisfied: nvidia-cufft-cu12==11.3.0.4 in /usr/local/lib/python3.12/dist-packages (from torch) (11.3.0.4)\n",
            "Requirement already satisfied: nvidia-curand-cu12==10.3.7.77 in /usr/local/lib/python3.12/dist-packages (from torch) (10.3.7.77)\n",
            "Requirement already satisfied: nvidia-cusolver-cu12==11.7.1.2 in /usr/local/lib/python3.12/dist-packages (from torch) (11.7.1.2)\n",
            "Requirement already satisfied: nvidia-cusparse-cu12==12.5.4.2 in /usr/local/lib/python3.12/dist-packages (from torch) (12.5.4.2)\n",
            "Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /usr/local/lib/python3.12/dist-packages (from torch) (0.7.1)\n",
            "Requirement already satisfied: nvidia-nccl-cu12==2.27.5 in /usr/local/lib/python3.12/dist-packages (from torch) (2.27.5)\n",
            "Requirement already satisfied: nvidia-nvshmem-cu12==3.3.20 in /usr/local/lib/python3.12/dist-packages (from torch) (3.3.20)\n",
            "Requirement already satisfied: nvidia-nvtx-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch) (12.6.77)\n",
            "Requirement already satisfied: nvidia-nvjitlink-cu12==12.6.85 in /usr/local/lib/python3.12/dist-packages (from torch) (12.6.85)\n",
            "Requirement already satisfied: nvidia-cufile-cu12==1.11.1.6 in /usr/local/lib/python3.12/dist-packages (from torch) (1.11.1.6)\n",
            "Requirement already satisfied: triton==3.5.0 in /usr/local/lib/python3.12/dist-packages (from torch) (3.5.0)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.12/dist-packages (from pandas) (2.9.0.post0)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.12/dist-packages (from pandas) (2025.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas) (2025.3)\n",
            "Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (1.16.3)\n",
            "Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (1.5.3)\n",
            "Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (3.6.0)\n",
            "Requirement already satisfied: hf-xet<2.0.0,>=1.1.3 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<1.0,>=0.34.0->transformers) (1.2.0)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)\n",
            "Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch) (1.3.0)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch) (3.0.3)\n",
            "Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests->transformers) (3.4.4)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests->transformers) (3.11)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests->transformers) (2.5.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests->transformers) (2025.11.12)\n"
          ]
        }
      ],
      "source": [
        "!pip install transformers torch pandas scikit-learn\n"
      ],
      "execution_count": 1
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JtxxpdWIOmjG"
      },
      "outputs": [],
      "source": [
        "import sys\n",
        "import os\n",
        "import glob\n",
        "import re\n",
        "import pandas as pd\n",
        "import torch\n",
        "from torch.utils.data import Dataset\n",
        "from sklearn.model_selection import train_test_split\n",
        "from transformers import Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification\n",
        "import matplotlib.pyplot as plt\n"
      ],
      "execution_count": 2
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "MOBoHKDMOmjJ"
      },
      "outputs": [],
      "source": [
        "# --- Data Loader ---\n",
        "def load_data(data_path):\n",
        "    if not os.path.exists(data_path):\n",
        "        print(f\"Warning: Data path {data_path} does not exist.\")\n",
        "        return pd.DataFrame()\n",
        "\n",
        "    csv_files = glob.glob(os.path.join(data_path, '*.csv'))\n",
        "\n",
        "    if not csv_files:\n",
        "        print(f\"Warning: No CSV files found in {data_path}\")\n",
        "        return pd.DataFrame()\n",
        "\n",
        "    df_list = []\n",
        "    for csv_file in csv_files:\n",
        "        try:\n",
        "            df = pd.read_csv(csv_file)\n",
        "            df_list.append(df)\n",
        "        except Exception as e:\n",
        "            print(f\"Error reading {csv_file}: {e}\")\n",
        "\n",
        "    if not df_list:\n",
        "        return pd.DataFrame()\n",
        "\n",
        "    full_df = pd.concat(df_list, ignore_index=True)\n",
        "    return full_df\n",
        "\n",
        "def clean_initial_data(df):\n",
        "    if df.empty:\n",
        "        return df\n",
        "\n",
        "    emotion_columns = ['anger', 'confusion', 'disgust', 'fear', 'joy', 'love', 'sadness', 'surprise']\n",
        "    available_cols = [col for col in emotion_columns if col in df.columns]\n",
        "    if not available_cols:\n",
        "        return df\n",
        "\n",
        "    df_clean = df.copy()\n",
        "    if 'text' in df_clean.columns:\n",
        "        df_clean = df_clean.dropna(subset=['text'])\n",
        "        df_clean = df_clean[df_clean['text'].str.strip() != '']\n",
        "\n",
        "    return df_clean\n"
      ],
      "execution_count": 3
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ARLj8yCXOmjL"
      },
      "outputs": [],
      "source": [
        "# --- Preprocessor ---\n",
        "class TextPreprocessor:\n",
        "    def __init__(self, model_name='bert-base-uncased'):\n",
        "        self.tokenizer = BertTokenizer.from_pretrained(model_name)\n",
        "        self.happy_emoticons = [':)', ': )', ':-)', '=)', ':]', ';)', ';-)']\n",
        "        self.sad_emoticons = [':(', ': (', ': [', '= [', ':[', ':-(', '=(', '; [', ';(']\n",
        "        self.flat_emoticons = [':/', ':|']\n",
        "        self.emoticon_pattern = re.compile(r'[:;xX=]-?[dDpP/\\\\]?\\s*[\\(\\)[\\]\\{\\}|]')\n",
        "\n",
        "    def replace_emoticons(self, text):\n",
        "        if not isinstance(text, str):\n",
        "            return \"\"\n",
        "        for emo in self.happy_emoticons:\n",
        "            text = text.replace(emo, '{happy_face}')\n",
        "        for emo in self.sad_emoticons:\n",
        "            text = text.replace(emo, '{sad_face}')\n",
        "        for emo in self.flat_emoticons:\n",
        "            text = text.replace(emo, '{flat_face}')\n",
        "        text = re.sub(self.emoticon_pattern, '', text)\n",
        "        return text\n",
        "\n",
        "    def preprocess(self, df, text_column='text'):\n",
        "        df = df.copy()\n",
        "        df['text_processed'] = df[text_column].apply(self.replace_emoticons)\n",
        "        return df\n"
      ],
      "execution_count": 4
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zOTHmNk9OmjN"
      },
      "outputs": [],
      "source": [
        "# --- Model & Dataset ---\n",
        "class EmotionDataset(Dataset):\n",
        "    def __init__(self, encodings, labels):\n",
        "        self.encodings = encodings\n",
        "        self.labels = labels.values\n",
        "\n",
        "    def __getitem__(self, idx):\n",
        "        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}\n",
        "        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)\n",
        "        return item\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.labels)\n",
        "\n",
        "def get_model(num_labels):\n",
        "    model = BertForSequenceClassification.from_pretrained(\n",
        "        'bert-base-uncased',\n",
        "        num_labels=num_labels,\n",
        "        problem_type=\"multi_label_classification\"\n",
        "    )\n",
        "    return model\n"
      ],
      "execution_count": 7
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "OjlwtCS0OmjP"
      },
      "outputs": [],
      "source": [
        "# --- Training Logic ---\n",
        "EMOTION_COLUMNS = ['anger', 'confusion', 'fear', 'joy', 'sadness']\n",
        "\n",
        "def train_model(data_path, output_dir, epochs=3, sample_size=None):\n",
        "    print(\"Loading data...\")\n",
        "    # For Colab, users might upload files directly to /content/\n",
        "    # So we check if data_path exists, if not assume /content/\n",
        "\n",
        "    df = load_data(data_path)\n",
        "    if df.empty:\n",
        "        print(f\"No data found in {data_path}. Please upload your CSV files (goemotions_*.csv).\")\n",
        "        return None\n",
        "\n",
        "    df = clean_initial_data(df)\n",
        "\n",
        "    available_emotions = [col for col in EMOTION_COLUMNS if col in df.columns]\n",
        "    if not available_emotions:\n",
        "        print(f\"Error: None of the expected emotion columns {EMOTION_COLUMNS} found in dataset.\")\n",
        "        return None\n",
        "\n",
        "    if sample_size:\n",
        "        print(f\"Sampling {sample_size} rows for debugging...\")\n",
        "        df = df.sample(n=min(sample_size, len(df)), random_state=42)\n",
        "\n",
        "    print(\"Preprocessing...\")\n",
        "    preprocessor = TextPreprocessor()\n",
        "    df = preprocessor.preprocess(df)\n",
        "\n",
        "    X = df['text_processed']\n",
        "    y = df[available_emotions]\n",
        "\n",
        "    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "\n",
        "    print(\"Tokenizing...\")\n",
        "    train_encodings = preprocessor.tokenizer(\n",
        "        X_train.tolist(), truncation=True, padding=True, max_length=128\n",
        "    )\n",
        "    val_encodings = preprocessor.tokenizer(\n",
        "        X_val.tolist(), truncation=True, padding=True, max_length=128\n",
        "    )\n",
        "\n",
        "    train_dataset = EmotionDataset(train_encodings, y_train)\n",
        "    val_dataset = EmotionDataset(val_encodings, y_val)\n",
        "\n",
        "    print(\"Initializing model...\")\n",
        "    model = get_model(len(available_emotions))\n",
        "\n",
        "    training_args = TrainingArguments(\n",
        "        output_dir=output_dir,\n",
        "        num_train_epochs=epochs,\n",
        "        per_device_train_batch_size=16,\n",
        "        per_device_eval_batch_size=16,\n",
        "        eval_strategy=\"epoch\",  # Updated from evaluation_strategy\n",
        "        save_strategy=\"epoch\",\n",
        "        logging_dir=os.path.join(output_dir, 'logs'),\n",
        "    )\n",
        "\n",
        "    trainer = Trainer(\n",
        "        model=model,\n",
        "        args=training_args,\n",
        "        train_dataset=train_dataset,\n",
        "        eval_dataset=val_dataset\n",
        "    )\n",
        "\n",
        "    print(\"Starting training...\")\n",
        "    trainer.train()\n",
        "\n",
        "    print(\"Saving model...\")\n",
        "    model.save_pretrained(output_dir)\n",
        "    preprocessor.tokenizer.save_pretrained(output_dir)\n",
        "    print(\"Training complete.\")\n",
        "    return preprocessor\n"
      ],
      "execution_count": 5
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tlXp0rHUOmjR"
      },
      "source": [
        "# Instructions\n",
        "# 1. Run the first cell to install dependencies.\n",
        "# 2. Upload your 'goemotions_*.csv' files to the 'Files' tab on the left (they will appear in /content/).\n",
        "# 3. Run the valid cells to define classes.\n",
        "# 4. Run the training cell below.\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "GJL_TqSrOmjV",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 314,
          "referenced_widgets": [
            "a133b7b90bfd4913b0d2ec8156688c1a",
            "e9d15a0c9225471680ef708dafd79424",
            "8dee92290c4e4deba1b9564022442c44",
            "1e84a614f100488491a5c1e722824469",
            "ba97615c233c43e5bc4c8c766230b659",
            "084be6fd447b4541a63c3d9312e1a2e5",
            "23cd19b83e924829b6ac1d3ed2e7fca8",
            "a1ea5a6d43d440ab929156f1711e2c8c",
            "43ecd8fd673b40e3848edf82e8988b2e",
            "ff2019f020604a1b92b5f6f073bc1263",
            "8ca68730006a467aa0fd0d79f72dea79"
          ]
        },
        "outputId": "d19845e4-52f4-4b7a-a3fa-5d0dfe86b3a2"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Loading data...\n",
            "Preprocessing...\n",
            "Tokenizing...\n",
            "Initializing model...\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "model.safetensors:   0%|          | 0.00/440M [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "a133b7b90bfd4913b0d2ec8156688c1a"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']\n",
            "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Starting training...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/notebook/notebookapp.py:191: SyntaxWarning: invalid escape sequence '\\/'\n",
            "  | |_| | '_ \\/ _` / _` |  _/ -_)\n",
            "\u001b[34m\u001b[1mwandb\u001b[0m: (1) Create a W&B account\n",
            "\u001b[34m\u001b[1mwandb\u001b[0m: (2) Use an existing W&B account\n",
            "\u001b[34m\u001b[1mwandb\u001b[0m: (3) Don't visualize my results\n",
            "\u001b[34m\u001b[1mwandb\u001b[0m: Enter your choice:"
          ]
        }
      ],
      "source": [
        "# Define paths\n",
        "data_dir = '/content'  # Default Colab directory\n",
        "output_dir = '/content/models'\n",
        "\n",
        "# START TRAINING\n",
        "# Set sample_size=100 for a quick test, or None for full training\n",
        "preprocessor = train_model(data_dir, output_dir, epochs=3, sample_size=None)\n"
      ],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ws7zIWYmOmjX"
      },
      "outputs": [],
      "source": [
        "# Zip the model for download\n",
        "import shutil\n",
        "from google.colab import files\n",
        "\n",
        "if os.path.exists(output_dir):\n",
        "    shutil.make_archive('emotion_model', 'zip', output_dir)\n",
        "    print(\"Model zipped. Downloading...\")\n",
        "    try:\n",
        "        files.download('emotion_model.zip')\n",
        "    except Exception as e:\n",
        "        print(\"Auto-download failed. Please download 'emotion_model.zip' from the files tab.\")\n"
      ],
      "execution_count": null
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.5"
    },
    "colab": {
      "provenance": [],
      "gpuType": "T4"
    },
    "accelerator": "GPU",
    "widgets": {
      "application/vnd.jupyter.widget-state+json": {
        "a133b7b90bfd4913b0d2ec8156688c1a": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HBoxModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_e9d15a0c9225471680ef708dafd79424",
              "IPY_MODEL_8dee92290c4e4deba1b9564022442c44",
              "IPY_MODEL_1e84a614f100488491a5c1e722824469"
            ],
            "layout": "IPY_MODEL_ba97615c233c43e5bc4c8c766230b659"
          }
        },
        "e9d15a0c9225471680ef708dafd79424": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HTMLModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_084be6fd447b4541a63c3d9312e1a2e5",
            "placeholder": "​",
            "style": "IPY_MODEL_23cd19b83e924829b6ac1d3ed2e7fca8",
            "value": "model.safetensors: 100%"
          }
        },
        "8dee92290c4e4deba1b9564022442c44": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "FloatProgressModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_a1ea5a6d43d440ab929156f1711e2c8c",
            "max": 440449768,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_43ecd8fd673b40e3848edf82e8988b2e",
            "value": 440449768
          }
        },
        "1e84a614f100488491a5c1e722824469": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HTMLModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_ff2019f020604a1b92b5f6f073bc1263",
            "placeholder": "​",
            "style": "IPY_MODEL_8ca68730006a467aa0fd0d79f72dea79",
            "value": " 440M/440M [00:06&lt;00:00, 53.1MB/s]"
          }
        },
        "ba97615c233c43e5bc4c8c766230b659": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "084be6fd447b4541a63c3d9312e1a2e5": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "23cd19b83e924829b6ac1d3ed2e7fca8": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "DescriptionStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "a1ea5a6d43d440ab929156f1711e2c8c": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "43ecd8fd673b40e3848edf82e8988b2e": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "ProgressStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "ff2019f020604a1b92b5f6f073bc1263": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "8ca68730006a467aa0fd0d79f72dea79": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "DescriptionStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        }
      }
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}