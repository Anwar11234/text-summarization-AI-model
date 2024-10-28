# Summarization Model for Programming and Computer Science Content

This repository contains a summarization model designed to generate concise summaries of educational content, specifically in the fields of programming and computer science. The model leverages various techniques for effective summarization and has been fine-tuned to perform well on lecture transcripts and similar text.

## Table of Contents
- [Data Collection](#data-collection)
- [Model Selection and Architecture](#model-selection-and-architecture)
  - [BART Architecture](#bart-architecture)
- [Fine-Tuning with LoRA and PEFT](#fine-tuning-with-lora-and-peft)
  - [Low-Rank Adaptation (LoRA)](#low-rank-adaptation-lora)
  - [Parameter-Efficient Fine-Tuning (PEFT)](#parameter-efficient-fine-tuning-peft)
- [Evaluation Metrics](#evaluation-metrics)
  - [ROUGE-1](#rouge-1)
  - [ROUGE-2](#rouge-2)
  - [ROUGE-L](#rouge-l)
  - [ROUGE-LSUM](#rouge-lsum)
- [Results](#results)
- [Deployment with FastAPI and Docker](#deployment-with-fastapi-and-docker)
  - [FastAPI Implementation](#fastapi-implementation)
  - [Containerization with Docker](#containerization-with-docker)
  - [Hosting on Hugging Face Spaces](#hosting-on-hugging-face-spaces)

## Data Collection

Using the `youtube_transcript_api` Python library, we collected transcripts from approximately 7,500 YouTube videos focused on programming and computer science. To create variety in our summaries, we employed OpenAI's GPT to generate summaries for each transcript. Instead of using a single prompt for all videos, we varied the prompts to ensure diversity in the summaries.

In addition, we augmented our dataset by incorporating samples from the widely-used CNN/DailyMail dataset, a benchmark for summarization tasks. The combined dataset was then shuffled and split into training and validation sets.

## Model Selection and Architecture

For this summarization task, we selected the `facebook/bart-base` model. BART (Bidirectional and Auto-Regressive Transformers) is a transformer-based model with an encoder-decoder architecture, particularly effective for text generation tasks.

### BART Architecture

BART combines the benefits of bidirectional encoding and autoregressive decoding:
- **Encoder**: Processes the input text bidirectionally, considering the full context of the input sequence.
- **Decoder**: Generates the output text autoregressively, predicting one token at a time.

Key Advantages of BART:
- **Flexibility**: Supports various text generation tasks, including summarization, translation, and question answering.
- **Performance**: Achieves state-of-the-art results on several benchmark datasets.
- **Robustness**: Well-suited for real-world applications as it handles input noise effectively.

## Fine-Tuning with LoRA and PEFT

To adapt the `facebook/bart-base` model to our specific summarization task, we fine-tuned it using Low-Rank Adaptation (LoRA) and Parameter-Efficient Fine-Tuning (PEFT).

### Low-Rank Adaptation (LoRA)

LoRA approximates the weight matrices with low-rank factorization, reducing the number of trainable parameters. This optimization speeds up training and lowers computational costs without significantly compromising model performance.

### Parameter-Efficient Fine-Tuning (PEFT)

PEFT selectively updates only the most relevant model parameters during fine-tuning. By focusing on these key parameters, PEFT allows for efficient fine-tuning, especially useful for large-scale models like BART.

## Evaluation Metrics

We used ROUGE (Recall-Oriented Understudy for Gisting Evaluation) scores to assess the performance of our summarization model. ROUGE measures the overlap between generated summaries and reference summaries and includes the following metrics:

### ROUGE-1

Calculates the overlap of unigrams (single words) between the generated and reference summaries.

### ROUGE-2

Measures the overlap of bigrams (two consecutive words) between the generated and reference summaries.

### ROUGE-L

Calculates the longest common subsequence (LCS) between the generated and reference summaries, considering word order and sentence structure.

### ROUGE-LSUM

Similar to ROUGE-L but evaluates the summary as a whole, assessing the overall coherence of the generated summary.

## Results

After fine-tuning, we evaluated our model on the validation dataset. The results are as follows:

- **ROUGE-1 F1**: 0.433
- **ROUGE-2 F1**: 0.191
- **ROUGE-L F1**: 0.292
- **ROUGE-LSUM F1**: 0.365

These results indicate that our model is effective in summarizing lecture transcripts, capturing a significant portion of important information while maintaining coherence and readability.

## Deployment with FastAPI and Docker

The summarization API was deployed using FastAPI and containerized with Docker. It was hosted on Hugging Face Spaces, providing an accessible interface for users.

### FastAPI Implementation

FastAPI was chosen for its performance, ease of use, and automatic documentation generation. The implementation includes:

1. **Setup**: Install FastAPI and Uvicorn.
2. **Define the API**: Create endpoints for the summarization functionality.
3. **Integrate the Model**: Call the summarization function within the API endpoint.
4. **Run the API**: Use Uvicorn to run the FastAPI application.

### Containerization with Docker

We used Docker to create a containerized environment for the API, ensuring consistency across deployment platforms. The Docker setup includes:

1. **Dockerfile**: Defines the image.
2. **Build the Image**: Use Docker to build the image.
3. **Run the Container**: Start the container from the built image.

### Hosting on Hugging Face Spaces

The containerized API was deployed on Hugging Face Spaces. Deployment steps include:

1. **Create a Space**: Set up a new Space on Hugging Face.
2. **Configure the Space**: Configure the Space to use Docker.
3. **Upload Docker Image**: Push the Docker image to Hugging Face Spaces.
4. **Run the Space**: Start the Space to make the API accessible online.

---

This repository provides an efficient and robust solution for generating summaries of educational content in programming and computer science. The combination of BART, LoRA, and PEFT allows for high-quality summarization with reduced computational resources.
