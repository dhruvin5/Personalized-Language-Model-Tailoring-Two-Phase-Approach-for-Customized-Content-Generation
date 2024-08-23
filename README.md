# Personalized Language Model Tailoring

### Two-Phase Approach for Customized Content Generation

This repository contains the implementation and experiments related to the paper "Personalized Language Model Tailoring: Two-Phase Approach for Customized Content Generation,". The project focuses on improving the personalization of Language Models (LLMs) for generating customized content such as news headlines and scholarly titles.

## Abstract

The study emphasizes the importance of personalizing Language Models (LLMs) to accommodate individual preferences. A two-phase approach is proposed, leveraging datasets from the LAMP repository, namely Personalized News Headline Generation and Personalized Scholarly Title Generation. The approach involves:
1. Generating relevant documents using generative retrieval from a user's historical data.
2. Extracting the most relevant chunks of information to enhance the inputs provided to the LLM.

The experiments show that this method significantly improves the LLM's ability to capture language style and semantics, thereby generating more customized and relevant outputs.

## Methodology

### 1. Dataset
The study utilizes two key datasets:
- **Personalized News Headline Generation**: Generates news article titles based on the user's previous articles and their titles.
- **Personalized Scholarly Title Generation**: Generates academic article titles using the user's history of abstracts and titles.

### 2. Proposed Two-Phase Approach
The methodology is divided into two phases:

- **Phase 1: Generative Task and Encoding**
  - **Generative Task**: Uses the PaLM language model to generate multiple hypothetical versions of the input content.
  - **Encoding**: Compares two encoding schemes, DistilBERT and Contriever, to determine which provides better performance on the given datasets.
  - **Dense Search**: Utilizes FAISS for creating an index of user profiles and performs dense searches to retrieve the top K documents.

- **Phase 2: Extracting Relevant Chunks**
  - **Sentence Segmentation and Encoding**: Divides documents into sentences and encodes them using the `bert-base-uncased` model.
  - **Similarity Calculation**: Employs cosine similarity to find the most relevant sentences from the user's history.
  - **LLM Integration**: The extracted sentences are fed into the LLM, which generates the final customized output.

### 3. Results
The results indicate that the two-phase approach, particularly with the use of sentence extraction, significantly enhances the quality of generated headlines compared to providing the entire user history as input.

| Model         | ROUGE-1 | ROUGE-L |
| ------------- | ------- | ------- |
| **FLAN T-5**  | 0.174   | 0.160   |
| **PaLM**      | 0.258   | 0.235   |

### 4. Conclusion
The two-phase approach for customizing LLM outputs demonstrates superior performance compared to a single-phase approach. The consistent use of the PaLM LLM across both phases was key to capturing user history for generating personalized content. The study highlights the potential of this methodology to be further refined and tested on larger datasets.

## Acknowledgments
Special thanks to Robert for the support with the project.

## References
- Salemi, A., Mysore, S., Bendersky, M., Zamani, H. (2023). [LaMP: When Large Language Models Meet Personalization](https://arxiv.org/abs/2304.11406). arXiv preprint.
- Gao, L., Ma, X., Lin, J., Callan, J. (2022). [Precise zero-shot dense retrieval without relevance labels](https://arxiv.org/abs/2212.10496). arXiv preprint.
- Qian, H., Dou, Z., Zhu, Y., Ma, Y., Wen, J.R. (2021). [Learning implicit user profile for personalized retrieval-based chatbot](https://dl.acm.org/doi/10.1145/3459637.3482393). ACM International Conference on Information and Knowledge Management.

## How to Use

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/Personalized-Language-Model-Tailoring.git
    ```
   
2. **Install Dependencies**:
   Make sure to install the required dependencies using `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Experiments**:
   Follow the instructions in the `experiments` folder to reproduce the results from the paper.
