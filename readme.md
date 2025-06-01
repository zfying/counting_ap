# Mechanism of Counting

This is a toy project investigates how LLMs internally represent and track counting information when performing categorical word counting tasks using activation patching.

## Task Description
We focus on counting tasks of the format below:
```
Count the number of words in the following list that match the given type, and put the numerical answer in parentheses.
Type: fruit
List: [dog apple cherry bus cat grape bowl]
Answer: (3)
```
The **RQs** are:
1. How do different sized language models perform on zero-shot counting tasks?
2. Is there evidence for hidden state layers containing running count representations during list processing?
3. Where in the model architecture does counting computation occur (layer/component)?

## Dataset Generation
We constructed a controlled dataset using four semantic **categories**. Each word is carefully selected to be as *non-controversial* as possible and to be *single-token* for the simplicity of implementation of activation patching:

```python
categories = {
    'clothing': ['shirt', 'pants', 'dress', 'coat', 'vest', 'suit'],
    'animal': ['dog', 'cat', 'bird', 'fish', 'wolf', 'fox', 'cow'],
    'color': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown'],
    'body part': ['hand', 'leg', 'ear', 'eye', 'mouth', 'arm', 'foot']
}
```
**Distractors**: 40 semantically neutral *single-token* words (e.g., 'book', 'stone', 'clock', 'water'). 

**Dataset details**
- List length: 3-12 words (uniformly distributed)
- Target count: 0 to min(5, list_length) (uniformly distributed)
- Total samples: 2000 examples
- Tokenization: All words verified as single tokens across tested tokenizers

**Corrupted samples**: We also corrupt each sample by randomly selecting a word and changing it from a target word to a distractor, or vice-versa.

**Prompt design**: After experimentation, we settled on this prompt structure to ensure reliable integer outputs:
```
Analyze the word list below. Count ONLY words that are {category}.
Follow these rules:
1. Strictly match the type definition (case-sensitive)
2. Ignore non-word elements
3. Output ONLY one integer count and nothing else

Type: {category}
List: [{word_list}]
Answer:
```
## Model Evalution
**Tested models**: Llama-3.2-1B-Instruct, Llama-3.1-8B-Instruct, Llama-3.3-70B-Instruct, Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-72B-Instruct.

![Model Accuracy Comparison](figs/model_accuracy_comparison.png)
![Model Accuracy by Categories](figs/accuracy_by_categories.png)
![Error Rate by Count](figs/error_rate_by_count.png)
![Error Rate by List Length](figs/error_rate_by_list_length.png)

**Takeaways:**
- Larger models significantly outperform smaller ones (70B >> 8B >> 1B). The accuracies range from ~30% to ~90%.
- Performance remains relatively consistent across semantic categories.
- Accuracy doesn't strictly correlate with true count, suggesting models have counting priors.
- Accuracy decreases with longer lists.

## Activation Patching

Following suggests from Zhang & Nanda (2024), we implemented **Symmetric Token Replacement (STR)** experiments. We measure effect of patching with normalized logit differences, for each layer, for each token since the first word of the list, on residual stream, attention layer, and MLP layer. 

Due to time and compute constraint, we only conduct experiments on Llama-70B with 21 samples and Llama-8B with 300 samples. The 21 samples on Llama-70B are all with list length of 5 and corruption on the 3rd word in the list. 

### Single Sample Qualitative Results
We show four representative samples from Llama-70B below:
Sample A: [word1] [word2] [*CORRUPT*] [distractor] [distractor] |  Sample B: [word1] [word2] [*CORRUPT*] [distractor] [*target*]
:-------------------------:|:-------------------------:
![Sample A](figs/single_sample_heatmap_vertical_test_001127.png)  |  ![Sample B](figs/single_sample_heatmap_vertical_test_001603.png)

Sample C: [word1] [word2] [*CORRUPT*] [*target*] [distractor] |  Sample D: [word1] [word2] [*CORRUPT*] [*target*] [*target*]
:-------------------------:|:-------------------------:
![Sample C](figs/single_sample_heatmap_vertical_test_001145.png)  |  ![Sample D](figs/single_sample_heatmap_vertical_test_000129.png)

**Residual stream observations:**
1. Strong patching effects at **corrupted word** at **early and mid** layers (Word 3 for all four samples).
2. Post-corruption, only **target words** show strong effects at **mid-layer** (W5 for Sample B, W4 for Sample C, and W4/W5 for Sample D), while patching at post-corruption distractors has basically no effect (W4/W5 for Sample A, W4 for Sample B, and W5 for Sample C).
3. Strong effects at **final token** in **later** layers, with weaker but notable effects at second-to-last token

**Attention layer observations:**
Notable effects only at specific **mid-layers** (around layer 35) at **final token** position. These effects appear at the same layers where residual stream effects transition from word list to final token. This suggests attention layers at mid-layers route information about counting from the word list to the last token for next token prediction.

**MLP layer:** patching exhibits some weak effects at the corruption word position at early layers and the final token position at later layers.

### Multi-Sample Aggregation
Averaging effects across 21 samples at Llama-70B reveals patterns consistent with the qualitative results above:
![Average Residual](figs/average_residual.png)

1. **Corrupted word**: Strong early-mid layer effects 
2. **Post-corruption distractors**: Minimal effect across all layers
3. **Post-corruption targets**: Strong mid-layer effects
4. **Second-to-last token**: Moderate late-layer effects 
5. **Final token**: Strong late-layer effects 

![Average Attention](figs/average_attn.png)
- Concentrated effects in layers 30-35
- Strong effects at final token position

### Discussion

Our results provide evidence supporting a running count hypothesis for how Llama processes counting tasks. The observed pattern strongly suggests that the model maintains an internal count state that is incrementally updated as it processes each word in the list. Specifically, corruption at position 3 disrupts this count state in early-to-mid layers. Crucially, only target words show strong patching effects post-corruption, while distractors show minimal effects, indicating that count updates occur selectively when task-relevant items are encountered. The strong effects at the final token in later layers suggest that the accumulated count information is ultimately routed for next-token prediction. The attention layer effects concentrated in mid-layers (30-35) at the final token position, likely reflecting the mechanism by which count information is aggregated and prepared for output. This pattern is inconsistent with end-of-sequence batch counting and instead supports a sequential updating mechanism where each fruit incrementally modifies an internal count representation that persists through the residual stream.

The computation mechnism is likely to be something like this:
```
Input Classification → Running Count Update → Final Aggregation → Output
    (layers 0-25)        (layers 25-45)        (layers 45-65)
```
