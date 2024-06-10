# GloVe summarization with python
here i have provided to you the code of glove model technique for summariizing text.<br>
Ofcourse the fact that this model is one of the embedding moels this is not as strong as transformer models. <br>
## How to use the code ? 
For using the code you need to download the embeddings from the stanford university or whereever you consider to download, but you will need the embeddings. <br>
You can download the __Stanford__ embeddings from [here.](https://nlp.stanford.edu/projects/glove/)<br>

## Code explanation
The code first extracts the informations of embeddings.<br> 
then tokenizes the sentences and words.<br>
and then it uses cosine similarity matrix and pagerank algorithm(for picking up the summary sentences similar to using the [LexRank](https://rishabh71510.medium.com/understanding-lexrank-text-summarization-algorithm-fb2c5415e0b6)).<br>
