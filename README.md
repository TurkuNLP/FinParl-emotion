### Code used for emotion analysis with Finnish parliamentary speeches.

We use GPT-4 to annotate sentences from the Finnish parliament and compare them against the human annotated gold standard. This data is the used to train a BERT model for emotion analysis. We compare the model trained on the GPT-4 annotated against models trained on machine translated datasets and find that GPT-4 annotations give better results. <br>

The emotions we use are:
- 0 = neutral
- 1 = happiness/success
- 2 = hopefulness/optimism/trust
- 3 = love/praise
- 4 = surprise (positive)
- 5 = sadness/disappointment
- 6 = fear/concern/mistrust
- 7 = hate/disgust/derision
- 8 = astonishment (negative)

The data folder contains the sentences annotated by GPT-4. The machine translated datasets are on Huggingface. <br>
- HunEmPoli: https://huggingface.co/datasets/TurkuNLP/HunEmPoli_finnish
- Many Emotions: https://huggingface.co/datasets/TurkuNLP/many_emotions_finnish
