
﻿
# Poem generator (for Polish / English language)

## Main idea
The main idea for this project was to create a model which would generate poems with rhymes and "nice" structure. 
As a base model, the papuGaPT2 has been used (https://huggingface.co/flax-community/papuGaPT2), which is the GPT2 model for Polish language. 
## Data
To perform this task we have used poems by Kochanowski and Tuwim (scrapped from https://poezja.org/wz/). For experiments, "Pan Tadeusz" has been used.
## Final solution
The final solution is based on the previously mentioned model papuGaPT2, fine tuned using poems by Kochanowski and Tuwim with an additional generator technique. The technique attempts to generate rhymes AABBCC... by:
1. First generating the first line, which ends with the "\n" symbol. 
2. Then the second line, cutting the last word when we reach the "\n" symbol again, which will give us a first verse and a second verse minus one word. 
3. The next step involves generation of **N** endings (in our case N=100). This step is performed by using **2 lines - 1 word** as context and generating of up to 9 tokens if one of them was the "\n" symbol or otherwise moving on to the next ending.
4. After that, the last 2 syllables are extracted from the first line of the generated poem, as well as from each of the candidates for the ending.
5. In the case of a difference in the length of the 2 last syllables they are reduced to a common length by deleting leading characters. 
6. The Levenshtein distance is calculated and divided by the length of final characters (to favour longer similar parts - if potential endings with length 4 and length 6 have distance equal 1 then we want to divide it by 4 and 6 respectively to get lower value for the longer string).
7. Now we choose endings with minimal score and randomly select on of them.

## Models
Model fine-tuned using ["Pan Tadeusz"](https://drive.google.com/drive/folders/1E6y89iFpTYdcMVrCg4BFLKPU3mYiCu1t?usp=sharing) and [poems by Kochanowski and Tuwim](https://drive.google.com/drive/folders/1JyFHdcOx88TKyjEvs8AsnuQO8uVD6EB8?usp=sharing).

## Code
We provide a Python file with a sample notebook. 
The best way of running code is creation of conda environment, activating it and running  notebook
````
conda env create --name NAME --file nlp.yml
conda activate NAME
jupyter notebook
````
In case of problems with creating the environment due to a difference in platforms run:
```
conda update --all
conda export --no-builds > nlp.yml
conda env create -f nlp.yml
```
The "dirty version" and approaches not used in final version are available [here](https://github.com/MarBry111/NLP_poem_generator).

