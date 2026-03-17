# Obligatory 2: Cross-lingual Word-in-Context

Deadline: 20. 03. 2026 at 21:59

[**Assignment PDF**](./pdf/assignment.pdf)

## `datacat.sh`

Script that takes (at least two) language strings in `[de, en, es, ru, zh]`, 
and combines the corresponding datasets in `/fp/projects01/ec403/IN5550/obligatories/2` into multilingual versions.

Example:
```bash
./datacat.sh en ru
```
The result of this command would be that we get two files `train.jsonl` and `dev.jsonl` which are placed in `data/en_ru/`.

## Notes

- use "/cluster/work/projects/ec403/\<USERNAME\>/\<SUBDIRNAME\>" as cache dir for hugginface models
- The datasets are available on Fox at: /fp/projects01/ec403/IN5550/obligatories/2
- It might also be a good idea to use a smaller language model for debugging (such as mmBERT-small, available at /fp/projects01/ec403/hf_models/mmBERT-small). For the same reason, try to limit for how long you train the models, there is no benefit in running them on more than 10 epochs – much less usually works even better with good hyperparameters.
- The language models are already downloaded on Fox, please save your user space and load them from /fp/projects01/ec403/hf_models/.
