# QFS
# dataset
链接: https://pan.baidu.com/s/1i7idMVsdEkve4WLE0TewHw?pwd=ksn1 提取码: ksn1

# model
链接: https://pan.baidu.com/s/1tSxFvNXH95LWBDokw9Pm3A?pwd=6j40 提取码: 6j40

transformers==3.0.5

```bash

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained('model_path')
tokenizer = AutoTokenizer.from_pretrained('model_path')
text = 'What is the main contribution of this work? </s> Hi, my name is Bonsun Kim, and I\'d like to introduce our work, Data Augmentation for Rare Symptoms in Vaccine Side Effect Detection. So in this study, we studied the problem of anti-detection and normalization on patient self-report symptoms. In our dataset, we have patient self-reports in text form and their corresponding symptom entities. For example, if a patient describes their condition like, I feel muscle pain, then the label is myalgia in this case. Patients usually have multiple symptoms, and also they usually describe their conditions in colloquial ways. So this makes the task more challenging. In our task, we have several challenges. First, in medical domain, entity names can be long and it contains a lot of common nouns. And second, the number of entity types is large, and the number of labels in each example varies widely. For example, the patient reports contain from a minimum of 1 to a maximum of 131 symptoms. And lastly, many symptoms are rare, which is resulting in a long tailed distribution. So this graph is the distribution of symptoms in our dataset. So the symptoms like headache or pyrexia are really common, and they have lots of examples. But most of the symptoms are really rare, and it shows long-tailed distribution in our dataset. So the contribution of this work is, first, we frame entity detection and normalization as an entity retrieval task, not a classification method. Second, we propose a data augmentation method for data sparsity problem of rare symptoms. So next is about method. Firstly, let me talk about autoregressive entity retrieval models. As I mentioned, we frame entity detection and normalization as an entity retrieval task instead of classification. So this is a traditional classification approach. So when the input text is given, the portrayed model like BERT or BioBERT predicts scores through the feedforward layers on the top of the model. And in our work, we use entity retrieval model using encoder-decoder based model. So the goal of this work is predicting symptom entities corresponding to the input description. So we generate the symptom entity as sentence. For example, if the input sentence is like, I have muscle pain and fever, then the output sentence should be myalgia and pyrexia. So given the input and output sentences, the model is trained to maximize the probability which is used in the training length model generally. Next is data augmentation. Because the distribution of symptoms shows long tail, and most symptoms are rare in our training sets, so we propose data augmentation methods using definition of symptoms. So we can regard symptom definition as synthetic patient reports and symptom name as the corresponding labels. So we use two ways for obtaining definition. The first is pre-trained length model. So we set the prompt to the definition of symptom name and then use the generated sentence as a synthetic patient report. Next method is UMLS medical dictionary. So we first search terms with symptom names and then choose the first top result definition. So this is example of our data augmentation method. So first we get definition of symptoms through pre-trained length model and UMLS dictionary. Then we make augmented data like definition as input text and symptom name as its label. Also to mimic the more realistic scenarios of multiple symptoms, we also generate synthetic reports with up to two symptoms by concatenating the definitions. And next is experiments...'
input = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True, padding="max_length")
print(input)
summaries = model.generate(input_ids = input['input_ids'],attention_mask=input['attention_mask'])
dec = tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(dec)
```
