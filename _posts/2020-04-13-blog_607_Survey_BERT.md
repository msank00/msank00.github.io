---
layout: post
title:  "Survey - BERT"
date:   2020-04-13 00:00:10 -0030
categories: jekyll update
mathjax: true
---

<link rel="stylesheet" href="/assets/css/markdownstyle.css">

# Content

1. TOC
{:toc}

----

<center>
<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/FKlPCK1uFrc" frameborder="0" allowfullscreen="true" width="100%" height="300"> </iframe>
</figure>
</center>

test line. 
{: .blue}

# Introduction

## History

2018 was a breakthrough year in NLP. Transfer learning, particularly models like Allen AI’s ELMO, OpenAI’s Open-GPT, and Google’s BERT allowed researchers to smash multiple benchmarks with minimal task-specific fine-tuning and provided the rest of the NLP community with pretrained models that could easily (with less data and less compute time) be fine-tuned and implemented to produce state of the art results. Unfortunately, for many starting out in NLP and even for some experienced practicioners, the theory and practical application of these powerful models is still not well understood.

## What is BERT?

<center>
<img src="/assets/images/image_39_bert_03.png" height="300" alt="image">
</center>

BERT (Bidirectional Encoder Representations from Transformers), released in late 2018, is the model we will use in this tutorial to provide readers with a better understanding of and practical guidance for using transfer learning models in NLP. BERT is a method of pretraining language representations that was used to create models that NLP practicioners can then download and use for free. You can either use these models to extract high quality language features from your text data, or you can fine-tune these models on a specific task (classification, entity recognition, question answering, etc.) with your own data to produce state of the art predictions.

This post will explain how you can modify and fine-tune BERT to create a powerful NLP model that quickly gives you state of the art results.


## A Shift in NLP

This shift to transfer learning parallels the same shift that took place in computer vision a few years ago. Creating a good deep learning network for computer vision tasks can take millions of parameters and be very expensive to train. Researchers discovered that deep networks learn hierarchical feature representations (simple features like edges at the lowest layers with gradually more complex features at higher layers). Rather than training a new network from scratch each time, the lower layers of a trained network with generalized image features could be copied and transfered for use in another network with a different task. It soon became common practice to download a pre-trained deep network and quickly retrain it for the new task or add additional layers on top - vastly preferable to the expensive process of training a network from scratch. For many, the introduction of deep pre-trained language models in 2018 (ELMO, BERT, ULMFIT, Open-GPT, etc.) signals the same shift to transfer learning in NLP that computer vision saw.



**Reference:**

- [Amazing blog by Chris McCormick](http://mccormickml.com/2019/07/22/BERT-fine-tuning/)
- [Amazing Youtube Lectures by Chris McCormick](https://www.youtube.com/playlist?list=PLam9sigHPGwOBuH4_4fr-XvDbe5uneaf6)
- [Important Colab Notebook by Chris McCormick](https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX)
- [Handson by Abhishek Thakur](https://www.youtube.com/watch?v=hinZO--TEk4&t=1454s)


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# BERT Shortcomings

1. BERT is very large. $\sim 109$M parameters with total size $\sim 417$MB.
   1. Slow fine-tuning.
   2. Slow inference.
2. Jargon (Domain specific language)
3. NOT all NLP applications can be solved using BERT
   1. :ballot_box_with_check: Classification, NER, POS Tagging, QnA
   2. :negative_squared_cross_mark: Language Modelling, Text Generation, Translation 

**Reference:**

- [Youtube video](https://www.youtube.com/watch?v=x66kkDnbzi4&list=PLam9sigHPGwOBuH4_4fr-XvDbe5uneaf6&index=3)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# BERT Word-Piece embedding

1. BERT is `pretrained` $\rightarrow$ Vocabulary is FIXED.
2. Breaks down `unknown words` into **sub-words**.

<center>
<img src="/assets/images/image_39_bert_01.png" width="600" alt="image">
</center>

3. A sub-word exists for every character.


```py
import torch
from pytorch_pretrained_bert import BertTokenizer

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

with open("vocabulary.txt", 'w') as f:
    
    # For each token...
    for token in tokenizer.vocab.keys():
        
        # Write it out and escape any unicode characters.            
        f.write(token + '\n')
```

From perusing the vocab, I'm seeing that:

```R
#    The first 999 tokens (1-indexed) appear to be reserved, and most are of the form [unused957].
  1 - [PAD]
  101 - [UNK]
  102 - [CLS]
  103 - [SEP]
  104 - [MASK]
# Rows 1000-1996 appear to be a dump of individual characters.
  They do not appear to be sorted by frequency (e.g., the letters of the alphabet are all in sequence).
# The first word is "the" at position 1997.
  From there, the words appear to be sorted by frequency.
# The top ~18 words are whole words, and then number 2016 is ##s, presumably the most common sub-word.
# The last whole word is at 29612, "necessitated"
```
<center>
<img src="/assets/images/image_39_bert_02.png" height="300" alt="image">
</center>


```R
Number of single character tokens: 997 

! $ % &  ( ) * + , - . / 0 1 2 3 4 5 6 7 8 9 : ; < = > ? @ [ \ ] ^ _ ` a b
c d e f g h i j k l m n o p q r s t u v w x y z { | } ~ ¡ ¢ £ ¤ ¥ ¦ § ¨ © ª « ¬
® ° ± ² ³ ´ µ ¶ · ¹ º » ¼ ½ ¾ ¿ × ß æ ð ÷ ø þ đ ħ ı ł ŋ œ ƒ ɐ ɑ ɒ ɔ ɕ ə ɛ ɡ ɣ ɨ
ɪ ɫ ɬ ɯ ɲ ɴ ɹ ɾ ʀ ʁ ʂ ʃ ʉ ʊ ʋ ʌ ʎ ʐ ʑ ʒ ʔ ʰ ʲ ʳ ʷ ʸ ʻ ʼ ʾ ʿ ˈ ː ˡ ˢ ˣ ˤ α β γ δ
ε ζ η θ ι κ λ μ ν ξ ο π ρ ς σ τ υ φ χ ψ ω а б в г д е ж з и к л м н о п р с т у
ф х ц ч ш щ ъ ы ь э ю я ђ є і ј љ њ ћ ӏ ա բ գ դ ե թ ի լ կ հ մ յ ն ո պ ս վ տ ր ւ
ք ־ א ב ג ד ה ו ז ח ט י ך כ ל ם מ ן נ ס ע ף פ ץ צ ק ר ש ת ، ء ا ب ة ت ث ج ح خ د
ذ ر ز س ش ص ض ط ظ ع غ ـ ف ق ك ل م ن ه و ى ي ٹ پ چ ک گ ں ھ ہ ی ے अ आ उ ए क ख ग च
ज ट ड ण त थ द ध न प ब भ म य र ल व श ष स ह ा ि ी ो । ॥ ং অ আ ই উ এ ও ক খ গ চ ছ জ
ট ড ণ ত থ দ ধ ন প ব ভ ম য র ল শ ষ স হ া ি ী ে க ச ட த ந ன ப ம ய ர ல ள வ ா ி ு ே
ை ನ ರ ಾ ක ය ර ල ව ා ก ง ต ท น พ ม ย ร ล ว ส อ า เ ་ ། ག ང ད ན པ བ མ འ ར ལ ས မ ა
ბ გ დ ე ვ თ ი კ ლ მ ნ ო რ ს ტ უ ᄀ ᄂ ᄃ ᄅ ᄆ ᄇ ᄉ ᄊ ᄋ ᄌ ᄎ ᄏ ᄐ ᄑ ᄒ ᅡ ᅢ ᅥ ᅦ ᅧ ᅩ ᅪ ᅭ ᅮ
ᅯ ᅲ ᅳ ᅴ ᅵ ᆨ ᆫ ᆯ ᆷ ᆸ ᆼ ᴬ ᴮ ᴰ ᴵ ᴺ ᵀ ᵃ ᵇ ᵈ ᵉ ᵍ ᵏ ᵐ ᵒ ᵖ ᵗ ᵘ ᵢ ᵣ ᵤ ᵥ ᶜ ᶠ ‐ ‑ ‒ – — ―
‖ ‘ ’ ‚ “ ” „ † ‡ • … ‰ ′ ″ › ‿ ⁄ ⁰ ⁱ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹ ⁺ ⁻ ⁿ ₀ ₁ ₂ ₃ ₄ ₅ ₆ ₇ ₈ ₉ ₊ ₍
₎ ₐ ₑ ₒ ₓ ₕ ₖ ₗ ₘ ₙ ₚ ₛ ₜ ₤ ₩ € ₱ ₹ ℓ № ℝ ™ ⅓ ⅔ ← ↑ → ↓ ↔ ↦ ⇄ ⇌ ⇒ ∂ ∅ ∆ ∇ ∈ − ∗
∘ √ ∞ ∧ ∨ ∩ ∪ ≈ ≡ ≤ ≥ ⊂ ⊆ ⊕ ⊗ ⋅ ─ │ ■ ▪ ● ★ ☆ ☉ ♠ ♣ ♥ ♦ ♭ ♯ ⟨ ⟩ ⱼ ⺩ ⺼ ⽥ 、 。 〈 〉
《 》 「 」 『 』 〜 あ い う え お か き く け こ さ し す せ そ た ち っ つ て と な に ぬ ね の は ひ ふ へ ほ ま み
む め も や ゆ よ ら り る れ ろ を ん ァ ア ィ イ ウ ェ エ オ カ キ ク ケ コ サ シ ス セ タ チ ッ ツ テ ト ナ ニ ノ ハ
ヒ フ ヘ ホ マ ミ ム メ モ ャ ュ ョ ラ リ ル レ ロ ワ ン ・ ー 一 三 上 下 不 世 中 主 久 之 也 事 二 五 井 京 人 亻 仁
介 代 仮 伊 会 佐 侍 保 信 健 元 光 八 公 内 出 分 前 劉 力 加 勝 北 区 十 千 南 博 原 口 古 史 司 合 吉 同 名 和 囗 四
国 國 土 地 坂 城 堂 場 士 夏 外 大 天 太 夫 奈 女 子 学 宀 宇 安 宗 定 宣 宮 家 宿 寺 將 小 尚 山 岡 島 崎 川 州 巿 帝
平 年 幸 广 弘 張 彳 後 御 德 心 忄 志 忠 愛 成 我 戦 戸 手 扌 政 文 新 方 日 明 星 春 昭 智 曲 書 月 有 朝 木 本 李 村
東 松 林 森 楊 樹 橋 歌 止 正 武 比 氏 民 水 氵 氷 永 江 沢 河 治 法 海 清 漢 瀬 火 版 犬 王 生 田 男 疒 発 白 的 皇 目
相 省 真 石 示 社 神 福 禾 秀 秋 空 立 章 竹 糹 美 義 耳 良 艹 花 英 華 葉 藤 行 街 西 見 訁 語 谷 貝 貴 車 軍 辶 道 郎
郡 部 都 里 野 金 鈴 镇 長 門 間 阝 阿 陳 陽 雄 青 面 風 食 香 馬 高 龍 龸 ﬁ ﬂ ！ （ ） ， － ． ／ ： ？ ～ 
" ' #
```

**Reference:**

- [Youtube - add link]()

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# BERT Tokenizer

To feed our text to BERT, it must be split into tokens, and then these tokens must be mapped to their index in the tokenizer vocabulary.

The **tokenization must be performed by the tokenizer included with BERT** -- the below cell will download this for us. We'll be using the BERT `base` model with `uncased` version.

**Note:** 

- BERT has `base` and `large` model. But `large` model size is quite big, so we work with `base` model.  
- `uncase` means no Upper case letters in the vocabulary

```py
from transformers import BertTokenizer

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Loading BERT tokenizer...
```

Let's apply the tokenizer to one sentence just to see the output.

```py
# Print the original sentence.
print(' Original: ', sentences[0])

# Print the sentence split into tokens.
print('Tokenized: ', tokenizer.tokenize(sentences[0]))

# Print the sentence mapped to token ids.
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))

# Output:
# Original:  Our friends won't buy this analysis, let alone the next one we propose.
# Tokenized:  ['our', 'friends', 'won', "'", 't', 'buy', 'this', 'analysis', ',', 'let', 'alone', 'the', 'next', 'one', 'we', 'propose', '.']
# Token IDs:  [2256, 2814, 2180, 1005, 1056, 4965, 2023, 4106, 1010, 2292, 2894, 1996, 2279, 2028, 2057, 16599, 1012]

```

**Reference:**

- [Colab Notebook ](https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX#scrollTo=-8kEDRvShcU5)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----

# Required Formatting


We are required to:

- **Add special tokens** to the start and end of each sentence.
- Pad & truncate all sentences to a single constant length.
- Explicitly differentiate real tokens from padding tokens with the `attention mask`.

## Special Tokens

- `[SEP]`

At the end of every sentence, we need to append the special [SEP] token.

This token is an artifact of **two-sentence tasks**, where BERT is given **two separate sentences** and asked to determine something (e.g., can the answer to the question in sentence A be found in sentence B?).

I am not certain yet why the token is still required when we have only single-sentence input, but it is!

- `[CLS]`

For **classification tasks**, we must pre-pend the special [CLS] token to the beginning of every sentence.

This token has special significance. **BERT consists of 12 Transformer layers** stacked on top of each other. Each transformer takes in a list of token embeddings, and produces the same number of embeddings on the output (but with the feature values changed, of course!).

<center>
<img src="/assets/images/image_39_bert_04.png" width="350" alt="image">
</center>

On the output of the final (12th) transformer, only the first embedding (corresponding to the [CLS] token) is used by the classifier.

>> "The first token of every sequence is always a special classification token ([CLS]). The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks." (from the [BERT paper](https://arxiv.org/pdf/1810.04805.pdf))

You might think to try some pooling strategy over the final embeddings, but this isn't necessary. Because BERT is trained to only use this [CLS] token for classification, we know that the model has been motivated to encode everything it needs for the classification step into that single 768-value embedding vector. It's already done the pooling for us!

**Reference:**

- [Colab Notebook by Chris McCormick](https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX#scrollTo=-8kEDRvShcU5)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----

# Sentence Length & Attention Mask

The sentences in our dataset obviously have varying lengths, so how does BERT handle this?

BERT has two constraints:

- All sentences **must be padded or truncated** to a single, fixed length.
- The maximum sentence length is $512$ tokens.

Padding is done with a special `[PAD]` token, which is at index $0$ in the BERT vocabulary. The below illustration demonstrates padding out to a `MAX_LEN` of $8$ tokens.

<center>
<img src="/assets/images/image_39_bert_05.png" width="450" alt="image">
</center>

The "Attention Mask" is simply an array of 1s and 0s indicating which tokens are padding and which aren't (seems kind of redundant, doesn't it?!). This mask tells the "Self-Attention" mechanism in BERT not to incorporate these PAD tokens into its interpretation of the sentence.

The maximum length does impact training and evaluation speed, however. For example, with a Tesla K80:

- MAX_LEN = $128$ --> Training epochs take $\sim 5:28$ each
- MAX_LEN = $64$ --> Training epochs take $\sim 2:57$ each

**Reference:**

- [Colab Notebook by Chris McCormick](https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX#scrollTo=-8kEDRvShcU5)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----

# BertForSequenceClassification

For this task, we first want to modify the pre-trained BERT model to give outputs for classification, and then we want to continue training the model on our dataset until that the entire model, end-to-end, is well-suited for our task.

Thankfully, the huggingface pytorch implementation includes a set of interfaces designed for a variety of NLP tasks. Though these interfaces are all built on top of a trained BERT model, each has different top layers and output types designed to accomodate their specific NLP task.

Here is the current list of classes provided for fine-tuning:


- BertModel
- BertForPreTraining
- BertForMaskedLM
- BertForNextSentencePrediction
- **BertForSequenceClassification** - The one we'll use.
- BertForTokenClassification
- BertForQuestionAnswering

The documentation for these can be found under [here](https://huggingface.co/transformers/v2.2.0/model_doc/bert.html).


- We'll be using [BertForSequenceClassification](https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#bertforsequenceclassification). This is the normal BERT model with an added `single linear layer` **on top for classification** that we will use as a sentence classifier. 
- As we feed input data, the entire pre-trained BERT model and the additional untrained classification layer is trained on our specific task. 

OK, let's load BERT! There are a few different pre-trained BERT models available. `bert-base-uncased` means the version that has only **lowercase letters** (`uncased`) and is the smaller version of the two (`base` vs `large`).

The documentation for from_pretrained can be found [here](https://huggingface.co/transformers/v2.2.0/main_classes/model.html#transformers.PreTrainedModel.from_pretrained), with the additional parameters defined [here](https://huggingface.co/transformers/v2.2.0/main_classes/configuration.html#transformers.PretrainedConfig).

```py
from transformers import BertForSequenceClassification, AdamW, BertConfig

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.cuda()
```

Just for curiosity's sake, we can browse all of the model's parameters by name here.

In the below cell, I've printed out the names and dimensions of the weights for:

- The embedding layer.
- The first of the twelve transformers.
- The output layer.


```py
# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
```


```py
The BERT model has 201 different named parameters.

==== Embedding Layer ====

bert.embeddings.word_embeddings.weight                  (30522, 768)
bert.embeddings.position_embeddings.weight                (512, 768)
bert.embeddings.token_type_embeddings.weight                (2, 768)
bert.embeddings.LayerNorm.weight                              (768,)
bert.embeddings.LayerNorm.bias                                (768,)

==== First Transformer ====

bert.encoder.layer.0.attention.self.query.weight          (768, 768)
bert.encoder.layer.0.attention.self.query.bias                (768,)
bert.encoder.layer.0.attention.self.key.weight            (768, 768)
bert.encoder.layer.0.attention.self.key.bias                  (768,)
bert.encoder.layer.0.attention.self.value.weight          (768, 768)
bert.encoder.layer.0.attention.self.value.bias                (768,)
bert.encoder.layer.0.attention.output.dense.weight        (768, 768)
bert.encoder.layer.0.attention.output.dense.bias              (768,)
bert.encoder.layer.0.attention.output.LayerNorm.weight        (768,)
bert.encoder.layer.0.attention.output.LayerNorm.bias          (768,)
bert.encoder.layer.0.intermediate.dense.weight           (3072, 768)
bert.encoder.layer.0.intermediate.dense.bias                 (3072,)
bert.encoder.layer.0.output.dense.weight                 (768, 3072)
bert.encoder.layer.0.output.dense.bias                        (768,)
bert.encoder.layer.0.output.LayerNorm.weight                  (768,)
bert.encoder.layer.0.output.LayerNorm.bias                    (768,)

==== Output Layer ====

bert.pooler.dense.weight                                  (768, 768)
bert.pooler.dense.bias                                        (768,)
classifier.weight                                           (2, 768)
classifier.bias                                                 (2,)
```

**Reference:**

- [Colab Notebook by Chris McCormick](https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX#scrollTo=-8kEDRvShcU5)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----

# Optimizer & Learning Rate Scheduler

Now that we have our model loaded we need to grab the training hyperparameters from within the stored model.

For the purposes of fine-tuning, the authors recommend choosing from the following values (from Appendix A.3 of the [BERT paper](https://arxiv.org/pdf/1810.04805.pdf)):

- Batch size: $16$, $32$
- Learning rate (Adam): $5e^{-5}$, $3e^{-5}$, $2e^{-5}$
- Number of epochs: $2$, $3$, $4$

We chose:

- Batch size: 32 (set when creating our DataLoaders)
- Learning rate: $2e^{-5}$
- Epochs: 4 (we'll see that this is probably too many...)

The epsilon parameter $eps = 1e^{-8}$ is "a very small number to prevent any division by zero in the implementation" (from [here](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)).

You can find the creation of the AdamW optimizer in `run_glue.py` [here](https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L109).

**NOTE:** Please check this amazing script from Huggingface :hugs: , [run_glue.py](https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L109). 

**Reference:**

- [Colab Notebook by Chris McCormick](https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX#scrollTo=-8kEDRvShcU5)


----

# Sentence Transformers: Sentence Embeddings using BERT a.k.a Sentence BERT

From the abstract of the original paper

**BERT** (Devlin et al., $2018$) and **RoBERTa** (Liu et al., 2019) has set a new state-of-the-art performance on **sentence-pair regression** tasks like `semantic textual similarity` (STS). However, it requires that both sentences are fed into the network, which causes a massive computational overhead: Finding the most similar pair in a collection of 10,000 sentences requires about 50 million inference computations (~65 hours) with BERT. The construction of BERT makes it unsuitable for semantic similarity search as well as for unsupervised tasks like clustering.

In this paper, we present **Sentence-BERT** (SBERT), a modification of the pretrained BERT network that use `siamese` and `triplet network` structures to derive semantically meaningful sentence embeddings that can be compared using cosine-similarity. This reduces the effort for finding the most similar pair from 65 hours with BERT / RoBERTa to about 5 seconds with SBERT, while maintaining the accuracy from BERT. 

![image](/assets/images/image_06_SBERT_1.png)


## How to use it in code

```py
# install the package
pip install -U sentence-transformers
```

```py
# download a pretrained model.
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Then provide some sentences to the model.
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']
sentence_embeddings = model.encode(sentences)

# And that's it already. We now have a list of numpy arrays with the embeddings.
for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
```

_**for more details check the pypi repository_


:paperclip: **Reference:**

- [PyPi sentence-transformers](https://pypi.org/project/sentence-transformers/#Training)
- [arXiv: Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) :fire:

----

# BERT Inner Workings 

**Original NMT**

<center>
<img src="/assets/images/image_39_bert_06.png" width="500" alt="image">
</center>


<center>
<img src="/assets/images/image_39_bert_08.png" width="500" alt="image">
</center>



2 RNN, Encoder and Decoder.

- Decoder generates probability distribution for each of the word in the vocab and you pick which has the highest probability.
- Next given the first decoder output what will be the next decoder output and so on.You can add `Teacher-Forcing` here if required.

**Google NMT**

<center>
<img src="/assets/images/image_39_bert_07.png" width="500" alt="image">
</center>


<center>
<img src="/assets/images/image_39_bert_09.svg" width="700" alt="image">
</center>


- Google added the `attention` (the concept existed already)
- First **major** use of attention in NLP.
- In Original NMT, entire sentence is encoded into one encoding vector. 
- But google, 
>> instead of looking at the single encoded output vector, they look at the individual hidden state for all the input words and **attention mechanism takes a linear combination of those hidden states** (gives different weights to different input words' hidden representation). :+1: :rocket: 

_**MUST watch** [lecture 16, prof. Mikesh IIT-M](https://www.cse.iitm.ac.in/~miteshk/CS7015.html) **for in-depth attention understanding**_ :fire: :fire:


- So when the decoder tries to output the word `European`, hope that the attention mechanism already giving lots of weight to the input word `European` and less weight to the other words.
- In general attention tells the model to which words it should focus on.
- The biggest benefit of Transformer model is it's ability to run in parallel. 

## BERT Architecture

- BERT is an enhancement of the Transformer model (**Attention is all you need**).
  - Transformer model is an encoder-decoder model with stack of 6 encode and decoder. 
- In BERT the decoder part is dropped and numbers of encoder layers are increased from 6 to 12


<center>
<img src="/assets/images/image_39_bert_10.png" width="700" alt="image">
</center>

- Now the main challenge for this model is how to train or on which kind of task you will train this model.
  - 2 **fake tasks** are created
  - **Predict the masked word**
  - Next sentence prediction (was `Sentence B` found immediately after `Sentence A` ?)

**BERT is conceptually not so simple but empirically very powerful**. 

The BERT Encoder block implements the base version of the BERT network. It is composed of 12 successive transformer layers, each having 12 attention heads.
The total number of parameters is 110 million.

<center>
<img src="https://peltarion.com/static/bert_encoder_block.svg" alt="image" height="600">
</center>

_The architecture is reverse. Input at the top and the output at the bottom._


Every token in the input of the block is first embedded into a learned `768-long` **embedding vector**.

Each embedding vector is then transformed progressively every time it traverses one of the BERT Encoder layers:

- Through linear projections, every embedding vector creates a **triplet** of `64-long vectors`, called the **key, query, and value vectors**
- The key, query, and value vectors from all the embeddings pass through a **self-attention head**, which outputs one `64-long vector` for each **input triplet**.
- Every output vector from the self-attention head is a function of the whole input sequence, which is what makes **BERT context-aware**.
- A single embedding vector uses **different linear projections** to create `12 unique` **triplets of key, query, and value** vectors, which all go through their own self-attention head.
- This allows each self-attention head to focus on different aspects of how the tokens interact with each other.
- The output from all the self-attention heads are first concatenated together, then they go through another linear projection and a feed-forward layer, which helps to utilize **deep non-linearity**
  - **Residual connections from previous states** are also used to increase robustness.



The result is a sequence of transformed embedding vectors, which are sent through the same layer structure 11 more times.

After the 12th encoding layer, the embedding vectors have been transformed to contain more accurate information about each token. You can choose if you want the BERT Encoder block to return all of them or only the first one (corresponding to the [CLS] token), which is often sufficient for classification tasks.


:paperclip: **Reference:**

- [How Google Translate Works - YT Video](https://www.youtube.com/watch?v=AIpXjFwVdIE) :fire:
- [BERT Research - Ep. 4 - Inner Workings I by  ChrisMcCormickAI
](https://www.youtube.com/watch?v=C4jmYHLLG3A&list=PLam9sigHPGwOBuH4_4fr-XvDbe5uneaf6&index=5) :fire: :rocket:
- [BERT Architecture](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/blocks/bert-encoder?fbclid=IwAR1t_a3no4BRylPk_29fZbKwmKB1mRdT0jFLSzXWL0t5fnSKKXTZlpKCVsA) :fire:

----


<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>