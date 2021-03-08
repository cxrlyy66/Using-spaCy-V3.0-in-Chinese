#spaCy V3.0.0 专业领域中文分词问题

>spaCy 3.0.0版本今年已经正式发布。非常幸运的是，其提供的5个最新transformer-based pipelines 模型中就包括中文预训练模型（zh_core_web_trf）。该模型在中文上的表现也相当突出:

PACKAGE|LANGUAGE|TRANSFORMER|TAGGER|PARSER| NER
-|-|-|-|-|-
en_core_web_trf|English|roberta-base|97.8|95.2|89.9
de_dep_news_trf|German|bert-base-german-cased|99.0|95.8|-
es_dep_news_trf|Spanish|bert-base-spanish-wwm-cased|98.2|94.6|-
fr_dep_news_trf|French|camembert-base|95.7|94.4|-
zh_core_web_trf|Chinese|bert-base-chinese|92.5|76.6|75.4

对于中文还有3个预训练模型：
1. zh_core_web_sm-3.0.0: 小型
1. zh_core_web_md-3.0.0: 中型
1. zh_core_web_lg-3.0.0: 大型

**注：这三者没有采用TRANSFORMER，而是tok2vec**

与 spaCy V2不同的是，增加了管道组件：
 - V2: tagger, parser, ner
 - V3: tok2vec/transformer, tagger, parser, ner, attribute_ruler

关于新版本的其他特性，我将结合以后个人在具体领域使用过程中的心得逐步展开。

## 关于中文专业词汇

在中文NLP中，首先遇到的就是中文分词问题。与英文不同，英文分词天然使用空格，但中文却没有这样的天然分隔。对于日常通用领域，各分词器的表现还不错，但涉及到特定专业领域，会存在许多专业词汇，需要用到用户自定义词典(user dictionary)。

在专业领域应用spaCy进行特定任务处理时，就会出现问题，比如这样一句话：

*"调整给水，注意给水流量与蒸汽流量相匹配，注意过热度，保证主蒸汽温度不超限。"*

**其正确断句应该是：**

*调整/给水/，/注意/给水流量/与/蒸汽流量/相匹配/，/注意/过热度/，/保证/主蒸汽/温度/不/超限/。*

**但直接使用spaCy得到的doc中的Token是：**

*调整/给水/，/注意/给/水流量/与/蒸汽/流量/相匹配/，/注意/过/热度/，/保证/主蒸/汽温度/不/超限/。*

    import spacy
    
    nlp = spacy.load('zh_core_web_sm')
    proper_nouns = ['给水流量','蒸汽流量','过热度','主蒸汽']
    nlp.tokenizer.pkuseg_update_user_dict(proper_nouns)
    
    doc = nlp('调整给水，注意给水流量与蒸汽流量相匹配，注意过热度，保证主蒸汽温度不超限。')
    print('/'.join([t.text for t in doc]))
    
这时，就得到了我们想要的结果：

*调整/给水/，/注意/给/水流量/与/蒸汽/流量/相匹配/，/注意/过/热度/，/保证/主蒸/汽温度/不/超限/。*


    

