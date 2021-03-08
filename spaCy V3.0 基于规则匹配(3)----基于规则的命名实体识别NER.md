#spaCy V3.0 基于规则匹配(3)----基于规则的命名实体识别NER

EntityRuler是一个spaCy管道组件，可以通过基于patterns字典添加命名实体，能够方便基于规则和统计方式的命名实体识别方法相结合，从而实现功能更强大的spaCy管道。

## 1 实体patterns

实体patterns是一个字典，包含两个键：“label”，指定模式匹配时的实体标签，“pattern”，匹配模式。EntityRuler接受两种类型的pattern：

- 1 用于精确字符匹配的短语模式(Phrase patterns)：
    
    
    {"label": "PROPER_N", "pattern": "喘振区"}
    
- 2 词符模式(Token patterns) ，用字典描述一个词符(Token)或词符(Token)列表：


    {"label": "T", "pattern": [{"POS": "NOUN", "OP":"?"},{"POS": "NOUN"}, {"ORTH": "温度"}]}
    
    #对于例句：  "冷渣器内部冷却水管泄漏造成灰渣板结。轴承绝缘击穿，电机漏磁电流通过轴承造成油膜破坏。"
    
    #其结果为：轴承温度,盘根温度,电机线圈温度
  
## 2 EntityRuler的用法

EntityRuler是一个spaCy管道组件，通常通过nlp.add_pipe添加到spaCy管道。 当对文本调用nlp对象时，会在doc中找到匹配项，将其作为实体添加到doc.ents，并使用指定的pattern标签作为实体标签。如果有匹配项存在重叠，则以长度优先的模式匹配。等长时，则选择文档中先出现的匹配项。

    nlp = spacy.load("zh_core_web_sm")
    user_dict = get_user_dict('c:/user_dict.txt')
    nlp.tokenizer.pkuseg_update_user_dict(user_dict)
    ruler = nlp.add_pipe('entity_ruler')
    patterns = [{"label": "A", "pattern": [{"POS": "NOUN", "OP":"?"},{"POS": "NOUN"}, {"ORTH": "电流"}]},\
                {"label": "T", "pattern": [{"POS": "NOUN", "OP":"?"},{"POS": "NOUN"}, {"ORTH": "温度"}]}]
    ruler.add_patterns(patterns)
    doc = nlp('启动一台真空泵，检查真空泵电机电流、入口压力、轴承温度、盘根温度、电机线圈温度、声音、振动正常。')
    print([(ent.text, ent.label_) for ent in doc.ents])
    
    #其结果为：[('真空泵电机电流', 'A'), ('轴承温度', 'T'), ('盘根温度', 'T'), ('电机线圈温度', 'T')]
    
EntityRuler的设计是通过与spaCy的现有管道组件集成，来增强命名实体识别器的功能。如果将其添加到“ner”组件之前，spaCy的实体识别器将保留EntityRuler已识别出的实体spans并调整在其周围的其他预测。在某些情况下，可以显著提高准确性。如果在“ner”组件之后添加，EntityRuler将只向doc中添加与模型预测出的不重叠的实体。如果要覆盖重叠的实体，可以在初始化时设置overwrite_ents=True。

## 3 EntityRuler patterns的验证和调试

    ruler = nlp.add_pipe("entity_ruler", config={"validate": True})
    
## 4 向patterns中加入IDs

EntityRuler还可以给每个pattern增加id属性。使用id属性可以将多个pattern与同一实体相关联。

    patterns = [{"label": "HEART", "pattern":"高压加热器", "id" : "hp_heater"},\
                {"label": "HEART", "pattern":"高加", "id" : "hp_heater"}]
                
    #对于例句：“解列高压加热器水侧，关闭高加入口三通阀，给水走旁路，注意给水压力、流量应无波动。关闭高加出口门。 ”           
    
    #[('高压加热器', 'HEART', 'hp_heater'), ('高加', 'HEART', 'hp_heater'), ('高加', 'HEART', 'hp_heater')]
    
其Entity Label为：'HEART', ent\_id__：'hp\_heater'

如果对 '低压加热器' 和 '低加' ，其Entity Label也是：'HEART', 但可以定义其ent\_id\_：'lp\_heater'

## 5 使用pattern文件

使用EntityRuler的 to_disk函数可以将patterns保存到以换行符分隔的Json文件中，也可以通过 from_disk函数从Json文件加载回来。注意Json文件的每一行包含一个pattern。

Json文件格式示例：

    {"label": "HEART", "pattern":"高压加热器", "id" : "hp_heater"}
    {"label": "HEART", "pattern":"高加", "id" : "hp_heater"}
    {"label": "A", "pattern": [{"POS": "NOUN", "OP":"?"},{"POS": "NOUN"}, {"ORTH": "电流"}]}
    {"label": "T", "pattern": [{"POS": "NOUN", "OP":"?"},{"POS": "NOUN"}, {"ORTH": "温度"}]}

patterns保存和加载：

    ruler.to_disk("./patterns.jsonl")
    new_ruler = nlp.add_pipe("entity_ruler").from_disk("./patterns.jsonl")
    
nlp对象如果包含EntityRuler，当保存nlp对象时，EntityRuler的patterns也将自动导出到pipeline 目录：

    nlp = spacy.load("zh_core_web_sm")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([{"label": "PROPER_N", "pattern": "喘振区"}])
    nlp.to_disk("/path/to/pipeline")
    
保存的pipeline会将entity_ruler包含到其config.cfg中，pipeline也会包括一个entityruler.jsonl。当重新加载pipeline时，所有pipeline组件都将被还原和反序列化，包括EntityRuler。可以让你交付功能强大的包括二进制权重和规则在内的pipeline packages！

## 6 大规模短语模式(phrase patterns)的用法

当使用大量短语模式（比如>10000）时，了解EntityRuler的add_patterns函数是如何工作的将非常有用。对于每个短语模式，EntityRuler调用nlp对象来构造doc对象。比如尝试在现有管道组件（POS tagger）后添加EntityRuler，并希望基于模式的POS属性提取匹配项，则会发生这种情况。这时，将为EntityRuler传递一个配置值“phrase_matcher_attr”：“POS”。

在一个庞大的列表中对每个模式运行完整的语言pipelines，可能需要很长时间。一个简单解决方法是在添加短语模式(phrase patterns)时禁用不需要的其他语言管道组件。

	ruler = nlp.add_pipe("entity_ruler")
	patterns = [{"label": "TEST", "pattern": str(i)} for i in range(100000)]
	with nlp.select_pipes(enable="tagger"):
		ruler.add_patterns(patterns)

>这里就只使用了"tagger"。