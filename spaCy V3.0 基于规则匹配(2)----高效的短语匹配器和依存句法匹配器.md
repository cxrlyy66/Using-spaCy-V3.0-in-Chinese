#spaCy V3.0 基于规则匹配(2)----高效的短语匹配器和依存句法匹配器

## 1 短语匹配器(PhraseMatcher)

### 1.1 基本用法
对于需要匹配大型术语列表的情况，可以通过PhraseMatcher和创建Doc对象来代替词符匹配模式(token patterns)，可以获得总体上**更高的效率**。Doc模式可以包含单个或多个词符。

	import spacy
	from spacy.matcher import PhraseMatcher

	nlp = spacy.load("zh_core_web_sm")
	matcher = PhraseMatcher(nlp.vocab)
	terms = ['失速区','喘振区','油膜破坏','电机漏磁']
	
	# 注意：只有使用 nlp.make_doc 才能加速
	
	patterns = [nlp.make_doc(text) for text in terms]
	matcher.add("TerminologyList", patterns)

	doc = nlp("轴承绝缘击穿，电机漏磁电流通过轴承造成油膜破坏。二次风系统挡板误关，引起系统阻力增大，造成风压与进入的风量不匹配，使风机进入喘振区。")
	matches = matcher(doc)
	for match_id, start, end in matches:
		span = doc[start:end]
		print(span.text)
		
> 注意：patterns 是一个Doc列表。
>
>创建patterns，每个短语都必须使用nlp对象进行处理。如果加载了预训练模型，则在循环或列表中执行此操作很容易变得低效和费时。如果您只需要分词和词法属性，那么可以运行nlp.make_doc,它只使用了分词器(tokenizer)。当然还可以使用nlp.tokenizer.pipe方法，将文本作为流进行处理，以得到进一步的速度提升。

> 错误用法：
> - patterns = [nlp(term) for term in LOTS_OF_TERMS]
>
>正确用法：
>+ patterns = [nlp.make_doc(term) for term in LOTS_OF_TERMS]
>+ patterns = list(nlp.tokenizer.pipe(LOTS_OF_TERMS))

### 1.2 匹配其他Token属性

默认情况下，PhraseMatcher将逐字匹配Token的文本，Token.text. 但通过在初始化时设置attr参数，可以更改匹配器(PhraseMatcher)在将短语模式(patterns)与文档进行比较时使用的Token属性。

> 注意：在前面的例子中，生成patterns列表用的是nlp.make_doc，它只使用了分词器(tokenizer)。对于本节需要匹配其他Token属性的情况，就要根据需要加入相应的组件。你可以直接使用nlp或通过 nlp.select_pipes()选择性的禁用某些组件。

比如：根据形状匹配数字Token（如IP地址）。使用Token的Shape属性将不必担心这些字符串如何分词，并且能够根据几个示例找到Tokens及其组合。下面我们将匹配形状ddd.d.d.d和ddd.ddd.d.d：

	matcher = PhraseMatcher(nlp.vocab, attr="SHAPE")
	matcher.add("IP", [nlp("127.0.0.1"), nlp("127.127.0.0")])

	doc = nlp("通常路由器有像'192.168.1.1'或'192.168.2.1'这样的IP地址。")
	for match_id, start, end in matcher(doc):
		print("Matched based on token shape:", doc[start:end])
		
当然从理论上讲，此方法对POS等属性也同样适用。例如，基于词性标签(tag)匹配的模式nlp(“我喜欢花”)将返回“我爱狗”的匹配。还可以匹配像IS_PUNCT这样的布尔属性，以匹配具有与模式相同的标点符号和非标点符号序列的短语。但是这么做很容易让人迷惑，且与编写一个或两个Token模式相比也没有太大的优势。

## 2 依存句法匹配器

DependencyMatcher使用Semgrex操作符匹配依存句法分析中的模式。它需要一个包含依存句法解析器的模型，比如DependencParser。DependencMatcher模式没有定义Matcher patterns中相邻Token的列表，而是匹配依存关系分析中的Roken并指定它们之间的关系。

依存句法匹配器的patterns由字典列表组成，每个字典描述要匹配的Token及其与patterns中现有Token的关系。除了第一个字典（它仅使用RIGHT_ID和RIGHT_ATTRS定义anchor token）之外，每个pattern 都应该具有以下4个键：

键名|说明
-|-
LEFT_ID|关系符左边的节点名称，该节点此前要出现在patters字典列表<br>str
REL_OP|表明左右两节点关系的操作符<br>str
RIGHT_ID|关系符右侧节点名称(该名称不能重复)<br>str
RIGHT_ATTRS|要匹配的关系符右侧节点的属性，其格式与Token Matcher中的patters相同<br>Dict[str, Any]

添加到patterns中的每个附加Token,都通过关系操作符REL_OP链接到现有名称为LEFT_ID的Token。新Token被命名为RIGHT_ID并由具有RIGHT_ATTRS描述的属性。

> 重要提示：由于用LEFT_ID和RIGHT_ID来作为识别Token的唯一名称，patters字典列表中的顺序就非常重要。所有作为LEFT_ID出现的节点，必须在前面的字典中作为RIGHT_ID被定义过！！！！

依存句法匹配器可用的操作符

符号|说明
-|-
A < B|A是B的直接子节点
A > B|A是B的直接头节点
A << B|A能够通过多个子节点到头节点关系跳转路径到达B
A >> B|A能够通过多个头节点到子节点关系跳转路径到达B
A . B|A是B的位置左邻节点, 即：A.i == B.i - 1 (A、B在同一依存解析树中,i是其Doc中的位置索引。 下同)
A .* B|A是B的位置前序节点, 即：A.i < B.i
A ; B|A是B的位置右邻节点, 即：A.i == B.i + 1
A ;* B|A是B的位置后序节点, 即：A.i > B.i
A $+ B|B是A的右邻同级节点, 即：A.head == B.head and A.i == B.i - 1
A $- B|B是A的左邻同级节点, 即：A.head == B.head and A.i == B.i + 1
A $++ B|B是A的位置后序同级节点, 即：A.head == B.head and A.i < B.i
A $-- B|B是A的位置前序同级节点, 即：A.head == B.head and A.i > B.i

- 依存句法匹配器的patterns

如果要从以下句子中找出“造成”什么后果：

*1 "轴承绝缘击穿，电机漏磁电流通过轴承造成油膜破坏。"
2 "冷渣器内部冷却水管泄漏造成灰渣板结。"*

我们要找到以下关系：

- 造成的直接宾语(dobj)
- 直接宾语(dobj)的复合名词修饰或形容词修饰(也可以没有修饰）


    nlp = spacy.load("zh_core_web_sm")
    matcher = DependencyMatcher(nlp.vocab)
    pattern = [
        {
            "RIGHT_ID": "anchor_word",       # 唯一名称
            "RIGHT_ATTRS": {"ORTH": "造成"}  # "造成"的token pattern
      },
      {
            "LEFT_ID": "anchor_word",
            "REL_OP": ">",
            "RIGHT_ID": "w_object",
            "RIGHT_ATTRS": {"DEP": "dobj"}
      },
      {
            "LEFT_ID": "w_object",
            "REL_OP": ">",
            "RIGHT_ID": "object_modifier",
            "RIGHT_ATTRS": {"DEP": {"IN":["compound:nn","amod"], "OP":"?"}}
            }      
    ]
    matcher.add("FOUNDED", [pattern])
    doc = nlp("冷渣器内部冷却水管泄漏造成灰渣板结。轴承绝缘击穿，电机漏磁电流通过轴承造成油膜破坏。")
    matches = matcher(doc)
    print(matches) 
    # 每一个token_id对应一个pattern字典
    for match_id, token_ids in matches:
        for i in range(len(token_ids)):
            print(pattern[i]["RIGHT_ID"] + ":", doc[token_ids[i]].text)    

运行结果：

    [(4851363122962674176, [5, 7, 6]), (4851363122962674176, [19, 21, 20])]
    anchor_word: 造成
    w_object: 板结
    object_modifier: 灰渣
    anchor_word: 造成
    w_object: 破坏
    object_modifier: 油膜
    
> **提高匹配速度的重要提示：**
>当token patterns能够潜在匹配句子中的许多token，或者当关系运算符在依存关系解析中的路径较长时（如<<、>>、*以及；*关系运算符），匹配速度可能会比较慢。

>为了提高匹配速度，操作符尽可能具体。例如，尽量使用 > 而不是 >> ，使用包含语义标签和其他Token属性，而不是像 {} 匹配句子中任何Token。





