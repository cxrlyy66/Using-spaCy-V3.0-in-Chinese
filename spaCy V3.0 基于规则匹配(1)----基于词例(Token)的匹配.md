#spaCy V3.0 基于规则匹配(1)----基于词符(Token)的匹配

> 用于发现短语、词符(tokens)、实体
> 

*相比于在普通文本上使用正则表达式，spaCy基于规则的匹配引擎和组件不仅可以找到要查找的单词和短语，还可以访问文档中的词符(tokens)及其关系。这意味着可以轻松地访问和分析被查找词符(tokens)周围的词符，将spans合并为单个Token，或者向doc.ents中的命名实体添加条目.*

**在介绍spaCy基于规则匹配的内容之前，首先回答两个问题。**
- 1 使用规则还是训练模型？
.
对于复杂的任务，通常更好的做法是训练一个统计实体识别模型。然而，统计模型需要大量训练数据，因此，在许多情况下，基于规则的方法更为实用，尤其对一个新开始的项目。可以使用基于规则的方法作为数据收集过程的一部分，采用“自举法”启动统计模型。
.
如果希望系统能够基于现有示例类推到更大范围时，那么训练模型是非常有用的。要是有局部上下文线索，效果更好。例如，如果你试图发现人名或公司名，则此应用会得益于统计命名实体识别模型。
.
如果要从数据中找到的示例数量是有限的，或这些示例具有非常清晰、结构化模式，可以使用标记规则或正则表达式来表示，那么基于规则的系统就是一个不错的选择。例如，使用纯基于规则的方法，或许就能够很好地处理国家名称、IP地址或URL。
.
当然也可以将这两种方法结合起来，并使用规则来改进统计模型，以处理非常具体的案例提高准确性。有关详细信息，请参见基于第4节《基于规则的实体识别》。

- 2 使用词符匹配器(Token Mather)还是短语匹配器(PhraseMatcher)？
.
如果你已经有了一个由单个或多个Token短语组成的大型术语列表或地名录，并且希望在数据中找到其准确实例，那么PhraseMatcher非常有用。
.
Matcher(Token Mather)没有PhraseMatcher快，因为它是对各个Token属性进行比较。但是，它能够对所需查找的Tokens编写非常抽象的表示，比如：词汇属性、模型预测的语言特征、运算符、集合操作和丰富的比较操作。例如，你可以找到一个名词，后跟一个动词“爱”或“喜欢”，再跟一个或不跟限定词，以及另一个至少10个字符长的Token。

## 1 基于词符(Token)的匹配

spaCy提供了一个很有特色的Token规则匹配引擎---Matcher，类似于正则表达式。这些规则可以引用Token注释（例如，Token的text、tag_），以及标志（例如，IS_PUNCT）。此规则匹配器还允许传入一个自定义回调函数来对匹配项进行操作。例如，合并实体和应用自定义标签。您还可以将模式与实体id相关联，进行一些基本的实体链接或消除歧义任务。对于大型术语列表匹配，可以使用PhraseMatcher，它可以使用Doc对象作为匹配模式。

### 1.1 添加模式

例如：写一个模板，匹配到'母管' 后面跟着一个名词"NOUN",

    pattern = [{'ORTH':'母管'},{'POS':'NOUN'},{'POS':'VERB','OP':"?"},{'ORTH':'报警'}]

>重要提示：
>在编写模式时，请记住模式中的每个字典代表一个词符(Token)。如果spaCy的词符(Token)与模式中定义的词符(Token)不匹配，那么该模式将不会产生任何结果。开发复杂模式时，请确保参照spaCy的词符(Token)来检查示例：

	import spacy
	from spacy.matcher import Matcher

	nlp = spacy.load('zh_core_web_trf')
	doc = nlp('辅机冷却水母管压力低报警。')
	#分词结果 
	[t for t in doc]
	#[辅机, 冷却, 水母, 管, 压力, 低, 报警, 。]

	patterns = [{'ORTH':'母管'},{'POS':'NOUN'},{'POS':'VERB','OP':"?"},{'ORTH':'报警'}]
	matcher = Matcher(nlp.vocab)
	matcher.add("MY_PATTERN", [patterns])
	matches = matcher(doc)

	for match_id, start, end in matches:
		string_id = nlp.vocab.strings[match_id]  # Get string representation
		span = doc[start:end]  # The matched span
		print(match_id, string_id, start, end, span.text)
		
	#结果为空，没有找到匹配的模式！！！
	
	
	#结合上篇《spaCy V3.0.0 专业领域中文分词问题》,改写代码：

	nlp = spacy.load('zh_core_web_trf')
	#添加用户专有名词
	proper_nouns = ['冷却水', '母管']
	nlp.tokenizer.pkuseg_update_user_dict(proper_nouns)

	doc = nlp('辅机冷却水母管压力低报警。')
	#分词结果 
	[t for t in doc]
	#[辅机, 冷却水, 母管, 压力, 低, 报警, 。]

	patterns = [{'ORTH':'母管'},{'POS':'NOUN'},{'POS':'VERB','OP':"?"},{'ORTH':'报警'}]
	matcher = Matcher(nlp.vocab)
	matcher.add("MY_PATTERN", [patterns])
	matches = matcher(doc)

	for match_id, start, end in matches:
		string_id = nlp.vocab.strings[match_id]  # Get string representation
		span = doc[start:end]  # The matched span
		print(match_id, string_id, start, end, span.text)
		
	#结果：
	#5420578651535644415 MY_PATTERN 2 6 母管压力低报警
	
首先，我们用vocab初始化匹配器。匹配器必须与其操作的文档共享同一个vocab。接着调用matcher.add()，需要传入两个参数：一个是唯一的ID用来识别匹配的是哪一个模板，另一个是需要匹配的模式列表。

matcher返回一个（match_id，start，end）三元组列表,在本例中是[('5420578651535644415'，2，6)]，即原始文档的span：doc[0:3]。match_id是字符串ID“MY_PATTERN”的哈希值。

默认情况下，matcher只返回匹配项，不执行任何其他操作，如合并实体或指定标签。但这些操作完全可以通过给add()的on_match参数传入回调函数，为每个模式分别定义。这对于编写完全自定义和特定于模式的逻辑非常有用。例如，你可能希望将一些模式合并到一个Token中，同时为其他模式类型添加实体标签。这样就不必为每个处理创建不同的匹配器。

属性| 说明
-|-
ORTH	|token的文本  <br>str
TEXT|token的文本<br>str
LENGTH|token文本的长度<br>int
 IS_ALPHA, IS_ASCII, IS_DIGIT|Token文本是否由字母、ASCII字符、数字组成<br>bool
 IS_PUNCT, IS_SPACE, IS_STOP|Token文本是否为标点、空格、停用词<br>bool
 IS_SENT_START|Token是否是句子开头<br>bool
 LIKE_NUM, LIKE_URL, LIKE_EMAIL|Token文本是否类似数字、URL、email<br>bool
 POS, TAG, MORPH, DEP, LEMMA, SHAPE|Token的注释属性<br>str
ENT_TYPE|Token的实体标签<br>str
_|Token的用户扩展属性<br>Dict[str, Any]
OP|运算符，用于确定匹配标记模式的频率<br>str

- 为什么不支持Token的所有属性？
.
spaCy并不能访问Token的所有属性，因为Matcher在Cython数据上循环的，而不是在Python对象上循环的。在matcher中，我们处理的是TokenC结构----此时我们还没有Token的实例。这意味着无法访问那些引用计算属性的属性。

### 1.2 pattern语法和属性的扩展

 token patterns除了映射单值，也可以映射一个属性字典。例如：
 
	# 匹配 "爱猫" 或 "喜欢花"
	pattern1 = [{"TEXT": {"IN": ["爱", "喜欢"]}},
				{"POS": "NOUN"}]

	# 匹配Token长度 >= 10
	pattern2 = [{"LENGTH": {">=": 10}}]

属性| 说明
-|-
IN|属性值在列表中<br>Any
NOT_IN|属性值在列表中<br>Any
==, >=, <=, >, <|属性值逻辑比较<br>Union[int, float]

### 1.3 正则表达式

在某些情况下，仅仅匹配Token和Token属性还不能满足我们的需求----–例如，您可能希望几种泵，而不必为每种泵写添加新的模式。你就可以利用正则表达式：

    pattern = [{"TEXT": {"REGEX": "[水油气]泵"}}]

而以下正则表达式表示可以匹配tag属性以“V”开头的Token：

    pattern = [{"TAG": {"REGEX": "^V"}}]
    
>重要提示：
>在使用REGEX操作符时，一定要记住：这些匹配操作是在单个Token上进行的！！！！

- 全文匹配正则表达式

如果表达式应用于多个标记，一个简单的解决方案是在doc.text上用re.finditer进行匹配，并使用匹配到的字符索引通过Doc.char_span方法来创建一个Span。但是，如果匹配的字符索引没有对应有效的Token边界，则Doc.char_span返回None。

如：对于TEXT='辅机冷却水母管压力低报警。'，要在TEXT全文匹配“母管”，匹配到的字符索引为(start=5,end=6)。
而分词是 [辅机, 冷却, 水母, 管, 压力, 低, 报警, 。]，
则doc.char_span(5,7)，返回None

- 如何将这种通过全文匹配得到的结果扩展为有效的Token序列呢？
.
一个简单的方法是对doc创建一个字符索引到Token索引的映射字典：


    chars_to_tokens = {}
    for token in doc:
		for i in range(token.idx, token.idx + len(token.text)):
			chars_to_tokens[i] = token.i
			
	#chars_to_tokens：{0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 4, 8: 4, 9: 5, 10: 6, 11: 6, 12: 7}
			
这样就可以在给定位置查找字符，并获得该字符所属的相应Token的索引。你的Span就是doc[token_start:token_end]。但这样的结果只能是最近似的Token序列。

	span = doc.char_span(5, 7)
	if span is not None:
		print("Found match:", span.text)
	else:
		start_token = chars_to_tokens.get(start)
		end_token = chars_to_tokens.get(end)
		if start_token is not None and end_token is not None:
			span = doc[start_token:end_token + 1]
			print("Found closest match:", span.text)
			
	#结果为：Found closest match: 水母管

### 1.4 数量操作符

在matcher中还可以使用“OP”数量限定关键字。能够用来定义要匹配的tokens序列，例如一个或多个标点符号，或指定token是否可选。注意，不能嵌套或限定数量范围，但可以通过给on_match参数指定回调函数来实现这些操作。

OP|说明
-|-
!|否定模式，只能0次匹配
?|可选, 匹配0或1次
+|匹配1或多次
*|匹配0到多次

### 1.5 使用通配符

虽然token属性为编写特定模式提供了许多选项，但也可以使用空字典{}作为表示任何Token的通配符。主要用于知道要匹配的内容的上下文，但对其本身却不清楚。例如，假设要从数据中提取用户的用户名。只知道被表示为“User name:{username}”。用户名本身可以包含任何字符，但不能包含空格，因此只将其看作一个Token。

    [{"ORTH": "User"}, {"ORTH": "name"}, {"ORTH": ":"}, {}]

### 1.6 patterns的验证和调试

    matcher = Matcher(nlp.vocab, validate=True)







