#spaCy V3.0 Tranformer模型WordEmbedding数据详解

对于Tranformer中文模型，官网给出的概要是：

项目|说明
-|-
LANGUAGE|ZH Chinese
TYPE|CORE Vocabulary, syntax, entities, vectors
GENRE|WEB written text (blogs, news, comments)
SIZE|TRF 398 MB
COMPONENTS |transformer, tagger, parser, ner, attribute_ruler
PIPELINE |transformer, tagger, parser, ner, attribute_ruler
VECTORS |0 keys, 0 unique vectors (0 dimensions)
SOURCES |OntoNotes 5
AUTHOR|Explosion
LICENSE|MIT

可以看出，其并不包含预定义的词典和词向量。新版本的pipelines如下：
![](spaCy_pipelines.png)

	nlp = spacy.load('zh_core_web_trf')

	def get_user_dict(f):
		ul = []
		with open(f, 'r', encoding='utf-8') as f:
			for l in f.readlines():
				ul.append(l.split('\t')[0])
		return ul

	udict = get_user_dict('c:/user_dict.txt')
	nlp.tokenizer.pkuseg_update_user_dict(udict)
	doc = nlp('调整给水，注意给水流量与蒸汽流量相匹配，注意过热度，保证主蒸汽温度不超限。')

	[t for t in doc]

	===============================================================
	[调整, 给水, ，, 注意, 给水流量, 与, 蒸汽, 流量, 相匹配, ，, 注意, 过热度, ，, 保证, 主蒸汽, 温度, 不, 超限, 。]
	
	
	doc._.trf_data
	===============================================================
		TransformerData(
		wordpieces=WordpieceBatch(
			strings=[['[CLS]', '调', '整', '给', '水', '，', '注', '意', '给', '水', '流', '量', '与', '蒸', '汽', '流', '量', '相', '匹', '配', '，', '注', '意', '过', '热', '度', '，', '保', '证', '主', '蒸', '汽', '温', '度', '不', '超', '限', '。', '[SEP]']], 
			input_ids=array([[ 101, 6444, 3146, 5314, 3717, 8024, 3800, 2692, 5314, 3717, 3837,
			7030,  680, 5892, 3749, 3837, 7030, 4685, 1276, 6981, 8024, 3800,
			2692, 6814, 4178, 2428, 8024,  924, 6395,  712, 5892, 3749, 3946,
			2428,  679, 6631, 7361,  511,  102]], dtype=int64), 
			attention_mask=array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=int64), 
			lengths=[39], 
			token_type_ids=array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int64)), 
			tensors=[array([[[(1, 39, 768]]], dtype=float32), array([[(1, 768)]], dtype=float32)], 
			align=Ragged(data=array([[ 1],[ 2],...[37]], dtype=int32), 
			lengths=array([2, 2, 1, 2, 4, 1, 2, 2, 3, 1, 2, 3, 1, 2, 3, 2, 1, 2, 1], dtype=int32), 
			data_shape=(-1,), 
			cumsums=None))
从中可以看出：
1. strings：是按字分隔的(bert)
2. input_ids: 是各字在Transformer的vocab.txt中的索引值
3. lengths=[39]：是给原始文本加上'[CLS]'，'[SEP]'后的字数
4. tensors分两部分：
  - doc._.trf_data.tensors[0] ：39个字符 * 768维向量
  - doc._.trf_data.tensors[1] ：是 1 * 768的矩阵(本例为向量)，表示整个句子
5. lengths=array([2, 2, 1, 2, 4, 1, 2, 2, 3, 1, 2, 3, 1, 2, 3, 2, 1, 2, 1]:表示各分词的字数

其中的tensors能够单独调用。

- input_ids (torch.LongTensor of shape (batch_size, sequence_length)) –

    	Indices of input sequence tokens in the vocabulary.
    	
    	Indices can be obtained using BertTokenizer. 
    	See transformers.PreTrainedTokenizer.encode() and
    	transformers.PreTrainedTokenizer.__call__() for details.

- attention_mask (torch.FloatTensor of shape (batch_size, sequence_length), optional) –

    Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]:
    
    - 1 for tokens that are not masked,
    
    - 0 for tokens that are masked.

- token_type_ids (torch.LongTensor of shape (batch_size, sequence_length), optional) –

    Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]:
    
    - 0 corresponds to a sentence A token,
    
    - 1 corresponds to a sentence B token.