#spaCy V3.0 的配置系统说明(config system)

> spaCy v3.0突显了全新的transformer-based pipelines，使其准确度达到了目前最先进的水平。你可以使用任何预训练的transformer来训练自己的pipelines，甚至可以通过多任务学习在多个组件之间共享一个transformer。模型的训练 现在已经是**完全可配置**和可扩展的，你可以使用PyTorch、TensorFlow和其他框架定义自己的定制模型。新的spaCy projects系统允许你在**一个文件**中描述整个端到端工作流（ end-to-end workflows），提供从原型到产品的简单途径，并使你能够轻松地克隆和改写最佳实践项目以满足自己的项目。

SpacyV3.0引入了一个全面的、可扩展的系统来配置模型训练。一个单独的配置文件描述了模型训练的每一个细节，由于没有使用任何隐藏默认值，这样可以很容易地重现实验和跟踪更改。你可以使用spaCy提供的“quickstart”小部件或init config命令来快速开始。无需提供大量命令行参数，只要将config.cfg文件传递给spacy train命令就可以开始模型训练。训练配置文件包括训练管道的所有设置和超参数。一些设置也可以是注册的函数，你甚至可以换出和自定义这些函数，从而可以轻松实现自己的自定义模型和体系结构。

SpacyV3.0的配置系统使用了Thinc的配置系统，使得用户能够方便描述任意树形对象。这些对象可以通过使用decorator语法注册的函数调用来创建。甚至可以对所创建的函数进行版本设置，从而在不破坏向后兼容性的情况下进行改进。Thinc的配置系统使用decorator将配置系统链接到代码中的函数。

比如下面的配置文件：

    [training]
    patience = 10
    dropout = 0.2
    use_vectors = false
    
    [training.logging]
    level = "INFO"
    
    [nlp]
    # This uses the value of training.use_vectors
    use_vectors = ${training.use_vectors}
    lang = "en"
    
    
其对应的json格式就是：

	{
		"training": {
			"patience": 10,
			"dropout": 0.2,
			"use_vectors": false,
			"logging": {
				"level": "INFO"
			}
		},
		"nlp": {
			"use_vectors": false,
			"lang": "en"
		}
	}

这个config文件被分为几个部分，方括号中表示各部分的名称，如[training]。在各部分中用“=”给键赋值。这些值可以是引用其他部分中的值，引用是通过“.”以及由“$”和“{}”表示的点位符实现的。比如：${training.use_vectors}  它就表示在[training]中的 use_vectors的值(本例为false)。这对于跨组件共享设置非常有用。

这种配置格式与Python自有的configparser主要有三种不同：
1. 它使用Json格式的值。
2. 各部分的结构化表示：使用.表示法构建各部分的嵌套结构。对于[section.subsection]，则意味着把subsection放到了section中。
3. 对注册函数的引用。对于各部分(section)中以"@"开头的键(key)，其值表示一个注册函数的名称。其下面的键则表示该注册函数的参数。如果函数定义带类型提示，则这些参数值会参照定义类型进行验证。

配置文件并没有预先定义好的需要遵循的scheme，如何设置顶级[section]取决于你自己的应用。最终得到的是一个字典，其中包含可以在脚本中使用的值—无论是完整初始化好的函数，还是一些基本设置值。

中文预训练模型zh_core_web_trf 的config.cfg片段：

    [nlp]
    lang = "zh"
    pipeline = ["transformer","tagger","parser","ner","attribute_ruler"]
    disabled = []
    before_creation = null
    after_creation = null
    after_pipeline_creation = null
    batch_size = 64
    
    [nlp.tokenizer]
    @tokenizers = "spacy.zh.ChineseTokenizer"
    segmenter = "pkuseg"
    
    [components]
    
    [components.attribute_ruler]
    factory = "attribute_ruler"
    validate = false
    
    [components.ner]
    factory = "ner"
    moves = null
    update_with_oracle_cut_size = 100
    
    [components.ner.model]
    @architectures = "spacy.TransitionBasedParser.v2"
    state_type = "ner"
    extra_state_tokens = false
    hidden_width = 64
    maxout_pieces = 2
    use_upper = false
    nO = null
    
    [components.ner.model.tok2vec]
    @architectures = "spacy-transformers.TransformerListener.v1"
    grad_factor = 1.0
    pooling = {"@layers":"reduce_mean.v1"}
    upstream = "*"
    
其中：
@tokenizers = "spacy.zh.ChineseTokenizer"
@architectures = "spacy.TransitionBasedParser.v2"
pooling = {"@layers":"reduce_mean.v1"}

这些都是spaCy中预定义好的。
