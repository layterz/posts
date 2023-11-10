"use strict";(self.webpackChunkposts=self.webpackChunkposts||[]).push([[163],{8394:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>h,contentTitle:()=>a,default:()=>c,frontMatter:()=>i,metadata:()=>r,toc:()=>l});var s=n(5893),o=n(1151);const i={slug:"teaching-an-llm-how-to-read",title:"Teaching an LLM How to Read"},a=void 0,r={permalink:"/posts/teaching-an-llm-how-to-read",source:"@site/blog/2023-11-10-teaching-an-llm-how-to-read/index.md",title:"Teaching an LLM How to Read",description:"cover",date:"2023-11-10T00:00:00.000Z",formattedDate:"November 10, 2023",tags:[],readingTime:7.87,hasTruncateMarker:!0,authors:[],frontMatter:{slug:"teaching-an-llm-how-to-read",title:"Teaching an LLM How to Read"},unlisted:!1,nextItem:{title:"Generating Narratives: Writing Coherent Stories with LLMs",permalink:"/posts/generating-narratives"}},h={authorsImageUrls:[]},l=[{value:"Setting up promptx",id:"setting-up-promptx",level:2},{value:"Storing quotes",id:"storing-quotes",level:2},{value:"Generating thoughts",id:"generating-thoughts",level:2},{value:"Question prompting",id:"question-prompting",level:2},{value:"Next steps",id:"next-steps",level:2}];function d(e){const t={a:"a",admonition:"admonition",blockquote:"blockquote",code:"code",em:"em",h2:"h2",img:"img",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",...(0,o.a)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(t.p,{children:(0,s.jsx)(t.img,{alt:"cover",src:n(1410).Z+"",width:"1792",height:"1024"})}),"\n",(0,s.jsxs)(t.p,{children:[(0,s.jsx)(t.a,{href:"https://arxiv.org/pdf/2205.14704.pdf",children:"Retrieval Augmented Generation (RAG)"})," has become a standard approach to building LLM interfaces on top of specific data, but it can struggle in cases where the input doesn't overlap meaningfully with the stored embeddings."]}),"\n",(0,s.jsxs)(t.p,{children:["In this post, I want to explore a solution to this by introducing a ",(0,s.jsx)(t.em,{children:(0,s.jsx)(t.strong,{children:"thought-generation"})})," step. The process works as follows:"]}),"\n",(0,s.jsxs)(t.ol,{children:["\n",(0,s.jsx)(t.li,{children:"Text chunks are first passed to a prompt to generate thoughts about the passage."}),"\n",(0,s.jsx)(t.li,{children:"The generated thoughts are then embedded"}),"\n",(0,s.jsx)(t.li,{children:"RAG interface uses the generated thoughts as context"}),"\n"]}),"\n",(0,s.jsxs)(t.p,{children:["All tests are done using ",(0,s.jsx)(t.code,{children:"chatgpt-3.5-turbo"})," and can be run using the ",(0,s.jsx)(t.a,{href:"https://github.com/layterz/promptx/blob/main/examples/arxiv-reader/arxiv-reader.ipynb",children:"notebook"}),"."]}),"\n",(0,s.jsx)(t.h2,{id:"setting-up-promptx",children:"Setting up promptx"}),"\n",(0,s.jsxs)(t.p,{children:["We're going to use ",(0,s.jsx)(t.a,{href:"https://github.com/layterz/promptx",children:"promptx"})," to handle the LLM interaction and embeddings."]}),"\n",(0,s.jsx)(t.pre,{children:(0,s.jsx)(t.code,{className:"language-bash",children:"pip install pxx\n"})}),"\n",(0,s.jsx)(t.p,{children:"And now we need to configure the LLM we want to use."}),"\n",(0,s.jsx)(t.pre,{children:(0,s.jsx)(t.code,{className:"language-bash",children:"export PXX_DEFAULT_LLM=chatgpt\nexport PXX_OPENAI_API_KEY=...\nexport PXX_OPENAI_ORG_ID=...\n"})}),"\n",(0,s.jsxs)(t.p,{children:["To see how to use other models take a look at the ",(0,s.jsx)(t.a,{href:"https://github.com/layterz/promptx#configuration",children:"documentation"}),"."]}),"\n",(0,s.jsx)(t.p,{children:"Now, let's test that it's working."}),"\n",(0,s.jsx)(t.pre,{children:(0,s.jsx)(t.code,{className:"language-python",children:"from promptx import prompt\n\noutput = prompt('Where is the capital of France?')\nassert 'Paris' in output\n"})}),"\n",(0,s.jsx)(t.h2,{id:"storing-quotes",children:"Storing quotes"}),"\n",(0,s.jsxs)(t.p,{children:["I'm using the paper ",(0,s.jsx)(t.a,{href:"https://arxiv.org/abs/2311.03319",children:"DAIL: Data Augmentation for In-Context Learning via Self-Paraphrase"}),". It uses an ensemble approach to boost the performance of prompting generative models."]}),"\n",(0,s.jsxs)(t.p,{children:["Let's start by simply storing each sentence from the paper as a quote. When you store something in ",(0,s.jsx)(t.strong,{children:"promptx"})," it creates an embedding, which will allow us to query the quotes to augment question answering and other tasks."]}),"\n",(0,s.jsxs)(t.p,{children:["First, we need to define ",(0,s.jsx)(t.code,{children:"Quote"})," and ",(0,s.jsx)(t.code,{children:"Document"})," entities. The ",(0,s.jsx)(t.code,{children:"Entity"})," class is a thin wrapper on top of ",(0,s.jsx)(t.code,{children:"pydantic.BaseModel"}),", which allows it to be stored as an embedding."]}),"\n",(0,s.jsx)(t.pre,{children:(0,s.jsx)(t.code,{className:"language-python",children:"from promptx import Entity\n\nclass Document(Entity):\n    title: str\n    authors: list[str]\n    abstract: str\n    url: str\n\nclass Quote(Entity):\n    text: str\n    source_id: str\n    start: int\n    end: int\n"})}),"\n",(0,s.jsxs)(t.p,{children:["Now, we need to iterate over the paper and create the quote embeddings. I'm using ",(0,s.jsx)(t.code,{children:"spacy"})," to split the paper into sentences, but you can use ",(0,s.jsx)(t.code,{children:"nltk"})," or whatever method you want as long as the text is split into small enough chunks to be meaningfully embedded."]}),"\n",(0,s.jsx)(t.pre,{children:(0,s.jsx)(t.code,{className:"language-python",children:"import spacy\nfrom promptx import store\n\npaper = 'DAIL: Data Augmentation for In-Context Learning via Self-Paraphrase...'\nnlp = spacy.load('en_core_web_sm')\ndoc = nlp(paper)\nquotes = [\n    Quote(text=sent.text, source_id=paper.id, start=sent.start_char, end=sent.end_char)\n    for sent in doc.sents\n]\n\nstore(*quotes, collection='quotes')\n"})}),"\n",(0,s.jsxs)(t.p,{children:["This stores the quotes in a ",(0,s.jsx)(t.code,{children:"promptx"})," collection. Let's query that to see what quotes we have."]}),"\n",(0,s.jsx)(t.pre,{children:(0,s.jsx)(t.code,{className:"language-python",children:"from promptx import query\n\nquotes = query('how does the paper use ensembles?', collection='quotes', limit=10)\n"})}),"\n",(0,s.jsx)(t.p,{children:(0,s.jsx)(t.img,{alt:"quote-query",src:n(2651).Z+"",width:"1850",height:"309"})}),"\n",(0,s.jsxs)(t.p,{children:["The response from ",(0,s.jsx)(t.code,{children:"query"})," is a ",(0,s.jsx)(t.code,{children:"Collection"}),", which extends ",(0,s.jsx)(t.code,{children:"pd.Dataframe"})," - allowing you to use standard pandas filtering and aggregation methods."]}),"\n",(0,s.jsx)(t.pre,{children:(0,s.jsx)(t.code,{className:"language-python",children:"# remove any quotes that are too short\nquotes = quotes[quotes['text'].str.len() > 25]\n"})}),"\n",(0,s.jsx)(t.h2,{id:"generating-thoughts",children:"Generating thoughts"}),"\n",(0,s.jsxs)(t.p,{children:["Next, let's consider what we do when we read something like a paper. We don't just remember quotes, we also have thoughts and ideas. We can do the same thing with ",(0,s.jsx)(t.strong,{children:"promptx"})," by defining a ",(0,s.jsx)(t.code,{children:"Thought"})," entity and crafting prompts to illicit interesting thoughts."]}),"\n",(0,s.jsx)(t.pre,{children:(0,s.jsx)(t.code,{className:"language-python",children:"from enum import Enum\nfrom promptx import Entity\n\nclass ThoughtType(Enum):\n    question = 'question'\n    idea = 'idea'\n    comment = 'comment'\n    critique = 'critique'\n    fact = 'fact'\n\nclass Thought(Entity):\n    text: str\n    thought_type: ThoughtType\n    source_id: str\n    start: int\n    end: int\n"})}),"\n",(0,s.jsxs)(t.p,{children:["Above, we define a ",(0,s.jsx)(t.code,{children:"Thought"})," entity, but this time we're using an ",(0,s.jsx)(t.code,{children:"enum"})," which will restrict the allowed values for ",(0,s.jsx)(t.code,{children:"thought_type"}),"."]}),"\n",(0,s.jsxs)(t.p,{children:["Now, we can use this to generate thoughts while we're reading the paper. By setting ",(0,s.jsx)(t.code,{children:"output=[Thought]"})," we're telling ",(0,s.jsx)(t.strong,{children:"promptx"})," to return a list of ",(0,s.jsx)(t.code,{children:"Thought"})," objects instead of the default string output."]}),"\n",(0,s.jsx)(t.pre,{children:(0,s.jsx)(t.code,{className:"language-python",children:"from promptx import prompt\n\ndef read(doc: list[str], bs=5, limit=None):\n    thoughts = []\n    for chunk in batch(doc, bs=bs, limit=limit):\n        output = prompt(\n            '''\n            Given a passage from a document, generate a list of thoughts \n            about the passage. Don't repeat yourself!\n            ''',\n            input=dict(\n                passage=chunk,\n            ),\n            output=[Thought],\n        ).objects\n\n        thoughts += output\n    return thoughts\n"})}),"\n",(0,s.jsx)(t.admonition,{type:"info",children:(0,s.jsxs)(t.p,{children:["The output from ",(0,s.jsx)(t.code,{children:"prompt"})," is a ",(0,s.jsx)(t.code,{children:"Collection"})," just like ",(0,s.jsx)(t.code,{children:"query"}),". Above, we convert it to a list of ",(0,s.jsx)(t.code,{children:"Entity"})," objects using the ",(0,s.jsx)(t.code,{children:"objects"})," property."]})}),"\n",(0,s.jsx)(t.p,{children:"To avoid reading the document again, we can use the stored quotes."}),"\n",(0,s.jsx)(t.pre,{children:(0,s.jsx)(t.code,{className:"language-python",children:"quotes = query(collection='quotes')\nread([q.text for q in quotes.objects])\n"})}),"\n",(0,s.jsx)(t.pre,{children:(0,s.jsx)(t.code,{children:"'Data Augmentation for In-Context Learning via Self-Paraphrase',\n'ICL requires high-quality annotated demonstrations which might not be available in real-world scenarios.',\n'DAIL leverages the intuition that large language models are more familiar with the content generated by themselves.',\n'It first utilizes the language model to generate paraphrases of the test sample and employs majority voting to determine the final result based on individual predictions.',\n'Our extensive empirical evaluation shows that DAIL outperforms the standard ICL method and other ensemble-based methods in the low-resource scenario.',\n'Additionally, we explore the use of voting consistency as a confidence score of the model when the logits of predictions are inaccessible.',\n'We believe our work will stimulate further research on ICL in low-resource settings.',\n'Recently, the rapid development of large language models (LLMs) and their striking skill and knowledge have sparked significant interest in In-Context Learning (ICL).',\n'Under ICL, there is no parameter adjustment to LLMs.',\n'The model is given an instruction, task description, in-context samples, and a test case.',\n'In-Context Learning (ICL) combined with pre-trained large language models has achieved promising results on various NLP tasks.',\n'ICL requires high-quality annotated demonstrations which might not be available in real-world scenarios.',\n'DataAugmentation for In-Context Learning (DAIL) is proposed to overcome this limitation.',\n'DAIL leverages the intuition that large language models are more familiar with the content generated by themselves.',\n'It first utilizes the language model to generate paraphrases of the test sample and employs majority voting to determine the final result based on individual predictions.',\n'Our extensive empirical evaluation shows that DAIL outperforms the standard ICL method and other ensemble-based methods in the low-resource scenario.',\n'Additionally, we explore the use of voting consistency as a confidence score of the model when the logits of predictions are inaccessible.',\n'We believe our work will stimulate further research on ICL in low-resource settings.',\n'Recently, the rapid development of large language models (LLMs) (Devlin et al., 2018; Radford et al., 2019) and their striking skill and knowledge have sparked significant interest in In-Context Learning (ICL) (Brown et al., 2020).',\n'Different from other training-based paradigms, under ICL, there is no parameter adjustment to LLMs.',\n'In-Context Learning (ICL) combined with pre-trained large language models has achieved promising results on various NLP tasks.',\n'ICL requires high-quality annotated demonstrations which might not be available in real-world scenarios.',\n'Data Augmentation for In-Context Learning (DAIL) is proposed to overcome the limitation of high-quality annotated demonstrations not being available in real-world scenarios.',\n'DAIL leverages the intuition that large language models are more familiar with the content generated by themselves.',\n'It first utilizes the language model to\\ngenerate paraphrases of the test sample and\\nemploys majority voting to determine the final\\nresult based on individual predictions.',\n'Our ex-\\ntensive empirical evaluation shows that DAIL\\noutperforms the standard ICL method and other\\nensemble-based methods in the low-resource\\nscenario.',\n'Additionally, we explore the use of\\nvoting consistency as a confidence score of the\\nmodel when the logits of predictions are inac-\\ncessible.',\n'We believe our work will stimulate\\nfurther research on ICL in low-resource set-\\ntings.',\n'Recently, the rapid development of large language\\nmodels (LLMs) (Devlin et al., 2018; Radford et al.,\\n2019) and their striking skill and knowledge have\\nsparked significant interest in In-Context Learning\\n(ICL) (Brown et al., 2020).',\n'Different from other training-based paradigms, under ICL, there is no parameter adjustment to LLMs.'\n"})}),"\n",(0,s.jsxs)(t.p,{children:["This could be taken a lot further. E.g. by using RAG during thought generation, using ",(0,s.jsx)(t.a,{href:"https://arxiv.org/abs/2310.06117",children:"step-back prompts"}),", or with user defined instructions to guide the types of thoughts to generate. However, without a solid way to evaluate the usefulness, it's easy to over-engineer so let's see how it performs first."]}),"\n",(0,s.jsx)(t.h2,{id:"question-prompting",children:"Question prompting"}),"\n",(0,s.jsx)(t.pre,{children:(0,s.jsx)(t.code,{className:"language-python",children:"def qa(question, collection):\n    context = query(question, collection=collection)\n    output = prompt(\n        '''\n        Given a question and a list of quotes, answer the question.\n        ''',\n        input=dict(\n            question=question,\n            context=[c.text for c in context.objects]\n        ),\n    )\n    return output\n"})}),"\n",(0,s.jsx)(t.p,{children:"Let's try the standard RAG approach first by just using quotes."}),"\n",(0,s.jsx)(t.pre,{children:(0,s.jsx)(t.code,{className:"language-python",children:"qa('Can you explain the main thesis of the paper?', collection='quotes')\n"})}),"\n",(0,s.jsxs)(t.blockquote,{children:["\n",(0,s.jsx)(t.p,{children:"The main thesis of the paper is that the proposed method, called DAIL, out-performs other ensemble-based methods for classification. The paper also presents experiments that demonstrate the effectiveness and generalization of the proposed method on coarse-grained and fine-grained classification tasks."}),"\n"]}),"\n",(0,s.jsx)(t.p,{children:"This isn't that helpful - it is able to pick out the name (DAIL) and some aspects of the paper, but isn't able to describe the core idea behind the paper."}),"\n",(0,s.jsx)(t.p,{children:"Let's try it with the generated thoughts."}),"\n",(0,s.jsx)(t.pre,{children:(0,s.jsx)(t.code,{className:"language-python",children:"qa('What is the name of the proposed method?', collection='thoughts')\n"})}),"\n",(0,s.jsxs)(t.blockquote,{children:["\n",(0,s.jsxs)(t.p,{children:["The main thesis of the paper is that a new method called DAIL (In-Context Learning) outperforms the standard ICL method and other ensemble-based methods in low-resource scenarios. ",(0,s.jsx)(t.strong,{children:"DAIL leverages the intuition that large language models are more familiar with the content generated by themselves"}),". It first utilizes the language model to generate paraphrases of the test sample and employs majority voting to determine the final result based on individual predictions. The authors believe that their work will stimulate further research on ICL in low-resource settings."]}),"\n"]}),"\n",(0,s.jsx)(t.p,{children:"This is much better. Crucially, it's identified the main idea behind the paper."}),"\n",(0,s.jsx)(t.h2,{id:"next-steps",children:"Next steps"}),"\n",(0,s.jsxs)(t.p,{children:["I haven't tested this on any benchmarks or using any objective metrics, but it seems like a promising approach. In a future post I'll explore how to evaluate the effectiveness of the generated thoughts using ",(0,s.jsx)(t.code,{children:"promptx.evaluate()"}),"."]})]})}function c(e={}){const{wrapper:t}={...(0,o.a)(),...e.components};return t?(0,s.jsx)(t,{...e,children:(0,s.jsx)(d,{...e})}):d(e)}},1410:(e,t,n)=>{n.d(t,{Z:()=>s});const s=n.p+"assets/images/cover-deda39facab07aff72ef1018530b4955.png"},2651:(e,t,n)=>{n.d(t,{Z:()=>s});const s=n.p+"assets/images/quote-query-acbbf570ddc4e49d2ac1d44efc986e21.png"}}]);