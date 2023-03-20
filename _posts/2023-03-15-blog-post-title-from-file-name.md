## A Language Model's Best Friend

It is astounding to see how rapidly the world of NLP is progressing and continuously expanding the limits of what can be achieved with language comprehension. What is more exciting is how researchers are now exploring the potential of few-shot learning to advance this field. With these advancements, we can expect NLP to revolutionize how humans use language to interact with machines.
One such paper that I recently had the opportunity to read is titled ['Making Pre-Trained Language Models Better Few-shot Learners'](https://arxiv.org/abs/2012.15723). Published in ACL 2021, this work by Tianyu Gao, Adam Fisch and Danqi Chen presents an interesting approach to improve few-shot learning capabilities of pre-trained language models.

Keep reading this article to delve deeper into their findings!

---

### Outline

In this blog post we will cover the following topics:
1.	Basic Terminology
2.	Motivation and Background
3.	Techniques
4.	Findings
5.	Limitations
6.	Further Research

----

### 1. Basic Terminology

To ensure that everyone is on the same page, let's briefly cover some basic definitions (in extremely simple terms for easy understanding) before diving into the paper's details.

#### 1.1 What is few-shot learning?

According to [AI Multiple](https://research.aimultiple.com/few-shot-learning/#:~:text=Few%2Dshot%20learning%20(FSL)%2C%20also%20referred%20to%20as%20low%2Dshot%20learning%20(LSL)%20in%20few%20sources%2C%20is%20a%20type%20of%20machine%20learning%20method%20where%20the%20training%20dataset%20contains%20limited%20information.), few-shot learning is a type of machine learning method where the training dataset contains limited information.

#### 1.2 What are Pre-trained Language Models?

In simple words, a Pre-trained Language Model is an extensive neural network that has been trained on a broad (general) dataset. And as you may already know, the training phase will fix some weights based on the features it has learned. This resulting model can then be fine-tuned (or rather customized) to a specific task like sentiment classification.

#### 1.3 What is fine-tuning?

It basically just means training the language model *again*, but this time with a dataset that is more specific to the task you want to perform. When this happens, the weights of the pre-trained model get adjusted to account for the new features being learnt.

#### 1.4 What are demonstrations?

They are just labeled samples from each class that are provided as examples for in-context learning without having to perform fine-tuning.

#### 1.5 What is the T5 model?

T5 refers to [Text-To-Text Transfer Transformer](https://arxiv.org/abs/1910.10683) where the model obtains a textual input along with what task is to be performed and provides a textual output. It can be used for text summarization, translation or even calculation of similarity. But in this paper, the T5 model is used to fill in the blanks of a given input statement as shown in the figure below.

![image](/images/t5_model.png)

#### 1.6 What is a development dataset?

We all maybe familiar with a training dataset and a test dataset. But there is something also known as a development set or as some of us may have heard - a validation set. This set is mostly used to fine-tune hyperparameters such as number of hidden layers, learning rates, etc. There are algorithms like k-fold cross validation that help in splitting the data into the different datasets.

---

### 2. Motivation and Background

The GPT-3 [(Brown et al.)](https://arxiv.org/abs/2005.14165) is one of the most well-known examples of a pre-trained language model achieving few-shot learning. To accomplish this, the model used natural language prompts and task demonstrations.
Due to its large size, GPT-3 is unfortunately impractical to use. As a result, the paper by Gao et al. was born. Their group was inspired to use few-shot learning on smaller pre-trained language models like BERT and RoBERTa.

This paper was inspired by two models:
1. Third-generation Generative Pre-trained Transformer (GPT-3) by [Brown et al.](https://arxiv.org/abs/2005.14165)
    * It has about 175B parameters.
    * There is no updating of weights when the model is fine-tuned.
    * The prompts used are manually designed.
    * Demonstrations are used to provide context.

2. Pattern-Exploiting Training (PET) by [Schick and Schuetze](https://arxiv.org/abs/2009.07118v2)
    * It uses a much smaller language model with about 340M parameters.
    * The model parameters are updated during fine-tuning.
    * The prompts used are designed manually.
    * No demonstrations are used.

This paper focuses on combining the benefits of both of these models, resulting in **Better Few-shot Fine-tuning of Language Models** or as the authors like to call it - **Language Models’ Best Friend Forever (LM-BFF)**.

Here is a visualization of what this approach entails:

![image](/images/lm_bff.png)

---

### 3. Techniques

The LM-BFF mainly comprises of a trio of techniques:

1. Prompt-based fine-tuning
2. Automatic prompt generation
3. Dynamic demonstration selection

Let's look at each one in detail.

#### 3.1 Prompt-based fine-tuning

If you are wondering what a prompt is, here is the answer - a template along with the label word is called Prompt. 

Let us look at the following example to understand this better.

![image](/images/prompt_based_ft.png)

In prompt-based fine-tuning, a MASK token is introduced after the input sentence in a template, hence the downstream task becomes a **Masked Language Model** problem.
The diagram above illustrates this method using an example where an input phrase "No reason to watch" is followed by a template with the \[MASK\] token, which is essentially a blank to be filled in. This token is then passed into an MLM head and *label mapping* is performed. Label mapping simply refers to assigning a label or class to a given word. In the example above, 'great' is labeled as a 'positive' word, while 'terrible' is labeled 'negative'.

A major advantage of this method is that it does not generate any new parameters and can easily be extended to both classification and regression.

If you are new to prompt-based learning, fret not! Here is an [article](https://engineering.rappi.com/prompting-the-new-era-of-natural-language-processing-6494d828a9b9) that is a starting point and can help you get on track.

#### 3.2 Automatic prompt generation

One of the experiments that was performed by [Gao et al.](https://arxiv.org/abs/2012.15723), showed that the type of prompt given as input greatly influenced the results obtained. For example, the placement of the \[MASK\] tokens, the order of the label words and so on.

Creating prompts manually is not only time-consuming but also requires domain expertise, which may not always be accessible. Hence the team of researchers came up with a smart strategy to automatically generate these prompts.

Since prompts are a combination of templates and label words, the generation is also done for each of these components separately. Thus, we have the following methods of automatic prompt generation:

#### 3.2.1 Automatic selection of label words

Here the template is manually generated and is fixed. The image below clearly explains the process of how the best label word is selected.

![image](/images/auto_sel_of_label_words.png)

What we see here is that once the classified input from the training dataset is passed through the MLM, the **top-k** vocabulary is selected based on the conditional likelihood using the initial Language Model (RoBERTa). All possible combinations are then enumerated and to narrow down the search space further, the **top-n** assignments that maximizes zero shot accuracy on the training dataset D<sub>train</sub> are selected. (Here n, k are hyperparameters.)
These top-n label words are fine-tuned and re-ranked to find the best label using the development set D<sub>dev</sub>.

#### 3.2.2 Automatic generation of templates

Since we are automating the generation of template, the label words are created manually in this method. The following figure from the paper demonstrates this technique effectively.

![image](/images/auto_gen_of_templates.png)

The method uses a fixed set of label words M(Y) and generates a set of templates {T}.

Input sentences are taken from the training dataset D<sub>train</sub> and are passed to the T5 model. This is where all the magic happens. Since this model is already pre-trained for filling in missing spans and create blanks for the remaining words (exactly what we need to do), it can construct templates without us specifying the number of tokens beforehand. Is this magic or what?

The decoding process then involves searching for an appropriate output that aligns with all instances of the training data D<sub>train</sub>. To achieve this, a search strategy referred to as **beam search** is employed. This technique utilizes a *beam* containing the N most probable sequences at each point of time. In this experiment, the authors opted for a wide beam width of 100, which generates a good number of diverse templates.

From this set, each generated template is fine-tuned on the training data D<sub>train</sub> while the development set D<sub>dev</sub> is used to pick the best template or to find the top-k templates to use as an ensemble (for multi-prompt - that's a story for another day).

#### 3.3 Dynamic demonstration selection

To understand dynamic selection of demonstration, it would be good to first be familiar with the 'naive' method of selecting demonstrations. The latter approach was used in GPT-3’s few-shot learning wherein the input is concatenated with up to 32 randomly drawn examples from the training set. This technique worked well due to its large context size of 2048. However, RoBERTa has a smaller context size of only 512 and using this same technique was not effective. To tackle this challenge, researchers improvised and introduced a dynamic selection process for demonstrations that could adapt to these differences in models.

To implement this technique, simply choose a **single** sample at random from each class and append it to the input. (Note: the authors experimented with selecting multiple examples per class, but found no noticeable improvements in doing so.)

Now the question arises: How exactly do you find that **ONE** example from each class? The solution is quite simple - compare the input sentence with a demonstration from every class in the training dataset, rank them by similarity and choose the top 50%. From there, randomly pick one example per class. That’s it - you have successfully performed dynamic selection. (Ha! Just theoretically though.)

The figure below depicts an example of dynamically selected demonstrations from the positive and the negative classes.

![image](/images/dynamic_demo.png)

---

### 4. Findings

Before taking a look at the results obtained in this paper, let us go through the assumptions that were considered during the setup. 

- The experiment was run on 15 English tasks. This includes 8 single sentence and 7 sentence pair tasks.
- The authors experimented with the RoBERTa-large model.
   They did run the experiment with both BERT and RoBERTa and found that the latter model performed better.
- When we saw the automatic generation of prompts in section 3, you would have noticed the automatic generation of templates and selection of label words were separate. The authors compared these two techniques along with a combined automatic generation of templates and label words and noticed that automatic templates worked best and hence they continue to use this for the rest of the experiment.
- In section 3.2, we observed the automatic generation of prompts. It was evident that generating templates and selecting label words were distinct processes. The authors conducted a comparison between these two approaches and a combination of template and label word generation. They found out that automatic templates had better performance compared to other techniques, thus they utilized this for the remainder of the experiment.
- The number of training samples used was set to 16.
- The size of the training and development sets are equal. This is done so that the experiment is run on a true few-shot setting.
- The performance was measured across 5 random splits of the training and developement sets D<sub>train</sub> and D<sub>dev</sub> to obtain better estimation of variance and robust performance.

Here are the major findings we see from this research:
1. The use of prompt-based fine-tuning, whether it was manually or automatically generated, proved to be more effective than standard fine-tuning.
2. Automatically generated prompts performed better than manually generated ones.
3. Including contextual demonstrations instead of omitting them, mostly led to a significant improvement in performance.

Which simply means LM-BFF performs better than vanilla fine-tuning!

***Note:** If you are interested in implementing the LM-BFF technique, the authors have also publically provided their code [here](https://github.com/princeton-nlp/LM-BFF). Unfortunately I could not implement it due to limitations of my device, but if I do get an opportunity to do so in the future, I will definitely update my findings here.*

---

### 5. Limitations

Since there is no such thing as absolute perfection in this world (except maybe that perfect pull of mozarella from the pizza or the smooth stroke of your eyeliner on the first go), let's take a look at certain aspects of this research that can be improved.

1. Unfortunately even though the results show better performance, it still suffers from high variance just like in standard fine-tuning.
2. Although we say that the prompt is generated automatically, one part of it is always manually designed (either the template or the label word). This is still a cause for poor generalization.
3. At least for now, the preferred task is the 'fill in the blanks' type task.
4. As seen in section 3.3, the demonstrations from each class of the training dataset are concatenated to the input sentence. This would work for binary classification, but in cases where there a more than 5 classes, the context size could potentially go beyond the 512 of RoBERTa. A possible solution that could work is the use of long transformers or long-formers. 

---

### 6. Further Research

From what we have seen so far, the paper "Making Pre-trained Language Models Better Few-shot Learners" presents a promising approach to improving few-shot learning with pre-trained language models.
1. For a small number of training examples, LM-BFF outperforms standard fine-tuning by 11% on average.
2. It is important to sample similar examples while incorporating demonstrations in context.
3. RoBERTa-large achieves about 90% accuracy on most binary sentence classification tasks with only 32 examples.

In 2022, Park et al. and their team extended the LM-BFF approach and they named it [LM-BFF-MS](https://aclanthology.org/2022.acl-short.34/) (Better Few-shot Fine-tuning of Language Models with Multiple Soft demonstrations). 

**A brief overview:** Instead of simply appending only one demonstration of each class to the input (as seen in Section 3.3), the extended approach appends multiple demonstrations in each class along with automatically generated *label phrases* instead of label words. They also use the concept of a *global demonstration memory* which shares soft tokens for various inputs. Based on their findings, the LM-BFF-MS shows improved and stable performance in five NLP tasks when compared to its previous version. Unfortunately the global memory did not always work as some tasks performed better with 'local' demonstrations for specific input sentences.

Another interesting feature introduced in their paper is the development of the Next Demonstration Prediction (NDP) task. According to this, the NDP *"predicts whether positive (or negative) examples in the demonstrations are correctly matched with a positive (or negative) label word for the prompted input."* 

What do you think would be the next extension of this approach? Maybe LM-BFF-MS-FAP? Fully Automated Prompt generation? 

I hope this article piqued your interest in the field of few-shot learning in NLP. 

*Thank you for reading!*

---

### References

Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. *In Advances in Neural Information Processing Systems (NeurIPS).*

Tianyu Gao, Adam Fisch, and Danqi Chen. 2021. Making pre-trained language models better few-shot learners. *In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3816–3830, Online. Association for Computational Linguistics.*

Eunhwan Park, Donghyeon Jeon, Seonhoon Kim, Inho Kang, and Seung-Hoon Na. 2022. LM-BFF-MS: Improving Few-Shot Fine-tuning of Language Models based on Multiple Soft Demonstration Memory. *In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 310–317, Dublin, Ireland. Association for Computational Linguistics.*

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits
of transfer learning with a unified text-to-text Transformer. *The Journal of Machine Learning Research (JMLR), 21(140).*

Timo Schick and Hinrich Schutze. 2021a. Exploiting cloze questions for few-shot text classification and natural language inference. *In European Chapter of the Association for Computational Linguistics (EACL).*

Timo Schick and Hinrich Schutze. 2021b. It’s not just size that matters: Small language models are also few-shot learners. *In North American Chapter of the Association for Computational Linguistics (NAACL).*

***Disclaimer:** This blog post only provides a brief overview of the paper, and not all information is included. It is recommended to read the full paper for a better understanding of the concepts discussed here, if you found this article interesting.*

