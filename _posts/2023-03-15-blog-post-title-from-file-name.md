## A Language Model's Best Friend?

Due to a plugin called `jekyll-titles-from-headings` which is supported by GitHub Pages by default. The above header (in the markdown file) will be automatically used as the pages title.

It is astounding to see how rapidly the world of NLP is progressing and continuously expanding the limits of what can be achieved with language comprehension. What is more exciting is how researchers are now exploring the potential of few-shot learning to advance this field. With these advancements, we can expect NLP to revolutionize how humans use language to interact with machines.
One such paper that I recently had the opportunity to read is titled ['Making Pre-Trained Language Models Better Few-shot Learners'](https://arxiv.org/abs/2012.15723). Published in ACL 2021, this work by Tianyu Gao, Adam Fisch and Danqi Chen presents an interesting approach to improve few-shot learning capabilities of pre-trained language models.

Keep reading this article to delve deeper into their findings!

---

### Outline

In this blog post we will cover the following topics:
1.	Basic Terminology
2.	Motivation and Background
3.	Techniques
4.	Limitations
5.	Outlook
6.	Conclusion
7.	References

----

### 1. Basic Terminology

To ensure that everyone is on the same page, let's briefly cover some basic definitions before diving into the paper's details.

#### 1.1 What is few-shot learning?

According to [AI Multiple](https://research.aimultiple.com/few-shot-learning/#:~:text=Few%2Dshot%20learning%20(FSL)%2C%20also%20referred%20to%20as%20low%2Dshot%20learning%20(LSL)%20in%20few%20sources%2C%20is%20a%20type%20of%20machine%20learning%20method%20where%20the%20training%20dataset%20contains%20limited%20information.), few-shot learning is a type of machine learning method where the training dataset contains limited information.

#### 1.2 What are Pre-trained Language Models?

In simple words, a Pre-trained Language Model is an extensive neural network that has been trained on a broad (general) dataset. And as you may already know, the training phase will fix some weights based on the features it has learned. This resulting model can then be fine-tuned (or rather customized) to a specific task like sentiment classification.

#### 1.3 What is fine-tuning?

It basically just means training the language model *again*, but this time with a dataset that is more specific to the task you want to perform. When this happens, the weights of the pre-trained model get adjusted to account for the new features being learnt.

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

Here is a visualization of what this set of techniques entails:
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

#### 3.2 Automatic prompt generation

One of the experiments that was performed by Gao et al., showed that the type of prompt given as input greatly influenced the results obtained. For example, the placement of the \[MASK\] tokens, the order of the label words and so on.

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

From this set, each generated template is fine-tuned on the training data D<sub>train</sub> while the development set D<sub>dev</sub> is used to pick the best template or to find the top-k templates to use as an ensemble (for multi-prompt).

#### 3.3 Dynamic demonstration selection

---

### 4. Limitations

---

### 5. Outlook

---

### 6. Conclusion

---

### 7. References

Tianyu Gao, Adam Fisch, Danqi Chen, 2021. Making pre-trained language models better few-shot learners.

Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. *In Advances in Neural Information Processing Systems (NeurIPS).*

