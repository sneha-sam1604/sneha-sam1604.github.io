## A Language Model's Best Friend?

Due to a plugin called `jekyll-titles-from-headings` which is supported by GitHub Pages by default. The above header (in the markdown file) will be automatically used as the pages title.

It is astounding to see how rapidly the world of NLP is progressing and continuously expanding the limits of what can be achieved with language comprehension. What is more exciting is how researchers are now exploring the potential of few-shot learning to advance this field. With these advancements, we can expect NLP to revolutionize how humans use language to interact with machines.
One such paper that I recently had the opportunity to read is titled ['Making Pre-Trained Language Models Better Few-shot Learners'](https://arxiv.org/abs/2012.15723). Published in ACL 2021, this work by Tianyu Gao, Adam Fisch and Danqi Chen presents an interesting approach to improve few-shot learning capabilities of pre-trained language models.

Keep reading this article to delve deeper into their findings!

---

### Outline

In this blog post we will cover the following topics:
1.	Basic Terminology
2.	Motivation
3.	Techniques
4.	Limitations
5.	Outlook
6.	Conclusion
7.	References

----

### 1. Basic Terminology

To ensure that everyone is on the same page, let's briefly cover some basic definitions before diving into the paper's details.

#### What is few-shot learning?

According to [AI Multiple](https://research.aimultiple.com/few-shot-learning/#:~:text=Few%2Dshot%20learning%20(FSL)%2C%20also%20referred%20to%20as%20low%2Dshot%20learning%20(LSL)%20in%20few%20sources%2C%20is%20a%20type%20of%20machine%20learning%20method%20where%20the%20training%20dataset%20contains%20limited%20information.), few-shot learning is a type of machine learning method where the training dataset contains limited information.

#### What are Pre-trained Language Models?

In simple words, a Pre-trained Language Model is an extensive neural network that has been trained on a broad (general) dataset. And as you may already know, the training phase will fix some weights based on the features it has learned. This resulting model can then be fine-tuned (or rather customized) to a specific task like sentiment classification.

Now, what is fine-tuning you may ask? 
It simply means training the language model *again*, but this time with a dataset that is more specific to the task you want to perform. When this happens, the weights of the pre-trained model get adjusted to account for the new features being learnt.

---

### 2. Motivation

The [GPT-3](https://arxiv.org/abs/2005.14165) is one of the most well-known examples of a pre-trained language model achieving few-shot learning. To accomplish this, the model used natural language prompts and task demonstrations.
Unfortunately, it is not very realistic to use GPT-3 due to its large size. And hence the paper by Gao et al. was born. Their group was inspired to implement few-shot learning on smaller pre-trained language models like RoBERTa.

---

### 3. Techniques

---

### 4. Limitations

---

### 5. Outlook

---

### 6. Conclusion

---

![image](/images/lm_bff.png)

---

### 7. References

https://research.aimultiple.com/few-shot-learning/

Gao et al., 2021. Making pre-trained language models better few-shot learners


