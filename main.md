# Intent Detection - TIFIN - AI Assignment  
#### By Naman Omar

---

####  Framing as a Machine Learning Task
Intent detection is a **multi-class text classification** problem in Natural Language Processing


#### Comparison of Approaches

| Approach                              | Description                                                    | Pros                                                                 | Cons                                                                  |
|--------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------------|-----------------------------------------------------------------------|
| **BoW / TF-IDF + Classifiers**       | Sparse vectorization + models like SVM, Logistic Regression    | - Simple & fast<br>- Easy to interpret<br>- Low resource usage       | - Ignores context<br>- Poor generalization<br>- Limited semantic understanding |
| **Word Embeddings + Classifiers**    | Use pretrained embeddings (e.g., Word2Vec, GloVe)              | - Captures semantic similarity<br>- Lightweight<br>- Easy integration | - Loses word order & context<br>- Aggregation can be naive           |
| **Deep Learning (LSTM / CNN)**       | Sequence models to learn from word order and dependencies      | - Learns patterns over time<br>- Better than traditional ML          | - Requires more data<br>- Slower to train<br>- Needs tuning          |
| **Transformers (BERT / DistilBERT)** | Fine-tuned transformer models for classification               | - Best performance<br>- Context-aware<br>- Little feature engineering | - High compute requirement<br>- Slower inference                     |


