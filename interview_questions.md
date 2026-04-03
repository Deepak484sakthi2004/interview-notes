# Interview Questions for Deepaksakthi Vellore Kumar
## Based on Resume + General Backend/AI Concepts

---

## Section 1: Backend Engineering (Kotlin, Spring Boot, JPA, Hibernate, jOOQ)

### General Concepts

1. What is JPA? Explain its internal working — persistence context, entity lifecycle, dirty checking, and flush modes.
2. What is JPQL? How does it differ from native SQL and Criteria API?
3. How does Hibernate work internally? Explain the session, first-level cache, second-level cache, and the proxy pattern for lazy loading.
4. JPA vs Hibernate — JPA is a specification, Hibernate is an implementation. But what does that actually mean in practice? Can you use JPA without Hibernate?
5. Why Hibernate? What are its alternatives (EclipseLink, OpenJPA, MyBatis, jOOQ)?
6. Hibernate vs jOOQ — What is the fundamental philosophical difference? When would you choose one over the other?
7. What is the N+1 query problem in Hibernate? How do you detect and fix it?
8. Explain Hibernate's dirty checking mechanism. How does it know which fields changed?
9. What are Hibernate's fetching strategies? Explain EAGER vs LAZY loading and their pitfalls.
10. What is the difference between `merge()`, `persist()`, `save()`, and `update()` in Hibernate?
11. How does transaction management work in Spring Boot? Explain `@Transactional`, propagation levels, and isolation levels.
12. What is jOOQ's approach to database access? How does its code generation work?
13. Why did SuperOps choose jOOQ over Hibernate/JPA? What are the tradeoffs?
14. Explain the Repository pattern in Spring Data JPA. How does Spring auto-generate implementations from interface method names?
15. What is connection pooling? How does HikariCP work in Spring Boot?

### Resume Deep-Dive (Backend)

16. You used Kotlin with Spring Boot — what advantages does Kotlin offer over Java for backend development? Any pain points with Spring?
17. You worked with Apache Pulsar at SuperOps. How does Pulsar differ from Kafka? Why was Pulsar chosen?
18. Explain how you designed your REST APIs at SuperOps. What was your approach to error handling, versioning, and authentication?
19. You used jOOQ at SuperOps — walk me through how you wrote a complex query (e.g., for ticket triaging or alerting). How did jOOQ's type safety help?
20. How did you handle database migrations in production at SuperOps?

---

## Section 2: AI / LLM Engineering

### ReACT Agent (SuperOps - AI Ticket First Response)

21. Walk me through the ReACT agent loop step by step. How does Reasoning + Acting work in your ticket first-response system?
22. What tools did you give the ReACT agent? How did it decide which tool to use?
23. How did you handle hallucinations in the agent's responses? What guardrails did you implement?
24. How did the agent handle multi-channel context (Slack vs Teams vs Email)? Did the prompt change per channel?
25. What was the fallback mechanism when the agent couldn't generate a confident response?
26. How did you evaluate the quality of agent-generated first responses? What metrics did you track?
27. What was the latency of the agent pipeline end-to-end? How did you optimize it?
28. How did you handle rate limits and token limits with the LLM API in a production ticket system?
29. Did you use function calling or tool-use APIs? How did you structure the function schemas?
30. How did you manage prompt versioning and A/B testing for the agent?

### Anomaly Detection (SuperOps - Intelligent Alerting)

31. What statistical models did you use for anomaly detection on asset time series data? Why not deep learning?
32. How did you define "anomaly" — point anomalies, contextual anomalies, or collective anomalies? Give examples from your system.
33. What features did you extract from the time series data? Did you use rolling statistics, decomposition, or frequency domain features?
34. How did you handle seasonality and trend in the time series data?
35. What was your alerting threshold strategy? How did you minimize false positives without missing real anomalies?
36. How did you handle different asset types with different baseline behaviors?
37. What was the data pipeline architecture — how did data flow from assets to your anomaly detection service?
38. How did you evaluate the anomaly detection system's performance? What metrics (precision, recall, F1) did you achieve?
39. How did you handle cold-start — when a new asset has no historical data?
40. Did you consider using autoencoders or LSTMs for anomaly detection? Why or why not?

### Ticket Triaging & Similar Tickets

41. How did you implement ticket classification — what model/approach did you use for auto-categorization by priority?
42. How did the similar ticket detection engine work? Did you use embeddings, TF-IDF, or something else?
43. What similarity metric did you use (cosine similarity, Euclidean, etc.)? What was the retrieval architecture?
44. How did you handle the cold-start problem when the system had few historical tickets?
45. How did you evaluate the ticket triaging accuracy? What was the baseline vs your system?

### RAG & Semantic Search

46. Explain your RAG pipeline architecture end-to-end — from document ingestion to response generation.
47. What is hybrid retrieval (dense + keyword)? How did you combine the scores from both?
48. What re-ranking strategy did you use? BM25 + cross-encoder? How did it improve results?
49. How did you chunk the 500+ documents? What chunking strategy (fixed size, semantic, recursive) and why?
50. How did you handle documents that update frequently? Did you have an incremental indexing pipeline?
51. What embedding model did you use? Why that specific one?
52. How did you evaluate retrieval quality? Did you use metrics like MRR, NDCG, or recall@k?
53. What was your strategy for handling queries that have no relevant documents in the knowledge base?
54. How did you achieve sub-200ms retrieval latency? What optimizations did you apply?
55. ChromaDB vs FAISS vs Pinecone — you've used ChromaDB and FAISS. When would you pick one over the other?

### OpenSearch (Product Search)

56. How does OpenSearch work internally? Explain the inverted index, segments, and the Lucene layer.
57. What is BM25 scoring? How does it differ from TF-IDF?
58. How did you configure the OpenSearch index for product search — analyzers, tokenizers, mappings?
59. Did you implement autocomplete/suggest functionality? How?
60. How did you handle relevance tuning? Did you use boosting, function scores, or custom scoring?
61. How would you add vector search (k-NN) to OpenSearch alongside BM25? When would you want both?

### LangChain & Prompt Engineering

62. Explain LangChain's architecture — chains, agents, tools, memory. How do they compose?
63. What is the difference between a Chain and an Agent in LangChain?
64. What is function calling in LLMs? How does it work under the hood (OpenAI's implementation)?
65. Explain prompt engineering techniques you've used — few-shot, chain-of-thought, self-consistency, ReACT.
66. How do you handle context window limits when the input exceeds the model's token limit?
67. What is LangFuse? How did you use it for LLM observability?
68. How do you evaluate LLM outputs in production? What metrics and frameworks did you use?

### CrewAI Multi-Agent System (KoworkerAI)

69. How does CrewAI work? Explain the concepts of agents, tasks, and crews.
70. Walk me through your 3-agent recruitment pipeline — how did the JD generator, candidate scorer, and outreach drafter communicate?
71. How did you handle failures — what if one agent produced bad output? Was there retry logic or human-in-the-loop?
72. What was the 12-criteria scoring rubric? How did you ensure the LLM followed it consistently?
73. How did you parse structured output from the LLM? What format (JSON, YAML) and what validation did you apply?
74. How did you integrate with the LinkedIn API? What data did you extract and what were the rate limit challenges?
75. How did you measure the "50% reduction in sourcing-to-shortlist time" and "30% improvement in conversion"?

### GPT from Scratch (PyTorch)

76. Explain multi-head self-attention from first principles. What is Q, K, V? Why do we scale by sqrt(d_k)?
77. What is positional embedding? Why do transformers need it? Did you use sinusoidal or learned embeddings?
78. Explain layer normalization — why LayerNorm and not BatchNorm in transformers?
79. What is the difference between encoder-only, decoder-only, and encoder-decoder architectures? Why did you choose decoder-only?
80. How did you handle the training loop — what loss function, optimizer, learning rate schedule?
81. What was the size of your model (parameters, layers, heads, embedding dim)? What literary corpus did you train on?
82. What was the quality of generated text? How did you evaluate it?
83. What is causal masking and why is it needed in decoder-only models?
84. Explain the residual connections and their importance in deep transformers.

### Masto AI Companion (ToToys AI)

85. How did you design the custom persona prompting? What made the AI companion feel "personalized"?
86. What was the multi-turn conversation memory layer? How did you manage context across sessions?
87. How did you handle rate-limit-aware API batching? What was the architecture?
88. What fine-tuning approach did you use for the LLM? LoRA, full fine-tuning, or prompt tuning?
89. How did you handle harmful/toxic content generation in a consumer-facing app?
90. What was the backend architecture? How did you scale Flask to handle 1000+ users?

---

## Section 3: ML Fundamentals

91. Explain the bias-variance tradeoff. How does it apply to model selection?
92. What is gradient descent? Explain SGD, Adam, and learning rate scheduling.
93. What is overfitting and how do you detect and prevent it?
94. Explain precision, recall, F1-score, and when you'd optimize for each.
95. What is cross-validation? When would you use k-fold vs stratified k-fold?
96. Explain the transformer architecture at a high level. Why did it replace RNNs and LSTMs?
97. What is transfer learning? How does fine-tuning a pre-trained model work?
98. What is the attention mechanism? Why was "Attention is All You Need" revolutionary?
99. Explain word embeddings — Word2Vec, GloVe, and contextual embeddings (BERT, GPT).
100. What is the difference between classification, regression, clustering, and ranking?

---

## Section 4: System Design & Architecture

101. How would you design a real-time anomaly detection system at scale?
102. How would you design a RAG-based customer support chatbot for a SaaS platform?
103. How would you design a ticket routing/triaging system that learns and improves over time?
104. How would you design a multi-agent LLM system with proper error handling, observability, and cost control?
105. How would you design a search system that combines keyword search and semantic search?

---

## Section 5: Infrastructure & DevOps

106. How do you containerize an ML model with Docker? What's in your Dockerfile?
107. Explain your CI/CD pipeline with GitHub Actions. What did you automate?
108. How do you handle secrets management in production?
109. How do you monitor an LLM-powered service in production?
110. How do you handle model versioning and rollback?

---

## Section 6: Behavioral / Situational

111. Tell me about a time when your anomaly detection system generated too many false positives. How did you fix it?
112. Describe a production incident you handled at SuperOps. What was the root cause and how did you resolve it?
113. How did you prioritize between multiple feature requests (alerting, ticket triaging, search)?
114. Tell me about a technical decision you made that you later realized was wrong. What did you learn?
115. How do you stay updated with the fast-evolving LLM/AI landscape?

---

*Total: 115 Questions*
*Estimated Interview Duration: 4-6 hours (full coverage) or pick sections for 1-hour focused rounds*
