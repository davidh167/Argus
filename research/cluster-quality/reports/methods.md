# Cluster quality vs author-labeled gold

270 distinct interest-set phrases; 56 gold SAME pairs; 518 NEAR pairs excluded.
Gold was labeled with LLM assistance and reviewed by the author -- treat small differences as noise.

| method | clusters | precision | recall | F1 | FP | FN |
|---|---|---|---|---|---|---|
| greedy centroid @0.65 | 110 | 0.091 | 0.768 | 0.163 | 427 | 13 |
| greedy centroid + normalize @0.65 | 110 | 0.084 | 0.714 | 0.150 | 437 | 16 |
| greedy centroid @0.7 | 145 | 0.126 | 0.750 | 0.215 | 292 | 14 |
| greedy centroid + normalize @0.7 | 145 | 0.129 | 0.750 | 0.220 | 284 | 14 |
| greedy centroid @0.75 | 170 | 0.158 | 0.732 | 0.260 | 218 | 15 |
| greedy centroid + normalize @0.75 | 172 | 0.176 | 0.732 | 0.284 | 192 | 15 |
| greedy centroid @0.8 | 201 | 0.562 | 0.732 | 0.636 | 32 | 15 |
| greedy centroid + normalize @0.8 | 201 | 0.562 | 0.732 | 0.636 | 32 | 15 |
| greedy centroid @0.85 | 222 | 0.857 | 0.536 | 0.659 | 5 | 26 |
| greedy centroid + normalize @0.85 | 222 | 0.857 | 0.536 | 0.659 | 5 | 26 |
| agglomerative avg-link @0.65 | 130 | 0.152 | 0.732 | 0.252 | 229 | 15 |
| agglomerative avg-link + normalize @0.65 | 128 | 0.134 | 0.732 | 0.227 | 264 | 15 |
| agglomerative avg-link @0.7 | 156 | 0.212 | 0.732 | 0.329 | 152 | 15 |
| agglomerative avg-link + normalize @0.7 | 157 | 0.252 | 0.732 | 0.374 | 122 | 15 |
| agglomerative avg-link @0.75 | 184 | 0.465 | 0.714 | 0.563 | 46 | 16 |
| agglomerative avg-link + normalize @0.75 | 186 | 0.482 | 0.714 | 0.576 | 43 | 16 |
| agglomerative avg-link @0.8 | 203 | 0.784 | 0.714 | 0.748 | 11 | 16 |
| agglomerative avg-link + normalize @0.8 | 203 | 0.784 | 0.714 | 0.748 | 11 | 16 |
| agglomerative avg-link @0.85 | 223 | 0.967 | 0.518 | 0.674 | 1 | 27 |
| agglomerative avg-link + normalize @0.85 | 223 | 0.967 | 0.518 | 0.674 | 1 | 27 |
| agglomerative + normalize + acronyms @0.75 | 182 | 0.417 | 0.804 | 0.549 | 63 | 11 |
| agglomerative + normalize + acronyms @0.8 | 199 | 0.703 | 0.804 | 0.750 | 19 | 11 |
| agglomerative + normalize + acronyms @0.85 | 219 | 0.917 | 0.589 | 0.717 | 3 | 23 |
| agglomerative on LLM-canonical names @0.75 | 169 | 0.164 | 0.768 | 0.270 | 219 | 13 |
| agglomerative on raw+canonical concat @0.75 | 183 | 0.228 | 0.679 | 0.341 | 129 | 18 |
| agglomerative on LLM-canonical names @0.8 | 197 | 0.188 | 0.679 | 0.295 | 164 | 18 |
| agglomerative on raw+canonical concat @0.8 | 204 | 0.552 | 0.661 | 0.602 | 30 | 19 |
| agglomerative on LLM-canonical names @0.85 | 213 | 0.423 | 0.589 | 0.493 | 45 | 23 |
| agglomerative on raw+canonical concat @0.85 | 226 | 0.853 | 0.518 | 0.644 | 5 | 27 |
| agglomerative on LLM-canonical names @0.9 | 230 | 0.762 | 0.571 | 0.653 | 10 | 24 |
| agglomerative on raw+canonical concat @0.9 | 237 | 0.931 | 0.482 | 0.635 | 2 | 29 |

Best by F1: **agglomerative + normalize + acronyms @0.8** (0.750)

False positives (merged but not SAME), best method:
- AI infrastructure ~ AI systems architecture
- GPU cluster optimization ~ Go
- GPU optimization ~ Go
- LLM Systems ~ LLM training systems
- LLM cost optimization ~ LLM performance optimization
- LLM deployment ~ LLMs
- LLM deployment ~ Large Language Models
- LLM deployment ~ large language models
- LLM deployment ~ multimodal large language models
- LLM serving ~ LLMs
- LLM serving ~ Large Language Models
- LLM serving ~ large language models
- LLM serving ~ multimodal large language models
- LLM systems design ~ LLM training systems
- ML deployment ~ on-device ML deployment
- ML pipeline deployment ~ on-device ML deployment
- ML systems design ~ Production ML systems
- Machine Learning Algorithms ~ Machine learning compilers
- Machine learning compilers ~ machine learning

False negatives (SAME but split), best method:
- ASR models / speech recognition
- Fine-tuning workflows / LLM fine-tuning
- Go / Golang
- RAG Pipeline Development / RAG systems
- RAG Pipeline Development / Retrieval Augmented Generation
- RAG Pipelines / RAG systems
- RAG Pipelines / Retrieval Augmented Generation
- RAG pipelines / RAG systems
- RAG pipelines / Retrieval Augmented Generation
- RAG systems / Retrieval Augmented Generation
- agent evaluation / agent evaluation frameworks
