"""Author-labeled gold for the interest-set key_skills phrases.

Labeled with LLM assistance and reviewed by the author, from the 270 distinct phrases
-- a small, single-labeler set, not ground truth.

Pair semantics:
  SAME   strict equivalence: a reader would call these the same skill.
  NEAR   related / hierarchical (React vs React/Next.js, LLM inference vs
         LLM serving). Either merge or split is defensible, so pairs inside
         a NEAR group that are not SAME are EXCLUDED from scoring.
  other  DIFFERENT (the default for every unlisted pair).
"""

from itertools import combinations

SAME = [
    ["LLMs", "Large Language Models", "large language models"],
    ["NLP", "Natural language processing"],
    ["Natural Language Understanding", "NLU", "Language Understanding"],
    ["PostgreSQL", "Postgres"],
    ["Go", "Golang"],
    [
        "distributed systems",
        "Distributed systems",
        "distributed systems design",
        "Distributed Systems Design",
    ],
    ["system design", "systems design"],
    [
        "RAG pipelines",
        "RAG Pipelines",
        "RAG systems",
        "RAG Pipeline Development",
        "Retrieval Augmented Generation",
    ],
    ["LLM integrations", "LLM Integrations"],
    ["infrastructure as code", "Infrastructure-as-Code"],
    ["CI/CD Pipelines", "CI/CD pipelines"],
    ["information retrieval", "Information Retrieval", "Information retrieval"],
    ["Recommender systems", "recommender systems", "recommendation systems"],
    ["computer vision", "Computer vision"],
    ["vector databases", "Vector databases", "Vector Databases"],
    ["Embedded Systems", "embedded systems"],
    ["sensor fusion", "Sensor Fusion"],
    ["agentic AI", "Agentic AI systems", "Agentic Systems", "agentic systems"],
    ["Generative AI", "generative AI"],
    ["cloud deployment", "Cloud deployment (AWS/GCP/Azure)"],
    ["deep learning frameworks", "Deep Learning frameworks"],
    ["agent evaluation", "agent evaluation frameworks"],
    ["quantization", "model quantization"],
    ["distributed training", "distributed ML training"],
    ["synthetic data", "synthetic data generation"],
    ["LLM fine-tuning", "Fine-tuning workflows"],
    ["speech recognition", "ASR models"],
]

NEAR = [
    ["C++", "C/C++"],
    ["JavaScript/TypeScript", "JavaScript", "TypeScript"],
    [
        "NLP",
        "Natural language processing",
        "NLP research",
        "Natural Language Understanding",
        "NLU",
        "Language Understanding",
        "Query understanding",
        "semantic text understanding",
        "computational linguistics",
    ],
    [
        "agentic AI",
        "Agentic AI systems",
        "Agentic Systems",
        "agentic systems",
        "agentic workflows",
        "AI agents",
        "LLM agents",
        "Agent Orchestration",
        "coding agent harnesses",
    ],
    ["Generative AI", "generative AI", "generative AI integration", "generative modeling"],
    [
        "cloud deployment",
        "Cloud deployment (AWS/GCP/Azure)",
        "cloud platforms",
        "Cloud data platforms",
        "cloud-native systems",
    ],
    [
        "Ranking systems",
        "ranking algorithms",
        "Text Ranking",
        "Learning-to-rank",
        "relevance metrics",
        "Search systems",
        "Enterprise Search",
        "information retrieval",
        "Information Retrieval",
        "Information retrieval",
    ],
    [
        "LLM inference",
        "LLM inference optimization",
        "inference optimization",
        "LLM serving",
        "model serving",
        "GPU serving",
        "Distributed inference systems",
        "inference systems",
        "Real-time inference optimization",
        "LLM performance optimization",
        "inference performance metrics",
        "LLM deployment",
        "ML deployment",
    ],
    ["agent evaluation", "agent evaluation frameworks", "evaluation frameworks"],
    [
        "LLM systems design",
        "LLM Systems",
        "ML systems design",
        "AI systems architecture",
        "system design",
        "systems design",
        "AI product engineering",
    ],
    [
        "On-device ML deployment",
        "On-device deployment/model compression",
        "edge AI optimization",
        "quantization",
        "model quantization",
        "model optimization",
    ],
    [
        "distributed training",
        "distributed ML training",
        "distributed training infrastructure",
        "LLM training systems",
        "tensor parallelism",
        "parallelism strategies",
        "PyTorch FSDP",
        "Horovod",
        "GPU training",
        "training optimization",
    ],
    ["hardware acceleration", "AI acceleration hardware", "Model acceleration"],
    [
        "GPU optimization",
        "GPU cluster optimization",
        "GPU fleet management",
        "GPU training",
        "GPU architecture",
        "CUDA",
        "CUDA kernels",
        "Triton kernels",
        "Low-level kernel optimization",
        "NVIDIA software/hardware stack",
        "TensorRT",
        "HPC optimization",
    ],
    [
        "machine learning",
        "Machine Learning Algorithms",
        "deep learning",
        "deep learning theory",
        "Machine learning classification/clustering/anomaly detection",
        "feature engineering",
    ],
    [
        "production deployment",
        "production systems",
        "ML pipeline deployment",
        "Production ML systems",
        "machine learning infrastructure",
        "AI infrastructure",
    ],
    ["REST APIs", "Web APIs", "APIs", "REST/gRPC APIs"],
    ["SQL", "SQL/large-scale data handling"],
    [
        "Spark",
        "Spark/Hive",
        "Distributed data frameworks (Spark/Ray/Daft)",
        "large-scale data processing",
    ],
    ["Data pipelines", "realtime ingestion pipelines", "ETL", "data infrastructure"],
    [
        "data collection and analysis",
        "data collection project management",
        "data mining",
        "statistical analysis",
    ],
    ["React", "React/Next.js", "React/TypeScript"],
    [
        "system reliability and scaling",
        "SLIs/SLOs",
        "systems monitoring",
        "infrastructure monitoring",
        "production monitoring",
        "LLM observability",
        "telemetry analysis",
        "OpenTelemetry",
    ],
    [
        "compiler design",
        "compiler engineering",
        "Machine learning compilers",
        "compiler toolchains (LLVM/GNU)",
    ],
    ["Document understanding", "document processing", "unstructured data processing"],
    ["performance profiling", "performance optimization", "runtime optimization"],
    ["autonomous driving", "robotic systems", "physical AI"],
    [
        "LLM fine-tuning",
        "Fine-tuning workflows",
        "instruction tuning",
        "LLM domain adaptation",
        "Domain-adapted language models",
    ],
    ["Annotation systems", "Data labeling platforms", "language data annotation"],
    ["speech recognition", "ASR models", "audio signal processing", "Signal Processing"],
    ["web development", "full-stack web development", "web applications"],
    ["mobile development", "iOS development"],
    # Added after the first run's false-positive list showed these were
    # hierarchy, not errors (gold amended once, post hoc -- see README).
    ["PyTorch", "PyTorch FSDP"],
    [
        "deep learning",
        "Deep Learning frameworks",
        "deep learning frameworks",
        "deep learning theory",
    ],
    [
        "JavaScript/TypeScript",
        "JavaScript",
        "TypeScript",
        "React/TypeScript",
        "React",
        "React/Next.js",
    ],
    ["ML deployment", "ML pipeline deployment", "On-device ML deployment", "production deployment"],
    ["SQL/large-scale data handling", "large-scale data processing", "Spark/Hive"],
    ["AWS", "AWS Neuron", "Trainium/AWS"],
    ["CI/CD Pipelines", "CI/CD pipelines", "continuous integration"],
    [
        "RAG pipelines",
        "RAG Pipelines",
        "RAG systems",
        "RAG Pipeline Development",
        "Retrieval Augmented Generation",
        "RAG/Graph-RAG",
        "LightRAG",
    ],
    [
        "LLMs",
        "Large Language Models",
        "large language models",
        "multimodal large language models",
        "transformer models",
        "model pretraining",
        "LLM pre-training data curation",
    ],
]


def _pairs(groups):
    out = set()
    for g in groups:
        for a, b in combinations(sorted(set(g)), 2):
            out.add((a, b))
    return out


SAME_PAIRS = _pairs(SAME)
IGNORED_PAIRS = _pairs(NEAR) - SAME_PAIRS


def label(a: str, b: str) -> str | None:
    """'same' | 'diff' | None (excluded)."""
    key = tuple(sorted((a, b)))
    if key in SAME_PAIRS:
        return "same"
    if key in IGNORED_PAIRS:
        return None
    return "diff"
