# Embedding benchmark

- phrases: 6985 distinct; repeats per scenario: 3
- machine: macOS-27.0.1-arm64-arm-64bit, Python 3.12.2

| scenario | median s | min s | max s |
|---|---|---|---|
| ollama cold | 44.87 | 44.54 | 46.27 |
| native cold, no persistence | 4.78 | 4.77 | 10.70 |
| native cold, persisting | 5.40 | 5.39 | 5.60 |
| native restart, persistence | 0.15 | 0.15 | 0.16 |
| native restart, no persistence | 4.78 | 4.77 | 4.78 |
| in-process warm | 0.01 | 0.01 | 0.01 |
