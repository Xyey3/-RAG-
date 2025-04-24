# 🧠 基于因果增强的RAG问答系统

本项目致力于构建一个融合因果推理机制的RAG（Retrieval-Augmented Generation）问答系统，集成因果图建模、策略强化学习检索与本地大语言模型生成模块，以提升问答系统在多跳推理、因果解释与知识抽取场景中的准确性与可解释性。

## 📌 项目特色 Highlights

- **因果图增强（Causal Graph Learning）**：基于文本构建因果结构图，挖掘关键因果链条辅助检索与生成。
- **策略检索优化（RL for Retrieval）**：采用强化学习训练策略网络，动态选择高质量证据节点进行知识增强。
- **本地部署模型（LLM on Ollama）**：使用 Ollama 平台运行如 Mistral、LLaMA 等本地大语言模型，实现私有化部署与安全可控。
- **多维度评估（RAGAS+Custom Metrics）**：融合 RAGAS 框架与因果图结构指标，进行多层次模型性能分析与可视化展示。
