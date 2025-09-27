<div align="center">

# 🧠 ToM-Reasoning
### *To Think or Not To Think, That is The Question for LLM Reasoning in Theory of Mind Tasks*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


*Exploring the depths of reasoning through Theory of Mind tasks*

</div>

---

## 🌟 Overview

This repository contains the official implementation of our research on **Theory of Mind (ToM) reasoning** in Large Language Models. We investigate whether LLMs should "think" before acting in complex cognitive tasks.

### ✨ Key Features

- 🎯 **Dynamic Intervention**: Adaptive reasoning control 
- ⚡ **Slow-to-Fast Reasoning**: Efficient transition from deliberate to intuitive reasoning
- 🎭 **Think-to-Match**: Step-by-step deduction-guided option matching


---

## 📁 Project Structure

```
📦 ToM-Reasoning/
├── 🔄 vllm_intervention.py    # Dynamic intervention framework
├── 🐌 S2F.py                  # Slow-to-Fast reasoning model
├── 🤔 T2M.py                  # Think-to-Match architecture
└── 📖 README.md               # You are here!
```

### 📋 File Descriptions

| File | Description | Key Features |
|------|-------------|--------------|
| `vllm_intervention.py` | 🔄 **Dynamic Intervention** | Core implementation of adaptive reasoning control |
| `S2F.py` | 🐌➡️⚡ **Slow-to-Fast Model** | Transitions from deliberate to fast reasoning |
| `T2M.py` | 🤔🎯 **Think-to-Match Model** | Step-by-step deduction-guided option matching |

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch transformers vllm
```
---

## 🔬 Models

### 🐌➡️⚡ Slow-to-Fast (S2F) 
A novel architecture that learns to transition from careful, deliberate reasoning to fast, intuitive responses as confidence increases.

### 🤔🎯 Think-to-Match (T2M)
An advanced model that explicitly models the "thinking" process before matching responses to cognitive patterns.

---


**Made with ❤️ for the AI Research Community**

</div>