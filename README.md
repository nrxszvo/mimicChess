# mimicChess

## A Neural Network Chess Bot Trained in the Style of an LLM to Predict Human-like Chess Moves

**mimicChess** is built with [Pytorch Lightning](https://lightning.ai) and utilizes the [Lichess database](https://database.lichess.org) to train a **transformer neural network** with approximately **1 billion parameters** on over **100 million chess games** between human opponents.

---

### 🔗 Demo

To learn more about how mimicChess was built and to watch it in action, visit the [mimicChess demo site](https://chessbot.michaelhorgan.me).

---

### 📦 Repository Contents

This repository contains the following:

- **A) Download and process the Lichess database**
- **B) Train mimicChess on the resulting dataset using Pytorch**

---

### 🤖 Related Projects

For the source code used to create the Lichess bot that wraps the mimicChess engine, check out the [mimicBot](https://github.com/nrxszvo/mimicBot) repository.

---

### 🔧 Code References

The transformer architecture used in this project is derived from the [Llama 3](https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/model.py) example codebase.

