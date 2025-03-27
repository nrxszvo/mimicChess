# mimicChess
## a nueral network chess bot trained in the style of an LLM to predict human-like chess moves

mimicChess is built with [Pytorch Lightning](https://lightning.ai) and leverages the [Lichess database](https://database.lichess.org) to train a transformer neural net with approximately 1 billion parameters on over 100 million chess games between human opponents.

See the [mimicChess demo site](https://chessbot.michaelhorgan.me) to learn more about how mimicChess was built and to watch mimicChess in action.

This repository contains all source code used to A) download and process the lichess database and B) train mimicChess on the resulting dataset using Pytorch.

For source code used to create the lichess bot that wraps the mimicChess engine, see the [mimicBot](https://github.com/nrxszvo/mimicBot) repository.

Code for the transformer architecture is derived from the [Llama 3](https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/model.py) example codebase.
