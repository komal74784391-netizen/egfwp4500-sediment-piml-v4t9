# egfwp4500-sediment-piml-v4t9

This repository contains the reference implementation for the paper:

**"Physics-Informed Machine Learning for Reach-Scale Sediment Transport"**
James Thomas, Petr Novak, Shreya Bhattacharya.
*Water Resources Research*, 2022.

The code couples a neural transport surrogate with an explicit sediment continuity
constraint and reproduces all figures in the manuscript.

## Layout
- `src/sediment_piml/model.py` -- hybrid transport network
- `src/sediment_piml/continuity.py` -- physics constraint layer
- `scripts/train.py` -- training entry point
- `requirements.txt`

## License
BSD-3-Clause. See `LICENSE`.
