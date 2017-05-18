# ADIOS: Architectures Deep In Output Space

ADIOS is implemented as a thin wrapper around Keras' `Model` model (i.e., multiple-input multiple-output deep architecture) by adding the adaptive thresholding functionality as described in the paper.
`adios.utils.assemble.assemble` helper function provides and handy way to construct ADIOS and MLP models from config dictionaries.
Configs can be loaded from  `adios.docs.configs.adios.yaml`.

You can run example scripts are given in `adios_train.py`.

**Note:** `tensorflow version` must be `1.1.0`.


### Requirements
- `gensim`
- `numpy`
- `tensorflow==1.1.0`
- `Keras==2.0.4`
- `scikit-learn`
- `h5py`
- `argparse`
- `PyYAML`

### How to use

- **install virtualenv**
```bash
$ pip install virtualenv

# cd project
$ cd /your/path/to/project/adios

# make env using python 2.7(system version)
$ virtualenv env --python=python

# you will see a new dir named “env”; the env by followed cmd:
$ source env/bin/activate
```

- **The requirements can be installed via `pip` as follows:**

```bash
$ pip install -r requirements.txt
```


### License

MIT (for details, please refer to [LICENSE](https://github.com/alshedivat/adios/blob/master/LICENSE))

Copyright (c) 2016 Moustapha Cisse, Maruan Al-Shedivat, Samy Bengio
