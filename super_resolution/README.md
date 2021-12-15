```
pip install scikit-image
```

error
```
AttributeError: module 'keras.utils' has no attribute 'Sequence'
```
keras.utils.Sequenc -> keras.utils.all_utils.Sequence
class DataGenerator(keras.utils.all_utils.Sequence):  # keras.utils.Sequence