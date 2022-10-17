"""Try to import all available dictionaries."""

try:
    from ._dictionary_spams import DictionarySpams
except ImportError:
    print("WARNING: Spams not installed, 'DictionarySpams' won't be available")

# try:
#     from ._dictionary_sklearn import DictionarySklearn
# except ImportError:
#     print("WARNING: Scikit-learn not installed, 'DictionarySklearn' won't be available")
