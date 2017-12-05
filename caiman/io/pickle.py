import pickle


def save_object(obj, filename):
    """Save to a pickle file."""
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    """Read a pickle file."""
    with open(filename, 'rb') as input_obj:
        obj = pickle.load(input_obj)
    return obj
