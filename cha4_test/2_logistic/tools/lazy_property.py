from functools import wraps

def lazy_property(function):
    attr = '_lazy_' + function.__name__

    @property
    @wraps(function)
    def _lazy(self):
        if not hasattr(self, attr):
            setattr(self, attr, function(self))
        return getattr(self, attr)
    return _lazy
