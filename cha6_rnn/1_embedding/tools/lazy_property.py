import functools
def lazy_property(function):
    attr_name = '_lazy_' + function.__name__

    @property
    @functools.wraps(function)
    def wraper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, function(self))
        return getattr(self, attr_name)

    return wraper