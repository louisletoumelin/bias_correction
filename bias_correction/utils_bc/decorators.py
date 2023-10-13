
def pass_if_doesnt_has_module():
    def decorator(function):
        def wrapper(*args, **kwargs):
            try:
                result = function(*args, **kwargs)
            except (NameError, ModuleNotFoundError, ImportError):
                result = None
                print(f"{function.__name__} doesn't have the appropriate modules")
            return result
        return wrapper
    return decorator


def pass_if_doesnt_have_seaborn_version():
    def decorator(function):
        def wrapper(*args, **kwargs):
            try:
                import seaborn as sns
                result = function(*args, **kwargs)
                if float(sns.__version__.split('.')[1]) < 12:
                    print(f"Seaborn version should be >= 0.12, here it is {sns.__version__}")
            except (NameError, ModuleNotFoundError, ImportError):
                result = None
                print(f"{function.__name__} doesn't have the appropriate modules")
            return result
        return wrapper
    return decorator
