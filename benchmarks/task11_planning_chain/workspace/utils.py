"""Helpers used by the pipeline."""


def load_config():
    print("utils: loading config from defaults")
    return {"workers": 4, "retries": 3}


def cleanup(resource):
    print(f"utils: cleanup {resource}")
