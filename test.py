from datasets import load_dataset
ds = load_dataset("derek-thomas/ScienceQA", split="train", trust_remote_code=True)
print(ds[0].keys())
print(ds[0]["answer"])        # should be an int
print(ds[0]["choices"])       # should be list[str]
print(type(ds[0]["image"]))   # PIL.Image or None