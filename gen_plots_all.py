import glob
import os
ROOT = "results"
models = glob.glob("{}/*".format(ROOT))
for model in models:
    print(model)
    results = glob.glob("{}/*".format(model))
    print(results)
    for result in results:
        print(result)
        os.system("python make_plots.py -r {} -t 100".format(result))

print("DONE")