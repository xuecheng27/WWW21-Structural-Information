import numpy as np 

from Q5_Extensibility.extensibility import ExtensibilityTest


if __name__ == "__main__":
    for i in range(5):
        t = ExtensibilityTest('datasets/synthetic/K1000.gml', np.linspace(1, 20, 20))
        t.run()
        t = ExtensibilityTest('datasets/synthetic/R1000.gml', np.linspace(1, 20, 20))
        t.run()
        t = ExtensibilityTest('datasets/zachary.gml', np.linspace(1, 20, 20))
        t.run()
        t = ExtensibilityTest('datasets/dolphins.gml', np.linspace(1, 20, 20))
        t.run()
        t = ExtensibilityTest('datasets/jazz.gml', np.linspace(1, 20, 20))
        t.run()