from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput
from main import main

with PyCallGraph(output=GraphvizOutput()):
    main()
